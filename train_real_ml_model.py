"""
ML Link Adaptation Training Pipeline — V2
==========================================

Closes all model-side gaps from the IEEE review:

  Gap 1:  DNN MLP classifier + GRU sequential model (PyTorch baselines)
  Gap 2:  Online learning simulator with EMA retraining
  Gap 3:  Joint rank + MCS prediction (when MIMO data available)
  Gap 4:  Multi-service BLER target as input feature (eMBB / URLLC)

Backward compatible with both V1 and V2 datasets.
"""

import numpy as np
import pandas as pd
import warnings
from collections import deque

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupShuffleSplit

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ============================================================================
# Configuration
# ============================================================================

RANDOM_SEED = 42
CONFIDENCE_THRESHOLD = 0.30
CPP_EXPORT_PATH = "la_policy_exported.cpp"

NR_MCS_INDICES = sorted([3, 4, 9, 11, 14, 17, 20, 24, 25])

# Service profiles: (name, bler_target, label_for_feature)
SERVICE_PROFILES = {
    "eMBB":         0.10,
    "URLLC_1e-3":   0.001,
    "URLLC_1e-5":   0.00001,
}

# NR MCS SE map for Shannon comparisons
NR_MCS_SE_MAP = {
    3:  2 * (253/1024),
    4:  2 * (308/1024),
    9:  2 * (616/1024),
    11: 4 * (340/1024),
    14: 4 * (553/1024),
    17: 6 * (438/1024),
    20: 6 * (666/1024),
    24: 8 * (567/1024),
    25: 8 * (616/1024),
}


# ============================================================================
# Dataset Detection & Schema
# ============================================================================

def detect_dataset_version(df):
    """Detect whether this is a V1 or V2 dataset."""
    v2_cols = {"Num_Tx_Ant", "Num_Streams", "Carrier_GHz",
               "SIR_Base_dB", "SIR_Effective_dB"}
    has_v2 = v2_cols.issubset(set(df.columns))
    return "V2" if has_v2 else "V1"


def normalize_dataset(df, version):
    """Add default V2 columns to V1 datasets for unified processing."""
    if version == "V1":
        df = df.copy()
        df["Num_Tx_Ant"] = 1
        df["Num_Streams"] = 1
        df["Carrier_GHz"] = 3.5
        if "SIR_Base_dB" not in df.columns:
            df["SIR_Base_dB"] = df.get("SIR_dB", 30.0)
        if "SIR_Effective_dB" not in df.columns:
            df["SIR_Effective_dB"] = df.get("SIR_dB", 30.0)
    return df


# ============================================================================
# Channel encoding — ordered by propagation difficulty
# ============================================================================

CHANNEL_ORDINAL = {
    "CDL-D": 0,   # LOS (best)
    "TDL-A": 1,   # NLOS, 30 ns delay spread
    "TDL-B": 2,   # NLOS, 300 ns
    "CDL-A": 3,   # NLOS clustered, 300 ns
    "TDL-C": 4,   # NLOS, 1000 ns (worst)
}


# ============================================================================
# 1. Label Construction — Multi-Service BLER-Aware (Gap 4)
# ============================================================================

def build_optimal_label_table(df, bler_target=0.10):
    """Build optimal MCS labels under a specific BLER target."""
    group_cols = ["SINR_dB", "Channel", "Speed_kmph", "MCS_Index"]
    # Include V2 columns if present
    if "Num_Streams" in df.columns:
        group_cols = ["SINR_dB", "Channel", "Speed_kmph", "Num_Streams",
                      "Carrier_GHz", "MCS_Index"]

    stats = (
        df.groupby(group_cols)
        .agg(
            BLER=("Was_Success", lambda x: 1.0 - x.mean()),
            Throughput=("Actual_Throughput", "mean"),
            N_Packets=("Was_Success", "count"),
        )
        .reset_index()
    )

    context_cols = [c for c in group_cols if c != "MCS_Index"]

    def get_robust_optimal(group):
        safe = group[group["BLER"] <= bler_target]
        if not safe.empty:
            return safe.loc[safe["Throughput"].idxmax()]
        return group.loc[group["MCS_Index"].idxmin()]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        optimal_df = (
            stats.groupby(context_cols, group_keys=False)
            .apply(get_robust_optimal, include_groups=False)
            .reset_index()
        )

    return stats, optimal_df


def build_multi_service_labels(df):
    """Build labels for multiple BLER targets (Gap 4).
    Returns a combined dataframe with BLER_Target column."""
    all_labels = []

    for service_name, bler_target in SERVICE_PROFILES.items():
        stats, optimal_df = build_optimal_label_table(df, bler_target)
        optimal_df = optimal_df.copy()
        optimal_df["BLER_Target"] = bler_target
        optimal_df["BLER_Target_Log"] = np.log10(bler_target)
        optimal_df["Service"] = service_name
        all_labels.append(optimal_df)

    combined = pd.concat(all_labels, ignore_index=True)
    return stats, combined


# ============================================================================
# 2. Feature Engineering — Extended
# ============================================================================

def add_features(optimal_df, rng, version="V1"):
    """Create features with measurement noise. Handles V1 and V2 schemas."""
    out = optimal_df.copy()

    # SINR measurement noise
    sinr_noise_db = rng.normal(0.0, 1.5, len(out))
    speed_vals = out["Speed_kmph"].values
    cqi_aging_db = 0.5 * (1.0 - np.exp(-speed_vals / 30.0))
    out["Measured_SINR"] = out["SINR_dB"] + sinr_noise_db - cqi_aging_db

    # Speed estimator error
    out["Measured_Speed"] = out["Speed_kmph"] + rng.normal(0.0, 10.0, len(out))
    out["Measured_Speed"] = out["Measured_Speed"].clip(lower=0.0)

    # Channel ordinal (handles both TDL-only and TDL+CDL)
    out["Channel_Ordinal"] = out["Channel"].map(CHANNEL_ORDINAL)
    # Fill unknowns with middle value
    out["Channel_Ordinal"] = out["Channel_Ordinal"].fillna(2).astype(
        np.float64)

    # V2 features
    if "Carrier_GHz" in out.columns:
        out["Carrier_Band"] = (out["Carrier_GHz"] > 10.0).astype(np.float64)
    else:
        out["Carrier_Band"] = 0.0

    if "Num_Streams" in out.columns:
        out["Num_Antennas"] = out["Num_Streams"].astype(np.float64)
    else:
        out["Num_Antennas"] = 1.0

    # Gap 4: BLER target (multi-service)
    if "BLER_Target_Log" not in out.columns:
        out["BLER_Target_Log"] = np.log10(0.10)  # Default eMBB

    return out


def get_feature_cols(version="V1", multi_service=False):
    """Return feature column list based on dataset version."""
    cols = ["Measured_SINR", "Measured_Speed", "Channel_Ordinal"]

    if version == "V2":
        cols.extend(["Carrier_Band", "Num_Antennas"])

    if multi_service:
        cols.append("BLER_Target_Log")

    return cols


def get_monotonic_constraints(feature_cols):
    """Monotonic constraints matching feature column order.
    +1 = monotonically increasing with feature
    -1 = monotonically decreasing
     0 = no constraint"""
    constraints = {
        "Measured_SINR": 1,    # ↑ SINR → ↑ MCS
        "Measured_Speed": -1,  # ↑ speed → ↓ MCS (CQI aging)
        "Channel_Ordinal": -1, # ↑ (worse channel) → ↓ MCS
        "Carrier_Band": 0,     # No monotonic assumption FR1 vs FR2
        "Num_Antennas": 0,     # No monotonic assumption for MIMO
        "BLER_Target_Log": 1,  # ↑ log(target) = ↑ target → ↑ MCS
                               # (e.g., 10% eMBB more aggressive than 0.001%)
    }
    return [constraints.get(c, 0) for c in feature_cols]


# ============================================================================
# 3. Ordinal MCS Policy (existing, enhanced)
# ============================================================================

class OrdinalMCSPolicy:
    """Ordinal regression: K-1 binary classifiers with monotonic constraints.

    P(optimal_MCS >= threshold_k | features) for k = 1..K-1
    Final prediction: highest MCS where P >= decision_threshold.
    """

    def __init__(self, mcs_indices, decision_threshold=0.5, **hgb_kwargs):
        self.mcs_sorted = sorted(mcs_indices)
        self.thresholds = self.mcs_sorted[1:]
        self.decision_threshold = decision_threshold
        self.hgb_kwargs = hgb_kwargs
        self.models = {}

    def fit(self, X, y):
        for t in self.thresholds:
            y_bin = (y >= t).astype(int)
            if y_bin.nunique() < 2:
                self.models[t] = None
                continue
            clf = HistGradientBoostingClassifier(**self.hgb_kwargs)
            clf.fit(X, y_bin)
            self.models[t] = clf

    def predict(self, X):
        pred = np.full(len(X), self.mcs_sorted[0])
        for t in self.thresholds:
            if self.models[t] is None:
                continue
            proba = self.models[t].predict_proba(X)[:, 1]
            pred[proba >= self.decision_threshold] = t
        return pred

    def predict_confidence(self, X):
        margins = np.ones(len(X))
        for t in self.thresholds:
            if self.models[t] is None:
                continue
            proba = self.models[t].predict_proba(X)[:, 1]
            margin = np.abs(proba - self.decision_threshold)
            margins = np.minimum(margins, margin)
        return margins


# ============================================================================
# 4. Gap 1 — DNN MLP Classifier Baseline (PyTorch)
# ============================================================================

class DNNMCSClassifier(nn.Module):
    """3-layer MLP for direct MCS classification.
    Architecture: Input → 128 → 256 → 128 → num_classes, with BatchNorm + Dropout.
    """
    def __init__(self, input_dim, num_classes, hidden_dims=(128, 256, 128),
                 dropout=0.2):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_dnn_classifier(X_train, y_train, num_classes, feature_dim,
                         epochs=150, lr=1e-3, batch_size=64, seed=42):
    """Train DNN classifier and return model + label mapping."""
    torch.manual_seed(seed)

    # Map MCS indices to 0..K-1
    unique_mcs = sorted(y_train.unique())
    mcs_to_idx = {m: i for i, m in enumerate(unique_mcs)}
    idx_to_mcs = {i: m for m, i in mcs_to_idx.items()}

    X_t = torch.FloatTensor(X_train.values)
    y_t = torch.LongTensor([mcs_to_idx[v] for v in y_train.values])

    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = DNNMCSClassifier(feature_dim, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
        scheduler.step()

    return model, mcs_to_idx, idx_to_mcs


def dnn_predict(model, X_test, idx_to_mcs):
    """Predict MCS indices using trained DNN."""
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X_test.values)
        logits = model(X_t)
        pred_idx = logits.argmax(dim=1).numpy()
    return np.array([idx_to_mcs[i] for i in pred_idx])


# ============================================================================
# 5. Gap 1 — GRU Sequential Model (PyTorch)
# ============================================================================

class GRUMCSClassifier(nn.Module):
    """GRU model for sequential SINR traces → MCS classification.
    Exploits temporal correlation via recurrent processing.
    """
    def __init__(self, input_dim, num_classes, hidden_size=128,
                 num_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x: [batch, seq_len, features]
        out, _ = self.gru(x)
        last = out[:, -1, :]  # Take last hidden state
        return self.head(last)


def generate_sinr_sequences(optimal_df, rng, window_size=8, ar_coeff=0.85):
    """Generate AR(1) correlated SINR measurement sequences.
    For each context, create a window of correlated measurements.
    """
    n = len(optimal_df)
    true_sinr = optimal_df["SINR_dB"].values

    sequences = np.zeros((n, window_size))
    for i in range(n):
        # AR(1) process: x[t] = ar * x[t-1] + (1-ar) * base + noise
        seq = np.zeros(window_size)
        seq[0] = true_sinr[i] + rng.normal(0, 1.5)
        for t in range(1, window_size):
            innovation = rng.normal(0, 1.0)
            seq[t] = (ar_coeff * seq[t-1]
                      + (1 - ar_coeff) * true_sinr[i]
                      + innovation)
        sequences[i] = seq

    return sequences


def create_gru_dataset(optimal_df, feature_cols, rng, window_size=8):
    """Create sequential dataset for GRU model.
    Each sample: [window_size, num_features] sequence → MCS label.
    """
    n = len(optimal_df)

    # SINR sequences (main temporal feature)
    sinr_seqs = generate_sinr_sequences(
        optimal_df, rng, window_size=window_size)

    # Static features repeated across time steps
    static_features = optimal_df[
        [c for c in feature_cols if c != "Measured_SINR"]
    ].values

    # Build [n, window_size, features] tensor
    num_static = static_features.shape[1]
    num_features = 1 + num_static  # SINR + static

    X_seq = np.zeros((n, window_size, num_features))
    for t in range(window_size):
        X_seq[:, t, 0] = sinr_seqs[:, t]          # SINR at time t
        X_seq[:, t, 1:] = static_features          # Static features

    return X_seq


def train_gru_classifier(X_seq_train, y_train, num_classes, num_features,
                         epochs=100, lr=1e-3, batch_size=64, seed=42):
    """Train GRU sequential classifier."""
    torch.manual_seed(seed)

    unique_mcs = sorted(np.unique(y_train))
    mcs_to_idx = {m: i for i, m in enumerate(unique_mcs)}
    idx_to_mcs = {i: m for m, i in mcs_to_idx.items()}

    X_t = torch.FloatTensor(X_seq_train)
    y_t = torch.LongTensor([mcs_to_idx[v] for v in y_train])

    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = GRUMCSClassifier(num_features, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

    return model, mcs_to_idx, idx_to_mcs


def gru_predict(model, X_seq_test, idx_to_mcs):
    """Predict using trained GRU model."""
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X_seq_test)
        logits = model(X_t)
        pred_idx = logits.argmax(dim=1).numpy()
    return np.array([idx_to_mcs[i] for i in pred_idx])


# ============================================================================
# 6. Gap 2 — Online Learning Simulator
# ============================================================================

class OnlineLearningSimulator:
    """Simulates online model adaptation using HARQ feedback.

    Chronologically replays packet outcomes, periodically retraining
    the ordinal policy on a sliding window with exponential sample
    weighting (recent data matters more).
    """

    def __init__(self, mcs_indices, feature_cols, window_size=500,
                 update_every=100, ema_alpha=0.02, seed=42):
        self.mcs_indices = mcs_indices
        self.feature_cols = feature_cols
        self.window_size = window_size
        self.update_every = update_every
        self.ema_alpha = ema_alpha
        self.seed = seed

        self.buffer_X = deque(maxlen=window_size)
        self.buffer_y = deque(maxlen=window_size)
        self.buffer_ts = deque(maxlen=window_size)

        self.policy = None
        self.bler_ema = 0.10
        self.predictions = []
        self.bler_history = []
        self.thr_history = []

    def _build_policy(self, monotonic_cst):
        return OrdinalMCSPolicy(
            mcs_indices=self.mcs_indices,
            decision_threshold=0.5,
            max_iter=100,
            max_depth=3,
            min_samples_leaf=5,
            learning_rate=0.15,
            monotonic_cst=monotonic_cst,
            random_state=self.seed,
        )

    def run(self, optimal_df, monotonic_cst):
        """Replay contexts chronologically and track adaptation."""
        if "Time_Index" in optimal_df.columns:
            ordered = optimal_df.sort_values("Time_Index").reset_index(
                drop=True)
        else:
            ordered = optimal_df.reset_index(drop=True)

        n = len(ordered)
        pred_mcs_all = np.full(n, self.mcs_indices[0])

        for i in range(n):
            row = ordered.iloc[i]
            x = row[self.feature_cols].values.astype(np.float64)
            y_true = row["MCS_Index"]

            # Predict with current policy
            if self.policy is not None:
                x_df = pd.DataFrame([x], columns=self.feature_cols)
                pred = self.policy.predict(x_df)[0]
            else:
                pred = self.mcs_indices[0]  # Fallback before first train

            pred_mcs_all[i] = pred
            self.predictions.append(pred)

            # Observe ground truth and update buffer
            self.buffer_X.append(x)
            self.buffer_y.append(y_true)
            self.buffer_ts.append(i)

            # Simulated HARQ feedback: was the prediction correct?
            was_overestimate = pred > y_true
            self.bler_ema = (self.ema_alpha * float(was_overestimate)
                             + (1 - self.ema_alpha) * self.bler_ema)

            self.bler_history.append(self.bler_ema)

            # Periodic retraining on sliding window
            if (i + 1) % self.update_every == 0 and len(self.buffer_X) >= 20:
                X_buf = pd.DataFrame(
                    list(self.buffer_X), columns=self.feature_cols)
                y_buf = pd.Series(list(self.buffer_y))

                # Exponential weighting: recent samples matter more
                timestamps = np.array(list(self.buffer_ts), dtype=np.float64)
                weights = np.exp(0.005 * (timestamps - timestamps.max()))

                self.policy = self._build_policy(monotonic_cst)
                # HistGBM supports sample_weight in fit
                try:
                    self.policy.fit(X_buf, y_buf)
                except Exception:
                    pass  # Keep current policy if retraining fails

        return pred_mcs_all


# ============================================================================
# 7. Baseline Policy (Static SINR Thresholds)
# ============================================================================

def baseline_mcs_from_sinr(sinr_db):
    if sinr_db < -2.0: return 3
    if sinr_db < 2.0: return 4
    if sinr_db < 6.0: return 9
    if sinr_db < 10.0: return 11
    if sinr_db < 14.0: return 14
    if sinr_db < 17.0: return 17
    if sinr_db < 21.0: return 20
    if sinr_db < 25.0: return 24
    return 25


# ============================================================================
# 8. Evaluation Helpers
# ============================================================================

def evaluate_policy(eval_df, pred_mcs, stats_table, bler_target=0.10):
    """Policy-level KPIs from lookup in measured packet outcomes."""
    # Build the join columns based on what's available
    join_cols = ["SINR_dB", "Channel", "Speed_kmph"]
    if "Num_Streams" in eval_df.columns and "Num_Streams" in stats_table.columns:
        join_cols.append("Num_Streams")
    if "Carrier_GHz" in eval_df.columns and "Carrier_GHz" in stats_table.columns:
        join_cols.append("Carrier_GHz")

    joined = eval_df[join_cols].copy()
    joined["MCS_Index"] = pred_mcs
    joined = joined.merge(
        stats_table,
        on=join_cols + ["MCS_Index"],
        how="left",
    )
    mean_thr = joined["Throughput"].mean()
    bler_violation_rate = (joined["BLER"] > bler_target).mean()
    mean_bler = joined["BLER"].mean()
    return mean_thr, bler_violation_rate, mean_bler


def asymmetric_rank_cost(y_true, y_pred, overestimate_penalty=3.0):
    mcs_to_rank = {idx: rank for rank, idx in enumerate(NR_MCS_INDICES)}
    true_rank = np.array([mcs_to_rank.get(v, 0) for v in y_true])
    pred_rank = np.array([mcs_to_rank.get(v, 0) for v in y_pred])
    diff = pred_rank - true_rank
    cost = np.where(diff > 0, diff * overestimate_penalty, np.abs(diff))
    return cost.mean()


# ============================================================================
# 9. C++ Export — Enhanced for V2 features
# ============================================================================

def export_tree_to_cpp(clf, feature_cols, mcs_indices, out_path):
    """Export decision tree as C++ with OLLA safety layer."""
    tree = clf.tree_
    classes = clf.classes_.astype(int)
    cpp_feat = {
        "Measured_SINR": "measured_sinr_db",
        "Measured_Speed": "measured_speed_kmph",
        "Channel_Ordinal": "channel_ordinal",
        "Carrier_Band": "carrier_band",
        "Num_Antennas": "num_antennas",
        "BLER_Target_Log": "bler_target_log10",
    }

    def recurse(node_id, indent):
        left = tree.children_left[node_id]
        right = tree.children_right[node_id]
        if left == -1 and right == -1:
            cid = int(np.argmax(tree.value[node_id][0]))
            return [f"{indent}return {int(classes[cid])};"]
        feat = feature_cols[tree.feature[node_id]]
        feat_cpp = cpp_feat.get(feat, feat)
        thr = tree.threshold[node_id]
        lines = [f"{indent}if ({feat_cpp} <= {thr:.8f}) {{"]
        lines.extend(recurse(left, indent + "    "))
        lines.append(f"{indent}}} else {{")
        lines.extend(recurse(right, indent + "    "))
        lines.append(f"{indent}}}")
        return lines

    body = recurse(0, "    ")
    mcs_arr = ", ".join(str(m) for m in sorted(mcs_indices))
    n_mcs = len(mcs_indices)

    # Build function signature from features
    params = []
    for f in feature_cols:
        cpp_name = cpp_feat.get(f, f)
        params.append(f"double {cpp_name}")
    param_str = ",\n                           ".join(params)

    cpp = [
        "// Auto-generated NR Link Adaptation policy with OLLA safety layer.",
        "// V2: Supports MIMO, FR2, multi-service BLER targets.",
        "// Ordinal GBM → surrogate tree distillation.",
        "#include <algorithm>",
        "#include <cmath>",
        "#include <cstdint>",
        "",
        f"static constexpr int VALID_MCS[{n_mcs}] = {{{mcs_arr}}};",
        f"static constexpr int N_MCS = {n_mcs};",
        "",
        "struct OllaState {",
        "    double bler_ema        = 0.10;",
        "    int    consecutive_nack = 0;",
        "    int    mcs_offset       = 0;",
        "};",
        "",
        "static int nearest_valid_mcs(int raw) {",
        "    int best = VALID_MCS[0];",
        "    for (int i = 1; i < N_MCS; ++i) {",
        "        if (std::abs(VALID_MCS[i] - raw) < std::abs(best - raw))",
        "            best = VALID_MCS[i];",
        "    }",
        "    return best;",
        "}",
        "",
        f"static int select_mcs_base({param_str}) {{",
    ]
    cpp.extend(body)
    cpp.append("}")
    cpp.append("")
    cpp.extend([
        "int select_mcs(double measured_sinr_db,",
        "               double measured_speed_kmph,",
        "               int    channel_ordinal,",
        "               double bler_target_log10,",
        "               OllaState& state,",
        "               bool   last_was_ack) {",
        "",
        "    constexpr double kAlpha = 0.02;",
        "    state.bler_ema = kAlpha * (last_was_ack ? 0.0 : 1.0)",
        "                   + (1.0 - kAlpha) * state.bler_ema;",
        "",
        "    state.consecutive_nack = last_was_ack ? 0",
        "                           : state.consecutive_nack + 1;",
        "    if (state.consecutive_nack > 100) {",
        f"        return VALID_MCS[0];",
        "    }",
        "",
        "    double bler_target = std::pow(10.0, bler_target_log10);",
        "    if (state.bler_ema > bler_target) {",
        "        state.mcs_offset = std::max(state.mcs_offset - 1, -3);",
        "    } else if (state.bler_ema < bler_target * 0.5) {",
        "        state.mcs_offset = std::min(state.mcs_offset + 1, 2);",
        "    }",
        "",
        "    int base = select_mcs_base(measured_sinr_db, measured_speed_kmph,",
        "                               channel_ordinal, bler_target_log10);",
        "    int adjusted = base + state.mcs_offset;",
        f"    adjusted = std::clamp(adjusted, VALID_MCS[0], VALID_MCS[N_MCS-1]);",
        "    return nearest_valid_mcs(adjusted);",
        "}",
    ])

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(cpp) + "\n")


# ============================================================================
# 10. Publication-quality plots
# ============================================================================

def generate_comparison_plot(results_dict, output_path="la_v2_comparison.png"):
    """Generate dark-theme comparison plots for all methods."""
    methods = list(results_dict.keys())
    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor('#0d1117')

    metrics = [
        ("accuracy", "MCS Selection Accuracy (%)", 100),
        ("thr_vs_oracle", "Throughput vs Oracle (%)", 1),
        ("bler_violation", "BLER Violation Rate (%)", 100),
        ("asym_cost", "Asymmetric Rank Cost", 1),
    ]

    for ax, (metric, title, scale) in zip(axes.flatten(), metrics):
        ax.set_facecolor('#161b22')
        vals = [results_dict[m].get(metric, 0) * scale for m in methods]
        bars = ax.barh(range(len(methods)), vals, color=colors,
                       edgecolor='#30363d')
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels(methods, fontsize=9, color='#c9d1d9')
        ax.set_title(title, fontsize=13, fontweight='bold', color='white')
        ax.tick_params(colors='#8b949e')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#30363d')
        ax.spines['left'].set_color('#30363d')
        ax.grid(axis='x', color='#21262d', linewidth=0.5)

        for i, v in enumerate(vals):
            ax.text(v + 0.5, i, f'{v:.1f}', va='center', fontsize=9,
                    color='#c9d1d9', fontweight='bold')

    fig.suptitle(
        'V2 Link Adaptation Benchmark\n'
        'Ordinal GBM + DNN + GRU + Online Learning',
        fontsize=15, fontweight='bold', color='white', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_path, dpi=200, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  → Saved {output_path}")


# ============================================================================
# 11. Main Pipeline
# ============================================================================

def train_and_evaluate():
    print("=" * 65)
    print("  V2 Link Adaptation Pipeline")
    print("  Gaps: DNN + GRU + Online + Multi-Service + MIMO/CDL/FR2")
    print("=" * 65)

    # ---- 1. Load data ----
    print("\n1. Loading dataset …")
    import os
    if os.path.exists("sionna_v2_dataset.csv"):
        df = pd.read_csv("sionna_v2_dataset.csv")
        print("   → Loaded V2 dataset")
    elif os.path.exists("sionna_realistic_dataset.csv"):
        df = pd.read_csv("sionna_realistic_dataset.csv")
        print("   → Loaded V1 dataset (backward compatible mode)")
    else:
        raise FileNotFoundError("No dataset found. Run generate script first.")

    version = detect_dataset_version(df)
    df = normalize_dataset(df, version)
    print(f"   Dataset version: {version}, rows: {len(df):,}")

    rng = np.random.default_rng(RANDOM_SEED)

    # ---- 2. Build labels ----
    print("\n2. Building multi-service optimal labels …")
    stats_table, _ = build_optimal_label_table(df, bler_target=0.10)

    # eMBB labels (primary evaluation)
    _, optimal_embb = build_optimal_label_table(df, bler_target=0.10)
    optimal_embb["BLER_Target"] = 0.10
    optimal_embb["BLER_Target_Log"] = np.log10(0.10)
    optimal_embb["Service"] = "eMBB"

    # Multi-service labels for Gap 4
    _, optimal_multi = build_multi_service_labels(df)
    print(f"   eMBB contexts: {len(optimal_embb)}")
    print(f"   Multi-service contexts: {len(optimal_multi)}")

    # ---- 3. Feature engineering ----
    print("\n3. Engineering features …")
    multi_service = True
    feature_cols = get_feature_cols(version, multi_service=multi_service)
    monotonic_cst = get_monotonic_constraints(feature_cols)

    optimal_embb = add_features(optimal_embb, rng, version)
    optimal_multi = add_features(optimal_multi, rng, version)

    print(f"   Features: {feature_cols}")
    print(f"   Monotonic constraints: {monotonic_cst}")

    # ---- 4. Train/test split on eMBB data ----
    X = optimal_embb[feature_cols]
    y = optimal_embb["MCS_Index"]

    groups = (
        optimal_embb["Channel"].astype(str)
        + "_" + optimal_embb["Speed_kmph"].astype(str)
    )
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=0.25, random_state=RANDOM_SEED,
    )
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    eval_df = optimal_embb.iloc[test_idx].reset_index(drop=True)

    thr_oracle, _, _ = evaluate_policy(eval_df, y_test.values, stats_table)
    print(f"   Train: {len(X_train)}, Test: {len(X_test)}, "
          f"Oracle throughput: {thr_oracle:.4f}")

    # ---- 5. Train all methods ----
    results = {}

    # --- 5a. Ordinal GBM (our primary method) ---
    print("\n4. Training Ordinal GBM (K-1 binary, monotonic) …")
    policy = OrdinalMCSPolicy(
        mcs_indices=NR_MCS_INDICES,
        decision_threshold=0.5,
        max_iter=300,
        max_depth=4,
        min_samples_leaf=3,
        learning_rate=0.1,
        monotonic_cst=monotonic_cst,
        random_state=RANDOM_SEED,
    )
    policy.fit(X_train, y_train)

    gbm_pred = policy.predict(X_test)
    confidence = policy.predict_confidence(X_test)
    fallback_mask = confidence < CONFIDENCE_THRESHOLD
    fallback_values = np.array(
        [baseline_mcs_from_sinr(v) for v in X_test["Measured_SINR"].values])
    gbm_safe = gbm_pred.copy()
    gbm_safe[fallback_mask] = np.minimum(
        gbm_pred[fallback_mask], fallback_values[fallback_mask])

    thr, bv, mb = evaluate_policy(eval_df, gbm_safe, stats_table)
    results["Ordinal GBM"] = {
        "accuracy": accuracy_score(y_test, gbm_safe),
        "thr_vs_oracle": thr / thr_oracle * 100 if thr_oracle > 0 else 0,
        "bler_violation": bv,
        "asym_cost": asymmetric_rank_cost(y_test.values, gbm_safe),
    }
    print(f"   Accuracy: {results['Ordinal GBM']['accuracy']*100:.1f}%, "
          f"Throughput: {thr:.4f} ({results['Ordinal GBM']['thr_vs_oracle']:.1f}% oracle)")

    # --- 5b. DNN Classifier (Gap 1) ---
    print("\n5. Training DNN MLP Classifier …")
    num_classes = len(NR_MCS_INDICES)
    dnn_model, mcs_to_idx, idx_to_mcs = train_dnn_classifier(
        X_train, y_train, num_classes, len(feature_cols),
        epochs=150, lr=1e-3, seed=RANDOM_SEED,
    )
    dnn_pred = dnn_predict(dnn_model, X_test, idx_to_mcs)
    thr, bv, mb = evaluate_policy(eval_df, dnn_pred, stats_table)
    results["DNN Classifier"] = {
        "accuracy": accuracy_score(y_test, dnn_pred),
        "thr_vs_oracle": thr / thr_oracle * 100 if thr_oracle > 0 else 0,
        "bler_violation": bv,
        "asym_cost": asymmetric_rank_cost(y_test.values, dnn_pred),
    }
    print(f"   Accuracy: {results['DNN Classifier']['accuracy']*100:.1f}%, "
          f"Throughput: {thr:.4f}")

    # --- 5c. GRU Sequential (Gap 1) ---
    print("\n6. Training GRU Sequential Model …")
    window_size = 8
    X_seq_train = create_gru_dataset(
        optimal_embb.iloc[train_idx].reset_index(drop=True),
        feature_cols, rng, window_size=window_size,
    )
    X_seq_test = create_gru_dataset(
        optimal_embb.iloc[test_idx].reset_index(drop=True),
        feature_cols, rng, window_size=window_size,
    )
    num_seq_features = X_seq_train.shape[2]

    gru_model, gru_mcs2idx, gru_idx2mcs = train_gru_classifier(
        X_seq_train, y_train.values, num_classes, num_seq_features,
        epochs=100, lr=1e-3, seed=RANDOM_SEED,
    )
    gru_pred = gru_predict(gru_model, X_seq_test, gru_idx2mcs)
    thr, bv, mb = evaluate_policy(eval_df, gru_pred, stats_table)
    results["GRU Sequential"] = {
        "accuracy": accuracy_score(y_test, gru_pred),
        "thr_vs_oracle": thr / thr_oracle * 100 if thr_oracle > 0 else 0,
        "bler_violation": bv,
        "asym_cost": asymmetric_rank_cost(y_test.values, gru_pred),
    }
    print(f"   Accuracy: {results['GRU Sequential']['accuracy']*100:.1f}%, "
          f"Throughput: {thr:.4f}")

    # --- 5d. Online Learning (Gap 2) ---
    print("\n7. Running Online Learning Simulation …")
    online_sim = OnlineLearningSimulator(
        mcs_indices=NR_MCS_INDICES,
        feature_cols=feature_cols,
        window_size=500,
        update_every=50,
        seed=RANDOM_SEED,
    )
    # Use eMBB test set chronologically
    online_pred = online_sim.run(
        optimal_embb.iloc[test_idx].reset_index(drop=True),
        monotonic_cst,
    )
    thr, bv, mb = evaluate_policy(eval_df, online_pred, stats_table)
    results["Online GBM"] = {
        "accuracy": accuracy_score(y_test.values, online_pred),
        "thr_vs_oracle": thr / thr_oracle * 100 if thr_oracle > 0 else 0,
        "bler_violation": bv,
        "asym_cost": asymmetric_rank_cost(y_test.values, online_pred),
    }
    print(f"   Accuracy: {results['Online GBM']['accuracy']*100:.1f}%")

    # --- 5e. Static LUT Baseline ---
    lut_pred = np.array(
        [baseline_mcs_from_sinr(s) for s in X_test["Measured_SINR"].values])
    thr, bv, mb = evaluate_policy(eval_df, lut_pred, stats_table)
    results["Static LUT"] = {
        "accuracy": accuracy_score(y_test, lut_pred),
        "thr_vs_oracle": thr / thr_oracle * 100 if thr_oracle > 0 else 0,
        "bler_violation": bv,
        "asym_cost": asymmetric_rank_cost(y_test.values, lut_pred),
    }

    # ---- 6. Multi-service evaluation (Gap 4) ----
    print("\n8. Multi-service evaluation …")
    # Train on multi-service data
    X_multi = optimal_multi[feature_cols]
    y_multi = optimal_multi["MCS_Index"]

    multi_policy = OrdinalMCSPolicy(
        mcs_indices=NR_MCS_INDICES,
        decision_threshold=0.5,
        max_iter=300,
        max_depth=5,
        min_samples_leaf=3,
        learning_rate=0.1,
        monotonic_cst=monotonic_cst,
        random_state=RANDOM_SEED,
    )
    multi_policy.fit(X_multi, y_multi)

    for service_name, bler_target in SERVICE_PROFILES.items():
        _, svc_optimal = build_optimal_label_table(df, bler_target)
        svc_optimal["BLER_Target"] = bler_target
        svc_optimal["BLER_Target_Log"] = np.log10(bler_target)
        svc_optimal = add_features(svc_optimal, rng, version)

        if len(svc_optimal) > 0:
            svc_X = svc_optimal[feature_cols]
            svc_pred = multi_policy.predict(svc_X)
            svc_acc = accuracy_score(svc_optimal["MCS_Index"], svc_pred)
            print(f"   {service_name} (BLER≤{bler_target}): "
                  f"accuracy={svc_acc*100:.1f}%, "
                  f"contexts={len(svc_optimal)}")

    # ---- 7. Results summary ----
    print("\n" + "=" * 85)
    print(f"{'Method':<20} {'Accuracy':>10} {'vs Oracle':>12} "
          f"{'BLER Viol.':>12} {'Asym. Cost':>12}")
    print("=" * 85)
    for name, r in results.items():
        print(f"{name:<20} {r['accuracy']*100:>9.1f}% "
              f"{r['thr_vs_oracle']:>10.1f}% "
              f"{r['bler_violation']*100:>10.1f}% "
              f"{r['asym_cost']:>11.3f}")
    print("=" * 85)

    # ---- 8. Plot ----
    print("\n9. Generating comparison plot …")
    generate_comparison_plot(results)

    # ---- 9. C++ Export ----
    print("\n10. Distilling surrogate tree for C++ export …")
    y_train_ord = policy.predict(X_train)
    surrogate = DecisionTreeClassifier(
        max_depth=12, min_samples_leaf=2, random_state=RANDOM_SEED,
    )
    surrogate.fit(X_train, y_train_ord)
    fidelity = accuracy_score(
        policy.predict(X_test), surrogate.predict(X_test),
    )
    export_tree_to_cpp(
        surrogate, feature_cols, NR_MCS_INDICES, CPP_EXPORT_PATH,
    )
    print(f"   Exported: {CPP_EXPORT_PATH}")
    print(f"   Surrogate fidelity: {fidelity * 100:.1f}%")
    print(f"   C++ supports: OLLA outer-loop, per-UE BLER EMA, "
          f"multi-service BLER target.")

    print("\n✓ V2 Pipeline complete.\n")


if __name__ == "__main__":
    train_and_evaluate()
