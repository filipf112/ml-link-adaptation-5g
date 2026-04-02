"""
Unified Benchmark: Paper DNN vs DNN Classifier vs GRU vs Ordinal GBM vs Shannon
================================================================================

Implements and evaluates all Link Adaptation strategies on the same
Sionna-generated dataset with proper 3GPP channels:

  1. Static LUT       — SINR-only threshold table (classical OLLA baseline)
  2. DNN Regressor    — Faithful reproduction of the paper's approach
                         (3-layer MLP, SINR→SE regression, nearest CQI mapping)
  3. DNN Classifier   — Direct MCS classification (cross-entropy loss)
  4. GRU Sequential   — Temporal SINR traces via recurrent network  
  5. Ordinal GBM      — Our approach (K-1 binary classifiers, monotonic, safety)
  6. Shannon Bound    — Theoretical upper bound on achievable SE

All evaluated with:
  - Throughput vs Oracle ratio
  - BLER violation rate
  - Asymmetric rank cost
  - 95% confidence intervals over multiple seeds
"""

import numpy as np
import pandas as pd
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

# --- sklearn ---
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupShuffleSplit

# --- PyTorch (for replicating the paper's DNN) ---
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
N_BOOTSTRAP_SEEDS = 10       # For confidence intervals
BLER_TARGET = 0.10
CONFIDENCE_THRESHOLD = 0.30

# NR MCS indices in the dataset
NR_MCS_INDICES = sorted([3, 4, 9, 11, 14, 17, 20, 24, 25])

# SE map for each MCS (bits/symbol × code_rate from TS 38.214)
NR_MCS_SE_MAP = {
    3:  2 * (253/1024),    # QPSK R≈0.247  → SE≈0.494
    4:  2 * (308/1024),    # QPSK R≈0.301  → SE≈0.602
    9:  2 * (616/1024),    # QPSK R≈0.602  → SE≈1.203
    11: 4 * (340/1024),    # 16QAM R≈0.332 → SE≈1.328
    14: 4 * (553/1024),    # 16QAM R≈0.540 → SE≈2.160
    17: 6 * (438/1024),    # 64QAM R≈0.428 → SE≈2.566
    20: 6 * (666/1024),    # 64QAM R≈0.650 → SE≈3.902
    24: 8 * (567/1024),    # 256QAM R≈0.554 → SE≈4.430
    25: 8 * (616/1024),    # 256QAM R≈0.602 → SE≈4.813
}

CHANNEL_ORDINAL = {"TDL-A": 0, "TDL-B": 1, "TDL-C": 2}


# ============================================================================
# 1. Data Preparation (shared across all methods)
# ============================================================================

def build_optimal_label_table(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate packet logs → (context, MCS) stats → optimal MCS per context."""
    stats = (
        df.groupby(["SINR_dB", "Channel", "Speed_kmph", "MCS_Index"])
        .agg(
            BLER=("Was_Success", lambda x: 1.0 - x.mean()),
            Throughput=("Actual_Throughput", "mean"),
            N_Packets=("Was_Success", "count"),
        )
        .reset_index()
    )

    def get_robust_optimal(group):
        safe = group[group["BLER"] <= BLER_TARGET]
        if not safe.empty:
            return safe.loc[safe["Throughput"].idxmax()]
        return group.loc[group["MCS_Index"].idxmin()]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        optimal_df = (
            stats.groupby(
                ["SINR_dB", "Channel", "Speed_kmph"], group_keys=False,
            )
            .apply(get_robust_optimal, include_groups=False)
            .reset_index()
        )
    return stats, optimal_df


def add_features(optimal_df: pd.DataFrame, rng) -> pd.DataFrame:
    """Add realistic, noisy observable features."""
    out = optimal_df.copy()
    sinr_noise_db = rng.normal(0.0, 1.5, len(out))
    speed_vals = out["Speed_kmph"].values
    cqi_aging_db = 0.5 * (1.0 - np.exp(-speed_vals / 30.0))
    out["Measured_SINR"] = out["SINR_dB"] + sinr_noise_db - cqi_aging_db
    out["Measured_Speed"] = out["Speed_kmph"] + rng.normal(0.0, 10.0, len(out))
    out["Measured_Speed"] = out["Measured_Speed"].clip(lower=0.0)
    out["Channel_Ordinal"] = out["Channel"].map(CHANNEL_ORDINAL).astype(np.float64)
    return out


# ============================================================================
# 2. Method 1 — Static LUT Baseline
# ============================================================================

def static_lut_predict(sinr_db_array: np.ndarray) -> np.ndarray:
    """Simple static SINR threshold table."""
    pred = np.full(len(sinr_db_array), NR_MCS_INDICES[0])
    thresholds = [
        (-2.0, 3), (2.0, 4), (6.0, 9), (10.0, 11),
        (14.0, 14), (17.0, 17), (21.0, 20), (25.0, 24),
    ]
    for sinr, mcs_val in zip(sinr_db_array, range(len(sinr_db_array))):
        chosen = 3
        for thr, mcs in thresholds:
            if sinr >= thr:
                chosen = mcs
        if sinr >= 25.0:
            chosen = 25
        pred[mcs_val] = chosen
    return pred


def baseline_mcs_from_sinr(sinr_db: float) -> int:
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
# 3. Method 2 — Paper's DNN Regressor (Faithful Reproduction)
# ============================================================================

class PaperDNNRegressor(nn.Module):
    """
    3-layer DNN regressor as described in the paper.
    Maps SINR → estimated SE, then nearest-CQI/MCS lookup.

    Architecture: Input(1) → 64 → 128 → 64 → Output(1), ReLU activations.
    (Dimensions inferred since paper says "three hidden layers" with
     unspecified widths — we use reasonable values.)
    """
    def __init__(self, hidden_dims=(64, 128, 64)):
        super().__init__()
        layers = []
        in_dim = 1
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class PaperDNNRegressorMultiInput(nn.Module):
    """
    Enhanced version: same architecture but accepts 3 inputs
    (SINR, Speed, Channel) to make a fairer comparison.
    """
    def __init__(self, input_dim=3, hidden_dims=(64, 128, 64)):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_dnn_regressor(X_train, y_train_se, lr=0.01, epochs=100,
                        multi_input=False, seed=42):
    """Train paper's DNN with MSE loss + Adam (paper's exact config)."""
    torch.manual_seed(seed)

    if multi_input:
        X_t = torch.FloatTensor(X_train.values)
        model = PaperDNNRegressorMultiInput(input_dim=X_t.shape[1])
    else:
        X_t = torch.FloatTensor(X_train[["Measured_SINR"]].values)
        model = PaperDNNRegressor()

    y_t = torch.FloatTensor(y_train_se.values.reshape(-1, 1))

    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()

    return model


def dnn_predict_mcs(model, X_test, multi_input=False) -> np.ndarray:
    """
    Paper's Algorithm 2: map predicted SE to nearest MCS.
    We implement their exact approach (nearest distance) AND
    a corrected version (largest supportable).
    """
    model.eval()
    with torch.no_grad():
        if multi_input:
            X_t = torch.FloatTensor(X_test.values)
        else:
            X_t = torch.FloatTensor(X_test[["Measured_SINR"]].values)
        se_pred = model(X_t).numpy().flatten()

    # Paper's approach: nearest SE (Algorithm 2 — argmin |SE_pred - SE_table|)
    se_values = np.array([NR_MCS_SE_MAP[m] for m in NR_MCS_INDICES])
    mcs_pred = np.zeros(len(se_pred), dtype=int)
    for i, se in enumerate(se_pred):
        distances = np.abs(se_values - se)
        mcs_pred[i] = NR_MCS_INDICES[np.argmin(distances)]
    return mcs_pred, se_pred


def dnn_predict_mcs_safe(model, X_test, multi_input=False) -> np.ndarray:
    """
    CORRECTED version of Algorithm 2: largest supportable MCS
    (rounds DOWN instead of nearest, respecting BLER constraint).
    """
    model.eval()
    with torch.no_grad():
        if multi_input:
            X_t = torch.FloatTensor(X_test.values)
        else:
            X_t = torch.FloatTensor(X_test[["Measured_SINR"]].values)
        se_pred = model(X_t).numpy().flatten()

    se_values = np.array([NR_MCS_SE_MAP[m] for m in NR_MCS_INDICES])
    mcs_pred = np.zeros(len(se_pred), dtype=int)
    for i, se in enumerate(se_pred):
        # Largest MCS whose SE ≤ predicted SE
        valid = [m for m, s in zip(NR_MCS_INDICES, se_values) if s <= se]
        mcs_pred[i] = max(valid) if valid else NR_MCS_INDICES[0]
    return mcs_pred, se_pred


# ============================================================================
# 4. Method 3 — Our Ordinal GBM Policy
# ============================================================================

class OrdinalMCSPolicy:
    """Ordinal regression: K-1 binary classifiers with monotonic constraints."""

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
# 5. Method 4 — DNN Multi-class Classifier (Gap 1 baseline)
# ============================================================================

class DNNMCSClassifier(nn.Module):
    """3-layer MLP for direct MCS classification (cross-entropy loss)."""
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


def train_dnn_classifier(X_train, y_train, mcs_indices, lr=1e-3,
                         epochs=150, seed=42):
    """Train DNN classifier with cross-entropy loss."""
    torch.manual_seed(seed)
    mcs_to_idx = {m: i for i, m in enumerate(sorted(mcs_indices))}
    idx_to_mcs = {i: m for m, i in mcs_to_idx.items()}

    X_t = torch.FloatTensor(X_train.values)
    y_t = torch.LongTensor([mcs_to_idx[v] for v in y_train.values])
    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = DNNMCSClassifier(X_t.shape[1], len(mcs_indices))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.train()
    for _ in range(epochs):
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
        scheduler.step()

    return model, idx_to_mcs


def dnn_classifier_predict(model, X_test, idx_to_mcs):
    model.eval()
    with torch.no_grad():
        logits = model(torch.FloatTensor(X_test.values))
        pred_idx = logits.argmax(dim=1).numpy()
    return np.array([idx_to_mcs[i] for i in pred_idx])


# ============================================================================
# 6. Method 5 — GRU Sequential Model (Gap 1 baseline)
# ============================================================================

class GRUMCSClassifier(nn.Module):
    """GRU for sequential SINR traces → MCS classification."""
    def __init__(self, input_dim, num_classes, hidden_size=128,
                 num_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(64, num_classes))

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])


def generate_sinr_sequences(optimal_df, rng, window_size=8, ar_coeff=0.85):
    """Generate AR(1) correlated SINR measurement sequences."""
    n = len(optimal_df)
    true_sinr = optimal_df["SINR_dB"].values
    seqs = np.zeros((n, window_size))
    for i in range(n):
        seqs[i, 0] = true_sinr[i] + rng.normal(0, 1.5)
        for t in range(1, window_size):
            seqs[i, t] = (ar_coeff * seqs[i, t-1]
                          + (1 - ar_coeff) * true_sinr[i]
                          + rng.normal(0, 1.0))
    return seqs


def create_gru_dataset(optimal_df, feature_cols, rng, window_size=8):
    sinr_seqs = generate_sinr_sequences(optimal_df, rng, window_size)
    static = optimal_df[
        [c for c in feature_cols if c != "Measured_SINR"]
    ].values
    n_feat = 1 + static.shape[1]
    X_seq = np.zeros((len(optimal_df), window_size, n_feat))
    for t in range(window_size):
        X_seq[:, t, 0] = sinr_seqs[:, t]
        X_seq[:, t, 1:] = static
    return X_seq


def train_gru_classifier(X_seq_train, y_train, mcs_indices, num_features,
                         epochs=100, lr=1e-3, seed=42):
    torch.manual_seed(seed)
    mcs_to_idx = {m: i for i, m in enumerate(sorted(mcs_indices))}
    idx_to_mcs = {i: m for m, i in mcs_to_idx.items()}

    X_t = torch.FloatTensor(X_seq_train)
    y_t = torch.LongTensor([mcs_to_idx[v] for v in y_train])
    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = GRUMCSClassifier(num_features, len(mcs_indices))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()

    return model, idx_to_mcs


def gru_predict(model, X_seq_test, idx_to_mcs):
    model.eval()
    with torch.no_grad():
        logits = model(torch.FloatTensor(X_seq_test))
        pred_idx = logits.argmax(dim=1).numpy()
    return np.array([idx_to_mcs[i] for i in pred_idx])


# ============================================================================
# 7. Method 6 — Shannon Capacity Upper Bound
# ============================================================================

def shannon_bound_mcs(sinr_db_array: np.ndarray) -> np.ndarray:
    """
    Shannon capacity C = log2(1 + SINR_linear).
    Map to highest MCS whose SE ≤ Shannon capacity.
    """
    sinr_lin = 10.0 ** (sinr_db_array / 10.0)
    shannon_se = np.log2(1.0 + sinr_lin)

    se_values = np.array([NR_MCS_SE_MAP[m] for m in NR_MCS_INDICES])
    mcs_pred = np.zeros(len(sinr_db_array), dtype=int)
    for i, cap in enumerate(shannon_se):
        valid = [m for m, s in zip(NR_MCS_INDICES, se_values) if s <= cap]
        mcs_pred[i] = max(valid) if valid else NR_MCS_INDICES[0]
    return mcs_pred, shannon_se


# ============================================================================
# 6. Evaluation Metrics
# ============================================================================

def evaluate_policy(eval_df, pred_mcs, stats_table):
    """Throughput + BLER violation rate from packet-level lookup."""
    joined = eval_df[["SINR_dB", "Channel", "Speed_kmph"]].copy()
    joined["MCS_Index"] = pred_mcs
    joined = joined.merge(
        stats_table,
        on=["SINR_dB", "Channel", "Speed_kmph", "MCS_Index"],
        how="left",
    )
    mean_thr = joined["Throughput"].mean()
    bler_violation_rate = (joined["BLER"] > BLER_TARGET).mean()
    mean_bler = joined["BLER"].mean()
    return mean_thr, bler_violation_rate, mean_bler


def asymmetric_rank_cost(y_true, y_pred, overestimate_penalty=3.0):
    """Over-estimation costs 3× more than under-estimation."""
    mcs_to_rank = {idx: rank for rank, idx in enumerate(NR_MCS_INDICES)}
    true_rank = np.array([mcs_to_rank.get(v, 0) for v in y_true])
    pred_rank = np.array([mcs_to_rank.get(v, 0) for v in y_pred])
    diff = pred_rank - true_rank
    cost = np.where(diff > 0, diff * overestimate_penalty, np.abs(diff))
    return cost.mean()


# ============================================================================
# 7. Main Benchmark
# ============================================================================

@dataclass
class BenchmarkResult:
    name: str
    accuracy: float = 0.0
    throughput: float = 0.0
    thr_vs_oracle: float = 0.0
    bler_violation: float = 0.0
    mean_bler: float = 0.0
    asym_cost: float = 0.0
    mse_se: float = 0.0        # Only for DNN methods


def run_single_seed(df, seed, feature_cols):
    """Run all methods on a single random seed, return results dict."""
    rng = np.random.default_rng(seed)
    stats_table, optimal_df = build_optimal_label_table(df)
    optimal_df = add_features(optimal_df, rng)

    X = optimal_df[feature_cols]
    y = optimal_df["MCS_Index"]

    # Compute ground-truth SE for DNN supervision
    y_se = y.map(NR_MCS_SE_MAP)

    groups = (
        optimal_df["Channel"].astype(str)
        + "_" + optimal_df["Speed_kmph"].astype(str)
    )
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=0.25, random_state=seed,
    )
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    y_se_train = y_se.iloc[train_idx]
    eval_df = optimal_df.iloc[test_idx].reset_index(drop=True)

    thr_oracle, _, _ = evaluate_policy(eval_df, y_test.values, stats_table)

    results = {}

    # --- 1. Static LUT ---
    lut_pred = np.array([baseline_mcs_from_sinr(s)
                         for s in X_test["Measured_SINR"].values])
    thr, bv, mb = evaluate_policy(eval_df, lut_pred, stats_table)
    results["Static LUT"] = BenchmarkResult(
        name="Static LUT",
        accuracy=accuracy_score(y_test, lut_pred),
        throughput=thr,
        thr_vs_oracle=thr / thr_oracle * 100 if thr_oracle > 0 else 0,
        bler_violation=bv,
        mean_bler=mb,
        asym_cost=asymmetric_rank_cost(y_test.values, lut_pred),
    )

    # --- 2a. Paper's DNN (SINR-only, paper's exact config) ---
    dnn_1d = train_dnn_regressor(X_train, y_se_train, lr=0.01,
                                 epochs=100, multi_input=False, seed=seed)
    dnn_1d_pred, dnn_1d_se = dnn_predict_mcs(dnn_1d, X_test, multi_input=False)
    thr, bv, mb = evaluate_policy(eval_df, dnn_1d_pred, stats_table)
    mse_1d = np.mean((y_se.iloc[test_idx].values - dnn_1d_se) ** 2)
    results["DNN 1D (Paper)"] = BenchmarkResult(
        name="DNN 1D (Paper)",
        accuracy=accuracy_score(y_test, dnn_1d_pred),
        throughput=thr,
        thr_vs_oracle=thr / thr_oracle * 100 if thr_oracle > 0 else 0,
        bler_violation=bv,
        mean_bler=mb,
        asym_cost=asymmetric_rank_cost(y_test.values, dnn_1d_pred),
        mse_se=mse_1d,
    )

    # --- 2b. Paper's DNN (SINR-only, safe rounding) ---
    dnn_1d_safe_pred, _ = dnn_predict_mcs_safe(dnn_1d, X_test, multi_input=False)
    thr, bv, mb = evaluate_policy(eval_df, dnn_1d_safe_pred, stats_table)
    results["DNN 1D (Safe)"] = BenchmarkResult(
        name="DNN 1D (Safe)",
        accuracy=accuracy_score(y_test, dnn_1d_safe_pred),
        throughput=thr,
        thr_vs_oracle=thr / thr_oracle * 100 if thr_oracle > 0 else 0,
        bler_violation=bv,
        mean_bler=mb,
        asym_cost=asymmetric_rank_cost(y_test.values, dnn_1d_safe_pred),
        mse_se=mse_1d,
    )

    # --- 2c. Enhanced DNN (3 inputs, same architecture) ---
    dnn_3d = train_dnn_regressor(X_train, y_se_train, lr=0.001,
                                 epochs=200, multi_input=True, seed=seed)
    dnn_3d_pred, dnn_3d_se = dnn_predict_mcs_safe(dnn_3d, X_test, multi_input=True)
    thr, bv, mb = evaluate_policy(eval_df, dnn_3d_pred, stats_table)
    mse_3d = np.mean((y_se.iloc[test_idx].values - dnn_3d_se) ** 2)
    results["DNN 3D (Enhanced)"] = BenchmarkResult(
        name="DNN 3D (Enhanced)",
        accuracy=accuracy_score(y_test, dnn_3d_pred),
        throughput=thr,
        thr_vs_oracle=thr / thr_oracle * 100 if thr_oracle > 0 else 0,
        bler_violation=bv,
        mean_bler=mb,
        asym_cost=asymmetric_rank_cost(y_test.values, dnn_3d_pred),
        mse_se=mse_3d,
    )

    # --- 3. Ordinal GBM (our method) ---
    policy = OrdinalMCSPolicy(
        mcs_indices=NR_MCS_INDICES,
        decision_threshold=0.5,
        max_iter=300,
        max_depth=4,
        min_samples_leaf=3,
        learning_rate=0.1,
        monotonic_cst=[1, -1, -1],
        random_state=seed,
    )
    policy.fit(X_train, y_train)
    gbm_pred = policy.predict(X_test)

    # Confidence-gated fallback
    confidence = policy.predict_confidence(X_test)
    fallback_mask = confidence < CONFIDENCE_THRESHOLD
    fallback_values = np.array(
        [baseline_mcs_from_sinr(v) for v in X_test["Measured_SINR"].values],
    )
    gbm_safe = gbm_pred.copy()
    gbm_safe[fallback_mask] = np.minimum(
        gbm_pred[fallback_mask], fallback_values[fallback_mask],
    )

    thr, bv, mb = evaluate_policy(eval_df, gbm_safe, stats_table)
    results["Ordinal GBM (Ours)"] = BenchmarkResult(
        name="Ordinal GBM (Ours)",
        accuracy=accuracy_score(y_test, gbm_safe),
        throughput=thr,
        thr_vs_oracle=thr / thr_oracle * 100 if thr_oracle > 0 else 0,
        bler_violation=bv,
        mean_bler=mb,
        asym_cost=asymmetric_rank_cost(y_test.values, gbm_safe),
    )

    # --- 4. DNN Classifier (direct MCS classification) ---
    dnn_clf, clf_idx2mcs = train_dnn_classifier(
        X_train, y_train, NR_MCS_INDICES, lr=1e-3, epochs=150, seed=seed)
    dnn_clf_pred = dnn_classifier_predict(dnn_clf, X_test, clf_idx2mcs)
    thr, bv, mb = evaluate_policy(eval_df, dnn_clf_pred, stats_table)
    results["DNN Classifier"] = BenchmarkResult(
        name="DNN Classifier",
        accuracy=accuracy_score(y_test, dnn_clf_pred),
        throughput=thr,
        thr_vs_oracle=thr / thr_oracle * 100 if thr_oracle > 0 else 0,
        bler_violation=bv,
        mean_bler=mb,
        asym_cost=asymmetric_rank_cost(y_test.values, dnn_clf_pred),
    )

    # --- 5. GRU Sequential ---
    gru_rng = np.random.default_rng(seed)
    X_seq_train = create_gru_dataset(
        optimal_df.iloc[train_idx].reset_index(drop=True),
        feature_cols, gru_rng)
    X_seq_test = create_gru_dataset(
        optimal_df.iloc[test_idx].reset_index(drop=True),
        feature_cols, gru_rng)
    gru_model, gru_idx2mcs = train_gru_classifier(
        X_seq_train, y_train.values, NR_MCS_INDICES,
        X_seq_train.shape[2], epochs=100, lr=1e-3, seed=seed)
    gru_pred = gru_predict(gru_model, X_seq_test, gru_idx2mcs)
    thr, bv, mb = evaluate_policy(eval_df, gru_pred, stats_table)
    results["GRU Sequential"] = BenchmarkResult(
        name="GRU Sequential",
        accuracy=accuracy_score(y_test, gru_pred),
        throughput=thr,
        thr_vs_oracle=thr / thr_oracle * 100 if thr_oracle > 0 else 0,
        bler_violation=bv,
        mean_bler=mb,
        asym_cost=asymmetric_rank_cost(y_test.values, gru_pred),
    )

    # --- 6. Shannon Bound ---
    shan_pred, shan_se = shannon_bound_mcs(X_test["Measured_SINR"].values)
    thr, bv, mb = evaluate_policy(eval_df, shan_pred, stats_table)
    results["Shannon Bound"] = BenchmarkResult(
        name="Shannon Bound",
        accuracy=accuracy_score(y_test, shan_pred),
        throughput=thr,
        thr_vs_oracle=thr / thr_oracle * 100 if thr_oracle > 0 else 0,
        bler_violation=bv,
        mean_bler=mb,
        asym_cost=asymmetric_rank_cost(y_test.values, shan_pred),
    )

    return results, thr_oracle


def aggregate_results(all_results: List[Dict[str, BenchmarkResult]]):
    """Compute mean ± 95% CI across seeds."""
    method_names = list(all_results[0].keys())
    metrics = ["accuracy", "throughput", "thr_vs_oracle",
               "bler_violation", "mean_bler", "asym_cost"]

    agg = {}
    for method in method_names:
        agg[method] = {}
        for metric in metrics:
            values = [r[method].__dict__[metric] for r in all_results]
            mean = np.mean(values)
            std = np.std(values, ddof=1) if len(values) > 1 else 0
            ci95 = 1.96 * std / np.sqrt(len(values))
            agg[method][metric] = (mean, ci95)
    return agg


def print_results_table(agg):
    """Pretty-print the results table."""
    methods = list(agg.keys())
    print("\n" + "=" * 110)
    print(f"{'Method':<22} {'Accuracy':>12} {'Throughput':>14} "
          f"{'vs Oracle':>12} {'BLER Viol.':>12} {'Mean BLER':>12} "
          f"{'Asym. Cost':>12}")
    print("=" * 110)

    for m in methods:
        a = agg[m]
        acc_str = f"{a['accuracy'][0]*100:.1f}±{a['accuracy'][1]*100:.1f}%"
        thr_str = f"{a['throughput'][0]:.4f}±{a['throughput'][1]:.4f}"
        orc_str = f"{a['thr_vs_oracle'][0]:.1f}±{a['thr_vs_oracle'][1]:.1f}%"
        blr_str = f"{a['bler_violation'][0]*100:.1f}±{a['bler_violation'][1]*100:.1f}%"
        mbl_str = f"{a['mean_bler'][0]*100:.1f}±{a['mean_bler'][1]*100:.1f}%"
        cst_str = f"{a['asym_cost'][0]:.3f}±{a['asym_cost'][1]:.3f}"
        print(f"{m:<22} {acc_str:>12} {thr_str:>14} "
              f"{orc_str:>12} {blr_str:>12} {mbl_str:>12} {cst_str:>12}")

    print("=" * 110)
    print("  (Values shown as mean ± 95% CI over "
          f"{N_BOOTSTRAP_SEEDS} seeds)\n")


def generate_plots(all_results, all_oracles, agg):
    """Generate publication-quality comparison plots."""
    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor('#0d1117')
    gs = GridSpec(2, 2, hspace=0.35, wspace=0.30)

    methods_order = ["Static LUT", "DNN 1D (Paper)", "DNN 1D (Safe)",
                     "DNN 3D (Enhanced)", "DNN Classifier", "GRU Sequential",
                     "Ordinal GBM (Ours)", "Shannon Bound"]
    colors = {
        "Static LUT": "#6e7681",
        "DNN 1D (Paper)": "#f85149",
        "DNN 1D (Safe)": "#ff7b72",
        "DNN 3D (Enhanced)": "#d29922",
        "DNN Classifier": "#bc8cff",
        "GRU Sequential": "#d2a8ff",
        "Ordinal GBM (Ours)": "#3fb950",
        "Shannon Bound": "#58a6ff",
    }

    def style_ax(ax, title, ylabel):
        ax.set_facecolor('#161b22')
        ax.set_title(title, fontsize=14, fontweight='bold', color='white', pad=12)
        ax.set_ylabel(ylabel, fontsize=11, color='#c9d1d9')
        ax.tick_params(colors='#8b949e', labelsize=9)
        ax.spines['bottom'].set_color('#30363d')
        ax.spines['left'].set_color('#30363d')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', color='#21262d', linewidth=0.5)

    # --- Plot 1: Throughput vs Oracle (%) ---
    ax1 = fig.add_subplot(gs[0, 0])
    vals = [agg[m]["thr_vs_oracle"][0] for m in methods_order if m in agg]
    errs = [agg[m]["thr_vs_oracle"][1] for m in methods_order if m in agg]
    labels = [m for m in methods_order if m in agg]
    bars = ax1.barh(range(len(vals)), vals, xerr=errs,
                    color=[colors[m] for m in labels],
                    edgecolor='#30363d', linewidth=0.5, capsize=3,
                    error_kw={'ecolor': '#c9d1d9', 'linewidth': 1})
    ax1.set_yticks(range(len(labels)))
    ax1.set_yticklabels(labels, fontsize=9, color='#c9d1d9')
    ax1.axvline(x=100, color='#3fb950', linestyle='--', alpha=0.5, linewidth=1)
    ax1.set_xlim(0, max(vals) * 1.15)
    for i, (v, e) in enumerate(zip(vals, errs)):
        ax1.text(v + e + 1, i, f'{v:.1f}%', va='center',
                 fontsize=9, color='#c9d1d9', fontweight='bold')
    style_ax(ax1, 'Throughput vs Oracle (%)', 'Method')

    # --- Plot 2: BLER Violation Rate (%) ---
    ax2 = fig.add_subplot(gs[0, 1])
    vals2 = [agg[m]["bler_violation"][0] * 100 for m in methods_order if m in agg]
    errs2 = [agg[m]["bler_violation"][1] * 100 for m in methods_order if m in agg]
    bars2 = ax2.barh(range(len(vals2)), vals2, xerr=errs2,
                     color=[colors[m] for m in labels],
                     edgecolor='#30363d', linewidth=0.5, capsize=3,
                     error_kw={'ecolor': '#c9d1d9', 'linewidth': 1})
    ax2.set_yticks(range(len(labels)))
    ax2.set_yticklabels(labels, fontsize=9, color='#c9d1d9')
    ax2.axvline(x=10, color='#f85149', linestyle='--', alpha=0.5, linewidth=1,
                label='10% BLER Target')
    for i, (v, e) in enumerate(zip(vals2, errs2)):
        ax2.text(v + e + 0.5, i, f'{v:.1f}%', va='center',
                 fontsize=9, color='#c9d1d9', fontweight='bold')
    style_ax(ax2, 'BLER Violation Rate (%)', '')
    ax2.legend(loc='lower right', fontsize=8, facecolor='#161b22',
               edgecolor='#30363d', labelcolor='#c9d1d9')

    # --- Plot 3: Asymmetric Rank Cost ---
    ax3 = fig.add_subplot(gs[1, 0])
    vals3 = [agg[m]["asym_cost"][0] for m in methods_order if m in agg]
    errs3 = [agg[m]["asym_cost"][1] for m in methods_order if m in agg]
    bars3 = ax3.barh(range(len(vals3)), vals3, xerr=errs3,
                     color=[colors[m] for m in labels],
                     edgecolor='#30363d', linewidth=0.5, capsize=3,
                     error_kw={'ecolor': '#c9d1d9', 'linewidth': 1})
    ax3.set_yticks(range(len(labels)))
    ax3.set_yticklabels(labels, fontsize=9, color='#c9d1d9')
    for i, (v, e) in enumerate(zip(vals3, errs3)):
        ax3.text(v + e + 0.02, i, f'{v:.3f}', va='center',
                 fontsize=9, color='#c9d1d9', fontweight='bold')
    style_ax(ax3, 'Asymmetric Rank Cost (lower = better)', '')

    # --- Plot 4: Accuracy ---
    ax4 = fig.add_subplot(gs[1, 1])
    vals4 = [agg[m]["accuracy"][0] * 100 for m in methods_order if m in agg]
    errs4 = [agg[m]["accuracy"][1] * 100 for m in methods_order if m in agg]
    bars4 = ax4.barh(range(len(vals4)), vals4, xerr=errs4,
                     color=[colors[m] for m in labels],
                     edgecolor='#30363d', linewidth=0.5, capsize=3,
                     error_kw={'ecolor': '#c9d1d9', 'linewidth': 1})
    ax4.set_yticks(range(len(labels)))
    ax4.set_yticklabels(labels, fontsize=9, color='#c9d1d9')
    ax4.set_xlim(0, 105)
    for i, (v, e) in enumerate(zip(vals4, errs4)):
        ax4.text(v + e + 1, i, f'{v:.1f}%', va='center',
                 fontsize=9, color='#c9d1d9', fontweight='bold')
    style_ax(ax4, 'MCS Selection Accuracy (%)', '')

    fig.suptitle(
        'Link Adaptation Benchmark: DNN Regressor (Paper) vs '
        'Ordinal GBM (Ours) vs Shannon Bound',
        fontsize=16, fontweight='bold', color='white', y=0.98,
    )
    fig.text(0.5, 0.01,
             f'Evaluated on 3GPP TDL channels with LDPC+HARQ-CC | '
             f'{N_BOOTSTRAP_SEEDS} seeds | 95% CI error bars',
             ha='center', fontsize=10, color='#8b949e')

    plt.savefig('la_benchmark_comparison.png', dpi=200,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    print("  → Saved la_benchmark_comparison.png")
    plt.close()


# ============================================================================
# 8. Entry Point
# ============================================================================

def main():
    print("=" * 60)
    print("  Link Adaptation Benchmark Suite — V2")
    print("  Paper DNN / DNN Clf / GRU / Ordinal GBM / Shannon")
    print("=" * 60)

    print("\n1. Loading packet-level dataset…")
    df = pd.read_csv("sionna_realistic_dataset.csv")
    print(f"   {len(df):,} rows loaded.")

    feature_cols = ["Measured_SINR", "Measured_Speed", "Channel_Ordinal"]

    print(f"\n2. Running {N_BOOTSTRAP_SEEDS} seeds for 95% CI…")
    all_results = []
    all_oracles = []
    for i in range(N_BOOTSTRAP_SEEDS):
        seed = RANDOM_SEED + i * 7
        print(f"   Seed {i+1}/{N_BOOTSTRAP_SEEDS} (seed={seed})…", end=" ")
        results, oracle_thr = run_single_seed(df, seed, feature_cols)
        all_results.append(results)
        all_oracles.append(oracle_thr)
        best = max(results.values(), key=lambda r: r.thr_vs_oracle)
        print(f"best={best.name} ({best.thr_vs_oracle:.1f}% oracle)")

    print("\n3. Aggregating results…")
    agg = aggregate_results(all_results)
    print_results_table(agg)

    print("4. Generating plots…")
    generate_plots(all_results, all_oracles, agg)

    print("\n5. Key Findings:")
    our_thr = agg["Ordinal GBM (Ours)"]["thr_vs_oracle"][0]
    paper_thr = agg["DNN 1D (Paper)"]["thr_vs_oracle"][0]
    paper_bler = agg["DNN 1D (Paper)"]["bler_violation"][0]
    our_bler = agg["Ordinal GBM (Ours)"]["bler_violation"][0]
    print(f"   • Ordinal GBM achieves {our_thr:.1f}% of oracle throughput "
          f"vs paper's DNN at {paper_thr:.1f}%")
    print(f"   • Paper's DNN BLER violation rate: {paper_bler*100:.1f}% "
          f"vs ours: {our_bler*100:.1f}%")

    gap = our_thr - paper_thr
    if gap > 0:
        print(f"   → Our approach outperforms by +{gap:.1f} percentage points")
    else:
        print(f"   → Paper's approach leads by {-gap:.1f} percentage points")

    print("\n✓ Benchmark complete.\n")


if __name__ == "__main__":
    main()
