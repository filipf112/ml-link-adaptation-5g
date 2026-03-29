import numpy as np
import pandas as pd
import warnings
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupShuffleSplit


RANDOM_SEED = 42
BLER_TARGET = 0.10
CONFIDENCE_THRESHOLD = 0.30   # min margin for confidence gating
CPP_EXPORT_PATH = "la_policy_exported.cpp"

# NR MCS indices that appear in the dataset (must match generate script)
NR_MCS_INDICES = sorted([3, 4, 9, 11, 14, 17, 20, 24, 25])


# ---------------------------------------------------------------------------
# Ordinal MCS Policy — K-1 cumulative binary classifiers
# ---------------------------------------------------------------------------

class OrdinalMCSPolicy:
    """Ordinal regression for MCS selection.

    Decomposes the K-class problem into K-1 binary classifiers, each answering:
        P(optimal_MCS >= threshold_k  |  features)

    Each binary classifier supports sklearn's monotonic_cst because it is
    binary classification — the multiclass limitation does not apply.

    Final prediction: highest MCS index where P >= decision_threshold.
    Confidence: minimum margin across all threshold decisions.
    """

    def __init__(self, mcs_indices, decision_threshold=0.5, **hgb_kwargs):
        self.mcs_sorted = sorted(mcs_indices)
        self.thresholds = self.mcs_sorted[1:]   # K-1 thresholds
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
        """Minimum margin (|P - 0.5|) across all threshold decisions."""
        margins = np.ones(len(X))
        for t in self.thresholds:
            if self.models[t] is None:
                continue
            proba = self.models[t].predict_proba(X)[:, 1]
            margin = np.abs(proba - self.decision_threshold)
            margins = np.minimum(margins, margin)
        return margins


# ---------------------------------------------------------------------------
# 1. Label construction — reward-aware with BLER constraint
# ---------------------------------------------------------------------------

def build_optimal_label_table(df):
    """Aggregate per-packet logs into (context, MCS) → (BLER, throughput)
    and select the optimal MCS per context under a BLER ≤ 10 % constraint."""
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


# ---------------------------------------------------------------------------
# 2. Feature engineering — richer than binary High_Delay_Spread
# ---------------------------------------------------------------------------

CHANNEL_ORDINAL = {"TDL-A": 0, "TDL-B": 1, "TDL-C": 2}


def add_features(optimal_df, rng):
    """Create realistic, observable features with measurement noise."""
    out = optimal_df.copy()

    # CQI / SINR measurement noise
    sinr_noise_db = rng.normal(0.0, 1.5, len(out))

    # Non-linear CQI aging: Doppler decorrelation grows sub-linearly
    speed_vals = out["Speed_kmph"].values
    cqi_aging_db = 0.5 * (1.0 - np.exp(-speed_vals / 30.0))
    out["Measured_SINR"] = out["SINR_dB"] + sinr_noise_db - cqi_aging_db

    # Speed estimator error
    out["Measured_Speed"] = out["Speed_kmph"] + rng.normal(0.0, 10.0, len(out))
    out["Measured_Speed"] = out["Measured_Speed"].clip(lower=0.0)

    # Ordinal channel encoding (0=best, 2=worst) — captures A vs B vs C
    out["Channel_Ordinal"] = out["Channel"].map(CHANNEL_ORDINAL).astype(np.float64)

    return out


# ---------------------------------------------------------------------------
# 3. Baseline policy (static SINR thresholds)
# ---------------------------------------------------------------------------

def baseline_mcs_from_sinr(sinr_db):
    """Simple static baseline for the 9-entry MCS table."""
    if sinr_db < -2.0:
        return 3
    if sinr_db < 2.0:
        return 4
    if sinr_db < 6.0:
        return 9
    if sinr_db < 10.0:
        return 11
    if sinr_db < 14.0:
        return 14
    if sinr_db < 17.0:
        return 17
    if sinr_db < 21.0:
        return 20
    if sinr_db < 25.0:
        return 24
    return 25


# ---------------------------------------------------------------------------
# 4. Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_policy(eval_df, pred_mcs, stats_table):
    """Policy-level KPIs from lookup in measured packet outcomes."""
    joined = eval_df[["SINR_dB", "Channel", "Speed_kmph"]].copy()
    joined["MCS_Index"] = pred_mcs
    joined = joined.merge(
        stats_table,
        on=["SINR_dB", "Channel", "Speed_kmph", "MCS_Index"],
        how="left",
    )
    mean_thr = joined["Throughput"].mean()
    bler_violation_rate = (joined["BLER"] > BLER_TARGET).mean()
    return mean_thr, bler_violation_rate


def asymmetric_rank_cost(y_true, y_pred, overestimate_penalty=3.0):
    """Over-estimation (MCS too high) costs 3× more than under-estimation."""
    mcs_to_rank = {idx: rank for rank, idx in enumerate(NR_MCS_INDICES)}
    true_rank = np.array([mcs_to_rank.get(v, 0) for v in y_true])
    pred_rank = np.array([mcs_to_rank.get(v, 0) for v in y_pred])
    diff = pred_rank - true_rank
    cost = np.where(diff > 0, diff * overestimate_penalty, np.abs(diff))
    return cost.mean()


def rolling_time_split_eval(optimal_df, stats_table, predictor_fn):
    """Expanding-window train + fixed-size test to approximate drift."""
    ordered = optimal_df.copy()
    if "Time_Index" in ordered.columns:
        ordered = ordered.sort_values("Time_Index").reset_index(drop=True)
    else:
        ordered = ordered.reset_index(drop=True)

    n = len(ordered)
    test_size = max(int(n * 0.15), 1)
    stride = max(int(n * 0.10), 1)

    rows = []
    start = max(int(n * 0.50), 1)
    while (start + test_size) <= n:
        window = ordered.iloc[start : start + test_size].reset_index(drop=True)
        pred = predictor_fn(window)
        thr, bler_v = evaluate_policy(window, pred, stats_table)
        rows.append((len(rows) + 1, thr, bler_v))
        start += stride
    return rows


# ---------------------------------------------------------------------------
# 5. C++ export — surrogate tree + OLLA safety layer
# ---------------------------------------------------------------------------

def export_tree_to_cpp(clf, feature_cols, mcs_indices, out_path):
    """Export a sklearn decision tree as C++ if/else logic, wrapped in an
    OLLA safety layer with per-UE BLER EMA tracking and kill-switch."""
    tree = clf.tree_
    classes = clf.classes_.astype(int)
    cpp_feat = {
        "Measured_SINR": "measured_sinr_db",
        "Measured_Speed": "measured_speed_kmph",
        "Channel_Ordinal": "channel_ordinal",
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

    cpp = [
        "// Auto-generated NR Link Adaptation policy with OLLA safety layer.",
        "// Ordinal GBM → surrogate tree distillation.",
        "#include <algorithm>",
        "#include <cmath>",
        "#include <cstdint>",
        "",
        f"static constexpr int VALID_MCS[{n_mcs}] = {{{mcs_arr}}};",
        f"static constexpr int N_MCS = {n_mcs};",
        "",
        "// Per-UE OLLA state — one instance per RNTI in the scheduler.",
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
        "// ML base policy (surrogate decision tree from ordinal GBM).",
        "static int select_mcs_base(double measured_sinr_db,",
        "                           double measured_speed_kmph,",
        "                           int    channel_ordinal) {",
        "    measured_speed_kmph = std::max(0.0, measured_speed_kmph);",
    ]
    cpp.extend(body)
    cpp.append("}")
    cpp.append("")
    cpp.extend([
        "// Full LA decision: ML base + OLLA outer-loop + emergency kill-switch.",
        "int select_mcs(double measured_sinr_db,",
        "               double measured_speed_kmph,",
        "               int    channel_ordinal,",
        "               OllaState& state,",
        "               bool   last_was_ack) {",
        "",
        "    // --- BLER EMA update (α = 0.02 ≈ 50 TTI window) ---",
        "    constexpr double kAlpha = 0.02;",
        "    state.bler_ema = kAlpha * (last_was_ack ? 0.0 : 1.0)",
        "                   + (1.0 - kAlpha) * state.bler_ema;",
        "",
        "    // --- Emergency kill-switch: 100 consecutive NACKs ---",
        "    state.consecutive_nack = last_was_ack ? 0",
        "                           : state.consecutive_nack + 1;",
        "    if (state.consecutive_nack > 100) {",
        f"        return VALID_MCS[0];  // fallback to MCS {sorted(mcs_indices)[0]}",
        "    }",
        "",
        "    // --- OLLA offset adjustment ---",
        "    constexpr double kBlerTarget = 0.10;",
        "    if (state.bler_ema > kBlerTarget) {",
        "        state.mcs_offset = std::max(state.mcs_offset - 1, -3);",
        "    } else if (state.bler_ema < kBlerTarget * 0.5) {",
        "        state.mcs_offset = std::min(state.mcs_offset + 1, 2);",
        "    }",
        "",
        "    // --- Combine ML base + OLLA offset ---",
        "    int base = select_mcs_base(measured_sinr_db, measured_speed_kmph,",
        "                               channel_ordinal);",
        "    int adjusted = base + state.mcs_offset;",
        f"    adjusted = std::clamp(adjusted, VALID_MCS[0], VALID_MCS[N_MCS - 1]);",
        "    return nearest_valid_mcs(adjusted);",
        "}",
    ])

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(cpp) + "\n")


# ---------------------------------------------------------------------------
# 6. Main training & evaluation pipeline
# ---------------------------------------------------------------------------

def train_and_evaluate():
    print("1. Loading packet-level logs …")
    df = pd.read_csv("sionna_realistic_dataset.csv")
    rng = np.random.default_rng(RANDOM_SEED)

    print("2. Building optimal labels (BLER ≤ 10 % constraint) …")
    stats_table, optimal_df = build_optimal_label_table(df)
    optimal_df = add_features(optimal_df, rng)

    feature_cols = ["Measured_SINR", "Measured_Speed", "Channel_Ordinal"]
    X = optimal_df[feature_cols]
    y = optimal_df["MCS_Index"]

    # Group split to avoid leakage across correlated contexts
    groups = (
        optimal_df["Channel"].astype(str)
        + "_" + optimal_df["Speed_kmph"].astype(str)
    )
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=0.25, random_state=RANDOM_SEED,
    )
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    eval_df = optimal_df.iloc[test_idx].reset_index(drop=True)

    # ---- Ordinal MCS Policy (K-1 binary classifiers) ----
    # Each binary classifier: P(MCS >= threshold_k | features)
    # Monotonic: SINR↑ ⇒ P↑  ;  speed↑ ⇒ P↓  ;  channel↑ (worse) ⇒ P↓
    print("3. Training Ordinal MCS Policy (8 binary classifiers, monotonic) …")
    policy = OrdinalMCSPolicy(
        mcs_indices=NR_MCS_INDICES,
        decision_threshold=0.5,
        max_iter=300,
        max_depth=4,
        min_samples_leaf=3,
        learning_rate=0.1,
        monotonic_cst=[1, -1, -1],
        random_state=RANDOM_SEED,
    )
    policy.fit(X_train, y_train)

    pred_raw = policy.predict(X_test)

    # Confidence-gated fallback using threshold margin
    confidence = policy.predict_confidence(X_test)
    fallback_mask = confidence < CONFIDENCE_THRESHOLD
    fallback_values = np.array(
        [baseline_mcs_from_sinr(v) for v in X_test["Measured_SINR"].values],
    )
    pred_safe = pred_raw.copy()
    pred_safe[fallback_mask] = np.minimum(
        pred_raw[fallback_mask], fallback_values[fallback_mask],
    )

    # ---- Metrics ----
    acc_raw = accuracy_score(y_test, pred_raw)
    acc_safe = accuracy_score(y_test, pred_safe)
    cost_raw = asymmetric_rank_cost(y_test.values, pred_raw)
    cost_safe = asymmetric_rank_cost(y_test.values, pred_safe)

    thr_oracle, bler_oracle = evaluate_policy(eval_df, y_test.values, stats_table)
    thr_raw, bler_raw = evaluate_policy(eval_df, pred_raw, stats_table)
    thr_safe, bler_safe = evaluate_policy(eval_df, pred_safe, stats_table)

    print("\n=== Evaluation Summary ===")
    print(f"Dataset rows           : {len(df)}")
    print(f"Policy contexts        : {len(optimal_df)}")
    print(f"Test contexts          : {len(X_test)}")
    print(f"Raw model accuracy     : {acc_raw * 100:.2f}%")
    print(f"Safety-gated accuracy  : {acc_safe * 100:.2f}%")
    print(f"Fallback activation    : {fallback_mask.mean() * 100:.2f}%")

    thr_ratio = thr_raw / thr_oracle * 100 if thr_oracle > 0 else 0
    print(f"\nThroughput vs Oracle   : {thr_ratio:.1f}%")

    print("\nAsymmetric rank cost (lower = better, over-est penalised 3×):")
    print(f"  Raw model : {cost_raw:.3f}")
    print(f"  Safe model: {cost_safe:.3f}")

    print("\nPolicy KPI vs Oracle:")
    print(f"  Oracle   — throughput: {thr_oracle:.4f},"
          f" BLER violations: {bler_oracle * 100:.2f}%")
    print(f"  Ord. raw — throughput: {thr_raw:.4f},"
          f" BLER violations: {bler_raw * 100:.2f}%")
    print(f"  Ord. safe— throughput: {thr_safe:.4f},"
          f" BLER violations: {bler_safe * 100:.2f}%")

    print("\nRolling time-split KPI (drift proxy):")
    rolling_rows = rolling_time_split_eval(
        optimal_df, stats_table,
        predictor_fn=lambda frame: policy.predict(
            frame[feature_cols],
        ),
    )
    if not rolling_rows:
        print("  Not enough contexts for rolling validation.")
    else:
        for fold_id, fold_thr, fold_bler_v in rolling_rows:
            print(f"  Fold {fold_id}: throughput={fold_thr:.4f},"
                  f" BLER violations={fold_bler_v * 100:.2f}%")

    # ---- Surrogate distillation for C++ export ----
    print("\n4. Distilling into surrogate tree (max_depth=12) for C++ …")
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
    print(f"Exported C++ policy   : {CPP_EXPORT_PATH}")
    print(f"Surrogate fidelity    : {fidelity * 100:.2f}%")
    print("C++ export includes OLLA outer-loop and per-UE BLER kill-switch.")


if __name__ == "__main__":
    train_and_evaluate()
