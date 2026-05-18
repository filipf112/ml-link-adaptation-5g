"""
Additional Experiments for ML Link Adaptation Report
=====================================================
1. SNR Sweep Analysis — MCS vs SINR staircase curve
2. Confusion Matrix — Teacher vs Student distillation quality
3. Per-Channel Breakdown — TDL-A vs TDL-B vs TDL-C performance
"""

import numpy as np
import pandas as pd
import warnings
from collections import defaultdict

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker

# Import from existing training script
from train_real_ml_model import (
    detect_dataset_version, normalize_dataset, build_multi_service_labels,
    add_features, NR_MCS_INDICES, CHANNEL_ORDINAL, NR_MCS_SE_MAP,
    CONFIDENCE_THRESHOLD
)

RANDOM_SEED = 42
FEATURES = ['Measured_SINR', 'Measured_Speed', 'Channel_Ordinal',
            'Carrier_Band', 'Num_Antennas', 'BLER_Target_Log']
MONO_CONSTRAINTS = [1, -1, -1, 0, 0, 1]


# ============================================================================
# 0. Data Loading & Model Training (reusable)
# ============================================================================

def load_and_train():
    """Load dataset, train teacher GBM and student tree."""
    print("=" * 70)
    print("  Loading dataset and training models...")
    print("=" * 70)

    # Load dataset
    for path in ["sionna_v2_dataset.csv", "sionna_realistic_dataset.csv"]:
        try:
            df = pd.read_csv(path)
            break
        except FileNotFoundError:
            continue
    else:
        raise FileNotFoundError("No dataset found!")

    version = detect_dataset_version(df)
    df = normalize_dataset(df, version)
    print(f"   Dataset: {version}, {len(df):,} rows")

    # Build labels
    stats, combined = build_multi_service_labels(df)
    rng = np.random.default_rng(RANDOM_SEED)
    combined = add_features(combined, rng, version)

    # Map to valid MCS indices
    valid_mcs_set = set(NR_MCS_INDICES)
    combined = combined[combined["MCS_Index"].isin(valid_mcs_set)].copy()

    X = combined[FEATURES].values
    y = combined["MCS_Index"].values

    # Build context groups for splitting
    context_cols = ["SINR_dB", "Channel", "Speed_kmph"]
    if "Num_Streams" in combined.columns:
        context_cols += ["Num_Streams", "Carrier_GHz"]
    if "BLER_Target_Log" in combined.columns:
        context_cols += ["BLER_Target_Log"]
    groups = combined[context_cols].apply(
        lambda row: "_".join(str(v) for v in row), axis=1
    ).values

    # Split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_SEED)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Train teacher (GBM — no monotonic_cst for multiclass in sklearn)
    teacher = HistGradientBoostingClassifier(
        max_iter=300, max_depth=8, learning_rate=0.05,
        random_state=RANDOM_SEED
    )
    teacher.fit(X_train, y_train)
    teacher_pred = teacher.predict(X_test)
    teacher_acc = accuracy_score(y_test, teacher_pred)
    print(f"   Teacher (GBM) test accuracy: {teacher_acc:.1%}")

    # Train student (Distilled Tree)
    teacher_train_pred = teacher.predict(X_train)
    student = DecisionTreeClassifier(max_depth=10, random_state=RANDOM_SEED)
    student.fit(X_train, teacher_train_pred)
    student_pred = student.predict(X_test)
    student_acc = accuracy_score(y_test, student_pred)
    fidelity = accuracy_score(teacher_pred, student_pred)
    print(f"   Student (Tree) test accuracy: {student_acc:.1%}")
    print(f"   Distillation fidelity: {fidelity:.1%}")

    return {
        "teacher": teacher, "student": student,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "teacher_pred": teacher_pred, "student_pred": student_pred,
        "combined": combined, "test_idx": test_idx
    }


# ============================================================================
# 1. SNR Sweep Analysis
# ============================================================================

def experiment_snr_sweep(models):
    """Generate MCS vs SINR staircase curves for different methods."""
    print("\n" + "=" * 70)
    print("  Experiment 1: SNR Sweep Analysis")
    print("=" * 70)

    teacher = models["teacher"]
    student = models["student"]

    sinr_range = np.linspace(-8, 30, 200)

    # Fixed scenario parameters
    scenarios = [
        {"name": "Pedestrian (3 km/h, TDL-A)",
         "speed": 3.0, "channel": 1.0, "band": 0.0, "ant": 1.0},
        {"name": "Urban (30 km/h, TDL-B)",
         "speed": 30.0, "channel": 2.0, "band": 0.0, "ant": 2.0},
        {"name": "Highway (120 km/h, TDL-C)",
         "speed": 120.0, "channel": 4.0, "band": 0.0, "ant": 2.0},
    ]

    bler_target_log = -1.0  # 10% BLER

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    fig.suptitle("MCS Selection vs SINR — SNR Sweep Analysis",
                 fontsize=14, fontweight='bold', y=1.02)

    for idx, scenario in enumerate(scenarios):
        ax = axes[idx]

        # Build feature matrix
        X_sweep = np.column_stack([
            sinr_range,
            np.full_like(sinr_range, scenario["speed"]),
            np.full_like(sinr_range, scenario["channel"]),
            np.full_like(sinr_range, scenario["band"]),
            np.full_like(sinr_range, scenario["ant"]),
            np.full_like(sinr_range, bler_target_log),
        ])

        # Predictions
        teacher_mcs = teacher.predict(X_sweep)
        student_mcs = student.predict(X_sweep)

        # Static LUT baseline (simple SINR-to-MCS mapping)
        lut_mcs = np.array([_static_lut(s) for s in sinr_range])

        ax.step(sinr_range, teacher_mcs, where='post', label='Ordinal GBM (Teacher)',
                color='#2196F3', linewidth=2.0, alpha=0.9)
        ax.step(sinr_range, student_mcs, where='post', label='Decision Tree (Student)',
                color='#FF5722', linewidth=1.5, linestyle='--', alpha=0.9)
        ax.step(sinr_range, lut_mcs, where='post', label='Static LUT',
                color='#9E9E9E', linewidth=1.2, linestyle=':', alpha=0.7)

        ax.set_xlabel("SINR (dB)", fontsize=11)
        if idx == 0:
            ax.set_ylabel("Selected MCS Index", fontsize=11)
        ax.set_title(scenario["name"], fontsize=11)
        ax.set_yticks(NR_MCS_INDICES)
        ax.set_xlim(-8, 30)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=8)

        # Add horizontal lines for valid MCS indices
        for mcs in NR_MCS_INDICES:
            ax.axhline(y=mcs, color='gray', alpha=0.1, linewidth=0.5)

    plt.tight_layout()
    plt.savefig("results/snr_sweep_analysis.png", dpi=150, bbox_inches='tight')
    print("   → Saved results/snr_sweep_analysis.png")
    plt.close()


def _static_lut(sinr_db):
    """Simple SINR→MCS lookup table baseline."""
    thresholds = [
        (-4.0, 3), (-1.0, 4), (2.0, 9), (5.0, 11),
        (8.0, 14), (12.0, 17), (16.0, 20), (20.0, 24), (24.0, 25)
    ]
    mcs = 3
    for threshold, m in thresholds:
        if sinr_db >= threshold:
            mcs = m
    return mcs


# ============================================================================
# 2. Confusion Matrix — Distillation Quality
# ============================================================================

def experiment_confusion_matrix(models):
    """Analyze teacher vs student disagreements."""
    print("\n" + "=" * 70)
    print("  Experiment 2: Distillation Confusion Matrix")
    print("=" * 70)

    teacher_pred = models["teacher_pred"]
    student_pred = models["student_pred"]
    y_test = models["y_test"]

    # Teacher vs Student confusion
    labels = sorted(set(teacher_pred) | set(student_pred))
    cm_fidelity = confusion_matrix(teacher_pred, student_pred, labels=labels)

    # Teacher vs Ground Truth confusion
    gt_labels = sorted(set(y_test) | set(teacher_pred))
    cm_teacher = confusion_matrix(y_test, teacher_pred, labels=gt_labels)

    # MCS error distribution (student - teacher)
    mcs_errors = student_pred.astype(int) - teacher_pred.astype(int)

    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    fig.suptitle("Distillation Quality Analysis",
                 fontsize=14, fontweight='bold', y=1.02)

    # Plot 1: Teacher vs Student confusion matrix
    ax = axes[0]
    cm_norm = cm_fidelity.astype(float) / cm_fidelity.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Student (Decision Tree)", fontsize=10)
    ax.set_ylabel("Teacher (Ordinal GBM)", fontsize=10)
    ax.set_title("Teacher vs Student Agreement", fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Add text annotations
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = cm_norm[i, j]
            if val > 0.01:
                color = 'white' if val > 0.5 else 'black'
                ax.text(j, i, f'{val:.0%}', ha='center', va='center',
                        fontsize=7, color=color)

    # Plot 2: MCS Error Distribution
    ax = axes[1]
    unique_errors, counts = np.unique(mcs_errors, return_counts=True)
    colors = ['#4CAF50' if e == 0 else ('#FF9800' if abs(e) <= 3 else '#F44336')
              for e in unique_errors]
    ax.bar(unique_errors, counts / len(mcs_errors) * 100, color=colors, width=0.8)
    ax.set_xlabel("MCS Error (Student − Teacher)", fontsize=10)
    ax.set_ylabel("Frequency (%)", fontsize=10)
    ax.set_title("MCS Prediction Error Distribution", fontsize=11)
    ax.axvline(x=0, color='green', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')

    # Statistics
    exact_match = np.mean(mcs_errors == 0)
    within_1step = np.mean(np.abs(mcs_errors) <= 2)
    mean_err = np.mean(np.abs(mcs_errors))
    ax.text(0.95, 0.95,
            f'Exact: {exact_match:.1%}\n±1 step: {within_1step:.1%}\nMAE: {mean_err:.1f}',
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 3: Cumulative error distribution
    ax = axes[2]
    abs_errors = np.abs(mcs_errors)
    max_err = max(abs_errors)
    error_thresholds = range(0, int(max_err) + 1)
    cum_pct = [np.mean(abs_errors <= t) * 100 for t in error_thresholds]
    ax.plot(error_thresholds, cum_pct, 'o-', color='#2196F3', linewidth=2, markersize=6)
    ax.fill_between(error_thresholds, cum_pct, alpha=0.15, color='#2196F3')
    ax.set_xlabel("MCS Error Tolerance (|Student − Teacher|)", fontsize=10)
    ax.set_ylabel("Cumulative Accuracy (%)", fontsize=10)
    ax.set_title("Cumulative Distillation Accuracy", fontsize=11)
    ax.axhline(y=90, color='red', linestyle='--', alpha=0.4, label='90% threshold')
    ax.axhline(y=95, color='orange', linestyle='--', alpha=0.4, label='95% threshold')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig("results/distillation_analysis.png", dpi=150, bbox_inches='tight')
    print("   → Saved results/distillation_analysis.png")
    plt.close()

    # Print summary
    print(f"   Exact match (teacher=student): {exact_match:.1%}")
    print(f"   Within ±1 MCS step:            {within_1step:.1%}")
    print(f"   Mean absolute MCS error:        {mean_err:.2f}")


# ============================================================================
# 3. Per-Channel Model Breakdown
# ============================================================================

def experiment_channel_breakdown(models):
    """Evaluate model performance per channel model."""
    print("\n" + "=" * 70)
    print("  Experiment 3: Per-Channel Model Breakdown")
    print("=" * 70)

    combined = models["combined"]
    test_idx = models["test_idx"]
    teacher = models["teacher"]
    student = models["student"]

    test_data = combined.iloc[test_idx].copy()
    X_test = test_data[FEATURES].values
    y_test = test_data["MCS_Index"].values

    teacher_pred = teacher.predict(X_test)
    student_pred = student.predict(X_test)

    channels = sorted(test_data["Channel"].unique())
    channel_results = []

    for ch in channels:
        mask = test_data["Channel"].values == ch
        if mask.sum() < 5:
            continue
        n = mask.sum()
        y_ch = y_test[mask]
        t_pred_ch = teacher_pred[mask]
        s_pred_ch = student_pred[mask]

        # Teacher metrics
        t_acc = accuracy_score(y_ch, t_pred_ch)
        t_se = np.mean([NR_MCS_SE_MAP.get(m, 0) for m in t_pred_ch])
        t_oracle_se = np.mean([NR_MCS_SE_MAP.get(m, 0) for m in y_ch])
        t_ratio = t_se / t_oracle_se if t_oracle_se > 0 else 0

        # Student metrics
        s_acc = accuracy_score(y_ch, s_pred_ch)
        s_se = np.mean([NR_MCS_SE_MAP.get(m, 0) for m in s_pred_ch])
        s_ratio = s_se / t_oracle_se if t_oracle_se > 0 else 0

        # Over/under estimation
        over_est = np.mean(t_pred_ch > y_ch)
        under_est = np.mean(t_pred_ch < y_ch)

        channel_results.append({
            "Channel": ch, "N": n,
            "Teacher_Acc": t_acc, "Teacher_Oracle%": t_ratio,
            "Student_Acc": s_acc, "Student_Oracle%": s_ratio,
            "Over_Est%": over_est, "Under_Est%": under_est,
            "Oracle_SE": t_oracle_se
        })

    results_df = pd.DataFrame(channel_results)

    # Print table
    print(f"\n   {'Channel':<10} {'N':>5} {'Teacher':>10} {'Student':>10} "
          f"{'vs Oracle':>10} {'Over-est':>10} {'Under-est':>10}")
    print("   " + "-" * 68)
    for _, row in results_df.iterrows():
        print(f"   {row['Channel']:<10} {row['N']:>5} "
              f"{row['Teacher_Acc']:>9.1%} {row['Student_Acc']:>9.1%} "
              f"{row['Teacher_Oracle%']:>9.1%} "
              f"{row['Over_Est%']:>9.1%} {row['Under_Est%']:>9.1%}")

    # Generate plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Per-Channel Model Performance Breakdown",
                 fontsize=14, fontweight='bold', y=1.02)

    ch_labels = results_df["Channel"].values
    x = np.arange(len(ch_labels))
    width = 0.35

    # Plot 1: Accuracy comparison
    ax = axes[0]
    b1 = ax.bar(x - width/2, results_df["Teacher_Acc"] * 100, width,
                label='Teacher (GBM)', color='#2196F3', alpha=0.85)
    b2 = ax.bar(x + width/2, results_df["Student_Acc"] * 100, width,
                label='Student (Tree)', color='#FF5722', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(ch_labels, fontsize=9)
    ax.set_ylabel("Accuracy (%)", fontsize=10)
    ax.set_title("Classification Accuracy", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 2: Oracle throughput ratio
    ax = axes[1]
    b1 = ax.bar(x - width/2, results_df["Teacher_Oracle%"] * 100, width,
                label='Teacher (GBM)', color='#2196F3', alpha=0.85)
    b2 = ax.bar(x + width/2, results_df["Student_Oracle%"] * 100, width,
                label='Student (Tree)', color='#FF5722', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(ch_labels, fontsize=9)
    ax.set_ylabel("% of Oracle Throughput", fontsize=10)
    ax.set_title("Throughput vs Oracle", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 3: Over/Under estimation
    ax = axes[2]
    b1 = ax.bar(x - width/2, results_df["Over_Est%"] * 100, width,
                label='Over-estimation (risk)', color='#F44336', alpha=0.85)
    b2 = ax.bar(x + width/2, results_df["Under_Est%"] * 100, width,
                label='Under-estimation (safe)', color='#4CAF50', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(ch_labels, fontsize=9)
    ax.set_ylabel("Frequency (%)", fontsize=10)
    ax.set_title("Prediction Error Direction", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig("results/channel_breakdown.png", dpi=150, bbox_inches='tight')
    print("   → Saved results/channel_breakdown.png")
    plt.close()

    return results_df


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    models = load_and_train()

    experiment_snr_sweep(models)
    experiment_confusion_matrix(models)
    channel_results = experiment_channel_breakdown(models)

    print("\n" + "=" * 70)
    print("  All experiments complete!")
    print("  Results saved to results/")
    print("=" * 70)
