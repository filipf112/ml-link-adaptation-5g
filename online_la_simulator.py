"""
Closed-Loop Link Adaptation Simulator — Digital Twin
=====================================================

Uses pre-computed BLER curves from Sionna dataset as PHY abstraction
(equivalent to 3GPP system-level link-level interface) to evaluate
LA strategies in a dynamic, time-varying channel environment.

This is how real gNB schedulers work:
  1. Channel measurements → SINR estimate
  2. LA agent selects MCS
  3. Transmission attempt → HARQ ACK/NACK
  4. Agent updates its internal state from feedback
  5. Repeat every TTI (1 ms)

Scenarios:
  1. Stationary      — steady-state performance comparison
  2. Interference     — new cell activation (SIR drop mid-simulation)
  3. Mobility change  — pedestrian → vehicular mid-simulation

Agents:
  - Static LUT:     Fixed SINR→MCS mapping (no adaptation)
  - OLLA:           Classical 3GPP outer-loop offset adjustment
  - Offline GBM:    Fixed ordinal model (no HARQ-based adaptation)
  - GBM + OLLA:     Ordinal model with online offset adaptation (our approach)
"""

import os
import numpy as np
import pandas as pd
import warnings

from scipy.interpolate import interp1d
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================================
# Configuration
# ============================================================================

RANDOM_SEED = 42
BLER_TARGET = 0.10
HARQ_MAX_ROUNDS = 4
HARQ_RTT_SLOTS = 4

NR_MCS_INDICES = sorted([3, 4, 9, 11, 14, 17, 20, 24, 25])
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

CHANNEL_ORDINAL = {"TDL-A": 0, "TDL-B": 1, "TDL-C": 2,
                   "CDL-D": 0, "CDL-A": 3}


# ============================================================================
# 1. BLER Lookup — Interpolated PHY Abstraction from Sionna Dataset
# ============================================================================

class BLERLookup:
    """Pre-computed BLER curves as the digital twin's PHY layer.

    For each (channel, speed, MCS) combination, builds an interpolated
    BLER(SINR) curve from the packet-level simulation data. This is the
    standard link-level interface used in 3GPP system-level simulators.
    """

    def __init__(self, df):
        self.curves = {}
        self._build_curves(df)

    def _build_curves(self, df):
        """Build interpolated BLER(SINR) curves per (channel, speed, MCS)."""
        stats = (
            df.groupby(["SINR_dB", "Channel", "Speed_kmph", "MCS_Index"])
            .agg(BLER=("Was_Success", lambda x: 1.0 - x.mean()),
                 N=("Was_Success", "count"))
            .reset_index()
        )

        for (ch, spd, mcs), group in stats.groupby(
                ["Channel", "Speed_kmph", "MCS_Index"]):
            sinr = group["SINR_dB"].values
            bler = group["BLER"].values
            order = np.argsort(sinr)
            sinr_s, bler_s = sinr[order], bler[order]

            if len(sinr_s) < 2:
                self.curves[(ch, spd, mcs)] = lambda _s, b=bler_s[0]: b
                continue

            f = interp1d(
                sinr_s, bler_s, kind='linear',
                bounds_error=False,
                fill_value=(min(bler_s[0], 0.99), max(bler_s[-1], 0.001)),
            )
            self.curves[(ch, spd, mcs)] = f

        print(f"   Built {len(self.curves)} BLER curves.")

    def get_bler(self, sinr_db, channel, speed_kmph, mcs_index):
        """Look up BLER for given conditions."""
        key = (channel, speed_kmph, mcs_index)
        if key in self.curves:
            return float(np.clip(self.curves[key](sinr_db), 0.0, 1.0))

        # Fallback: find nearest speed
        available = [k for k in self.curves if k[0] == channel
                     and k[2] == mcs_index]
        if available:
            nearest = min(available, key=lambda k: abs(k[1] - speed_kmph))
            return float(np.clip(self.curves[nearest](sinr_db), 0.0, 1.0))

        return 0.5  # No data at all


# ============================================================================
# 2. SINR Trace Generator — Time-Varying Channel
# ============================================================================

class SINRTraceGenerator:
    """Generates correlated time-varying SINR traces.

    Models:
      - Shadow fading: AR(1) process with speed-dependent decorrelation
      - Measurement noise: i.i.d. Gaussian
      - Interference variation: AR(1) with optional drift events
    """

    def __init__(self, rng, tti_ms=1.0):
        self.rng = rng
        self.tti_s = tti_ms / 1000.0

    def generate(self, num_tti, mean_sinr_db, speed_kmph,
                 shadow_std_db=4.0, meas_noise_std_db=1.5,
                 shadow_decorr_m=50.0):
        """Generate a correlated SINR trace."""
        v_ms = speed_kmph / 3.6
        dt = self.tti_s

        # Shadow fading autocorrelation
        if v_ms > 0 and shadow_decorr_m > 0:
            alpha = np.exp(-v_ms * dt / shadow_decorr_m)
        else:
            alpha = 0.999  # Near-static

        # AR(1) shadow fading
        shadow = np.zeros(num_tti)
        innovation_std = shadow_std_db * np.sqrt(1 - alpha**2)
        for t in range(1, num_tti):
            shadow[t] = alpha * shadow[t-1] + self.rng.normal(
                0, innovation_std)

        # Measurement noise (uncorrelated)
        noise = self.rng.normal(0, meas_noise_std_db, num_tti)

        # True SINR (what the channel actually is)
        true_sinr = mean_sinr_db + shadow

        # Measured SINR (what the gNB observes — noisy)
        measured_sinr = true_sinr + noise

        return true_sinr, measured_sinr


# ============================================================================
# 3. HARQ Simulation — Closed-Loop TTI Processor
# ============================================================================

def simulate_tti(bler_lookup, true_sinr_db, mcs_index, channel, speed,
                 rng):
    """Simulate one TTI with HARQ Chase Combining.

    Returns: (was_ack, throughput, num_tx_rounds)
    """
    se = NR_MCS_SE_MAP.get(mcs_index, 0)
    if se == 0:
        return False, 0.0, 1

    total_rounds = 0
    for round_idx in range(HARQ_MAX_ROUNDS):
        total_rounds += 1
        # Chase combining gain: ~10*log10(round+1) dB of combining
        combining_gain_db = 10 * np.log10(round_idx + 1)
        effective_sinr = true_sinr_db + combining_gain_db

        bler = bler_lookup.get_bler(effective_sinr, channel, speed,
                                    mcs_index)

        if rng.random() > bler:
            # ACK — successful decode
            effective_slots = 1 + round_idx * HARQ_RTT_SLOTS
            throughput = se / effective_slots
            return True, throughput, total_rounds

    # All rounds failed
    return False, 0.0, total_rounds


# ============================================================================
# 4. LA Agent Interfaces
# ============================================================================

class StaticLUTAgent:
    """Fixed SINR→MCS threshold table. No adaptation."""

    def __init__(self):
        self.name = "Static LUT"

    def select_mcs(self, measured_sinr, **kw):
        if measured_sinr < -2.0: return 3
        if measured_sinr < 2.0: return 4
        if measured_sinr < 6.0: return 9
        if measured_sinr < 10.0: return 11
        if measured_sinr < 14.0: return 14
        if measured_sinr < 17.0: return 17
        if measured_sinr < 21.0: return 20
        if measured_sinr < 25.0: return 24
        return 25

    def update(self, was_ack, **kw):
        pass  # No adaptation


class OLLAAgent:
    """Classical 3GPP Outer Loop Link Adaptation.

    Adjusts a dB offset based on HARQ feedback:
      - ACK  → increase offset by step_up  (be more aggressive)
      - NACK → decrease offset by step_down (be more conservative)

    The BLER target determines the ratio: step_down/step_up ≈ (1-target)/target
    For BLER=10%: step_down = 9 × step_up (one NACK undoes 9 ACKs)
    """

    def __init__(self, bler_target=0.10, step_up_db=0.1):
        self.name = "OLLA"
        self.offset_db = 0.0
        self.bler_target = bler_target
        self.step_up = step_up_db
        # step_down = step_up * (1-target)/target → BLER converges to target
        self.step_down = step_up_db * (1 - bler_target) / bler_target

    def select_mcs(self, measured_sinr, **kw):
        adjusted = measured_sinr + self.offset_db
        if adjusted < -2.0: return 3
        if adjusted < 2.0: return 4
        if adjusted < 6.0: return 9
        if adjusted < 10.0: return 11
        if adjusted < 14.0: return 14
        if adjusted < 17.0: return 17
        if adjusted < 21.0: return 20
        if adjusted < 25.0: return 24
        return 25

    def update(self, was_ack, **kw):
        if was_ack:
            self.offset_db += self.step_up
        else:
            self.offset_db -= self.step_down
        # Clamp to reasonable range
        self.offset_db = np.clip(self.offset_db, -6.0, 6.0)


class OfflineGBMAgent:
    """Pre-trained ordinal GBM without online adaptation."""

    def __init__(self, policy, feature_cols):
        self.name = "Offline GBM"
        self.policy = policy
        self.feature_cols = feature_cols

    def select_mcs(self, measured_sinr, measured_speed=0, channel_ord=0,
                   carrier_band=0, num_antennas=1, **kw):
        feats = [measured_sinr, measured_speed, channel_ord]
        if len(self.feature_cols) > 3:
            feats += [carrier_band, num_antennas]
        x = pd.DataFrame([feats], columns=self.feature_cols)
        return int(self.policy.predict(x)[0])

    def update(self, was_ack, **kw):
        pass  # No adaptation


class DNNClassifierNet(nn.Module):
    """MLP classifier for MCS selection."""
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, n_classes),
        )
    def forward(self, x):
        return self.net(x)


class OfflineDNNAgent:
    """Pre-trained DNN classifier without online adaptation."""
    def __init__(self, model, feature_cols, mcs_list):
        self.name = "Offline DNN"
        self.model = model
        self.feature_cols = feature_cols
        self.mcs_list = mcs_list
        self.model.eval()

    def select_mcs(self, measured_sinr, measured_speed=0, channel_ord=0,
                   carrier_band=0, num_antennas=1, **kw):
        feats = [measured_sinr, measured_speed, channel_ord]
        if len(self.feature_cols) > 3:
            feats += [carrier_band, num_antennas]
        x = torch.FloatTensor([feats])
        with torch.no_grad():
            logits = self.model(x)
        idx = logits.argmax(dim=1).item()
        return self.mcs_list[idx]

    def update(self, was_ack, **kw):
        pass


class DNNWithOLLAAgent:
    """DNN classifier + OLLA offset adaptation."""
    def __init__(self, model, feature_cols, mcs_list,
                 bler_target=0.10, step_up_db=0.1):
        self.name = "DNN + OLLA"
        self.model = model
        self.feature_cols = feature_cols
        self.mcs_list = mcs_list
        self.model.eval()
        self.offset_db = 0.0
        self.step_up = step_up_db
        self.step_down = step_up_db * (1 - bler_target) / bler_target
        self.consecutive_nack = 0

    def select_mcs(self, measured_sinr, measured_speed=0, channel_ord=0,
                   carrier_band=0, num_antennas=1, **kw):
        if self.consecutive_nack > 50:
            return NR_MCS_INDICES[0]
        adjusted = measured_sinr + self.offset_db
        feats = [adjusted, measured_speed, channel_ord]
        if len(self.feature_cols) > 3:
            feats += [carrier_band, num_antennas]
        x = torch.FloatTensor([feats])
        with torch.no_grad():
            logits = self.model(x)
        idx = logits.argmax(dim=1).item()
        return self.mcs_list[idx]

    def update(self, was_ack, **kw):
        if was_ack:
            self.offset_db += self.step_up
            self.consecutive_nack = 0
        else:
            self.offset_db -= self.step_down
            self.consecutive_nack += 1
        self.offset_db = np.clip(self.offset_db, -6.0, 6.0)


class GBMWithOLLAAgent:
    """Our approach: pre-trained ordinal GBM + OLLA offset adaptation.

    The GBM provides the base MCS selection (learned from data).
    The OLLA offset fine-tunes in real-time based on HARQ feedback.
    This is exactly what the C++ export implements.
    """

    def __init__(self, policy, feature_cols, bler_target=0.10,
                 step_up_db=0.1):
        self.name = "GBM + OLLA"
        self.policy = policy
        self.feature_cols = feature_cols
        self.offset_db = 0.0
        self.bler_target = bler_target
        self.step_up = step_up_db
        self.step_down = step_up_db * (1 - bler_target) / bler_target
        self.bler_ema = bler_target
        self.consecutive_nack = 0

    def select_mcs(self, measured_sinr, measured_speed=0, channel_ord=0,
                   carrier_band=0, num_antennas=1, **kw):
        # Kill switch: 50 consecutive NACKs → fallback
        if self.consecutive_nack > 50:
            return NR_MCS_INDICES[0]

        # GBM base prediction with offset-adjusted SINR
        adjusted_sinr = measured_sinr + self.offset_db
        feats = [adjusted_sinr, measured_speed, channel_ord]
        if len(self.feature_cols) > 3:
            feats += [carrier_band, num_antennas]
        x = pd.DataFrame([feats], columns=self.feature_cols)
        return int(self.policy.predict(x)[0])

    def update(self, was_ack, **kw):
        # OLLA offset update
        if was_ack:
            self.offset_db += self.step_up
            self.consecutive_nack = 0
        else:
            self.offset_db -= self.step_down
            self.consecutive_nack += 1

        self.offset_db = np.clip(self.offset_db, -6.0, 6.0)

        # BLER EMA for monitoring
        alpha = 0.02
        self.bler_ema = alpha * (0.0 if was_ack else 1.0) + (
            1 - alpha) * self.bler_ema


# ============================================================================
# 5. Ordinal GBM Policy (for agent training)
# ============================================================================

class OrdinalMCSPolicy:
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


# ============================================================================
# 6. Closed-Loop Simulator
# ============================================================================

class ClosedLoopSimulator:
    """Run LA agents in closed loop over a time-varying SINR trace."""

    def __init__(self, bler_lookup, rng):
        self.bler_lookup = bler_lookup
        self.rng = rng

    def run_scenario(self, agents, sinr_trace_true, sinr_trace_measured,
                     channel, speed, measured_speed=None,
                     carrier_band=0.0, num_antennas=1.0):
        """Run all agents over the same SINR trace.

        Returns dict of {agent_name: {throughput, bler, mcs_selected, ...}}
        """
        if measured_speed is None:
            measured_speed = speed + self.rng.normal(0, 10)
            measured_speed = max(0, measured_speed)

        ch_ord = CHANNEL_ORDINAL.get(channel, 1)
        num_tti = len(sinr_trace_true)
        results = {}

        for agent in agents:
            thr_history = np.zeros(num_tti)
            bler_history = np.zeros(num_tti)
            mcs_history = np.zeros(num_tti, dtype=int)
            ack_history = np.zeros(num_tti, dtype=bool)
            offset_history = np.zeros(num_tti)

            bler_running = 0.0
            total_acks = 0
            total_ttis = 0

            for t in range(num_tti):
                # Agent selects MCS based on measured SINR
                mcs = agent.select_mcs(
                    measured_sinr=sinr_trace_measured[t],
                    measured_speed=measured_speed,
                    channel_ord=ch_ord,
                    carrier_band=carrier_band,
                    num_antennas=num_antennas,
                )
                mcs_history[t] = mcs

                # PHY simulator evaluates the actual transmission
                was_ack, thr, n_rounds = simulate_tti(
                    self.bler_lookup, sinr_trace_true[t], mcs,
                    channel, speed, self.rng,
                )

                ack_history[t] = was_ack
                thr_history[t] = thr

                # Running BLER (windowed)
                total_ttis += 1
                total_acks += int(was_ack)
                bler_running = 1.0 - total_acks / total_ttis
                bler_history[t] = bler_running

                # Agent observes HARQ feedback
                agent.update(was_ack=was_ack)

                # Record offset if applicable
                if hasattr(agent, 'offset_db'):
                    offset_history[t] = agent.offset_db

            results[agent.name] = {
                "throughput": thr_history,
                "bler": bler_history,
                "mcs": mcs_history,
                "ack": ack_history,
                "offset": offset_history,
                "mean_thr": np.mean(thr_history),
                "final_bler": bler_running,
                "mean_mcs": np.mean(mcs_history),
            }

        return results


# ============================================================================
# 7. Scenario Definitions
# ============================================================================

def run_all_scenarios(bler_lookup, gbm_policy, feature_cols, rng,
                     dnn_model=None):
    """Run 3 scenarios comparing all agents."""
    trace_gen = SINRTraceGenerator(rng)
    sim = ClosedLoopSimulator(bler_lookup, rng)

    all_scenario_results = {}

    # --- Scenario 1: Stationary Channel ---
    print("\n   Scenario 1: Stationary channel (TDL-A, 30 km/h, 15 dB)")
    num_tti = 3000
    true_sinr, meas_sinr = trace_gen.generate(
        num_tti, mean_sinr_db=15.0, speed_kmph=30.0)

    agents = [
        StaticLUTAgent(),
        OLLAAgent(bler_target=0.10),
        OfflineGBMAgent(gbm_policy, feature_cols),
        GBMWithOLLAAgent(gbm_policy, feature_cols, bler_target=0.10),
    ]
    if dnn_model is not None:
        agents.extend([
            OfflineDNNAgent(dnn_model, feature_cols, NR_MCS_INDICES),
            DNNWithOLLAAgent(dnn_model, feature_cols, NR_MCS_INDICES),
        ])

    results = sim.run_scenario(
        agents, true_sinr, meas_sinr,
        channel="TDL-A", speed=30.0)
    all_scenario_results["Stationary"] = {
        "results": results,
        "true_sinr": true_sinr,
        "meas_sinr": meas_sinr,
    }

    # --- Scenario 2: Interference Drift ---
    print("   Scenario 2: Interference drift (SIR drops at TTI 1500)")
    num_tti = 3000
    # First half: high SINR (low interference), second half: low SINR
    true_sinr_1, meas_sinr_1 = trace_gen.generate(
        1500, mean_sinr_db=20.0, speed_kmph=30.0,
        shadow_std_db=3.0)
    true_sinr_2, meas_sinr_2 = trace_gen.generate(
        1500, mean_sinr_db=8.0, speed_kmph=30.0,
        shadow_std_db=3.0)
    true_sinr = np.concatenate([true_sinr_1, true_sinr_2])
    meas_sinr = np.concatenate([meas_sinr_1, meas_sinr_2])

    agents = [
        StaticLUTAgent(),
        OLLAAgent(bler_target=0.10),
        OfflineGBMAgent(gbm_policy, feature_cols),
        GBMWithOLLAAgent(gbm_policy, feature_cols, bler_target=0.10),
    ]
    if dnn_model is not None:
        agents.extend([
            OfflineDNNAgent(dnn_model, feature_cols, NR_MCS_INDICES),
            DNNWithOLLAAgent(dnn_model, feature_cols, NR_MCS_INDICES),
        ])

    results = sim.run_scenario(
        agents, true_sinr, meas_sinr,
        channel="TDL-A", speed=30.0)
    all_scenario_results["Interference Drift"] = {
        "results": results,
        "true_sinr": true_sinr,
        "meas_sinr": meas_sinr,
        "drift_tti": 1500,
    }

    # --- Scenario 3: Mobility Change ---
    print("   Scenario 3: Mobility change (3 km/h → 120 km/h at TTI 1500)")
    # Use same mean SINR but different fading statistics
    true_sinr_slow, meas_sinr_slow = trace_gen.generate(
        1500, mean_sinr_db=15.0, speed_kmph=3.0,
        shadow_std_db=2.0)
    true_sinr_fast, meas_sinr_fast = trace_gen.generate(
        1500, mean_sinr_db=15.0, speed_kmph=120.0,
        shadow_std_db=6.0)
    true_sinr = np.concatenate([true_sinr_slow, true_sinr_fast])
    meas_sinr = np.concatenate([meas_sinr_slow, meas_sinr_fast])

    agents = [
        StaticLUTAgent(),
        OLLAAgent(bler_target=0.10),
        OfflineGBMAgent(gbm_policy, feature_cols),
        GBMWithOLLAAgent(gbm_policy, feature_cols, bler_target=0.10),
    ]
    if dnn_model is not None:
        agents.extend([
            OfflineDNNAgent(dnn_model, feature_cols, NR_MCS_INDICES),
            DNNWithOLLAAgent(dnn_model, feature_cols, NR_MCS_INDICES),
        ])

    results = sim.run_scenario(
        agents, true_sinr, meas_sinr,
        channel="TDL-B", speed=30.0,  # Mixed speed - use average
        measured_speed=60.0)
    all_scenario_results["Mobility Change"] = {
        "results": results,
        "true_sinr": true_sinr,
        "meas_sinr": meas_sinr,
        "drift_tti": 1500,
    }

    return all_scenario_results


# ============================================================================
# 8. Publication-Quality Plots
# ============================================================================

def plot_scenario_results(all_scenarios, output_path="online_la_results.png"):
    """Generate a 3×3 panel figure: one row per scenario."""

    fig, axes = plt.subplots(3, 3, figsize=(22, 16))
    fig.patch.set_facecolor('#0d1117')

    agent_colors = {
        "Static LUT": "#6e7681",
        "OLLA": "#f0883e",
        "Offline GBM": "#58a6ff",
        "GBM + OLLA": "#3fb950",
        "Offline DNN": "#bc8cff",
        "DNN + OLLA": "#ff7b72",
    }

    smooth_window = 50  # Rolling average for cleaner plots

    for row, (scenario_name, scenario_data) in enumerate(
            all_scenarios.items()):
        results = scenario_data["results"]
        true_sinr = scenario_data["true_sinr"]
        drift_tti = scenario_data.get("drift_tti", None)
        num_tti = len(true_sinr)

        # Column 1: Throughput time series
        ax_thr = axes[row, 0]
        ax_thr.set_facecolor('#161b22')
        for agent_name, r in results.items():
            thr_smooth = pd.Series(r["throughput"]).rolling(
                smooth_window, min_periods=1).mean()
            ax_thr.plot(thr_smooth, color=agent_colors[agent_name],
                        label=agent_name, linewidth=1.2, alpha=0.9)
        if drift_tti:
            ax_thr.axvline(x=drift_tti, color='#f85149',
                           linestyle='--', alpha=0.7, label='Drift event')
        ax_thr.set_ylabel('Throughput (SE)', color='#c9d1d9', fontsize=10)
        ax_thr.set_title(f'{scenario_name}: Throughput',
                         color='white', fontsize=12, fontweight='bold')

        # Column 2: BLER time series
        ax_bler = axes[row, 1]
        ax_bler.set_facecolor('#161b22')
        for agent_name, r in results.items():
            # Use windowed BLER (last 200 TTIs)
            bler_window = pd.Series(
                (~r["ack"]).astype(float)).rolling(
                    200, min_periods=1).mean()
            ax_bler.plot(bler_window, color=agent_colors[agent_name],
                         label=agent_name, linewidth=1.2, alpha=0.9)
        ax_bler.axhline(y=0.10, color='#f85149', linestyle='--',
                        alpha=0.7, linewidth=1, label='10% target')
        if drift_tti:
            ax_bler.axvline(x=drift_tti, color='#f85149',
                            linestyle=':', alpha=0.5)
        ax_bler.set_ylabel('BLER (200-TTI window)', color='#c9d1d9',
                           fontsize=10)
        ax_bler.set_title(f'{scenario_name}: BLER',
                          color='white', fontsize=12, fontweight='bold')
        ax_bler.set_ylim(-0.01, 0.5)

        # Column 3: MCS selection
        ax_mcs = axes[row, 2]
        ax_mcs.set_facecolor('#161b22')
        for agent_name, r in results.items():
            mcs_smooth = pd.Series(r["mcs"].astype(float)).rolling(
                smooth_window, min_periods=1).mean()
            ax_mcs.plot(mcs_smooth, color=agent_colors[agent_name],
                        label=agent_name, linewidth=1.2, alpha=0.9)
        if drift_tti:
            ax_mcs.axvline(x=drift_tti, color='#f85149',
                           linestyle='--', alpha=0.7)
        ax_mcs.set_ylabel('Selected MCS index', color='#c9d1d9',
                          fontsize=10)
        ax_mcs.set_title(f'{scenario_name}: MCS Selection',
                         color='white', fontsize=12, fontweight='bold')

        # Style all axes
        for ax in [ax_thr, ax_bler, ax_mcs]:
            ax.tick_params(colors='#8b949e', labelsize=8)
            ax.spines['bottom'].set_color('#30363d')
            ax.spines['left'].set_color('#30363d')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(color='#21262d', linewidth=0.5, alpha=0.5)
            if row == 2:
                ax.set_xlabel('TTI', color='#8b949e', fontsize=10)

    # Legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5,
               fontsize=10, facecolor='#161b22', edgecolor='#30363d',
               labelcolor='#c9d1d9', framealpha=0.9,
               bbox_to_anchor=(0.5, 0.01))

    fig.suptitle(
        'Closed-Loop Link Adaptation: Digital Twin Evaluation\n'
        'HARQ Feedback-Driven Adaptation in Time-Varying Channels',
        fontsize=16, fontweight='bold', color='white', y=0.98)

    plt.tight_layout(rect=[0, 0.04, 1, 0.94])
    plt.savefig(output_path, dpi=200, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"   → Saved {output_path}")


def print_scenario_summary(all_scenarios):
    """Print quantitative summary table."""
    print("\n" + "=" * 80)
    print(f"{'Scenario':<22} {'Agent':<16} {'Mean Thr':>10} "
          f"{'Final BLER':>12} {'Mean MCS':>10}")
    print("=" * 80)

    for scenario_name, scenario_data in all_scenarios.items():
        results = scenario_data["results"]
        for agent_name, r in results.items():
            print(f"{scenario_name:<22} {agent_name:<16} "
                  f"{r['mean_thr']:>10.4f} "
                  f"{r['final_bler']*100:>10.1f}% "
                  f"{r['mean_mcs']:>10.1f}")
        print("-" * 80)

    print("=" * 80)


# ============================================================================
# 9. Main — Train GBM, Build Lookup, Run Scenarios
# ============================================================================

def main():
    print("=" * 65)
    print("  Closed-Loop LA Simulator — Digital Twin Evaluation")
    print("  HARQ feedback-driven adaptation in dynamic channels")
    print("=" * 65)

    # ---- 1. Load dataset ----
    print("\n1. Loading dataset for PHY abstraction …")
    if os.path.exists("sionna_v2_dataset.csv"):
        df = pd.read_csv("sionna_v2_dataset.csv")
        print(f"   V2 dataset: {len(df):,} rows")
    elif os.path.exists("sionna_realistic_dataset.csv"):
        df = pd.read_csv("sionna_realistic_dataset.csv")
        print(f"   V1 dataset: {len(df):,} rows")
    else:
        raise FileNotFoundError("No dataset found. Run generator first.")

    rng = np.random.default_rng(RANDOM_SEED)

    # Quantize SINR_dB to 1 dB bins (V2 SIR jitter is per-MCS)
    df["SINR_dB"] = df["SINR_dB"].round(0)

    # ---- 2. Build BLER lookup curves ----
    print("\n2. Building interpolated BLER curves (PHY abstraction) …")
    bler_lookup = BLERLookup(df)

    # ---- 3. Train offline GBM for agent use ----
    print("\n3. Training offline Ordinal GBM …")
    # Detect V2 columns
    has_v2 = "Num_Streams" in df.columns and "Carrier_GHz" in df.columns

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ctx_cols = ["SINR_dB", "Channel", "Speed_kmph"]
        if has_v2:
            ctx_cols += ["Num_Streams", "Carrier_GHz"]

        stats = (
            df.groupby(ctx_cols + ["MCS_Index"])
            .agg(BLER=("Was_Success", lambda x: 1.0 - x.mean()),
                 Throughput=("Actual_Throughput", "mean"),
                 N=("Was_Success", "count"))
            .reset_index()
        )

        def get_optimal(g):
            safe = g[g["BLER"] <= BLER_TARGET]
            if not safe.empty:
                return safe.loc[safe["Throughput"].idxmax()]
            return g.loc[g["MCS_Index"].idxmin()]

        optimal = (
            stats.groupby(ctx_cols, group_keys=False)
            .apply(get_optimal, include_groups=False)
            .reset_index()
        )

    # Feature engineering
    fn_rng = np.random.default_rng(42)
    sinr_noise = fn_rng.normal(0, 1.5, len(optimal))
    speed_vals = optimal["Speed_kmph"].values
    cqi_aging = 0.5 * (1 - np.exp(-speed_vals / 30.0))
    optimal["Measured_SINR"] = optimal["SINR_dB"] + sinr_noise - cqi_aging
    optimal["Measured_Speed"] = optimal["Speed_kmph"] + fn_rng.normal(
        0, 10, len(optimal))
    optimal["Measured_Speed"] = optimal["Measured_Speed"].clip(lower=0)
    optimal["Channel_Ordinal"] = optimal["Channel"].map(
        CHANNEL_ORDINAL).fillna(1).astype(float)

    feature_cols = ["Measured_SINR", "Measured_Speed", "Channel_Ordinal"]
    mono_cst = [1, -1, -1]

    if has_v2:
        optimal["Carrier_Band"] = (optimal["Carrier_GHz"] > 10.0).astype(float)
        optimal["Num_Antennas"] = optimal["Num_Streams"].astype(float)
        feature_cols += ["Carrier_Band", "Num_Antennas"]
        mono_cst += [0, 0]

    X = optimal[feature_cols]
    y = optimal["MCS_Index"]

    policy = OrdinalMCSPolicy(
        mcs_indices=NR_MCS_INDICES,
        decision_threshold=0.5,
        max_iter=300, max_depth=4, min_samples_leaf=3,
        learning_rate=0.1, monotonic_cst=mono_cst,
        random_state=RANDOM_SEED,
    )
    policy.fit(X, y)
    train_acc = accuracy_score(y, policy.predict(X))
    print(f"   GBM training accuracy: {train_acc*100:.1f}%")

    # ---- 3b. Train DNN classifier for comparison ----
    print("\n3b. Training DNN Classifier …")
    mcs_to_idx = {mcs: i for i, mcs in enumerate(NR_MCS_INDICES)}
    y_idx = y.map(mcs_to_idx).values

    X_t = torch.FloatTensor(X.values)
    y_t = torch.LongTensor(y_idx)
    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    dnn_model = DNNClassifierNet(n_features=len(feature_cols),
                                  n_classes=len(NR_MCS_INDICES))
    optimizer = optim.Adam(dnn_model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    dnn_model.train()
    for epoch in range(100):
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(dnn_model(bx), by)
            loss.backward()
            optimizer.step()

    dnn_model.eval()
    with torch.no_grad():
        dnn_pred_idx = dnn_model(X_t).argmax(dim=1).numpy()
    dnn_pred = np.array([NR_MCS_INDICES[i] for i in dnn_pred_idx])
    dnn_acc = accuracy_score(y, dnn_pred)
    print(f"   DNN training accuracy: {dnn_acc*100:.1f}%")

    # ---- 4. Run scenarios ----
    print("\n4. Running closed-loop scenarios …")
    all_scenarios = run_all_scenarios(
        bler_lookup, policy, feature_cols, rng, dnn_model=dnn_model)

    # ---- 5. Results ----
    print("\n5. Results:")
    print_scenario_summary(all_scenarios)

    # ---- 6. Plots ----
    print("\n6. Generating plots …")
    plot_scenario_results(all_scenarios)

    print("\n✓ Digital twin evaluation complete.")
    print("  Key insight: GBM+OLLA combines the best of both worlds —")
    print("  accurate base prediction from data + real-time adaptation")
    print("  from HARQ feedback. This is what the C++ export implements.\n")


if __name__ == "__main__":
    main()
