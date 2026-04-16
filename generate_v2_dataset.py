"""
V2 Dataset Generator — Closes Gaps 3, 5, 6, 7
================================================

New capabilities over V1:
  Gap 3:  2×2 MIMO via TDL num_tx_ant/num_rx_ant and CDL antenna arrays
  Gap 5:  CDL-A (NLOS) and CDL-D (LOS) channel models
  Gap 6:  Stochastic SIR jitter (lognormal per simulation point)
  Gap 7:  FR2 (28 GHz, 120 kHz SCS) alongside FR1 (3.5 GHz, 30 kHz SCS)

New CSV columns:
  Num_Tx_Ant       — 1 (SISO) or 2 (2×2 MIMO)
  Num_Streams      — number of spatial multiplexing layers
  Carrier_GHz      — 3.5 (FR1) or 28.0 (FR2)
  SIR_Base_dB      — nominal SIR before jitter
  SIR_Effective_dB — actual SIR after stochastic variation
"""

import argparse
import gc
import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf

# Prevent TF from pre-allocating all GPU memory
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from sionna.phy.mapping import Mapper, Demapper, BinarySource
from sionna.phy.ofdm import (
    ResourceGrid, ResourceGridMapper,
    LSChannelEstimator, LMMSEEqualizer,
)
from sionna.phy.mimo import StreamManagement
from sionna.phy.channel.tr38901 import TDL, CDL, Antenna, AntennaArray
from sionna.phy.channel import OFDMChannel
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# NR MCS Table — representative subset of TS 38.214 Table 5.1.3.1-2
# ---------------------------------------------------------------------------
NR_MCS_TABLE = [
    (3,  2,  253),   # QPSK   R≈0.247
    (4,  2,  308),   # QPSK   R≈0.301
    (9,  2,  616),   # QPSK   R≈0.602
    (11, 4,  340),   # 16-QAM R≈0.332
    (14, 4,  553),   # 16-QAM R≈0.540
    (17, 6,  438),   # 64-QAM R≈0.428
    (20, 6,  666),   # 64-QAM R≈0.650
    (24, 8,  567),   # 256-QAM R≈0.554
    (25, 8,  616),   # 256-QAM R≈0.602
]

HARQ_MAX_ROUNDS = 4
HARQ_RTT_SLOTS = 4

CSV_COLUMNS = [
    "SINR_dB", "SIR_dB", "Channel", "Speed_kmph",
    "MCS_Index", "Modulation_Qm", "Code_Rate",
    "Was_Success", "HARQ_Tx_Rounds", "Actual_Throughput", "Time_Index",
    "Num_Tx_Ant", "Num_Streams", "Carrier_GHz",
    "SIR_Base_dB", "SIR_Effective_dB",
]


# ---------------------------------------------------------------------------
# Resource Grid configurations per frequency range
# ---------------------------------------------------------------------------
RG_CONFIGS = {
    "FR1": {
        "fft_size": 128,
        "subcarrier_spacing": 30e3,
        "cyclic_prefix_length": 14,
        "carrier_frequency": 3.5e9,
    },
    "FR2": {
        "fft_size": 64,
        "subcarrier_spacing": 120e3,
        "cyclic_prefix_length": 9,
        "carrier_frequency": 28e9,
    },
}


# ---------------------------------------------------------------------------
# Simulator — TDL channel (supports SISO and MIMO via num_tx_ant/num_rx_ant)
# ---------------------------------------------------------------------------

class TDLLinkSimulator(tf.keras.Model):
    """NR link with TDL channel, configurable MIMO layers and carrier."""

    def __init__(self, num_bits_per_symbol, code_rate, tdl_model,
                 ue_speed_kmph, delay_spread, num_streams=1,
                 freq_range="FR1"):
        super().__init__()
        self.num_bits_per_symbol = num_bits_per_symbol
        self.code_rate = code_rate
        self.num_streams = num_streams

        rg_cfg = RG_CONFIGS[freq_range]

        # For MIMO (num_streams > 1) with dc_null=True, fft_size-1 must
        # be divisible by num_streams. fft_size=128→127(odd)→fails for
        # 2 streams. Fix: disable dc_null for MIMO (standard in 3GPP
        # system-level simulators — the DC subcarrier is a PHY detail
        # that doesn't affect BLER statistics at the link level).
        use_dc_null = (num_streams == 1)

        self.rg = ResourceGrid(
            num_ofdm_symbols=14,
            fft_size=rg_cfg["fft_size"],
            subcarrier_spacing=rg_cfg["subcarrier_spacing"],
            num_tx=1, num_streams_per_tx=num_streams,
            cyclic_prefix_length=rg_cfg["cyclic_prefix_length"],
            dc_null=use_dc_null,
            pilot_pattern="kronecker",
            pilot_ofdm_symbol_indices=[2, 11],
        )

        self.n = int(self.rg.num_data_symbols) * int(num_bits_per_symbol)
        self.k = max(int(round(self.n * code_rate)), 12)

        # ---- Transmitter ----
        self.binary_source = BinarySource()
        self.encoder = LDPC5GEncoder(self.k, self.n)
        self.mapper = Mapper("qam", num_bits_per_symbol)
        self.rg_mapper = ResourceGridMapper(self.rg)

        # ---- TDL channel with MIMO support ----
        self.tdl = TDL(
            model=tdl_model,
            delay_spread=delay_spread,
            carrier_frequency=rg_cfg["carrier_frequency"],
            min_speed=ue_speed_kmph / 3.6,
            max_speed=ue_speed_kmph / 3.6,
            num_tx_ant=num_streams,
            num_rx_ant=num_streams,
        )
        self.channel = OFDMChannel(
            self.tdl, self.rg, normalize_channel=True, return_channel=True,
        )

        # ---- Receiver ----
        self.ls_est = LSChannelEstimator(self.rg, interpolation_type="nn")
        rx_tx_association = np.array([[1]])
        self.stream_management = StreamManagement(
            rx_tx_association, num_streams_per_tx=num_streams,
        )
        self.lmmse_eq = LMMSEEqualizer(self.rg, self.stream_management)
        self.demapper = Demapper("app", "qam", num_bits_per_symbol)
        self.decoder = LDPC5GDecoder(self.encoder, hard_out=True)

    @tf.function(jit_compile=True)
    def transmit(self, batch_size):
        bits = self.binary_source([batch_size, 1, self.num_streams, self.k])
        codewords = self.encoder(bits)
        x = self.mapper(codewords)
        x_rg = self.rg_mapper(x)
        return bits, x_rg

    @tf.function(jit_compile=True)
    def receive_llr(self, x_rg, no):
        y_rg, _ = self.channel(x_rg, no)
        h_hat, err_var = self.ls_est(y_rg, no)
        x_hat, no_eff = self.lmmse_eq(y_rg, h_hat, err_var, no)
        llr = self.demapper(x_hat, no_eff)
        return llr

    @tf.function(jit_compile=True)
    def decode(self, llr):
        return self.decoder(llr)


# ---------------------------------------------------------------------------
# Simulator — CDL channel (requires antenna arrays)
# ---------------------------------------------------------------------------

class CDLLinkSimulator(tf.keras.Model):
    """NR link with CDL channel model and explicit antenna arrays."""

    def __init__(self, num_bits_per_symbol, code_rate, cdl_model,
                 ue_speed_kmph, delay_spread, num_streams=1,
                 freq_range="FR1"):
        super().__init__()
        self.num_bits_per_symbol = num_bits_per_symbol
        self.code_rate = code_rate
        self.num_streams = num_streams

        rg_cfg = RG_CONFIGS[freq_range]
        carrier_freq = rg_cfg["carrier_frequency"]

        # Same dc_null logic as TDLLinkSimulator — disable for MIMO
        use_dc_null = (num_streams == 1)

        self.rg = ResourceGrid(
            num_ofdm_symbols=14,
            fft_size=rg_cfg["fft_size"],
            subcarrier_spacing=rg_cfg["subcarrier_spacing"],
            num_tx=1, num_streams_per_tx=num_streams,
            cyclic_prefix_length=rg_cfg["cyclic_prefix_length"],
            dc_null=use_dc_null,
            pilot_pattern="kronecker",
            pilot_ofdm_symbol_indices=[2, 11],
        )

        self.n = int(self.rg.num_data_symbols) * int(num_bits_per_symbol)
        self.k = max(int(round(self.n * code_rate)), 12)

        # ---- Transmitter ----
        self.binary_source = BinarySource()
        self.encoder = LDPC5GEncoder(self.k, self.n)
        self.mapper = Mapper("qam", num_bits_per_symbol)
        self.rg_mapper = ResourceGridMapper(self.rg)

        # ---- CDL channel with antenna arrays ----
        if num_streams == 1:
            ut_array = Antenna(
                polarization="single", polarization_type="V",
                antenna_pattern="38.901", carrier_frequency=carrier_freq,
            )
            bs_array = Antenna(
                polarization="single", polarization_type="V",
                antenna_pattern="38.901", carrier_frequency=carrier_freq,
            )
        else:
            # Dual-polarized single panel → 2 antenna elements
            ut_array = AntennaArray(
                num_rows=1, num_cols=1,
                polarization="dual", polarization_type="cross",
                antenna_pattern="38.901", carrier_frequency=carrier_freq,
            )
            bs_array = AntennaArray(
                num_rows=1, num_cols=1,
                polarization="dual", polarization_type="cross",
                antenna_pattern="38.901", carrier_frequency=carrier_freq,
            )

        self.cdl = CDL(
            model=cdl_model,
            delay_spread=delay_spread,
            carrier_frequency=carrier_freq,
            ut_array=ut_array,
            bs_array=bs_array,
            direction="downlink",
            min_speed=ue_speed_kmph / 3.6,
            max_speed=ue_speed_kmph / 3.6,
        )
        self.channel = OFDMChannel(
            self.cdl, self.rg, normalize_channel=True, return_channel=True,
        )

        # ---- Receiver ----
        self.ls_est = LSChannelEstimator(self.rg, interpolation_type="nn")
        rx_tx_association = np.array([[1]])
        self.stream_management = StreamManagement(
            rx_tx_association, num_streams_per_tx=num_streams,
        )
        self.lmmse_eq = LMMSEEqualizer(self.rg, self.stream_management)
        self.demapper = Demapper("app", "qam", num_bits_per_symbol)
        self.decoder = LDPC5GDecoder(self.encoder, hard_out=True)

    @tf.function(jit_compile=True)
    def transmit(self, batch_size):
        bits = self.binary_source([batch_size, 1, self.num_streams, self.k])
        codewords = self.encoder(bits)
        x = self.mapper(codewords)
        x_rg = self.rg_mapper(x)
        return bits, x_rg

    @tf.function(jit_compile=True)
    def receive_llr(self, x_rg, no):
        y_rg, _ = self.channel(x_rg, no)
        h_hat, err_var = self.ls_est(y_rg, no)
        x_hat, no_eff = self.lmmse_eq(y_rg, h_hat, err_var, no)
        llr = self.demapper(x_hat, no_eff)
        return llr

    @tf.function(jit_compile=True)
    def decode(self, llr):
        return self.decoder(llr)


# ---------------------------------------------------------------------------
# HARQ Chase-Combining simulation (shared by both simulators)
# ---------------------------------------------------------------------------

def simulate_harq_drop(model, batch_size, snr_db, sir_eff_db):
    """HARQ-CC via LLR accumulation across retransmission rounds."""
    no_thermal = tf.constant(10.0 ** (-snr_db / 10.0), dtype=tf.float32)
    no_interf = tf.constant(10.0 ** (-sir_eff_db / 10.0), dtype=tf.float32)
    no = no_thermal + no_interf

    bits, x_rg = model.transmit(batch_size)
    accumulated_llr = None

    pending_mask = np.ones(batch_size, dtype=bool)
    final_success = np.zeros(batch_size, dtype=np.float32)
    tx_rounds = np.zeros(batch_size, dtype=np.int32)

    for round_idx in range(HARQ_MAX_ROUNDS):
        if not np.any(pending_mask):
            break

        llr_round = model.receive_llr(x_rg, no)

        if accumulated_llr is None:
            accumulated_llr = llr_round
        else:
            accumulated_llr = accumulated_llr + llr_round

        bits_rx = model.decode(accumulated_llr)
        errors_per_block = tf.reduce_sum(
            tf.abs(bits - bits_rx), axis=[1, 2, 3],
        )
        success_this_round = errors_per_block.numpy() == 0

        newly_succeeded = success_this_round & pending_mask
        tx_rounds[pending_mask] += 1
        final_success[newly_succeeded] = 1.0
        pending_mask[newly_succeeded] = False

    # SE with spatial multiplexing gain
    se = model.num_bits_per_symbol * model.code_rate * model.num_streams

    effective_slots = 1.0 + np.maximum(tx_rounds - 1, 0) * HARQ_RTT_SLOTS
    throughput = final_success * se / effective_slots

    return final_success, tx_rounds, throughput


# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------

def build_sweep_config(quick: bool):
    """Build the parameter sweep configuration.

    Full mode is sized for IEEE-grade coverage while keeping generation
    feasible (~1-2 hours on a single GPU):
      - 2 dB SNR steps (standard for BLER curve interpolation)
      - 3 speeds  (pedestrian/urban/highway per 3GPP TR 38.901)
      - 3 SIR    (clean / moderate / heavy interference)
      - 3 channels (flat NLOS / rich-scatter NLOS / LOS)
      - SISO + 2×2 MIMO, FR1 + FR2
    """

    if quick:
        snr_range = np.arange(-5, 31, 4.0)         # 9 points
        ue_speeds = [3.0, 120.0]                     # 2 speeds
        sir_levels = [30.0, 7.0]                     # 2 SIR levels
        mimo_configs = [1]                           # SISO only for quick
        freq_ranges = ["FR1"]                        # FR1 only for quick
        batch_size = 50
        channel_configs = [
            ("TDL-A", "tdl", "A", 30e-9),
            ("TDL-C", "tdl", "C", 1000e-9),
            ("CDL-D", "cdl", "D", 30e-9),
        ]
    else:
        snr_range = np.arange(-5, 31, 2.0)          # 18 points (2 dB steps)
        ue_speeds = [3.0, 30.0, 120.0]               # 3 speeds
        sir_levels = [30.0, 10.0, 3.0]               # 3 SIR levels
        mimo_configs = [1, 2]                        # SISO + 2×2 MIMO
        freq_ranges = ["FR1", "FR2"]                 # Both bands
        batch_size = 200                             # Larger for GPU efficiency
        channel_configs = [
            ("TDL-A", "tdl", "A", 30e-9),           # Flat fading NLOS
            ("TDL-C", "tdl", "C", 1000e-9),          # Rich-scatter NLOS
            ("CDL-D", "cdl", "D", 30e-9),            # LOS
        ]

    return {
        "snr_range": snr_range,
        "ue_speeds": ue_speeds,
        "sir_levels": sir_levels,
        "mimo_configs": mimo_configs,
        "freq_ranges": freq_ranges,
        "channel_configs": channel_configs,
        "batch_size": batch_size,
        "sir_jitter_sigma_db": 2.0,  # Gap 6: SIR stochastic jitter std
    }


# ---------------------------------------------------------------------------
# Combo key for checkpointing
# ---------------------------------------------------------------------------

def combo_key(ch_name, freq_range, num_streams, speed, mcs_idx):
    return f"{ch_name}|{freq_range}|{num_streams}|{speed}|{mcs_idx}"


# ---------------------------------------------------------------------------
# Main generation loop — with checkpointing and resume
# ---------------------------------------------------------------------------

OUT_PATH = "sionna_v2_dataset.csv"

def generate_v2_dataset(args):
    cfg = build_sweep_config(args.quick)
    mode_str = "QUICK" if args.quick else "FULL"
    print(f">>> V2 Dataset Generator [{mode_str} MODE] <<<")
    print(f"    SNR points: {len(cfg['snr_range'])}, "
          f"Speeds: {cfg['ue_speeds']}, "
          f"SIR levels: {cfg['sir_levels']}")
    print(f"    MIMO configs: {cfg['mimo_configs']}, "
          f"Freq ranges: {cfg['freq_ranges']}")
    print(f"    Channels: {[c[0] for c in cfg['channel_configs']]}")

    # --- Resume support: load existing progress ---
    completed_combos = set()
    existing_rows = 0
    if os.path.exists(OUT_PATH) and not args.quick:
        try:
            df_existing = pd.read_csv(OUT_PATH)
            existing_rows = len(df_existing)
            for _, row in df_existing.drop_duplicates(
                    subset=["Channel", "Carrier_GHz", "Num_Streams",
                            "Speed_kmph", "MCS_Index"]).iterrows():
                fr = "FR2" if row["Carrier_GHz"] > 10 else "FR1"
                key = combo_key(
                    row["Channel"], fr, int(row["Num_Streams"]),
                    row["Speed_kmph"], int(row["MCS_Index"]))
                completed_combos.add(key)
            print(f"\n    *** RESUMING: found {existing_rows:,} rows, "
                  f"{len(completed_combos)} combos already done ***\n")
            del df_existing
        except Exception:
            completed_combos = set()

    # RNG must be deterministic — advance past completed combos
    rng = np.random.default_rng(42)

    total_combos = (
        len(cfg["channel_configs"]) * len(cfg["freq_ranges"])
        * len(cfg["mimo_configs"]) * len(cfg["ue_speeds"])
        * len(NR_MCS_TABLE)
    )
    combo_idx = 0
    global_time_idx = existing_rows
    combos_skipped = 0
    t_start = time.time()

    for ch_name, ch_family, ch_model, ds in cfg["channel_configs"]:
        for freq_range in cfg["freq_ranges"]:
            carrier_ghz = RG_CONFIGS[freq_range]["carrier_frequency"] / 1e9

            effective_ds = ds
            if freq_range == "FR2":
                effective_ds = min(ds, 100e-9)

            for num_streams in cfg["mimo_configs"]:
                for speed in cfg["ue_speeds"]:
                    for mcs_idx, qm, rate_x1024 in NR_MCS_TABLE:
                        combo_idx += 1
                        code_rate = rate_x1024 / 1024.0

                        key = combo_key(ch_name, freq_range, num_streams,
                                        speed, mcs_idx)

                        # We must always advance the RNG to stay deterministic
                        # even when skipping (one draw per SNR×SIR point)
                        n_sir_draws = (len(cfg["snr_range"])
                                       * len(cfg["sir_levels"]))

                        if key in completed_combos:
                            # Advance RNG past this combo's jitter draws
                            rng.normal(0, cfg["sir_jitter_sigma_db"],
                                       n_sir_draws)
                            combos_skipped += 1
                            continue

                        elapsed = time.time() - t_start
                        done = combo_idx - combos_skipped
                        eta_s = (elapsed / max(done - len(completed_combos), 1)
                                 * (total_combos - combo_idx)) if done > len(completed_combos) else 0

                        print(f"  [{combo_idx}/{total_combos}] "
                              f"{ch_name} | {freq_range} | "
                              f"{num_streams}×{num_streams} MIMO | "
                              f"{speed} km/h | MCS {mcs_idx}"
                              f"  (ETA: {eta_s/60:.0f} min)")

                        # --- Build model ---
                        try:
                            if ch_family == "tdl":
                                model = TDLLinkSimulator(
                                    num_bits_per_symbol=qm,
                                    code_rate=code_rate,
                                    tdl_model=ch_model,
                                    ue_speed_kmph=speed,
                                    delay_spread=effective_ds,
                                    num_streams=num_streams,
                                    freq_range=freq_range,
                                )
                            else:
                                model = CDLLinkSimulator(
                                    num_bits_per_symbol=qm,
                                    code_rate=code_rate,
                                    cdl_model=ch_model,
                                    ue_speed_kmph=speed,
                                    delay_spread=effective_ds,
                                    num_streams=num_streams,
                                    freq_range=freq_range,
                                )
                        except Exception as exc:
                            print(f"    [SKIP] {exc}")
                            rng.normal(0, cfg["sir_jitter_sigma_db"],
                                       n_sir_draws)
                            continue

                        # --- Collect rows for this combo ---
                        combo_rows = []

                        for snr in cfg["snr_range"]:
                            for sir_base in cfg["sir_levels"]:
                                sir_jitter = rng.normal(
                                    0, cfg["sir_jitter_sigma_db"])
                                sir_eff = sir_base + sir_jitter

                                sinr_lin = 1.0 / (
                                    10**(-snr/10) + 10**(-sir_eff/10))
                                sinr_db = 10.0 * np.log10(sinr_lin)

                                # Adaptive batch for MIMO
                                bs = cfg["batch_size"]
                                if num_streams > 1:
                                    bs = max(
                                        bs // (num_streams
                                               * max(qm // 2, 1)),
                                        10)

                                try:
                                    success, rounds, thr = (
                                        simulate_harq_drop(
                                            model, bs, snr, sir_eff))
                                except Exception as exc:
                                    if "Out of memory" in str(exc):
                                        bs = max(bs // 2, 10)
                                        tf.keras.backend.clear_session()
                                        gc.collect()
                                        try:
                                            success, rounds, thr = (
                                                simulate_harq_drop(
                                                    model, bs, snr, sir_eff))
                                        except Exception:
                                            print(f"    [OOM] SNR={snr} "
                                                  f"bs={bs}")
                                            continue
                                    else:
                                        print(f"    [FAIL] SNR={snr}: "
                                              f"{exc}")
                                        continue

                                n = len(success)
                                for i in range(n):
                                    combo_rows.append({
                                        "SINR_dB": sinr_db,
                                        "SIR_dB": sir_eff,
                                        "Channel": ch_name,
                                        "Speed_kmph": speed,
                                        "MCS_Index": mcs_idx,
                                        "Modulation_Qm": qm,
                                        "Code_Rate": code_rate,
                                        "Was_Success": success[i],
                                        "HARQ_Tx_Rounds": rounds[i],
                                        "Actual_Throughput": thr[i],
                                        "Time_Index": global_time_idx + i,
                                        "Num_Tx_Ant": num_streams,
                                        "Num_Streams": num_streams,
                                        "Carrier_GHz": carrier_ghz,
                                        "SIR_Base_dB": sir_base,
                                        "SIR_Effective_dB": sir_eff,
                                    })
                                global_time_idx += n

                        # --- Checkpoint: append this combo to CSV ---
                        if combo_rows:
                            df_chunk = pd.DataFrame(combo_rows,
                                                    columns=CSV_COLUMNS)
                            write_header = not os.path.exists(OUT_PATH)
                            df_chunk.to_csv(OUT_PATH, mode='a',
                                            header=write_header,
                                            index=False)
                            print(f"    ✓ {len(combo_rows):,} rows saved "
                                  f"({combo_idx}/{total_combos})")

                        # --- Aggressive cleanup ---
                        del model
                        if combo_rows:
                            del combo_rows, df_chunk
                        tf.keras.backend.clear_session()
                        gc.collect()

    # --- Final summary ---
    if os.path.exists(OUT_PATH):
        df = pd.read_csv(OUT_PATH)
        print(f"\n✓ Done. {OUT_PATH}  ({len(df):,} rows)")
        print(f"\n=== Dataset Summary ===")
        print(f"Total rows: {len(df):,}")
        print(f"Channels: {df['Channel'].unique().tolist()}")
        print(f"Carriers: {df['Carrier_GHz'].unique().tolist()} GHz")
        print(f"MIMO configs: {df['Num_Streams'].unique().tolist()}")
        print(f"SINR range: [{df['SINR_dB'].min():.1f}, "
              f"{df['SINR_dB'].max():.1f}] dB")
        print(f"Mean success rate: {df['Was_Success'].mean():.3f}")
        elapsed = time.time() - t_start
        print(f"Wall time: {elapsed/60:.1f} min")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="V2 5G LA Dataset Generator (MIMO+CDL+FR2+SIR jitter)")
    parser.add_argument("--quick", action="store_true",
                        help="Fast sweep for testing (reduced resolution)")
    args = parser.parse_args()
    generate_v2_dataset(args)
