import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from sionna.phy.mapping import Mapper, Demapper, BinarySource
from sionna.phy.ofdm import (
    ResourceGrid, ResourceGridMapper,
    LSChannelEstimator, LMMSEEqualizer,
)
from sionna.phy.mimo import StreamManagement
from sionna.phy.channel.tr38901 import TDL
from sionna.phy.channel import OFDMChannel
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# NR MCS Table — representative subset of TS 38.214 Table 5.1.3.1-2
# Each entry: (mcs_index, Qm (bits/symbol), target_code_rate × 1024)
# ---------------------------------------------------------------------------
NR_MCS_TABLE = [
    (3,  2,  253),   # QPSK   R≈0.247  (lowest rate above LDPC 1/5 floor)
    (4,  2,  308),   # QPSK   R≈0.301
    (9,  2,  616),   # QPSK   R≈0.602
    (11, 4,  340),   # 16-QAM R≈0.332
    (14, 4,  553),   # 16-QAM R≈0.540
    (17, 6,  438),   # 64-QAM R≈0.428
    (20, 6,  666),   # 64-QAM R≈0.650
    (25, 8,  616),   # 256-QAM R≈0.602
    (24, 8,  567),   # 256-QAM R≈0.554  (capped to keep k ≤ 8448)
]

HARQ_MAX_ROUNDS = 4
HARQ_RTT_SLOTS = 4          # Feedback loop latency in slots (K1 ≈ 4 for 30 kHz SCS)
BATCH_SIZE = 100             # Packets per drop (reduced for larger sweep)


class NRLinkSimulator(tf.keras.Model):
    """Single-user NR link with configurable MCS and 3GPP TDL channel."""

    def __init__(self, num_bits_per_symbol, code_rate, tdl_model,
                 ue_speed_kmph, delay_spread):
        super().__init__()
        self.num_bits_per_symbol = num_bits_per_symbol
        self.code_rate = code_rate

        # ---- OFDM Resource Grid (≈10 PRBs, 30 kHz SCS) ----
        self.rg = ResourceGrid(
            num_ofdm_symbols=14,
            fft_size=128,
            subcarrier_spacing=30e3,
            num_tx=1, num_streams_per_tx=1,
            cyclic_prefix_length=14,
            dc_null=True,
            pilot_pattern="kronecker",
            pilot_ofdm_symbol_indices=[2, 11],
        )

        # Codeword dimensions from grid and MCS
        self.n = int(self.rg.num_data_symbols) * int(num_bits_per_symbol)
        self.k = max(int(round(self.n * code_rate)), 12)

        # ---- Transmitter ----
        self.binary_source = BinarySource()
        self.encoder = LDPC5GEncoder(self.k, self.n)
        self.mapper = Mapper("qam", num_bits_per_symbol)
        self.rg_mapper = ResourceGridMapper(self.rg)

        # ---- 3GPP TDL channel (TS 38.901) ----
        self.tdl = TDL(
            model=tdl_model,
            delay_spread=delay_spread,
            carrier_frequency=3.5e9,
            min_speed=ue_speed_kmph / 3.6,
            max_speed=ue_speed_kmph / 3.6,
        )
        self.channel = OFDMChannel(
            self.tdl, self.rg, normalize_channel=True, return_channel=True,
        )

        # ---- Receiver ----
        self.ls_est = LSChannelEstimator(self.rg, interpolation_type="nn")
        rx_tx_association = np.array([[1]])
        self.stream_management = StreamManagement(
            rx_tx_association, num_streams_per_tx=1,
        )
        self.lmmse_eq = LMMSEEqualizer(self.rg, self.stream_management)
        self.demapper = Demapper("app", "qam", num_bits_per_symbol)
        self.decoder = LDPC5GDecoder(self.encoder, hard_out=True)

    # --- Split into transmit / receive_llr / decode for HARQ combining ---

    @tf.function(jit_compile=True)
    def transmit(self, batch_size):
        """Generate random info bits, encode, map to OFDM resource grid."""
        bits = self.binary_source([batch_size, 1, 1, self.k])
        codewords = self.encoder(bits)
        x = self.mapper(codewords)
        x_rg = self.rg_mapper(x)
        return bits, x_rg

    @tf.function(jit_compile=True)
    def receive_llr(self, x_rg, no):
        """Pass transmitted grid through the channel; return soft LLRs."""
        y_rg, _ = self.channel(x_rg, no)
        h_hat, err_var = self.ls_est(y_rg, no)
        x_hat, no_eff = self.lmmse_eq(y_rg, h_hat, err_var, no)
        llr = self.demapper(x_hat, no_eff)
        return llr

    @tf.function(jit_compile=True)
    def decode(self, llr):
        """LDPC-decode from (accumulated) LLRs."""
        return self.decoder(llr)


def simulate_harq_drop(model, batch_size, snr_db, sir_db, rng_np):
    """
    HARQ Chase-Combining via LLR accumulation.

    Each retransmission draws a *new* channel realisation (different slot)
    and the demapped LLRs are summed before decoding — which is the correct
    soft-combining operation for Chase Combining with independent noise.
    """
    # Effective noise power: thermal + interference  →  SINR
    no_thermal = tf.constant(10.0 ** (-snr_db / 10.0), dtype=tf.float32)
    no_interf = tf.constant(10.0 ** (-sir_db / 10.0), dtype=tf.float32)
    no = no_thermal + no_interf

    # Transmit once (same codeword for all HARQ rounds)
    bits, x_rg = model.transmit(batch_size)

    accumulated_llr = None  # Will be set to first round's LLRs

    pending_mask = np.ones(batch_size, dtype=bool)
    final_success = np.zeros(batch_size, dtype=np.float32)
    tx_rounds = np.zeros(batch_size, dtype=np.int32)

    for round_idx in range(HARQ_MAX_ROUNDS):
        if not np.any(pending_mask):
            break

        # New channel realisation per retransmission (different slot)
        llr_round = model.receive_llr(x_rg, no)

        # Chase Combining: sum LLRs across rounds
        if accumulated_llr is None:
            accumulated_llr = llr_round
        else:
            accumulated_llr = accumulated_llr + llr_round

        # Attempt decoding on combined LLRs
        bits_rx = model.decode(accumulated_llr)

        errors_per_block = tf.reduce_sum(
            tf.abs(bits - bits_rx), axis=[1, 2, 3],
        )
        success_this_round = errors_per_block.numpy() == 0

        newly_succeeded = success_this_round & pending_mask
        tx_rounds[pending_mask] += 1
        final_success[newly_succeeded] = 1.0
        pending_mask[newly_succeeded] = False

    # Spectral efficiency for this MCS
    se = model.num_bits_per_symbol * model.code_rate

    # Slot-aware throughput:  1st TX = 1 slot, each retransmission adds
    # HARQ_RTT_SLOTS of latency (feedback + scheduling).
    effective_slots = 1.0 + np.maximum(tx_rounds - 1, 0) * HARQ_RTT_SLOTS
    throughput = final_success * se / effective_slots

    return final_success, tx_rounds, throughput


def generate_ml_dataset(args):
    print("Initialising NR link simulation (HARQ-CC with LLR accumulation)…")

    if args.quick:
        print(">>> RUNNING IN QUICK MODE (Reduced resolution sweep) <<<")
        snr_range = np.arange(-5, 31, 2.0)
        ue_speeds = [3.0, 30.0, 120.0]
        sir_levels = [30.0, 7.0]
        batch_size = 50
    else:
        print(">>> RUNNING IN FULL MODE (High resolution sweep) <<<")
        snr_range = np.arange(-5, 31, 0.5)   # 0.5 dB steps → 72 values (was 1.5 dB)
        ue_speeds = [3.0, 10.0, 30.0, 60.0, 90.0, 120.0]   # 6 speeds (was 4)
        sir_levels = [30.0, 20.0, 15.0, 7.0, 3.0]           # 5 levels (was 3)
        batch_size = 100

    channel_configs = {
        "A": 30e-9,      # 30 ns  — Indoor / short echoes
        "B": 300e-9,     # 300 ns — Urban
        "C": 1000e-9,    # 1 μs   — Dense urban / harsh
    }

    rng = np.random.default_rng(42)

    columns = {
        "SINR_dB": [], "SIR_dB": [], "Channel": [], "Speed_kmph": [],
        "MCS_Index": [], "Modulation_Qm": [], "Code_Rate": [],
        "Was_Success": [], "HARQ_Tx_Rounds": [], "Actual_Throughput": [],
        "Time_Index": [],
    }
    global_time_idx = 0

    for ch_model, ds in channel_configs.items():
        for speed in ue_speeds:
            print(f"--- TDL-{ch_model} (DS={ds*1e9:.0f} ns), "
                  f"Speed={speed} km/h ---")

            for mcs_idx, qm, rate_x1024 in NR_MCS_TABLE:
                code_rate = rate_x1024 / 1024.0
                try:
                    model = NRLinkSimulator(
                        num_bits_per_symbol=qm,
                        code_rate=code_rate,
                        tdl_model=ch_model,
                        ue_speed_kmph=speed,
                        delay_spread=ds,
                    )
                except Exception as exc:
                    print(f"  [SKIP] MCS {mcs_idx} (Qm={qm}, R={code_rate:.3f})"
                          f": {exc}")
                    continue

                for snr in snr_range:
                    for sir in sir_levels:
                        # Compute recorded SINR (what the gNB would estimate)
                        sinr_linear = 1.0 / (10**(-snr/10) + 10**(-sir/10))
                        sinr_db = 10.0 * np.log10(sinr_linear)

                        success, rounds, thr = simulate_harq_drop(
                            model, batch_size, snr, sir, rng,
                        )

                        n = batch_size
                        columns["SINR_dB"].extend(np.full(n, sinr_db))
                        columns["SIR_dB"].extend(np.full(n, sir))
                        columns["Channel"].extend(
                            np.full(n, f"TDL-{ch_model}"),
                        )
                        columns["Speed_kmph"].extend(np.full(n, speed))
                        columns["MCS_Index"].extend(np.full(n, mcs_idx))
                        columns["Modulation_Qm"].extend(np.full(n, qm))
                        columns["Code_Rate"].extend(np.full(n, code_rate))
                        columns["Was_Success"].extend(success)
                        columns["HARQ_Tx_Rounds"].extend(rounds)
                        columns["Actual_Throughput"].extend(thr)
                        columns["Time_Index"].extend(
                            np.arange(global_time_idx, global_time_idx + n),
                        )
                        global_time_idx += n

                # Free model memory between MCS entries
                del model
                tf.keras.backend.clear_session()

    df = pd.DataFrame(columns)
    df.to_csv("sionna_realistic_dataset.csv", index=False)
    print(f"\nDone. Saved sionna_realistic_dataset.csv  ({len(df)} rows)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 5G LA dataset.")
    parser.add_argument("--quick", action="store_true",
                        help="Run a fast, low-resolution sweep for testing.")
    args = parser.parse_args()

    generate_ml_dataset(args)
