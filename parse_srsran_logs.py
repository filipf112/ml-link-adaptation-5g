#!/usr/bin/env python3
"""
srsRAN Log Parser & Analyzer
==============================
Parses srsUE PHY logs to extract MCS decisions, CQI reports,
and generates publication-quality analysis plots.

Usage:
    python3 parse_srsran_logs.py <ue_log_file>
    python3 parse_srsran_logs.py /tmp/ue.log
"""

import sys
import re
import numpy as np
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def parse_ue_log(filepath):
    """Parse srsUE log file for PDCCH DCI grants and PUCCH reports."""
    dl_grants = []   # (timestamp_ms, mcs, rnti, harq_id)
    ul_grants = []   # (timestamp_ms, mcs, rnti, harq_id)
    cqi_reports = [] # (timestamp_ms, cqi)
    pdsch_rx = []    # (timestamp_ms, mcs, tbs, snr, crc)

    # Patterns — srsUE format: [  321] PDCCH: cc=0, c-rnti=0x4601 dci=1_0 ... mcs=28
    dl_dci_pat = re.compile(
        r'\[\s*(\d+)\].*PDCCH.*c-rnti=0x(\w+)\s+dci=1_0\s+.*mcs=(\d+)'
    )
    ul_dci_pat = re.compile(
        r'\[\s*(\d+)\].*PDCCH.*c-rnti=0x(\w+)\s+dci=0_0\s+.*mcs=(\d+)'
    )
    cqi_pat = re.compile(
        r'\[\s*(\d+)\].*PUCCH.*cqi=(\d+)'
    )
    pdsch_pat = re.compile(
        r'\[\s*(\d+)\].*PDSCH.*mcs=(\d+).*tbs=(\d+).*snr=([+-]?\d+\.?\d*).*CRC=(\w+)'
    )

    with open(filepath, 'r') as f:
        for line in f:
            # DL DCI
            m = dl_dci_pat.search(line)
            if m:
                ts = float(m.group(1))
                rnti = m.group(2)
                mcs = int(m.group(3))
                # Skip SIB/RAR grants (low MCS during setup)
                dl_grants.append({'ts': ts, 'mcs': mcs, 'rnti': rnti})
                continue

            # UL DCI
            m = ul_dci_pat.search(line)
            if m:
                ts = float(m.group(1))
                mcs = int(m.group(3))
                ul_grants.append({'ts': ts, 'mcs': mcs})
                continue

            # CQI
            m = cqi_pat.search(line)
            if m:
                ts = float(m.group(1))
                cqi = int(m.group(2))
                cqi_reports.append({'ts': ts, 'cqi': cqi})
                continue

            # PDSCH
            m = pdsch_pat.search(line)
            if m:
                ts = float(m.group(1))
                mcs = int(m.group(2))
                tbs = int(m.group(3))
                snr = float(m.group(4))
                crc = m.group(5)
                pdsch_rx.append({'ts': ts, 'mcs': mcs, 'tbs': tbs,
                                 'snr': snr, 'crc_ok': crc == 'OK'})

    return dl_grants, ul_grants, cqi_reports, pdsch_rx


def plot_mcs_timeseries(dl_grants, ul_grants, cqi_reports, output_dir):
    """Plot MCS selection over time."""
    if not dl_grants:
        print("   No DL grants found — skipping time series")
        return

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("srsRAN Live Test — MCS & CQI Time Series",
                 fontsize=14, fontweight='bold')

    # Normalize timestamps to start from 0
    t0 = dl_grants[0]['ts']

    # DL MCS
    ax = axes[0]
    dl_ts = [(g['ts'] - t0) / 1000.0 for g in dl_grants]  # Convert to seconds
    dl_mcs = [g['mcs'] for g in dl_grants]
    ax.plot(dl_ts, dl_mcs, '.', color='#2196F3', markersize=1.5, alpha=0.5)

    # Moving average
    if len(dl_mcs) > 50:
        window = 50
        dl_mcs_arr = np.array(dl_mcs, dtype=float)
        dl_ts_arr = np.array(dl_ts)
        ma = np.convolve(dl_mcs_arr, np.ones(window)/window, mode='valid')
        ax.plot(dl_ts_arr[:len(ma)], ma, color='#1565C0', linewidth=1.5,
                label=f'Moving avg ({window})')
        ax.legend(fontsize=9)

    ax.set_ylabel("DL MCS Index", fontsize=11)
    ax.set_title("Downlink MCS (ML Policy)", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1, 30)

    # UL MCS
    ax = axes[1]
    if ul_grants:
        ul_ts = [(g['ts'] - t0) / 1000.0 for g in ul_grants]
        ul_mcs = [g['mcs'] for g in ul_grants]
        ax.plot(ul_ts, ul_mcs, '.', color='#4CAF50', markersize=1.5, alpha=0.5)

        if len(ul_mcs) > 50:
            ul_mcs_arr = np.array(ul_mcs, dtype=float)
            ma = np.convolve(ul_mcs_arr, np.ones(50)/50, mode='valid')
            ax.plot(np.array(ul_ts)[:len(ma)], ma, color='#2E7D32',
                    linewidth=1.5, label='Moving avg (50)')
            ax.legend(fontsize=9)

    ax.set_ylabel("UL MCS Index", fontsize=11)
    ax.set_title("Uplink MCS (Standard OLLA)", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1, 30)

    # CQI
    ax = axes[2]
    if cqi_reports:
        cqi_ts = [(r['ts'] - t0) / 1000.0 for r in cqi_reports]
        cqi_vals = [r['cqi'] for r in cqi_reports]
        ax.plot(cqi_ts, cqi_vals, 's-', color='#FF9800', markersize=3,
                linewidth=1, alpha=0.7)

    ax.set_ylabel("Wideband CQI", fontsize=11)
    ax.set_xlabel("Time (seconds)", fontsize=11)
    ax.set_title("UE CQI Reports", fontsize=11)
    ax.set_ylim(0, 16)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = f"{output_dir}/srsran_mcs_timeseries.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"   → Saved {path}")
    plt.close()


def plot_mcs_distribution(dl_grants, ul_grants, output_dir):
    """Plot MCS distribution histograms."""
    if not dl_grants:
        print("   No DL grants found — skipping distribution")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("srsRAN Live Test — MCS Distribution",
                 fontsize=14, fontweight='bold', y=1.02)

    # DL
    ax = axes[0]
    dl_mcs = [g['mcs'] for g in dl_grants if g['mcs'] > 2]  # Skip SIB/RAR
    if dl_mcs:
        mcs_vals, counts = np.unique(dl_mcs, return_counts=True)
        pcts = counts / len(dl_mcs) * 100
        bars = ax.bar(mcs_vals, pcts, color='#2196F3', alpha=0.85, width=0.8)
        for bar, pct in zip(bars, pcts):
            if pct > 1:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{pct:.1f}%', ha='center', fontsize=8)
    ax.set_xlabel("MCS Index", fontsize=11)
    ax.set_ylabel("Frequency (%)", fontsize=11)
    ax.set_title(f"DL MCS Distribution (n={len(dl_mcs)})", fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # UL
    ax = axes[1]
    ul_mcs = [g['mcs'] for g in ul_grants]
    if ul_mcs:
        mcs_vals, counts = np.unique(ul_mcs, return_counts=True)
        pcts = counts / len(ul_mcs) * 100
        bars = ax.bar(mcs_vals, pcts, color='#4CAF50', alpha=0.85, width=0.8)
        for bar, pct in zip(bars, pcts):
            if pct > 1:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{pct:.1f}%', ha='center', fontsize=8)
    ax.set_xlabel("MCS Index", fontsize=11)
    ax.set_ylabel("Frequency (%)", fontsize=11)
    ax.set_title(f"UL MCS Distribution (n={len(ul_mcs)})", fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = f"{output_dir}/srsran_mcs_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"   → Saved {path}")
    plt.close()


def print_summary(dl_grants, ul_grants, cqi_reports, pdsch_rx):
    """Print summary statistics."""
    print("\n" + "=" * 60)
    print("  srsRAN Log Analysis Summary")
    print("=" * 60)

    if dl_grants:
        dl_mcs = [g['mcs'] for g in dl_grants]
        dl_data = [g['mcs'] for g in dl_grants if g['mcs'] > 2]
        duration = (dl_grants[-1]['ts'] - dl_grants[0]['ts']) / 1000.0
        print(f"\n  Duration:          {duration:.1f} seconds")
        print(f"  DL grants total:   {len(dl_grants)}")
        print(f"  DL grants (data):  {len(dl_data)}")
        print(f"  UL grants:         {len(ul_grants)}")
        print(f"  CQI reports:       {len(cqi_reports)}")
        print(f"  PDSCH receptions:  {len(pdsch_rx)}")

        if dl_data:
            print(f"\n  DL MCS (data only):")
            print(f"    Mean:   {np.mean(dl_data):.1f}")
            print(f"    Median: {np.median(dl_data):.1f}")
            print(f"    Min:    {min(dl_data)}")
            print(f"    Max:    {max(dl_data)}")

        if cqi_reports:
            cqis = [r['cqi'] for r in cqi_reports]
            print(f"\n  CQI:")
            print(f"    Mean:   {np.mean(cqis):.1f}")
            print(f"    Min:    {min(cqis)}")
            print(f"    Max:    {max(cqis)}")

        if pdsch_rx:
            crc_ok = sum(1 for p in pdsch_rx if p['crc_ok'])
            print(f"\n  PDSCH CRC:")
            print(f"    OK:     {crc_ok}/{len(pdsch_rx)} ({crc_ok/len(pdsch_rx)*100:.1f}%)")

    print("=" * 60)


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 parse_srsran_logs.py <ue_log_file>")
        print("Example: python3 parse_srsran_logs.py /tmp/ue.log")
        sys.exit(1)

    logfile = sys.argv[1]
    output_dir = "results"

    print(f"Parsing {logfile}...")
    dl, ul, cqi, pdsch = parse_ue_log(logfile)

    print_summary(dl, ul, cqi, pdsch)
    plot_mcs_timeseries(dl, ul, cqi, output_dir)
    plot_mcs_distribution(dl, ul, output_dir)

    print("\n✓ Log analysis complete.")


if __name__ == "__main__":
    main()
