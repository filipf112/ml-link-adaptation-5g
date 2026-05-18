#include <iostream>
#include <vector>
#include <iomanip>
#include <chrono>
#include <random>
#include <algorithm>
#include <numeric>
#include "srsran_integration/scheduler/ml_link_adaptation.h"
#include "srsran_integration/scheduler/ml_link_adaptation.cpp"

using namespace std;
using namespace srsran;

int main() {
    cout << "==================================================" << endl;
    cout << "   ML Link Adaptation — C++ Inference Benchmark    " << endl;
    cout << "==================================================" << endl;

    // ── Test 1: Functional correctness ──────────────────────────
    cout << "\n[1] Functional Test — SNR Sweep" << endl;
    cout << "--------------------------------------------------" << endl;
    cout << " SINR (dB) | EMA BLER | OLLA Offset | Selected MCS" << endl;
    cout << "--------------------------------------------------" << endl;

    OllaState state;
    double speed = 30.0, channel = 1.0, band = 0.0, ant = 2.0, bler_log = -1.0;
    vector<double> sinr_trace = {-5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0};

    for (double sinr : sinr_trace) {
        int mcs = select_mcs(sinr, speed, channel, band, ant, bler_log, state);
        cout << " " << setw(9) << fixed << setprecision(1) << sinr
             << " | " << setw(7) << fixed << setprecision(3) << state.bler_ema
             << "  | " << setw(11) << state.mcs_offset
             << " | " << setw(12) << mcs << endl;

        // Simulate ACKs (no BLER)
        for (int i = 0; i < 10; i++)
            update_olla_state(state, true);
    }

    // ── Test 2: Monotonicity check ─────────────────────────────
    cout << "\n[2] Monotonicity Verification" << endl;
    cout << "    Testing: higher SINR → equal or higher MCS..." << endl;

    bool monotonic = true;
    int prev_mcs = 0;
    OllaState mono_state;
    for (double sinr = -10.0; sinr <= 35.0; sinr += 0.5) {
        int mcs = select_mcs(sinr, 30.0, 1.0, 0.0, 1.0, -1.0, mono_state);
        if (mcs < prev_mcs) {
            monotonic = false;
            cout << "    VIOLATION at SINR=" << sinr << " dB: MCS "
                 << prev_mcs << " → " << mcs << endl;
        }
        prev_mcs = mcs;
    }
    cout << "    Result: " << (monotonic ? "PASS ✓" : "FAIL ✗") << endl;

    // ── Test 3: Inference latency benchmark ────────────────────
    cout << "\n[3] Inference Latency Benchmark" << endl;

    // Prepare random inputs
    mt19937 gen(42);
    uniform_real_distribution<> sinr_dist(-10.0, 35.0);
    uniform_real_distribution<> speed_dist(0.0, 150.0);
    uniform_int_distribution<>  chan_dist(0, 4);
    uniform_int_distribution<>  band_dist(0, 1);
    uniform_int_distribution<>  ant_dist(1, 2);

    const int N_WARMUP = 10000;
    const int N_ITER   = 1000000;

    OllaState bench_state;

    // Warmup (avoid cold-cache effects)
    for (int i = 0; i < N_WARMUP; i++) {
        select_mcs(sinr_dist(gen), speed_dist(gen), chan_dist(gen),
                   band_dist(gen), ant_dist(gen), -1.0, bench_state);
    }

    // Timed benchmark
    vector<double> latencies_ns;
    latencies_ns.reserve(N_ITER);

    for (int i = 0; i < N_ITER; i++) {
        double s = sinr_dist(gen);
        double sp = speed_dist(gen);
        double ch = chan_dist(gen);
        double b = band_dist(gen);
        double a = ant_dist(gen);

        auto t0 = chrono::high_resolution_clock::now();
        volatile int mcs = select_mcs(s, sp, ch, b, a, -1.0, bench_state);
        auto t1 = chrono::high_resolution_clock::now();

        double ns = chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count();
        latencies_ns.push_back(ns);

        // Occasional OLLA update
        if (i % 10 == 0) update_olla_state(bench_state, true);
    }

    // Statistics
    sort(latencies_ns.begin(), latencies_ns.end());
    double mean_ns = accumulate(latencies_ns.begin(), latencies_ns.end(), 0.0) / N_ITER;
    double median_ns = latencies_ns[N_ITER / 2];
    double p95_ns = latencies_ns[(int)(N_ITER * 0.95)];
    double p99_ns = latencies_ns[(int)(N_ITER * 0.99)];
    double max_ns = latencies_ns.back();
    double min_ns = latencies_ns.front();

    cout << "    Iterations:  " << N_ITER << endl;
    cout << "    -------------------------------------------" << endl;
    cout << "    Mean:        " << fixed << setprecision(0) << mean_ns << " ns ("
         << setprecision(3) << mean_ns / 1000.0 << " μs)" << endl;
    cout << "    Median:      " << fixed << setprecision(0) << median_ns << " ns" << endl;
    cout << "    P95:         " << fixed << setprecision(0) << p95_ns << " ns" << endl;
    cout << "    P99:         " << fixed << setprecision(0) << p99_ns << " ns" << endl;
    cout << "    Min:         " << fixed << setprecision(0) << min_ns << " ns" << endl;
    cout << "    Max:         " << fixed << setprecision(0) << max_ns << " ns" << endl;
    cout << "    -------------------------------------------" << endl;

    double slot_budget_ns = 1000000.0; // 1ms slot = 1,000,000 ns
    cout << "    L2 slot budget: 1,000,000 ns (1 ms)" << endl;
    cout << "    ML inference:   " << fixed << setprecision(0) << mean_ns
         << " ns (" << setprecision(4) << (mean_ns / slot_budget_ns * 100)
         << "% of budget)" << endl;
    cout << "    Verdict:     " << (p99_ns < 10000 ? "PASS ✓ (<10μs P99)" : "MARGINAL")
         << endl;

    cout << "\n==================================================" << endl;
    cout << "   All tests complete." << endl;
    cout << "==================================================" << endl;
    return 0;
}
