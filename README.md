# Machine Learning-Driven Formulation of Link Adaptation for 5G PHY/MAC Systems

## Abstract

In modern 5G and Beyond-5G (B5G) networks, the Base Station (gNB) Medium Access Control (MAC) layer faces the critical challenge of selecting the optimal Modulation and Coding Scheme (MCS) for downstream users within strict sub-millisecond latency budgets. Traditional approaches, such as Outer Loop Link Adaptation (OLLA), rely on heuristic margin adjustments that often fail to capture the highly nonlinear dynamics of fast-fading multipath channels, leading to suboptimal spectral efficiency and Block Error Rate (BLER) violations.

This project investigates a novel machine learning framework for predictive MCS selection. By simulating a fully 3GPP-compliant physical layer (multipath fading, LDPC coding, HARQ with Chase Combining), we formulate the link adaptation problem as an **Ordinal Regression task with Monotonic Constraints**. To satisfy the stringent execution time requirements of a real-time gNB, we propose a methodology based on Knowledge Distillation, where a complex, robust gradient-boosting ensemble is compressed into a computationally lightweight surrogate decision tree.

## I. System Model and Simulation Environment

The data generation pipeline relies on **NVIDIA Sionna**, a GPU-accelerated framework for 5G/6G physical layer simulations. We model the complete radio access link, inclusive of:
- **Channel Models:** 3GPP TR 38.901 TDL-A, TDL-B, and TDL-C profiles to reflect varying delay spreads in indoor, urban, and severe multipath environments.
- **Mobility:** Doppler effects induced by variable User Equipment (UE) velocities (from pedestrian 3 km/h to high-speed vehicular 120 km/h).
- **FEC and HARQ:** Transport Block (TB) encoding via 3GPP NR LDPC, alongside Hybrid Automatic Repeat reQuest with Chase Combining (HARQ-CC) utilizing Log-Likelihood Ratio (LLR) accumulation.

## II. Methodology

### A. Ordinal Target Formulation

MCS indices (0 through 28) are inherently ordinal: a higher MCS increases spectral efficiency but degrades error robustness. Treating this as a standard nominal classification problem is mathematically flawed. We decompose the $K$-class problem into $K-1$ binary classifiers, optimizing an asymmetric rank-cost function where over-estimation (causing packet loss) is penalized more heavily than under-estimation (suboptimal throughput).

### B. Monotonic Constraints

To embed domain knowledge and ensure physical robustness, we employ `HistGradientBoosting` algorithms subject to monotonic constraints. The hypothesis guarantees that strictly superior channel conditions ($\uparrow$ SNR, $\downarrow$ Doppler spread) mathematically guarantee a monotonically non-decreasing MCS selection.

### C. Latency-Constrained Inference via Distillation 

Complex ensemble models violate the strict microsecond inference limits required by L2 schedulers. We implement a surrogate distillation technique, converting the ensemble's continuous decision surfaces into a bounded-depth decision tree (`max_depth=10`). This tree is exported to deterministic `O(1)` C++ logic, operating alongside a classical OLLA safety fallback.

## III. Repository Structure

*Repository structural layout is under active development. Current tracking involves submodule integration for the physical layer baseline.*

- `sionna/` : Git submodule linking the upstream NVIDIA Sionna repository utilized for PHY simulation pipelines.

## IV. Prerequisites and Dependencies

- Python 3.10+
- TensorFlow 2.14+ (for Sionna hardware acceleration)
- scikit-learn >= 1.3
- GCC/Clang (Supporting C++17)

---
*Developed for the academic course: Decision Support Algorithms (Algorytmy Wspomagania Decyzji).*
