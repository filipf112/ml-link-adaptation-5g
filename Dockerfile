# ==============================================================================
# Dockerfile — ML Link Adaptation for 5G NR
# ==============================================================================
# Multi-stage build:
#   Stage 1: GPU image for dataset generation + training (Sionna + TensorFlow)
#   Stage 2: Lightweight image for inference-only / benchmark
#
# Usage:
#   # Build the full GPU image (for dataset generation + training)
#   docker build -t ml-la-5g .
#
#   # Run dataset generation (needs GPU — use nvidia-container-toolkit)
#   docker run --gpus all ml-la-5g python3 generate_v2_dataset.py --quick
#
#   # Run training
#   docker run --gpus all -v $(pwd)/data:/app/data ml-la-5g \
#       python3 train_real_ml_model.py
#
#   # Run benchmark (CPU is sufficient)
#   docker run ml-la-5g python3 benchmark_la_approaches.py
#
#   # Run closed-loop digital twin
#   docker run ml-la-5g python3 online_la_simulator.py
#
# Prerequisites:
#   - NVIDIA Container Toolkit (for GPU passthrough)
#     https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
#   - Docker >= 19.03
# ==============================================================================

# --- Base: NVIDIA CUDA 12.x + cuDNN 9 on Ubuntu 22.04 -----------------------
# TensorFlow 2.18 requires CUDA 12.x and cuDNN 9.
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04 AS base

# Prevent interactive prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive
ENV TF_CPP_MIN_LOG_LEVEL=2

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        && rm -rf /var/lib/apt/lists/*

# Ensure 'python' points to python3
RUN ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

# Install Python dependencies (layer cached separately from source code)
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# --- Full image: copy all source code ----------------------------------------
FROM base AS full

COPY . .

# Default command: show available scripts
CMD ["bash", "-c", "echo '=== ML Link Adaptation for 5G NR ===' && \
     echo '' && \
     echo 'Available scripts:' && \
     echo '  python3 generate_v2_dataset.py --quick   # Generate dataset (GPU)' && \
     echo '  python3 train_real_ml_model.py            # Train all models' && \
     echo '  python3 benchmark_la_approaches.py        # Run benchmark suite' && \
     echo '  python3 online_la_simulator.py            # Closed-loop digital twin' && \
     echo '' && \
     echo 'Example:' && \
     echo '  docker run --gpus all ml-la-5g python3 generate_v2_dataset.py --quick'"]
