# syntax=docker/dockerfile:1.4

# ─── Build Stage ─────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    pkg-config \
    libssl-dev \
    ca-certificates \
    python3 \
    && rm -rf /var/lib/apt/lists/*

# Install Rust via rustup
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain stable
ENV PATH="/root/.cargo/bin:${PATH}"

# CUDA environment
ENV CUDA_PATH=/usr/local/cuda
ENV CUDATKDIR=/usr/local/cuda
ENV CUDA_COMPUTE_CAP=89
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

WORKDIR /app

# See .dockerignore
COPY . .

# Use BuildKit cache mounts for external build cache
RUN --mount=type=cache,target=/root/.cargo/registry \
    --mount=type=cache,target=/root/.cargo/git \
    --mount=type=cache,target=/app/target \
    cargo build --release --features cuda && \
    # Copy out the binary before the cache mount disappears
    cp $(cargo metadata --no-deps --format-version 1 | \
         python3 -c "import sys,json; pkgs=json.load(sys.stdin)['packages']; print(pkgs[0]['targets'][0]['name'])") \
       /app/binary || \
    # Fallback: copy all release binaries
    find target/release -maxdepth 1 -type f -executable -exec cp {} /app/binary \;

# ─── Runtime Stage ────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
# Should mount at runtime
ENV HF_HOME=/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl3 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy the built binary
COPY --from=builder /app/binary /usr/local/bin/app

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}

ENTRYPOINT ["/usr/local/bin/app"]
