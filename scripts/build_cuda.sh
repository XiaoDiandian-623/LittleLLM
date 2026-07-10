#!/usr/bin/env bash
set -euo pipefail

CUDA_ARCHS="${KLLM_CUDA_ARCHS:-60;61;70;75;80;86}"

cmake -S . -B build-cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DKLLM_USE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}"
cmake --build build-cuda -j"$(nproc)"
