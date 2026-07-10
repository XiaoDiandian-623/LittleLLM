#!/usr/bin/env bash
set -euo pipefail

cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release -DKLLM_USE_CUDA=OFF
cmake --build build-cpu -j"$(nproc)"
