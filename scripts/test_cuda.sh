#!/usr/bin/env bash
set -euo pipefail

bash scripts/build_cuda.sh
./build-cuda/kllm_tests

