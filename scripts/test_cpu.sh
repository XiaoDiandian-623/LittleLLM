#!/usr/bin/env bash
set -euo pipefail

bash scripts/build_cpu.sh
./build-cpu/kllm_tests

