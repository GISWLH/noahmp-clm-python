#!/usr/bin/env bash
# xuanze RTX4090 = sm_89；可用 CUDA_ARCH 覆盖。
set -eu
cd "$(dirname "$0")"
nvcc -O3 --fmad=false --prec-div=false -std=c++14 -arch="${CUDA_ARCH:-sm_89}" \
  --compiler-options=-fPIC -shared column_kernel.cu -o column_kernel.so
