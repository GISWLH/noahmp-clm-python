#!/usr/bin/env bash
# [未实测] 需要带 nvcc 的 Linux/CUDA Toolkit 主机；本机 nvcc 不可用。
set -eu
nvcc -O3 --use_fast_math -std=c++14 -arch=sm_89 \
  --compiler-options=-fPIC -shared column_kernel.cu -o column_kernel.so
