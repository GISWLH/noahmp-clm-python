"""Bounded CUDA counter probe; diagnostic only, never an accepted benchmark.

Use ncu --profile-from-start off --kernel-name regex:step_columns --launch-count 1.
Three complete warmups precede the captured call. The profiler selects its first
column step, so counters cannot be extrapolated to all 48 steps without evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from bench_gpu import _gpu_runner
from gpu_checksum import consume_device_result
from noahmp_clm.io import synthetic_forcing
from profile_gpu import _profiler_runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100000)
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.n < 1 or args.steps < 1 or args.output.exists():
        parser.error("Positive counts and a fresh output path are required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    import cupy as cp
    from noahmp_clm.gpu.cuda.loader import load_column_kernel
    root = Path(__file__).resolve().parents[1]
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
              for name in ("src/noahmp_clm/gpu/cuda/column_kernel.cu",
                           "src/noahmp_clm/gpu/cuda/loader.py",
                           "benchmarks/profile_cuda_kernel.py")}
    forcing = synthetic_forcing(n_days=args.steps / 48., dt=1800.)
    if len(forcing) != args.steps:
        raise ValueError("Forcing length mismatch")
    runner = _gpu_runner("cuda", args.n, forcing)
    for _ in range(3):
        result = runner()
        del result
    runtime, runtime_path = _profiler_runtime()
    if runtime.cudaProfilerStart() != 0:
        raise RuntimeError("Profiler start failed")
    try:
        result = runner()
    finally:
        stop_status = runtime.cudaProfilerStop()
    if stop_status:
        raise RuntimeError("Profiler stop failed")
    digest, gate = consume_device_result(result, args.n, args.steps)
    row = {"status": "diagnostic_only_h4_unvalidated", "accepted_benchmark": False,
           "n": args.n, "steps": args.steps, "sourcehash": hashes,
           "cupy": cp.__version__, "device": str(cp.cuda.Device()),
           "runtime": runtime_path, "kernel_attributes": load_column_kernel().attributes,
           "digest": digest, "finite_and_shape_gate": gate,
           "reference_parity_checked_here": False}
    args.output.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: row[key] for key in ("status", "n", "steps", "kernel_attributes")}))
    return 0 if gate["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
