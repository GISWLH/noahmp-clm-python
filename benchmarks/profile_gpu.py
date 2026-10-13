"""One bounded JAX64 GPU profile using the benchmark's unchanged workload.

Run under ``nsys profile --trace=cuda --sample=none --cpuctxsw=none
--capture-range=cudaProfilerApi --capture-range-end=stop``. Three synchronized
warmups precede the single captured call. Large traces belong in profile_output;
this runner writes only compact JSONL metadata and measurements.
"""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import sysconfig
import time

from bench_gpu import _gpu_runner
from noahmp_clm.io import synthetic_forcing


def _profiler_runtime():
    """Load the venv CUDA runtime only to mark the Nsight capture interval."""
    path = (Path(sysconfig.get_paths()["purelib"]) / "nvidia" / "cuda_runtime"
            / "lib" / "libcudart.so.12")
    runtime = ctypes.CDLL(str(path))
    for name in ("cudaProfilerStart", "cudaProfilerStop"):
        function = getattr(runtime, name)
        function.argtypes = []
        function.restype = ctypes.c_int
    return runtime, str(path)


def _usage():
    value = resource.getrusage(resource.RUSAGE_SELF)
    return {"cpu_user_seconds": value.ru_utime,
            "cpu_system_seconds": value.ru_stime,
            "voluntary_context_switches": value.ru_nvcsw,
            "involuntary_context_switches": value.ru_nivcsw}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100000)
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--output", default="profile_output/jax64-profile.jsonl")
    args = parser.parse_args()
    if min(args.n, args.steps, args.warmups) <= 0:
        parser.error("n, steps, and warmups must be positive")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        parser.error("output exists; use a new path to preserve profile evidence")

    def emit(event, **values):
        row = {"event": event, "utc": datetime.now(timezone.utc).isoformat(),
               "backend": "jax64", "n": args.n, "steps": args.steps, **values}
        line = json.dumps(row, ensure_ascii=False)
        print(line, flush=True)
        with output.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    import jax
    import numpy as np
    emit("environment", python=platform.python_version(), numpy=np.__version__,
         jax=importlib.metadata.version("jax"),
         jaxlib=importlib.metadata.version("jaxlib"),
         cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
         warmups=args.warmups, captured_calls=1)
    forcings = synthetic_forcing(n_days=args.steps / 48., dt=1800.)
    if len(forcings) != args.steps:
        raise ValueError("synthetic forcing length differs from requested steps")
    runner = _gpu_runner("jax64", args.n, forcings)
    device = jax.devices()[0]
    emit("device", description=str(device), kind=device.device_kind,
         input_state_bytes=args.n * (70 * 8 + 33 * 4),
         shared_forcing_bytes=args.steps * 8 * 8,
         output_bytes=args.n * (70 * 8 + 33 * 4 + args.steps * 12 * 8))
    for index in range(args.warmups):
        start = time.perf_counter()
        result = runner()
        elapsed = time.perf_counter() - start
        del result
        emit("warmup", index=index, wall_seconds=elapsed)

    runtime, runtime_path = _profiler_runtime()
    emit("capture_ready", runtime=runtime_path, memory=device.memory_stats())
    before = _usage()
    status = runtime.cudaProfilerStart()
    if status != 0:
        raise RuntimeError(f"cudaProfilerStart failed with CUDA error {status}")
    start = time.perf_counter()
    try:
        result = runner()
        elapsed = time.perf_counter() - start
    finally:
        stop_status = runtime.cudaProfilerStop()
    after = _usage()
    if stop_status != 0:
        raise RuntimeError(f"cudaProfilerStop failed with CUDA error {stop_status}")
    emit("captured_call", wall_seconds=elapsed,
         us_per_column_step=elapsed / (args.n * args.steps) * 1e6,
         usage_delta={key: after[key] - before[key] for key in before},
         memory=device.memory_stats(),
         result_shapes=[list(value.shape) for value in result],
         result_dtypes=[str(value.dtype) for value in result])
    # Small post-capture readback is evidence of completed output, not parity.
    emit("output_sample", first_column_last_diagnostics=
         np.asarray(jax.device_get(result[2][0, -1])).tolist())


if __name__ == "__main__":
    main()
