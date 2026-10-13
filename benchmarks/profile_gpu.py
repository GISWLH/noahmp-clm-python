"""One bounded JAX GPU profile using the benchmark's unchanged workload.

Run under ``nsys profile --trace=cuda --sample=none --cpuctxsw=none
--capture-range=cudaProfilerApi --capture-range-end=stop``. Three synchronized
warmups precede the single captured call. Large traces belong in profile_output;
this runner writes only compact JSONL metadata and measurements.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
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
    parser.add_argument("--backend", choices=("jax64", "jax32"), default="jax64")
    parser.add_argument("--dump-hlo", type=Path,
                        help="Save optimized HLO outside the capture; static costs are not executed FLOPs")
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
               "backend": args.backend, "n": args.n, "steps": args.steps,
               "status": "diagnostic_profile_only", **values}
        line = json.dumps(row, ensure_ascii=False)
        print(line, flush=True)
        with output.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    import jax
    import numpy as np
    from gpu_checksum import as_cupy_result, consume_device_result, assert_gate
    source = Path(__file__).resolve().parents[1] / "src/noahmp_clm/gpu/jax_core.py"
    emit("environment", python=platform.python_version(), numpy=np.__version__,
         jax=importlib.metadata.version("jax"),
         jaxlib=importlib.metadata.version("jaxlib"),
         cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
         warmups=args.warmups, captured_calls=1,
         jax_core_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    forcings = synthetic_forcing(n_days=args.steps / 48., dt=1800.)
    if len(forcings) != args.steps:
        raise ValueError("synthetic forcing length differs from requested steps")
    runner = _gpu_runner(args.backend, args.n, forcings)
    device = jax.devices()[0]
    itemsize = 8 if args.backend == "jax64" else 4
    emit("device", description=str(device), kind=device.device_kind,
         input_state_bytes=args.n * (70 * itemsize + 33 * 4),
         shared_forcing_bytes=args.steps * 8 * itemsize,
         output_bytes=args.n * (70 * itemsize + 33 * 4 + args.steps * 12 * itemsize))
    reference = None
    for index in range(args.warmups):
        start = time.perf_counter()
        result = runner()
        elapsed = time.perf_counter() - start
        reference = result
        emit("warmup", index=index, wall_seconds=elapsed)

    reference_arrays = as_cupy_result(reference)
    reference_digest, reference_gate = consume_device_result(reference, args.n, args.steps)
    assert_gate(reference_gate)
    if args.dump_hlo:
        from noahmp_clm.gpu.jax_core import _BATCH_JIT
        dtype = np.float64 if itemsize == 8 else np.float32
        executable = _BATCH_JIT.lower(
            jax.ShapeDtypeStruct((args.n, 70), dtype),
            jax.ShapeDtypeStruct((args.n, 33), np.int32),
            jax.ShapeDtypeStruct((args.n, args.steps, 8), dtype),
        ).compile()
        args.dump_hlo.parent.mkdir(parents=True, exist_ok=True)
        if args.dump_hlo.exists():
            raise FileExistsError(args.dump_hlo)
        hlo = executable.as_text()
        args.dump_hlo.write_text(hlo, encoding="utf-8")
        emit("optimized_hlo", path=str(args.dump_hlo),
             sha256=hashlib.sha256(hlo.encode()).hexdigest(),
             static_cost_analysis=executable.cost_analysis(),
             costs_are_not_executed_flop_counts=True)

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
    digest, gate = consume_device_result(result, args.n, args.steps,
                                         reference_arrays, reference_digest)
    emit("post_capture_full_output_gate", digest=digest, gate=gate,
         timing_includes_reduce_readback=False, reference="last synchronized warmup")
    assert_gate(gate)


if __name__ == "__main__":
    main()
