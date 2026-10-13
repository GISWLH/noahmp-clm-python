"""Diagnostic-only CUDA timing/consumption reproduction against frozen sources.

Can run under nsys with capture-range=cudaProfilerApi. Results remain diagnostic
observations even if validation fails; no row claims accepted benchmark timing.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

from bench_gpu import _gpu_runner
from gpu_checksum import as_cupy_result, consume_device_result, nondegenerate_change, json_safe
from noahmp_clm.io import synthetic_forcing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100000)
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capture", action="store_true")
    args = parser.parse_args()
    if args.n <= 0 or args.steps <= 0:
        parser.error("n and steps must be positive")
    if args.output.exists():
        parser.error("preserve prior evidence: use a new output path")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def emit(row):
        line = json.dumps(json_safe(row), allow_nan=False)
        print(line, flush=True)
        with args.output.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    import cupy as cp
    import jax
    root = Path(__file__).resolve().parent.parent
    names = ("benchmarks/bench_gpu.py", "benchmarks/gpu_checksum.py", "benchmarks/audit_gpu_timing.py",
             "src/noahmp_clm/gpu/jax_core.py", "src/noahmp_clm/gpu/cuda/loader.py",
             "src/noahmp_clm/gpu/cuda/column_kernel.cu")
    def source_hashes():
        return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}

    hashes = source_hashes()
    emit({"event": "environment", "n": args.n, "steps": args.steps,
          "status": "diagnostic_only", "cupy": cp.__version__, "jax": jax.__version__,
          "sourcehash": hashes})
    forcing = synthetic_forcing(n_days=args.steps / 48.0, dt=1800.0)
    reference = _gpu_runner("jax32", args.n, forcing)()
    ref_arrays = as_cupy_result(reference)
    ref_digest, ref_gate = consume_device_result(reference, args.n, args.steps, expected_dtype="float32")
    emit({"event": "jax32_reference", "digest": ref_digest, "gate": ref_gate})
    runner = _gpu_runner("cuda", args.n, forcing)
    from noahmp_clm.gpu.cuda.loader import KERNEL_SOURCE
    loaded_hash = hashlib.sha256(KERNEL_SOURCE.encode("utf-8")).hexdigest()
    source_text = (root / "src/noahmp_clm/gpu/cuda/column_kernel.cu").read_text(encoding="utf-8")
    if loaded_hash != hashlib.sha256(source_text.encode("utf-8")).hexdigest():
        raise AssertionError("Loaded CUDA source differs from the snapshot on disk")
    emit({"event": "loaded_cuda_source", "sha256_normalized_text": loaded_hash,
          "sha256_raw_bytes": hashes["src/noahmp_clm/gpu/cuda/column_kernel.cu"],
          "matches_snapshot": True})
    runtime = None
    if args.capture:
        from profile_gpu import _profiler_runtime
        runtime, _ = _profiler_runtime()
    timings, gates = [], []
    baseline_digest = None
    for phase in ("warmup", "diagnostic_sample"):
        for index in range(3):
            cp.cuda.runtime.deviceSynchronize()
            capture = runtime is not None and phase == "diagnostic_sample" and index == 0
            if capture and runtime.cudaProfilerStart() != 0:
                raise RuntimeError("CUDA profiler start failed")
            start_event, stop_event = cp.cuda.Event(), cp.cuda.Event()
            wall_start = time.perf_counter()
            start_event.record()
            result = runner()
            stop_event.record()
            stop_event.synchronize()
            compute_wall = time.perf_counter() - wall_start
            cuda_seconds = cp.cuda.get_elapsed_time(start_event, stop_event) / 1000.0
            digest, gate = consume_device_result(result, args.n, args.steps, ref_arrays, ref_digest, expected_dtype="float32")
            end_to_end = time.perf_counter() - wall_start
            if capture and runtime.cudaProfilerStop() != 0:
                raise RuntimeError("CUDA profiler stop failed")
            emit({"event": phase, "index": index, "status": "diagnostic_only",
                  "synchronized_compute_wall_seconds": compute_wall,
                  "cuda_event_span_seconds": cuda_seconds,
                  "end_to_end_seconds_including_reduce_readback": end_to_end,
                  "digest": digest, "gate": gate})
            timings.append((compute_wall, cuda_seconds, end_to_end))
            gates.append(gate["passed"])
            baseline_digest = digest
            del result
    changed = [replace(f, swdown=f.swdown * 1.17 + 31.0) for f in forcing]
    result = _gpu_runner("cuda", args.n, changed)()
    changed_digest, changed_gate = consume_device_result(result, args.n, args.steps, expected_dtype="float32")
    sensitivity = nondegenerate_change(baseline_digest, changed_digest)
    emit({"event": "sensitivity", "forcing_change": "swdown * 1.17 + 31 W/m2",
          "digest": changed_digest, "finite_gate": changed_gate, "sensitivity": sensitivity})
    unchanged = source_hashes() == hashes
    success = all(gates) and ref_gate["passed"] and changed_gate["passed"] and sensitivity["passed"] and unchanged
    emit({"event": "summary", "status": "diagnostic_validation_passed" if success else "diagnostic_validation_failed",
          "accepted_benchmark": False, "all_six_strict_gates_passed": all(gates),
          "sensitivity_passed": sensitivity["passed"], "source_unchanged": unchanged})
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
