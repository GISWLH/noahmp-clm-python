"""实际同步的numpy单核/JAX-CUDA x64/x32/手写CUDA基准，逐项输出JSON。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import statistics
import time

import numpy as np

from noahmp_clm.column import ColumnState
from noahmp_clm.driver import run_column
from noahmp_clm.gpu.pack import column_state_to_vec, forcing_to_vec
from noahmp_clm.io import synthetic_forcing

DEFAULT_SIZES = (100, 1000, 10000, 100000, 300000)


def _numpy_runner(n, forcings):
    """每柱都真实调用普通版；及时释放诊断，避免大N保留数千万Python字典。"""
    seed = ColumnState()

    def run():
        checksum = 0.0
        for _ in range(n):
            diagnostics, final = run_column(seed, forcings)
            checksum += final.ts + diagnostics[-1]["et_mm"]
        return checksum

    return run


def _gpu_runner(backend, n, forcings):
    dtype = np.float64 if backend == "jax64" else np.float32
    f, i = column_state_to_vec(ColumnState())
    f = np.tile(f.astype(dtype), (n, 1))
    i = np.tile(i, (n, 1))
    u = np.asarray([forcing_to_vec(x) for x in forcings], dtype=dtype)
    if backend.startswith("jax"):
        import jax
        jax.config.update("jax_enable_x64", True)
        from noahmp_clm.gpu.jax_core import run_columns_jax
        if jax.default_backend() != "gpu":
            raise RuntimeError("本基准要求JAX-CUDA；禁止静默使用CPU")
        f, i, u = jax.device_put((f, i, u))

        def run():
            result = run_columns_jax(f, i, u)
            jax.block_until_ready(result)
            return result

        return run
    import cupy as cp
    from noahmp_clm.gpu.cuda.loader import run_columns_cuda
    f, i, u = cp.asarray(f), cp.asarray(i), cp.asarray(u)

    def run():
        result = run_columns_cuda(f, i, u)
        cp.cuda.get_current_stream().synchronize()
        return result

    return run


def _timed(function, repeats, progress):
    """Consume the NumPy scalar checksum and require deterministic repeats."""
    values = []
    expected_checksum = None
    for phase, count in (("warmup", 3), ("measured_run", repeats)):
        for index in range(count):
            start = time.perf_counter()
            result = np.asarray(function())
            if result.ndim != 0:
                raise ValueError("NumPy benchmark must return a scalar checksum")
            checksum = float(result)
            if not np.isfinite(checksum):
                raise ValueError(f"NumPy {phase} {index}: nonfinite checksum {checksum}")
            if expected_checksum is None:
                expected_checksum = checksum
            elif checksum != expected_checksum:
                raise ValueError(f"NumPy {phase} {index}: checksum changed from {expected_checksum} to {checksum}")
            elapsed = time.perf_counter() - start
            progress(phase, index, elapsed, checksum)
            if phase == "measured_run":
                values.append(elapsed)
    return statistics.median(values), values


def _source_hashes():
    root = Path(__file__).resolve().parents[1]
    names = (
        "benchmarks/bench_gpu.py", "benchmarks/gpu_checksum.py",
        "src/noahmp_clm/gpu/jax_core.py", "src/noahmp_clm/gpu/cuda/loader.py",
        "src/noahmp_clm/gpu/cuda/column_kernel.cu",
    )
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}


def _sensitivity_preflight(backend, n, forcings, reference_arrays, progress):
    """Perturb the full N columns outside timing; compare each changed result."""
    import cupy as cp
    import jax
    if __package__:
        from .gpu_checksum import consume_device_result, as_cupy_result, nondegenerate_change, assert_gate
    else:
        from gpu_checksum import consume_device_result, as_cupy_result, nondegenerate_change, assert_gate
    from noahmp_clm.gpu.jax_core import run_columns_jax

    count = n
    dtype = np.float64 if backend == "jax64" else np.float32
    f, i = column_state_to_vec(ColumnState())
    f = np.tile(f.astype(dtype), (count, 1))
    i = np.tile(i, (count, 1))
    u = np.tile(np.asarray([forcing_to_vec(x) for x in forcings], dtype=dtype), (count, 1, 1))
    # Per-column forcing and initial-state changes expose cached, constant and
    # broadcast-one-column implementations that homogeneous timings would miss.
    ids = np.arange(count)
    f[:, 25] += ((ids % 23) + 1) * 0.25
    u[:, :, 5] = (u[:, :, 5] * (1.05 + 0.04 * (ids % 4))[:, None]
                  + 31.0 + (ids % 11)[:, None])
    expected = run_columns_jax(*jax.device_put((f, i, u)))
    jax.block_until_ready(expected)
    expected_arrays = as_cupy_result(expected)
    expected_digest, expected_gate = consume_device_result(expected, count, len(forcings), expected_dtype=dtype)
    assert_gate(expected_gate)
    if backend == "cuda":
        from noahmp_clm.gpu.cuda.loader import run_columns_cuda
        actual = run_columns_cuda(cp.asarray(f), cp.asarray(i), cp.asarray(u))
    else:
        actual = run_columns_jax(*jax.device_put((f, i, u)))
        jax.block_until_ready(actual)
    digest, gate = consume_device_result(actual, count, len(forcings), expected_arrays, expected_digest, expected_dtype=dtype)
    base = tuple(array[:count] for array in reference_arrays)
    base_digest, base_gate = consume_device_result(base, count, len(forcings), expected_dtype=dtype)
    assert_gate(base_gate)
    sensitivity = nondegenerate_change(base_digest, digest)
    actual_diag = as_cupy_result(actual)[2]
    changed = cp.any(cp.abs(actual_diag[:, :, :6] - base[2][:, :, :6]) > 1e-6, axis=(1, 2))
    changed_columns = int(cp.asnumpy(cp.sum(changed)))
    passed = gate["passed"] and sensitivity["passed"] and changed_columns == count
    progress("sensitivity_preflight", 0, {
        "status": "passed" if passed else "failed", "columns": count,
        "scope": "full_benchmark_N_outside_timing",
        "independently_changed_columns": changed_columns, "gate": gate,
        "sensitivity": sensitivity, "digest": digest,
    })
    assert_gate(gate)
    if not sensitivity["passed"] or changed_columns != count:
        raise AssertionError("GPU benchmark failed independent-column/forcing sensitivity")


def _audited_gpu_timed(backend, n, forcings, repeats, progress):
    """GPU benchmark times are accepted only after full-output device consumption."""
    import cupy as cp
    if __package__:
        from .gpu_checksum import consume_device_result, as_cupy_result, assert_gate
    else:
        from gpu_checksum import consume_device_result, as_cupy_result, assert_gate

    hashes = _source_hashes()
    cuda_source_path = Path(__file__).resolve().parents[1] / "src/noahmp_clm/gpu/cuda/column_kernel.cu"
    normalized_cuda_hash = hashlib.sha256(cuda_source_path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
    reference_backend = "jax32" if backend == "cuda" else backend
    dtype = np.dtype("float64" if backend == "jax64" else "float32")
    reference = _gpu_runner(reference_backend, n, forcings)()
    reference_arrays = as_cupy_result(reference)
    reference_digest, reference_gate = consume_device_result(reference, n, len(forcings), expected_dtype=dtype)
    assert_gate(reference_gate)
    progress("reference", 0, {"reference_backend": reference_backend,
             "sourcehash": hashes, "digest": reference_digest, "gate": reference_gate})
    runner = _gpu_runner(backend, n, forcings)
    if backend == "cuda":
        from noahmp_clm.gpu.cuda.loader import KERNEL_SOURCE
        loaded_hash = hashlib.sha256(KERNEL_SOURCE.encode("utf-8")).hexdigest()
        if loaded_hash != normalized_cuda_hash:
            raise AssertionError("Loaded CUDA source differs from the recorded source hash")
        progress("loaded_cuda_source", 0, {
            "sha256_normalized_text": loaded_hash,
            "sha256_raw_bytes": hashes["src/noahmp_clm/gpu/cuda/column_kernel.cu"],
            "matches_disk_normalized_text": True,
        })
    _sensitivity_preflight(backend, n, forcings, reference_arrays, progress)
    samples = []
    for phase, count in (("warmup", 3), ("measured_run", repeats)):
        for index in range(count):
            cp.cuda.runtime.deviceSynchronize()
            start_event = cp.cuda.Event() if backend == "cuda" else None
            stop_event = cp.cuda.Event() if backend == "cuda" else None
            start = time.perf_counter()
            if start_event is not None:
                start_event.record()
            result = runner()  # Existing synchronized API; profile_gpu stays compatible.
            if stop_event is not None:
                stop_event.record()
                stop_event.synchronize()
            compute_wall = time.perf_counter() - start
            event_seconds = (cp.cuda.get_elapsed_time(start_event, stop_event) / 1000.0
                             if start_event is not None else None)
            digest, gate = consume_device_result(result, n, len(forcings), reference_arrays, reference_digest, expected_dtype=dtype)
            elapsed = time.perf_counter() - start
            timings = {"end_to_end_seconds": elapsed,
                       "synchronized_compute_wall_seconds": compute_wall,
                       "cuda_event_span_seconds": event_seconds}
            if not gate["passed"]:
                progress("validation_failed", index, {"status": "rejected", "phase": phase,
                         "diagnostic_timings_only": timings, "digest": digest, "gate": gate})
                assert_gate(gate)
            if _source_hashes() != hashes:
                raise AssertionError("Source changed during benchmark; timing is not accepted")
            progress(phase, index, {"status": "validated", **timings, "seconds": elapsed,
                                   "digest": digest, "gate": gate})
            if phase == "measured_run":
                samples.append(timings)
            del result
    elapsed = statistics.median(sample["end_to_end_seconds"] for sample in samples)
    return elapsed, samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=list(DEFAULT_SIZES))
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--backends", nargs="+", choices=("numpy", "jax64", "jax32", "cuda"),
                        default=["numpy", "jax64", "jax32", "cuda"])
    parser.add_argument("--output", help="持续追加的JSONL原始结果")
    parser.add_argument("--cuda-parity-passed", action="store_true",
                        help="仅在独立H4已通过时使用；该声明不能绕过本基准的逐次checksum门")
    args = parser.parse_args()
    if args.steps <= 0 or args.repeats <= 0 or any(n <= 0 for n in args.sizes):
        parser.error("sizes/steps/repeats必须为正")

    def emit(row):
        if __package__:
            from .gpu_checksum import json_safe
        else:
            from gpu_checksum import json_safe
        line = json.dumps(json_safe(row), ensure_ascii=False, allow_nan=False)
        print(line, flush=True)
        if args.output:
            with open(args.output, "a", encoding="utf-8") as output:
                output.write(line + "\n")

    emit({"event": "environment", "cpu": platform.processor() or platform.machine(),
          "python": platform.python_version(), "numpy": np.__version__,
          "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
          "warmups": 3, "repeats": args.repeats,
          "gpu_timing_basis": "synchronized execution plus full device reduction and host checksum readback",
          "gpu_gate": "pointwise and per-component reductions, rtol=atol=1e-6; exact integer state",
          "checksum_does_not_replace_h4": True, "sourcehash": _source_hashes()})
    forcings = synthetic_forcing(n_days=args.steps / 48., dt=1800.)
    if len(forcings) != args.steps:
        raise ValueError("合成强迫步数与请求不一致")
    failures = 0
    for n in args.sizes:
        for backend in args.backends:
            base = {"backend": backend, "n": n, "steps": args.steps}
            emit({**base, "event": "start"})
            try:
                if backend == "numpy":
                    runner = _numpy_runner(n, forcings)
                    elapsed, samples = _timed(
                        runner, args.repeats,
                        lambda phase, index, seconds, checksum: emit({
                            **base, "event": phase, "index": index, "seconds": seconds,
                            "checksum": checksum, "status": "validated",
                        }),
                    )
                    details = {"samples_seconds": samples, "checksum_gate_passed": True,
                               "checksum_gate": "finite scalar; exact equality across all warmups and measured repeats"}
                    del runner
                else:
                    elapsed, samples = _audited_gpu_timed(
                        backend, n, forcings, args.repeats,
                        lambda phase, index, payload: emit({**base, "event": phase, "index": index, **payload}),
                    )
                    details = {
                        "samples_seconds": [row["end_to_end_seconds"] for row in samples],
                        "median_compute_wall_seconds": statistics.median(row["synchronized_compute_wall_seconds"] for row in samples),
                        "sample_timings": samples, "checksum_gate_passed": True,
                        "h4_independently_passed": bool(args.cuda_parity_passed) if backend == "cuda" else None,
                    }
                emit({**base, "event": "result", "status": "timing_validated_h4_unverified"
                      if backend == "cuda" and not args.cuda_parity_passed else "measured",
                      "seconds": elapsed, **details,
                      "us_per_column_step": elapsed / (n * args.steps) * 1e6,
                      "column_steps_per_second": n * args.steps / elapsed})
            except Exception as exc:
                failures += 1
                emit({**base, "event": "result", "status": "rejected",
                      "error": f"{type(exc).__name__}: {exc}"})
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
