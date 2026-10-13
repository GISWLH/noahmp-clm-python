"""实际同步的numpy单核/JAX-CUDA x64/x32/手写CUDA基准，逐项输出JSON。"""
from __future__ import annotations

import argparse
import json
import os
import platform
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
    for index in range(3):
        start = time.perf_counter()
        result = function()
        del result
        progress("warmup", index, time.perf_counter() - start)
    values = []
    for index in range(repeats):
        start = time.perf_counter()
        result = function()
        del result
        elapsed = time.perf_counter() - start
        values.append(elapsed)
        progress("measured_run", index, elapsed)
    return statistics.median(values), values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=list(DEFAULT_SIZES))
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--backends", nargs="+", choices=("numpy", "jax64", "jax32", "cuda"),
                        default=["numpy", "jax64", "jax32", "cuda"])
    parser.add_argument("--output", help="持续追加的JSONL原始结果")
    parser.add_argument("--cuda-parity-passed", action="store_true",
                        help="仅在H4已通过时使用；否则CUDA标为measured_unvalidated")
    args = parser.parse_args()
    if args.steps <= 0 or args.repeats <= 0 or any(n <= 0 for n in args.sizes):
        parser.error("sizes/steps/repeats必须为正")

    def emit(row):
        line = json.dumps(row, ensure_ascii=False)
        print(line, flush=True)
        if args.output:
            with open(args.output, "a", encoding="utf-8") as output:
                output.write(line + "\n")

    emit({"event": "environment", "cpu": platform.processor() or platform.machine(),
          "python": platform.python_version(), "numpy": np.__version__,
          "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
          "warmups": 3, "repeats": args.repeats})
    forcings = synthetic_forcing(n_days=args.steps / 48., dt=1800.)
    if len(forcings) != args.steps:
        raise ValueError("合成强迫步数与请求不一致")
    for n in args.sizes:
        for backend in args.backends:
            base = {"backend": backend, "n": n, "steps": args.steps}
            emit({**base, "event": "start"})
            try:
                runner = _numpy_runner(n, forcings) if backend == "numpy" else _gpu_runner(backend, n, forcings)
                elapsed, samples = _timed(
                    runner, args.repeats,
                    lambda phase, index, seconds: emit({**base, "event": phase, "index": index, "seconds": seconds}),
                )
                emit({**base, "event": "result", "status": "measured_unvalidated"
                      if backend == "cuda" and not args.cuda_parity_passed else "measured",
                      "seconds": elapsed, "samples_seconds": samples,
                      "us_per_column_step": elapsed / (n * args.steps) * 1e6,
                      "column_steps_per_second": n * args.steps / elapsed})
                del runner
            except Exception as exc:
                emit({**base, "event": "result", "status": "unavailable",
                      "error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    main()
