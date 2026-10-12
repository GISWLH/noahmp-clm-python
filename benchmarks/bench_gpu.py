"""阶段 D：numpy/JAX/CuPy 单柱批量基准。

默认规模与阶段 3 计划一致；用 ``--sizes`` 可先做小规模 smoke。脚本只记录
实际完成的后端，JAX/CuPy 初始化失败会输出 ``unavailable`` 而不会伪造吞吐。
"""

from __future__ import annotations

import argparse
import platform
import statistics
import time
from typing import Callable

import numpy as np

from noahmp_clm.column import ColumnState, Forcing
from noahmp_clm.driver import run_columns
from noahmp_clm.gpu.pack import column_state_to_vec, forcing_to_vec
from noahmp_clm.io import synthetic_forcing


DEFAULT_SIZES = (100, 1000, 10000, 100000, 300000)


def _timed(function: Callable[[], object], repeats: int) -> float:
    """预热三次后返回 measured runs 的中位秒数。"""
    for _ in range(3):
        function()
    values = []
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        function()
        values.append(time.perf_counter() - start)
    return statistics.median(values)


def _numpy_runner(n: int, forcings: list[Forcing]) -> Callable[[], object]:
    states = [ColumnState() for _ in range(n)]
    forcing_lists = [forcings for _ in range(n)]
    return lambda: run_columns(states, forcing_lists)


def _jax_runner(n: int, forcings: list[Forcing]) -> Callable[[], object]:
    import jax
    from noahmp_clm.gpu.jax_core import run_columns_jax

    states = [ColumnState() for _ in range(n)]
    packed = [column_state_to_vec(state) for state in states]
    fvecs = np.asarray([item[0] for item in packed])
    ivecs = np.asarray([item[1] for item in packed])
    forcing_vecs = np.asarray([forcing_to_vec(item) for item in forcings])

    def run() -> object:
        result = run_columns_jax(fvecs, ivecs, forcing_vecs)
        return jax.tree_util.tree_map(lambda value: value.block_until_ready(), result)

    return run


def _cupy_runner(n: int, forcings: list[Forcing]) -> Callable[[], object]:
    import cupy as cp
    from noahmp_clm.gpu.cupy_core import run_columns_cupy

    states = [ColumnState() for _ in range(n)]
    packed = [column_state_to_vec(state) for state in states]
    fvecs = cp.asarray([item[0] for item in packed])
    ivecs = cp.asarray([item[1] for item in packed])
    forcing_vecs = cp.asarray([forcing_to_vec(item) for item in forcings])

    def run() -> object:
        result = run_columns_cupy(fvecs, ivecs, forcing_vecs)
        cp.cuda.Stream.null.synchronize()
        return result

    return run


def _measure(name: str, factory: Callable[[], Callable[[], object]], n: int, steps: int, repeats: int) -> dict:
    try:
        forcings = synthetic_forcing(n_days=steps / 48.0, dt=1800.0)
        elapsed = _timed(factory()(n, forcings), repeats)
    except Exception as exc:  # pragma: no cover - hardware-dependent path
        return {"backend": name, "status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}
    cells = n * steps
    return {
        "backend": name,
        "status": "measured",
        "seconds": elapsed,
        "us_per_column_step": elapsed / cells * 1.0e6,
        "column_steps_per_second": cells / elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=list(DEFAULT_SIZES))
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    print(f"CPU: {platform.processor() or platform.machine()}")
    print(f"Python: {platform.python_version()}")
    for n in args.sizes:
        results = [
            _measure("numpy", lambda: _numpy_runner, n, args.steps, args.repeats),
            _measure("jax", lambda: _jax_runner, n, args.steps, args.repeats),
            _measure("cupy", lambda: _cupy_runner, n, args.steps, args.repeats),
        ]
        print({"n": n, "steps": args.steps, "results": results})


if __name__ == "__main__":
    main()
