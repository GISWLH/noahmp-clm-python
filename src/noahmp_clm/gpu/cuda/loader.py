"""CUDA RawModule loader；编译/运行结果需在带 nvcc 的主机验证。"""

from __future__ import annotations

from pathlib import Path

try:  # loader 导入本身不应强制安装 CUDA runtime。
    import cupy as cp
    _CUPY_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - 依赖缺失环境
    cp = None
    _CUPY_IMPORT_ERROR = exc


KERNEL_SOURCE = Path(__file__).with_name("column_kernel.cu").read_text(encoding="utf-8")


def _require_cupy() -> None:
    """对应普通版: driver.run_columns。"""
    if _CUPY_IMPORT_ERROR is not None:
        raise ImportError("CUDA loader 需要 CuPy") from _CUPY_IMPORT_ERROR


def load_column_kernel() -> cp.RawKernel:
    """对应普通版: driver.step_column。

    延迟编译 CUDA 源码，避免普通安装在没有 CUDA Toolkit 时导入失败。
    """
    _require_cupy()
    module = cp.RawModule(
        code=KERNEL_SOURCE,
        options=("--std=c++14", "--use_fast_math"),
        name_expressions=("step_columns",),
    )
    return module.get_function("step_columns")


def launch_columns(
    state_soa: cp.ndarray,
    options_soa: cp.ndarray,
    forcing_soa: cp.ndarray,
    state_out_soa: cp.ndarray,
    options_out_soa: cp.ndarray,
    diag_soa: cp.ndarray,
) -> None:
    """对应普通版: driver.run_columns。

    1 thread 对应 1 column，block 固定 256，grid-stride loop 处理任意 N。
    """
    _require_cupy()
    if state_soa.ndim != 2 or options_soa.ndim != 2 or forcing_soa.ndim != 2:
        raise ValueError("CUDA loader 需要二维 SoA 数组")
    n_columns = int(state_soa.shape[1])
    blocks = (n_columns + 255) // 256
    kernel = load_column_kernel()
    kernel(
        (blocks,), (256,),
        (state_soa, options_soa, forcing_soa, state_out_soa, options_out_soa, diag_soa, n_columns),
    )


__all__ = ["KERNEL_SOURCE", "load_column_kernel", "launch_columns"]
