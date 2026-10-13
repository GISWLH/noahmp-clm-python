"""12个物理字段/35值的 float32 CUDA SoA loader；数值验收见 GPU.md。"""

from __future__ import annotations

from pathlib import Path
from functools import lru_cache
import numpy as np

from ...options import PhysicsOptions
from ..pack import OPTION_FIELDS

try:  # loader 导入本身不应强制安装 CUDA runtime。
    import cupy as cp
    _CUPY_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - 依赖缺失环境
    cp = None
    _CUPY_IMPORT_ERROR = exc


KERNEL_SOURCE = Path(__file__).with_name("column_kernel.cu").read_text(encoding="utf-8")

# Only implemented physics is advertised. Other ABI selectors retain their
# default values; accepting them would not implement their named physics.
PHYSICS_OPTION_VALUES = {
    "OptRainSnowPartition": (1, 2, 3),
    "OptSoilWaterTranspiration": (1, 2, 3),
    "OptGroundResistanceEvap": (1, 2, 3, 4),
    "OptSurfaceDrag": (1, 2),
    "OptStomataResistance": (1, 2),
    "OptSnowAlbedo": (1, 2),
    "OptSnowThermConduct": (1, 2, 3, 4, 5),
    "OptSoilSupercoolWater": (1, 2),
    "OptRunoffSurface": (1, 2, 3, 4),
    "OptRunoffSubsurface": (1, 2, 3, 4),
    "OptSoilPermeabilityFrozen": (1, 2),
    "OptSnowCompaction": (1, 2),
}
_DEFAULTS = PhysicsOptions()
_OPTION_RANGES = tuple(PHYSICS_OPTION_VALUES.get(name, (getattr(_DEFAULTS, name),))
                       for name in OPTION_FIELDS)
_OPTION_MIN = np.asarray([values[0] for values in _OPTION_RANGES], dtype=np.int32)
_OPTION_MAX = np.asarray([values[-1] for values in _OPTION_RANGES], dtype=np.int32)


def _require_cupy() -> None:
    """对应普通版: driver.run_columns。"""
    if _CUPY_IMPORT_ERROR is not None:
        raise ImportError("CUDA loader 需要 CuPy") from _CUPY_IMPORT_ERROR


@lru_cache(maxsize=1)
def load_column_kernel() -> cp.RawKernel:
    """对应普通版: driver.step_column。

    延迟编译 CUDA 源码，避免普通安装在没有 CUDA Toolkit 时导入失败。
    """
    _require_cupy()
    module = cp.RawModule(
        code=KERNEL_SOURCE,
        # J2 probes match XLA GPU's separate multiply/add and approximate
        # divide lowering; keep the same arithmetic in NVRTC and nvcc builds.
        options=("--std=c++14", "--fmad=false", "--prec-div=false"),
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
    if state_soa.ndim != 2:
        raise ValueError("CUDA loader 需要二维 SoA 数组")
    n_columns = int(state_soa.shape[1])
    specs = ((state_soa, 70, cp.float32), (options_soa, 33, cp.int32),
             (forcing_soa, 8, cp.float32), (state_out_soa, 70, cp.float32),
             (options_out_soa, 33, cp.int32), (diag_soa, 12, cp.float32))
    for array, fields, dtype in specs:
        if array.shape != (fields, n_columns) or array.dtype != dtype or not array.flags.c_contiguous:
            raise ValueError("CUDA SoA 形状/dtype/连续性必须满足 STATE_SPEC（float32/int32）")
        if array.device.id != state_soa.device.id:
            raise ValueError("所有 CUDA 数组必须位于同一设备")
    ranges = [(a.data.ptr, a.data.ptr + a.nbytes) for a, _, _ in specs]
    for index, (begin, end) in enumerate(ranges):
        if any(begin < other_end and other_begin < end for other_begin, other_end in ranges[:index]):
            raise ValueError("__restrict__ 输入/输出不得重叠")
    _check_options(options_soa)
    _launch_unchecked(state_soa, options_soa, forcing_soa, state_out_soa, options_out_soa, diag_soa)


def _check_options(options_soa) -> None:
    """对应普通版: options.PhysicsOptions。Validate every column before launch."""
    # All advertised ranges are contiguous. One device reduction checks all
    # selectors/columns, avoiding a host synchronization for each option field.
    low, high = cp.asarray(_OPTION_MIN), cp.asarray(_OPTION_MAX)
    valid = (options_soa[:32] >= low[:, None]) & (options_soa[:32] <= high[:, None])
    if not bool(cp.all(valid)):
        field, column = np.argwhere(~cp.asnumpy(valid))[0]
        name = OPTION_FIELDS[field]
        value = int(options_soa[field, column].item())
        scope = "已实现物理值" if name in PHYSICS_OPTION_VALUES else "默认 PhysicsOptions ABI保留值"
        raise ValueError(f"CUDA {name}={value} (column={column}) 尚不支持；"
                         f"{scope}为 {_OPTION_RANGES[field]}")
    if not bool(cp.all((options_soa[32] >= 1) & (options_soa[32] <= 3))):
        raise ValueError("雪层数必须为 1--3")


def _launch_unchecked(state, options, forcing, state_out, options_out, diagnostics) -> None:
    """对应普通版: driver.step_column。由已校验的批量驱动复用。"""
    n_columns = state.shape[1]
    if n_columns == 0:
        return
    blocks = min((n_columns + 255) // 256, 4096)
    kernel = load_column_kernel()
    kernel(
        (blocks,), (256,),
        (state, options, forcing, state_out, options_out, diagnostics, np.int32(n_columns)),
    )


def run_columns_cuda(fvecs, ivecs, forcings):
    """对应普通版: driver.run_columns。返回AoS最终状态和(N,T,12)诊断。

    物理在GPU执行；输入可为numpy/CuPy数组，强迫可共享(T,8)或独立(N,T,8)。
    选项只在入口校验一次，时间步复用两个互不重叠的SoA缓冲区。
    """
    _require_cupy()
    f = cp.asarray(fvecs, dtype=cp.float32)
    i = cp.asarray(ivecs, dtype=cp.int32)
    u = cp.asarray(forcings, dtype=cp.float32)
    if f.ndim != 2 or f.shape[1] != 70 or i.shape != (f.shape[0], 33):
        raise ValueError("批量状态形状必须是(N,70)/(N,33)")
    n = f.shape[0]
    if u.ndim == 2 and u.shape[1] == 8:
        u = cp.broadcast_to(u, (n, *u.shape))
    if u.ndim != 3 or u.shape[0] != n or u.shape[2] != 8:
        raise ValueError("强迫形状必须是(T,8)或(N,T,8)")
    # Own both ping-pong buffers even for N=1, where a transpose can already
    # be contiguous and ascontiguousarray would alias the caller's input.
    state, options = f.T.copy(order="C"), i.T.copy(order="C")
    _check_options(options)
    forcing_soa = cp.ascontiguousarray(u.transpose(1, 2, 0))
    state_out, options_out = cp.empty_like(state), cp.empty_like(options)
    diag = cp.empty((u.shape[1], 12, n), dtype=cp.float32)
    for index in range(u.shape[1]):
        _launch_unchecked(state, options, forcing_soa[index], state_out, options_out, diag[index])
        state, state_out = state_out, state
        options, options_out = options_out, options
    return state.T, options.T, diag.transpose(2, 0, 1)


__all__ = ["KERNEL_SOURCE", "load_column_kernel", "launch_columns", "run_columns_cuda"]
