"""CuPy 对照后端。

阶段 E 的目标是给 CUDA kernel 一个同 ABI 的正确性锚点。普通版物理目前仍含
可变长度 Python dataclass 和动态分支，因此本文件提供安全的 host-reference
适配器：输入/输出使用 CuPy 定长数组，单柱物理调用普通版纯函数。后续 kernel
可逐模块替换这里的 host round-trip，而不改变测试和内存 ABI。
"""

from __future__ import annotations

import cupy as cp

from ..driver import step_column
from .pack import forcing_vec_to_forcing, vec_to_column_state, column_state_to_vec


def step_column_cupy(
    fvec: cp.ndarray, ivec: cp.ndarray, forcing_vec: cp.ndarray
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    """对应普通版: driver.step_column。

    这是 CuPy 正确性锚点而非性能实现：状态解包和参考物理在 host 上完成，
    再把结果拷回 CuPy。这样不会伪装成 CUDA 加速，也能先锁定 ABI parity。
    """
    state = vec_to_column_state(cp.asnumpy(fvec), cp.asnumpy(ivec), [0.1, 0.3, 0.6, 1.0])
    forcing = forcing_vec_to_forcing(cp.asnumpy(forcing_vec))
    new_state, diag = step_column(state, forcing)
    new_fvec, new_ivec = column_state_to_vec(new_state)
    diag_vec = cp.asarray(
        [
            diag["ts"],
            diag["rnet"],
            diag["h"],
            diag["le"],
            diag["g"],
            diag["et_mm"],
            diag["prcp_mm"],
            diag["runoff_mm"],
            diag["melt_mm"],
            diag["swe"],
            diag["canw"],
            diag["residual"],
        ],
        dtype=cp.asarray(fvec).dtype,
    )
    return cp.asarray(new_fvec, dtype=fvec.dtype), cp.asarray(new_ivec), diag_vec


def run_columns_cupy(
    fvecs: cp.ndarray, ivecs: cp.ndarray, forcings: cp.ndarray
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    """对应普通版: driver.run_columns。

    维持 ``(N, T, 8)`` 或共享 ``(T, 8)`` 强迫接口；循环只用于建立 E 阶段
    对照基线，CUDA kernel 的 grid-stride 循环将在阶段 F 接管。
    """
    fvecs = cp.asarray(fvecs)
    ivecs = cp.asarray(ivecs)
    forcing_array = cp.asarray(forcings)
    if forcing_array.ndim == 2:
        forcing_array = cp.broadcast_to(
            forcing_array,
            (fvecs.shape[0], forcing_array.shape[0], forcing_array.shape[1]),
        )
    diagnostics = []
    current_f, current_i = fvecs, ivecs
    for step in range(forcing_array.shape[1]):
        step_outputs = [
            step_column_cupy(current_f[index], current_i[index], forcing_array[index, step])
            for index in range(fvecs.shape[0])
        ]
        current_f = cp.stack([item[0] for item in step_outputs])
        current_i = cp.stack([item[1] for item in step_outputs])
        diagnostics.append(cp.stack([item[2] for item in step_outputs]))
    return current_f, current_i, cp.stack(diagnostics, axis=1)


__all__ = ["step_column_cupy", "run_columns_cupy"]
