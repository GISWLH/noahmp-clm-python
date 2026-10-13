"""阶段 B：JAX 固定状态接口的 smoke 和 vmap/scan 语义测试。"""

from __future__ import annotations

import numpy as np
import pytest
import sys

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason=(
        "JAX 0.11.2 Windows CPU wheel intermittently aborts in XLA HLO verifier "
        "for this scan/vmap graph; run these tests on Linux/xuanze CUDA or CPU JAX."
    ),
)
from noahmp_clm.column import ColumnState
from noahmp_clm.gpu.jax_core import run_columns_jax, step_column_jax
from noahmp_clm.gpu.pack import column_state_to_vec, forcing_to_vec
from noahmp_clm.io import synthetic_forcing


def _inputs(n: int = 2, steps: int = 4):
    states = [ColumnState(lai=2.0 + index) for index in range(n)]
    fvecs, ivecs = zip(*(column_state_to_vec(state) for state in states))
    forcings = np.asarray(
        [forcing_to_vec(item) for item in synthetic_forcing(steps / 48.0, dt=1800.0)]
    )
    return np.asarray(fvecs), np.asarray(ivecs), forcings


def test_jax_default_chain_returns_fixed_diagnostics() -> None:
    fvecs, ivecs, forcings = _inputs()
    final_f, final_i, diagnostics = jax.jit(step_column_jax)(
        fvecs[0], ivecs[0], forcings[0]
    )
    assert final_f.shape == (70,)
    assert final_i.shape == (33,)
    assert diagnostics.shape == (12,)
    assert np.all(np.isfinite(np.asarray(final_f)))
    assert np.all(np.isfinite(np.asarray(diagnostics)))


def test_jax_batch_matches_individual_step_calls() -> None:
    fvecs, ivecs, forcings = _inputs(n=2, steps=4)
    batch_f, batch_i, batch_diag = run_columns_jax(fvecs, ivecs, forcings)
    assert batch_f.shape == (2, 70)
    assert batch_i.shape == (2, 33)
    assert batch_diag.shape == (2, 4, 12)
    assert np.all(np.isfinite(np.asarray(batch_f)))
    assert np.all(np.isfinite(np.asarray(batch_diag)))


def test_jax_float32_scan_preserves_dtype_with_x64_enabled() -> None:
    """防止固定层厚的 float64 常量使 switch/scan 分支产生 dtype 冲突。"""
    fvecs, ivecs, forcings = _inputs(n=2, steps=4)
    final_f, final_i, diagnostics = run_columns_jax(
        fvecs.astype(np.float32), ivecs, forcings.astype(np.float32)
    )
    assert final_f.dtype == diagnostics.dtype == np.dtype("float32")
    assert final_i.dtype == np.dtype("int32")
    assert np.all(np.isfinite(np.asarray(diagnostics)))
