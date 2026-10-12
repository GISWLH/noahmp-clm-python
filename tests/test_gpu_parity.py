"""阶段 C parity 门槛。

在 Linux/JAX wheel 上运行时比较 8 柱、48 步；Windows JAX 0.11.2 的 XLA
verifier 会在混合选项的 scan/vmap 图中进程级中止，因此该平台明确跳过，不能
把跳过写成通过。
"""

from __future__ import annotations

from dataclasses import replace
import sys

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="当前 Windows JAX CPU wheel 的混合选项 HLO verifier 中止；请在 xuanze/Linux 运行 parity",
)

from noahmp_clm.column import ColumnState, Forcing, default_soil_params
from noahmp_clm.driver import run_column
from noahmp_clm.gpu.jax_core import _run_one_jax
from noahmp_clm.gpu.pack import column_state_to_vec, forcing_to_vec, vec_to_column_state
from noahmp_clm.options import PhysicsOptions


def _forcing(kind: str, steps: int = 48) -> list[Forcing]:
    result = []
    for index in range(steps):
        if kind == "cold":
            tair, swdown, lwdown, prcp = 268.0, 100.0, 280.0, 1.0e-5 if index < 12 else 0.0
        elif kind == "warm":
            tair, swdown, lwdown, prcp = 295.0, 500.0, 350.0, 0.0
        elif kind == "rain":
            tair, swdown, lwdown, prcp = 285.0, 200.0, 330.0, 2.0e-5 if 10 <= index < 20 else 0.0
        else:
            tair, swdown, lwdown, prcp = 280.0, 300.0, 320.0, 0.0
        result.append(Forcing(1800.0, tair, 0.004, 2.0, 1.0e5, swdown, lwdown, prcp))
    return result


def _cases() -> tuple[list[ColumnState], list[list[Forcing]]]:
    states = []
    forcings = []
    for index, soil_name in enumerate(["sand", "loam", "clay"] * 3):
        options = PhysicsOptions()
        if index % 4 == 1:
            options = replace(options, OptRainSnowPartition=2, OptSnowAlbedo=2)
        elif index % 4 == 2:
            options = replace(options, OptSoilWaterTranspiration=2, OptGroundResistanceEvap=1)
        elif index % 4 == 3:
            options = replace(options, OptSurfaceDrag=2, OptStomataResistance=2)
        soil = default_soil_params(soil_name)
        states.append(ColumnState(soil=soil, lai=1.0 + index % 3, options=options))
        forcings.append(_forcing(["cold", "warm", "rain", "dry"][index % 4]))
    return states, forcings


def test_jax_float64_parity_8_columns_48_steps() -> None:
    states, forcings = _cases()
    keys = ["ts", "rnet", "h", "le", "g", "et_mm", "prcp_mm", "runoff_mm", "melt_mm", "swe", "canw", "residual"]
    for state, forcing in zip(states, forcings):
        fvec, ivec = column_state_to_vec(state)
        forcing_vec = np.asarray([forcing_to_vec(item) for item in forcing])
        final_f, final_i, jax_diag = jax.jit(_run_one_jax)(fvec, ivec, forcing_vec)
        numpy_diag, numpy_state = run_column(state, forcing)
        expected = np.asarray([[item[key] for key in keys] for item in numpy_diag])
        actual = np.asarray(jax_diag)
        # 雪/冻土字段的离散层 mask 与固定三层布局允许 1e-4；其余字段是阶段 C 的 1e-6 门槛。
        np.testing.assert_allclose(actual[:, :7], expected[:, :7], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(actual[:, 7:10], expected[:, 7:10], rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(actual[:, 10:], expected[:, 10:], rtol=1e-6, atol=1e-6)
        assert float(np.max(np.abs(actual[:, 11]))) < 0.5
        restored = vec_to_column_state(final_f, final_i, state.dz)
        assert np.all(np.isfinite(restored.tsoil))
