"""阶段 E：CuPy ABI 对照测试。"""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from noahmp_clm.column import ColumnState, Forcing
from noahmp_clm.driver import step_column
from noahmp_clm.gpu.cupy_core import step_column_cupy
from noahmp_clm.gpu.pack import column_state_to_vec, forcing_to_vec


def test_cupy_reference_matches_numpy_step() -> None:
    state = ColumnState()
    forcing = Forcing(1800.0, 285.0, 0.004, 2.0, 1.0e5, 500.0, 320.0, 0.0)
    expected_state, expected_diag = step_column(state, forcing)
    fvec, ivec = column_state_to_vec(state)
    out_f, out_i, out_diag = step_column_cupy(
        cp.asarray(fvec), cp.asarray(ivec), cp.asarray(forcing_to_vec(forcing))
    )
    expected_f, expected_i = column_state_to_vec(expected_state)
    expected_vector = np.asarray(
        [
            expected_diag["ts"], expected_diag["rnet"], expected_diag["h"],
            expected_diag["le"], expected_diag["g"], expected_diag["et_mm"],
            expected_diag["prcp_mm"], expected_diag["runoff_mm"], expected_diag["melt_mm"],
            expected_diag["swe"], expected_diag["canw"], expected_diag["residual"],
        ]
    )
    np.testing.assert_allclose(cp.asnumpy(out_f), expected_f, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(cp.asnumpy(out_i), expected_i)
    np.testing.assert_allclose(cp.asnumpy(out_diag), expected_vector, rtol=1e-6, atol=1e-6)
