"""CUDA driver must own ping-pong buffers even for one contiguous column."""
from __future__ import annotations

import os

import numpy as np
import pytest

try:
    import cupy as cp
except ImportError:
    if os.environ.get("NOAHMP_REQUIRE_GPU") == "1":
        raise
    pytest.skip("CuPy is optional", allow_module_level=True)

from noahmp_clm.column import ColumnState, Forcing
from noahmp_clm.gpu.cuda.loader import run_columns_cuda
from noahmp_clm.gpu.pack import column_state_to_vec, forcing_to_vec


def test_single_column_cuda_does_not_mutate_device_inputs():
    """The second step formerly wrote into the caller's contiguous SoA view."""
    try:
        devices = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError:
        if os.environ.get("NOAHMP_REQUIRE_GPU") == "1":
            raise
        pytest.skip("CUDA device unavailable")
    if devices == 0:
        if os.environ.get("NOAHMP_REQUIRE_GPU") == "1":
            pytest.fail("Actual CUDA device required")
        pytest.skip("CUDA device unavailable")
    state = ColumnState(tsoil=np.full(4, 268.0))
    f0, i0 = column_state_to_vec(state)
    u0 = np.array([forcing_to_vec(Forcing(1800., 268., .002, 2., 1e5,
                                         100., 260., 2e-5))] * 3, dtype=np.float32)
    inputs = (cp.asarray(f0[None], dtype=cp.float32),
              cp.asarray(i0[None], dtype=cp.int32), cp.asarray(u0))
    before = tuple(cp.asnumpy(value) for value in inputs)
    # N=1 is special: the transposed (70,1)/(33,1) views are C contiguous.
    assert inputs[0].T.flags.c_contiguous and inputs[1].T.flags.c_contiguous
    first = tuple(cp.asnumpy(value) for value in run_columns_cuda(*inputs))
    for value, original in zip(inputs, before):
        np.testing.assert_array_equal(cp.asnumpy(value), original)
    assert not np.array_equal(first[0], before[0]), "Workload must update the state"
    assert first[1][0, 32] > before[1][0, 32], "Snowfall must exercise integer-state updates"
    second = tuple(cp.asnumpy(value) for value in run_columns_cuda(*inputs))
    for actual, expected in zip(second, first):
        np.testing.assert_array_equal(actual, expected)
