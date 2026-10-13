"""J2 same-precision gate, independent of the original NumPy64 H4 gate."""
from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
jax = pytest.importorskip("jax")

from noahmp_clm.column import ColumnState, Forcing, default_soil_params
from noahmp_clm.gpu.cuda import loader
from noahmp_clm.gpu.jax_core import run_columns_jax
from noahmp_clm.gpu.pack import column_state_to_vec, forcing_to_vec


DIAGNOSTICS = ("ts", "rnet", "h", "le", "g", "et_mm", "prcp_mm",
               "runoff_mm", "melt_mm", "swe", "canw", "residual")


def _report(actual, expected, *, steps, group, names):
    failures = []
    for field, name in enumerate(names):
        a, b = actual[..., field], expected[..., field]
        delta = np.abs(a - b)
        allowed = 1e-6 + 1e-6 * np.abs(b)
        ratio = delta / allowed
        worst = np.unravel_index(np.argmax(ratio), ratio.shape)
        count = int(np.sum(~np.isclose(a, b, rtol=1e-6, atol=1e-6)))
        row = {"group": group, "steps": steps, "field": name,
               "failed": count, "max_abs": float(np.max(delta)),
               "max_tolerance_ratio": float(ratio[worst]),
               "worst_index": [int(value) for value in worst],
               "cuda": float(a[worst]), "jax32": float(b[worst])}
        print(json.dumps(row), flush=True)
        if count:
            failures.append(row)
    return failures


@pytest.mark.skipif(shutil.which("nvcc") is None, reason="GPU toolchain required")
@pytest.mark.parametrize("steps", [1, 48])
def test_cuda_jax_float32_same_precision(steps):
    """Keep every diagnostic, including closure residual, in the strict gate."""
    jax.config.update("jax_enable_x64", True)
    assert jax.default_backend() == "gpu"
    states, integers, forcing = [], [], []
    for index in range(1024):
        kind = index % 4
        state = ColumnState(
            soil=default_soil_params(("sand", "loam", "clay")[index % 3]),
            lai=1.0 + index % 13 / 4,
            tsoil=np.full(4, 268.0 if kind == 0 else 285.0),
            swe=0.2 if kind == 0 else (0.008 if kind == 1 else 0.0),
            snow_t=268.0 if kind == 0 else 273.15,
            canw=0.0004 if kind == 2 else 0.0,
        )
        f, i = column_state_to_vec(state)
        states.append(f)
        integers.append(i)
        tair, sw, lw, rain = ((268., 100., 280., 1e-5), (295., 500., 350., 0.),
                            (285., 200., 330., 2e-5), (280., 300., 320., 0.))[kind]
        forcing.append([forcing_to_vec(Forcing(1800., tair, .004, 2., 1e5, sw, lw,
                                             rain if step < 12 else 0.))
                        for step in range(steps)])
    f = np.asarray(states, np.float32)
    i = np.asarray(integers, np.int32)
    u = np.asarray(forcing, np.float32)
    cuda_result = tuple(cp.asnumpy(value) for value in loader.run_columns_cuda(f, i, u))
    jax_result = tuple(np.asarray(value) for value in run_columns_jax(f, i, u))
    assert cuda_result[0].dtype == jax_result[0].dtype == np.float32
    assert cuda_result[2].dtype == jax_result[2].dtype == np.float32
    failures = _report(cuda_result[2], jax_result[2], steps=steps,
                       group="diagnostics", names=DIAGNOSTICS)
    failures += _report(cuda_result[0], jax_result[0], steps=steps,
                        group="state", names=[f"fvec_{index}" for index in range(70)])
    # Print both complete reports before any failure so NumPy or the first
    # diagnostic cannot hide later component or final-state discrepancies.
    np.testing.assert_array_equal(cuda_result[1], jax_result[1])
    assert not failures, json.dumps(failures, indent=2)
    np.testing.assert_allclose(cuda_result[2], jax_result[2], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(cuda_result[0], jax_result[0], rtol=1e-6, atol=1e-6)
