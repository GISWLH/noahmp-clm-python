"""v1 scientific aggregate bias gate; retired H4 evidence remains in docs/."""

from __future__ import annotations

import shutil
import numpy as np

import pytest

cp = pytest.importorskip("cupy")
from noahmp_clm.gpu.cuda import loader
from noahmp_clm.column import ColumnState, Forcing, default_soil_params
from noahmp_clm.driver import run_column
from noahmp_clm.validation import compare_series
from noahmp_clm.gpu.pack import column_state_to_vec, forcing_to_vec, vec_to_column_state, forcing_vec_to_forcing


def test_cuda_loader_imports_source() -> None:
    assert "step_columns" in loader.KERNEL_SOURCE
    assert "gridDim.x" in loader.KERNEL_SOURCE


@pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc 不可用；需要CUDA环境")
@pytest.mark.parametrize("steps", [1, 48])
def test_cuda_kernel_compiles_and_runs(steps: int) -> None:  # pragma: no cover - CUDA-only
    jax = pytest.importorskip("jax")
    from noahmp_clm.gpu.jax_core import run_columns_jax
    states, forcings = [], []
    for index in range(1024):
        soil = default_soil_params(["sand", "loam", "clay"][index % 3])
        kind = index % 4
        cold = kind == 0
        state = ColumnState(
            soil=soil, lai=1.0 + (index % 13) / 4,
            tsoil=np.full(4, 268.0 if cold else 285.0),
            swe=0.2 if cold else (0.008 if kind == 1 else 0.0),
            snow_t=268.0 if cold else 273.15,
            canw=0.0004 if kind == 2 else 0.0,
        )
        tair, sw, lw, prcp = [(268.,100.,280.,1e-5), (295.,500.,350.,0.),
                               (285.,200.,330.,2e-5), (280.,300.,320.,0.)][kind]
        states.append(state)
        forcings.append([Forcing(1800.,tair,0.004,2.,1e5,sw,lw,prcp if t < 12 else 0.)
                         for t in range(steps)])
    packed = [column_state_to_vec(s) for s in states]
    f = np.asarray([x[0] for x in packed], dtype=np.float32)
    i = np.asarray([x[1] for x in packed], dtype=np.int32)
    u = np.asarray([[forcing_to_vec(x) for x in row] for row in forcings], dtype=np.float32)
    final_f, final_i, diag = loader.run_columns_cuda(f,i,u)
    actual = cp.asnumpy(diag)
    keys = ["ts","rnet","h","le","g","et_mm","prcp_mm","runoff_mm","melt_mm","swe","canw","residual"]
    reference, cache = [], {}
    for index in range(1024):
        # Only identical ordinary-reference inputs share work; all 1024 GPU
        # trajectories execute independently and are compared at every step.
        key = f[index].tobytes() + i[index].tobytes() + u[index].tobytes()
        if key not in cache:
            state = vec_to_column_state(f[index],i[index],states[index].dz)
            ds, _ = run_column(state,[forcing_vec_to_forcing(row) for row in u[index]])
            cache[key] = [[d[k] for k in keys] for d in ds]
        reference.append(cache[key])
    expected = np.asarray(reference)
    jax_result = run_columns_jax(f,i,u)
    jax_diag = np.asarray(jax_result[2])
    print("CUDA parity N=1024, steps=", steps, "JAX devices:", jax_result[2].devices(), flush=True)
    for index, key in enumerate(keys):
        print(key, "CUDA-numpy max_abs", np.max(abs(actual[:,:,index]-expected[:,:,index])),
              "CUDA-JAX32 max_abs", np.max(abs(actual[:,:,index]-jax_diag[:,:,index])), flush=True)
    # Wang retired the old pointwise 1e-4 H4 on 2026-10-13. This test now
    # measures integrated physical bias, including an absolute near-zero rule.
    # Exact CUDA32/JAX32 checks remain a separate mandatory 1e-6 gate.
    for column in range(1024):
        for key, variable, scale in (("h", "h_wm2", 1.), ("le", "le_wm2", 1.),
                ("g", "g_wm2", 1.), ("et_mm", "et_mm_day", 48.),
                ("runoff_mm", "runoff_mm_day", 48.), ("swe", "swe_m", 1.),
                ("canw", "canw_m", 1.)):
            index = keys.index(key)
            metric = compare_series(actual[column,:,index] * scale,
                expected[column,:,index] * scale, variable)
            assert metric["bias_pass"], (column, key, metric)
    np.testing.assert_allclose(actual,jax_diag,rtol=1e-6,atol=1e-6)
    np.testing.assert_allclose(cp.asnumpy(final_f),np.asarray(jax_result[0]),rtol=1e-6,atol=1e-6)
    np.testing.assert_array_equal(cp.asnumpy(final_i),np.asarray(jax_result[1]))
    assert np.all(np.isfinite(actual))
    assert np.max(abs(actual[:,:,11])) < 0.5


@pytest.mark.skipif(shutil.which("nvcc") is None, reason="需要CUDA环境")
def test_cuda_loader_rejects_invalid_abi_and_aliasing() -> None:
    f, i = column_state_to_vec(ColumnState())
    state = cp.asarray(f[:, None], dtype=cp.float32)
    options = cp.asarray(i[:, None], dtype=cp.int32)
    forcing = cp.zeros((8, 1), dtype=cp.float32)
    state_out, options_out = cp.empty_like(state), cp.empty_like(options)
    diag = cp.empty((12, 1), dtype=cp.float32)
    with pytest.raises(ValueError, match="dtype"):
        loader.launch_columns(state.astype(cp.float64), options, forcing, state_out, options_out, diag)
    with pytest.raises(ValueError, match="重叠"):
        loader.launch_columns(state, options, forcing, state, options_out, diag)
    # First option is fixed dynamic vegetation; changing it must not silently
    # run the default kernel under the name of a different option combination.
    options[0] = 1
    with pytest.raises(ValueError, match="默认 PhysicsOptions"):
        loader.launch_columns(state, options, forcing, state_out, options_out, diag)
