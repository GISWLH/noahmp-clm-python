"""Cached EHPC NumPy references; no NumPy forward physics on GPU hosts."""
from pathlib import Path

import numpy as np
import pytest


def data(dtype):
    suffix = '32' if dtype is np.float32 else '64'
    with np.load(Path(__file__).parent/'fixtures'/'clm_water_stress_gpu_reference.npz',
                 allow_pickle=False) as z:
        arrays = tuple(z[k+suffix].copy() for k in ('liquid', 'roots', 'parameters'))
        options = z['options'].copy()
        reference = z['reference'+suffix].copy()
    assert len(options) == 270
    return (*arrays, options), reference


def science(actual, reference):
    assert actual.shape == reference.shape == (270, 5)
    assert np.all(np.isfinite(actual)) and np.all((actual >= 0.) & (actual <= 1.))
    tolerance = np.where(reference <= .01, .001, .1*np.abs(reference))
    assert np.all(abs(actual-reference) <= tolerance)
    print('Actual-soil-head cached NumPy maximum beta difference',
          float(np.max(abs(actual-reference))), flush=True)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_actual_soil_head_jax_cached_numpy(dtype):
    jax = pytest.importorskip('jax')
    if dtype is np.float64 and not jax.config.x64_enabled:
        pytest.skip('JAX x64 required')
    from noahmp_clm.gpu.clm_water_stress import run_water_stress_jax
    arrays, reference = data(dtype)
    result = run_water_stress_jax(*arrays)
    actual = np.column_stack((np.asarray(result['root_factor']), np.asarray(result['layer_factor'])))
    assert actual.dtype == dtype
    science(actual, reference)


def test_actual_soil_head_cuda_cached_numpy_and_same_precision():
    cp = pytest.importorskip('cupy')
    pytest.importorskip('jax')
    from noahmp_clm.gpu.clm_water_stress import run_water_stress_jax
    from noahmp_clm.gpu.cuda.clm_water_stress import run_water_stress_cuda
    arrays, reference = data(np.float32)
    a, b = run_water_stress_cuda(*arrays), run_water_stress_jax(*arrays)
    science(np.column_stack((cp.asnumpy(a['root_factor']), cp.asnumpy(a['layer_factor']))), reference)
    for key in a:
        x, y = cp.asnumpy(a[key]), np.asarray(b[key])
        assert x.dtype == y.dtype == np.dtype('float32')
        print('Actual-soil-head raw same precision', key, float(np.max(abs(x-y))), flush=True)
        np.testing.assert_allclose(x, y, rtol=1e-6, atol=1e-6)
