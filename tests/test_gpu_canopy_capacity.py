"""Native capacity and all retained memory values against EHPC references."""
from pathlib import Path
import os
import numpy as np
import pytest


def reference(precision):
    with np.load(Path(__file__).parent / 'fixtures/canopy_capacity_reference.npz', allow_pickle=False) as data:
        prefix = precision + '/'
        return {k[len(prefix):]: v for k,v in data.items() if k.startswith(prefix)}


def host(values):
    return {k: v.get() if hasattr(v, 'get') else np.asarray(v) for k,v in values.items()}


@pytest.mark.parametrize('precision', ['32', '64'])
def test_capacity_native_against_ehpc_quantized_reference(precision):
    jax = pytest.importorskip('jax')
    if precision == '64' and not jax.config.x64_enabled: pytest.skip('JAX x64 required')
    from noahmp_clm.gpu.canopy_capacity import run_capacity_jax, run_temperature_memory_jax
    case = reference(precision)
    got = host(run_capacity_jax(*(case[k] for k in ['leaf','environment','traits','template'])))
    differences = {}
    for key, value in got.items():
        expected = case['expected/' + key]
        # Exact trait/profile formulas have a smaller numerical error than the
        # 10% science threshold; absolute tolerance covers zero-area/zero rates.
        np.testing.assert_allclose(value, expected, rtol=.1, atol=1e-5)
        differences[key] = float(np.max(abs(value.astype(float)-expected)))
    np.testing.assert_allclose(got['leaf_parameters'][:, :, 8], case['expected/leaf_parameters'][:, :, 8], rtol=0., atol=.001)
    for k in [4,5,6,7,9]:
        np.testing.assert_array_equal(got['leaf_parameters'][:, :, k], np.broadcast_to(case['template'][:, None, k], (32,10)))
    memory = host(run_temperature_memory_jax(*(case[k] for k in ['memory_values','memory_controls','memory_forcing'])))
    np.testing.assert_array_equal(memory['values'], case['expected_memory_values'])
    np.testing.assert_array_equal(memory['controls'], case['expected_memory_controls'])
    np.testing.assert_allclose(memory['means'], case['expected_memory_means'], rtol=0., atol=.001)
    if os.environ.get('NOAHMP_CAPACITY_DIAGNOSTICS'):
        folder=Path(os.environ['NOAHMP_CAPACITY_DIAGNOSTICS']);folder.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(folder/('reference'+precision+'.npz'), **got, **{'memory/'+k:v for k,v in memory.items()})
    print('capacity native reference', precision, dict(maximum_absolute_differences=differences,
          memory_temperature_error_K=float(np.max(abs(memory['means']-case['expected_memory_means'])))), flush=True)


def test_capacity_cuda_jax_every_raw_field_and_memory_sample_same_precision():
    pytest.importorskip('jax'); cp = pytest.importorskip('cupy')
    from noahmp_clm.gpu.canopy_capacity import run_capacity_jax, run_temperature_memory_jax
    from noahmp_clm.gpu.cuda.canopy_capacity import run_capacity_cuda, run_temperature_memory_cuda
    case = reference('32'); inputs = [case[k] for k in ['leaf','environment','traits','template']]
    actual = host(run_capacity_cuda(*inputs)); expected = host(run_capacity_jax(*inputs))
    maximum = {}
    for key in expected:
        np.testing.assert_allclose(actual[key], expected[key], rtol=1e-6, atol=1e-6)
        maximum[key] = float(np.max(abs(actual[key].astype(float)-expected[key].astype(float))))
    inputs = [case[k] for k in ['memory_values','memory_controls','memory_forcing']]
    a, b = host(run_temperature_memory_cuda(*inputs)), host(run_temperature_memory_jax(*inputs))
    for key in ['values','means']:
        np.testing.assert_allclose(a[key], b[key], rtol=1e-6, atol=1e-6)
        maximum['memory/' + key] = float(np.max(abs(a[key].astype(float)-b[key].astype(float))))
    np.testing.assert_array_equal(a['controls'], b['controls'])
    np.testing.assert_array_equal(a['values'], case['expected_memory_values'])
    if os.environ.get('NOAHMP_CAPACITY_DIAGNOSTICS'):
        folder=Path(os.environ['NOAHMP_CAPACITY_DIAGNOSTICS']);folder.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(folder/'allraw32.npz', **{'cuda/'+k:v for k,v in actual.items()},
                            **{'jax/'+k:v for k,v in expected.items()},
                            **{'cuda/memory/'+k:v for k,v in a.items()}, **{'jax/memory/'+k:v for k,v in b.items()})
    cp.cuda.runtime.deviceSynchronize()
    print('all raw capacity and temperature memory same precision', maximum, flush=True)
