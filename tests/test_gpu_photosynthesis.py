"""Real backend checks; skipped dependencies are not acceptance evidence."""
from itertools import product
import numpy as np
import pytest
from noahmp_clm.physics.photosynthesis import LeafParameters, leaf_gas_exchange
from noahmp_clm.gpu.photosynthesis import LEAF_OUTPUTS


def leaf_cases(dtype):
    return np.asarray(list(product([0., 50., 200., 1000., 2000.],
        [268., 280., 298.15, 310., 320.], [280., 420., 800.],
        [.1, .7, 1.2], [.2, 1., 3.], [0., .2, 1.], [.0, .3, 1.])), dtype=dtype)


def random_leaves():
    rng = np.random.default_rng(711)
    return rng.uniform([0., 260., 200., 0., .1, 0., 0.],
                       [2500., 325., 1200., 1.2, 4., 1., 1.], (2048, 7)).astype(np.float32)


def ordinary(inputs, parameters):
    x = inputs.astype(np.float64)
    r = leaf_gas_exchange(x[:,0], x[:,1], co2_ppm=x[:,2],
        relative_humidity=x[:,3], boundary_conductance=x[:,4],
        nitrogen_factor=x[:,5], water_stress=x[:,6], parameters=parameters)
    return np.stack([r[k] for k in LEAF_OUTPUTS], axis=1)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_jax_leaf_biochemistry_and_diffusion(dtype):
    jax = pytest.importorskip("jax")
    if dtype is np.float64 and not jax.config.x64_enabled:
        pytest.skip("JAX_ENABLE_X64=true required for actual float64 check")
    from noahmp_clm.gpu.photosynthesis import run_leaves_jax
    u = leaf_cases(dtype)
    p = LeafParameters(tpu25=8.)
    expected = ordinary(u, p)
    actual = np.asarray(run_leaves_jax(u, p))
    assert actual.dtype == dtype
    assert np.all(np.isfinite(actual))
    # Leaf near-zero rates use an explicit absolute tolerance. Physiology
    # allows 10%; the tighter float64 check diagnoses equations, not old H4.
    np.testing.assert_allclose(actual[:,:11], expected[:,:11],
        rtol=.1 if dtype is np.float32 else 1e-9, atol=1e-3 if dtype is np.float32 else 1e-9)
    assert np.max(abs(actual[:,11])) < (1e-3 if dtype is np.float32 else 1e-9)
    print("leaf JAX", dtype, "cases", len(u), "max_abs_by_output",
          dict(zip(LEAF_OUTPUTS, np.max(abs(actual-expected),axis=0).tolist())), flush=True)


@pytest.mark.parametrize("randomized", [False, True])
def test_cuda_leaf_matches_jax_same_precision(randomized):
    cp = pytest.importorskip("cupy")
    pytest.importorskip("jax")
    from noahmp_clm.gpu.photosynthesis import run_leaves_jax
    from noahmp_clm.gpu.cuda.photosynthesis import run_leaves_cuda
    u = random_leaves() if randomized else leaf_cases(np.float32)
    p = (LeafParameters(vcmax25=85., jmax25=155., tpu25=6.5,
        respiration25=.7, ball_berry_slope=7., growth_temperature=293.15)
        if randomized else LeafParameters(tpu25=8.))
    actual = cp.asnumpy(run_leaves_cuda(u, p))
    expected = np.asarray(run_leaves_jax(u, p))
    print("leaf CUDA/JAX32 max_abs_by_output",
          dict(zip(LEAF_OUTPUTS, np.max(abs(actual-expected),axis=0).tolist())), flush=True)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(run_leaves_cuda(u[:0]).shape, (0,12))


@pytest.mark.parametrize("dtype",[np.float32,np.float64])
def test_dark_analytic_numpy_against_native_iterative_leaf(dtype):
    jax=pytest.importorskip('jax')
    if dtype is np.float64 and not jax.config.x64_enabled:
        pytest.skip('JAX_ENABLE_X64=true required')
    from noahmp_clm.gpu.photosynthesis import run_leaves_jax
    all_inputs=leaf_cases(dtype);u=all_inputs[all_inputs[:,0]==0.]
    p=LeafParameters(vcmax25=80.,respiration25=1.,minimum_conductance=.002)
    expected=ordinary(u,p);actual=np.asarray(run_leaves_jax(u,p))
    np.testing.assert_allclose(actual,expected,rtol=.1 if dtype is np.float32 else 1e-9,
                               atol=1e-3 if dtype is np.float32 else 1e-9)
    same=None
    if dtype is np.float32:
        cp=pytest.importorskip('cupy')
        from noahmp_clm.gpu.cuda.photosynthesis import run_leaves_cuda
        cuda=cp.asnumpy(run_leaves_cuda(u,p))
        np.testing.assert_allclose(cuda,actual,rtol=1e-6,atol=1e-6)
        same=float(np.max(abs(cuda-actual)))
    print('dark analytic',dtype,'cases',len(u),'same_precision',same,
          'max_abs_by_output',dict(zip(LEAF_OUTPUTS,np.max(abs(actual-expected),axis=0).tolist())),flush=True)
