from itertools import product
import numpy as np
import pytest
from noahmp_clm.physics.canopy_radiation import two_stream_shortwave,canopy_longwave


def cases(dtype):
    # 0.5+0.5 is exactly passive in both float32 and float64. Quantizing
    # 0.6 and 0.4 separately makes their float64 sum exceed one.
    specs = list(product([0.,.1,3.,8.],[.05,.5,1.],[(.1,.05),(.45,.25),(.5,.5)],
                         [0.,.3,.9],[(400.,200.),(0.,0.),(600.,0.),(0.,600.)]))
    leaf = np.array([[lai/10.]*10 for lai,mu,opt,ag,flux in specs],dtype=dtype)
    u = np.array([[*flux,ag,mu,*opt] for lai,mu,opt,ag,flux in specs],dtype=dtype)
    return leaf,u


def random_cases():
    rng = np.random.default_rng(422)
    leaf = rng.uniform(.001,1.,(256,10))
    leaf *= rng.uniform(.1,10.,(256,1))/leaf.sum(axis=1,keepdims=True)
    u = rng.uniform([0.,0.,.05,.05,.05,.01],[900.,900.,.8,1.,.5,.4],(256,6))
    return leaf.astype(np.float32),u.astype(np.float32)


@pytest.mark.parametrize("dtype",[np.float32,np.float64])
def test_jax_two_stream_against_numpy(dtype):
    jax = pytest.importorskip("jax")
    if dtype is np.float64 and not jax.config.x64_enabled:
        pytest.skip("JAX_ENABLE_X64=true required")
    from noahmp_clm.gpu.canopy_radiation import run_shortwave_jax
    leaf,u = cases(dtype)
    actual = run_shortwave_jax(leaf,u)
    expected = [two_stream_shortwave(l.astype(float),float(v[0]),float(v[1]),
        ground_albedo=float(v[2]),cosine_zenith=float(v[3]),leaf_reflectance=float(v[4]),
        leaf_transmittance=float(v[5])) for l,v in zip(leaf,u)]
    for key in actual:
        a = np.asarray(actual[key]);e = np.asarray([r[key] for r in expected])
        print("radiation JAX",dtype,key,"max_abs",np.max(abs(a-e)),flush=True)
        np.testing.assert_allclose(a,e,rtol=.1 if dtype is np.float32 else 1e-8,
                                   atol=.1 if dtype is np.float32 else 1e-8)
    assert np.max(abs(np.asarray(actual["residual"]))) < (.002 if dtype is np.float32 else 1e-9)
    conservative = u[:,4]+u[:,5] == 1.
    assert np.max(abs(np.asarray(actual["absorbed"])[conservative])) < (.001 if dtype is np.float32 else 1e-9)


@pytest.mark.parametrize("randomized",[False,True])
def test_cuda_two_stream_matches_jax_same_precision(randomized):
    cp = pytest.importorskip("cupy")
    pytest.importorskip("jax")
    from noahmp_clm.gpu.canopy_radiation import run_shortwave_jax
    from noahmp_clm.gpu.cuda.canopy_radiation import run_shortwave_cuda
    leaf,u = random_cases() if randomized else cases(np.float32)
    j = run_shortwave_jax(leaf,u)
    c = run_shortwave_cuda(leaf,u)
    for key in j:
        actual,expected=cp.asnumpy(c[key]),np.asarray(j[key])
        print("radiation CUDA/JAX32",key,"max_abs",np.max(abs(actual-expected)),flush=True)
        np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=1e-6)


def longwave_cases(dtype):
    rng=np.random.default_rng(601)
    lai=rng.uniform(0.,1.,(256,10)).astype(dtype)
    temp=rng.uniform(250.,320.,(256,10)).astype(dtype)
    u=rng.uniform([200.,260.,.9],[500.,315.,1.],(256,3)).astype(dtype)
    lai[0]=0.
    temp[1]=290.
    u[1]=[5.670374419e-8*290.**4,290.,.97]
    return lai,temp,u


@pytest.mark.parametrize("dtype",[np.float32,np.float64])
def test_jax_longwave_budget_and_numpy(dtype):
    jax=pytest.importorskip("jax")
    if dtype is np.float64 and not jax.config.x64_enabled:
        pytest.skip("JAX_ENABLE_X64=true required")
    from noahmp_clm.gpu.canopy_radiation import run_longwave_jax
    lai,temp,u=longwave_cases(dtype)
    actual=run_longwave_jax(lai,temp,u)
    expected=[canopy_longwave(l.astype(float),t.astype(float),*map(float,v[:2]),ground_emissivity=float(v[2]))
              for l,t,v in zip(lai,temp,u)]
    for key in actual:
        a,e=np.asarray(actual[key]),np.asarray([x[key] for x in expected])
        print("longwave JAX",dtype,key,"max_abs",np.max(abs(a-e)),flush=True)
        np.testing.assert_allclose(a,e,rtol=.1 if dtype is np.float32 else 1e-9,
            atol=.002 if dtype is np.float32 else 1e-9)
    assert np.max(abs(np.asarray(actual["residual"]))) < (.002 if dtype is np.float32 else 1e-9)


def test_cuda_longwave_matches_jax_same_precision():
    cp=pytest.importorskip("cupy")
    pytest.importorskip("jax")
    from noahmp_clm.gpu.canopy_radiation import run_longwave_jax
    from noahmp_clm.gpu.cuda.canopy_radiation import run_longwave_cuda
    lai,temp,u=longwave_cases(np.float32)
    j=run_longwave_jax(lai,temp,u);c=run_longwave_cuda(lai,temp,u)
    for key in j:
        a,e=cp.asnumpy(c[key]),np.asarray(j[key])
        print("longwave CUDA/JAX32",key,"max_abs",np.max(abs(a-e)),flush=True)
        np.testing.assert_allclose(a,e,rtol=1e-6,atol=1e-6)
