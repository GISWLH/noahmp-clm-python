"""Quantized atmosphere, independent equations and 1e-6 device parity."""
import numpy as np
import pytest
from noahmp_clm.physics.wetbulb import wetbulb_temperature,wetbulb_snow_fraction
from noahmp_clm.gpu.wetbulb import run_wetbulb_jax


def cases(dtype):
    from test_wetbulb import humidity
    rng=np.random.default_rng(90319);tc=rng.uniform(-70.,50.,1000)
    tc[:7]=[-70.,-50.,0.,.01,7.999,8.,8.001]
    p=rng.uniform(40000.,110000.,len(tc));rh=rng.uniform(0.,1.05,len(tc));rh[:2]=[0.,1.]
    q=humidity(tc,rh,p)
    return np.column_stack((tc[q<.2]+273.15,q[q<.2],p[q<.2])).astype(dtype)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_native_wetbulb_reference_and_equation(dtype):
    jax=pytest.importorskip('jax')
    if dtype is np.float64 and not jax.config.x64_enabled:pytest.skip('JAX x64 required')
    a=cases(dtype);got={k:np.asarray(v) for k,v in run_wetbulb_jax(a).items()}
    ref=dict(temperature=wetbulb_temperature(*a.astype(float).T),snow_fraction=wetbulb_snow_fraction(*a.astype(float).T))
    maximum={k:float(abs(got[k]-ref[k]).max()) for k in ref}
    np.testing.assert_allclose(got['temperature'],ref['temperature'],rtol=0,atol=5e-5 if dtype is np.float32 else 1e-10)
    np.testing.assert_allclose(got['snow_fraction'],ref['snow_fraction'],rtol=.1,atol=1e-5 if dtype is np.float32 else 1e-10)
    t,q,p=a.astype(float).T;tw=got['temperature'].astype(float)-273.15
    gamma=1004.*p/(.622*np.where(t>273.15,2500000.,2834000.))
    es=610.8*np.exp(17.27*tw/(237.3+tw))
    residual=es-q*p/(.622+.378*q)+gamma*(tw-(t-273.15))
    # Backward bound: stored Tw half-ULP plus evaluation/root arithmetic.
    derivative=es*17.27*237.3/(237.3+tw)**2+gamma
    bound=16.*np.finfo(dtype).eps*np.maximum(abs(t)*derivative,1.)
    assert np.all(abs(residual)<=bound)
    print('native wetbulb',str(dtype),len(a),maximum,'equation_pa',abs(residual).max(),'bound_pa',bound.max(),flush=True)


def test_native_wetbulb_same_precision():
    cp=pytest.importorskip('cupy');pytest.importorskip('jax')
    from noahmp_clm.gpu.cuda.wetbulb import run_wetbulb_cuda
    a=cases(np.float32);cuda=run_wetbulb_cuda(a);jax=run_wetbulb_jax(a);maximum={}
    for k in cuda:
        left,right=cp.asnumpy(cuda[k]),np.asarray(jax[k])
        maximum[k]=float(abs(left-right).max())
        np.testing.assert_allclose(left,right,rtol=1e-6,atol=1e-6,err_msg=k)
    print('wetbulb same precision',maximum,flush=True)
