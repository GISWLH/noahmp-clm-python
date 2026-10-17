"""Quantized physical inputs, independent NumPy reference and native parity."""
import os
import numpy as np
import pytest
from noahmp_clm.physics.clm_freezing import equilibrium_liquid_water
from noahmp_clm.gpu.clm_freezing import pack_freezing,run_freezing_jax

def cases(dtype):
    rng=np.random.default_rng(20261017);parts=[]
    for sat,b,psi in [(.339,2.79,.069),(.439,5.25,.355),(.465,8.17,.263),(.468,11.55,.468)]:
        total=rng.uniform(0.,sat,2000);temperature=rng.uniform(180.,300.,2000)
        center=273.15/(1.+9.80616*psi/334000.*(sat/(.6*sat))**b)
        total[:8]=[0.,sat,.6*sat,.6*sat,.6*sat,sat,sat,sat]
        temperature[:8]=[260.,273.15,center-.00001,center,center+.00001,273.14,273.149,280.]
        parts.append(pack_freezing(total,temperature,sat,b,psi,dtype))
    return np.concatenate(parts)

def require_jax():
    jax=pytest.importorskip('jax')
    if os.environ.get('NOAHMP_REQUIRE_GPU')=='1':assert jax.default_backend()=='gpu'
    return jax

@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_native_freezing_reference(dtype):
    jax=require_jax()
    if dtype is np.float64 and not jax.config.x64_enabled:pytest.skip('JAX x64 required')
    a=cases(dtype);reference=equilibrium_liquid_water(*a.astype(float).T)
    got={k:np.asarray(v) for k,v in run_freezing_jax(a).items()}
    maximum={k:float(np.max(abs(got[k]-reference[k]))) for k in got}
    # Float32 melting-temperature storage is explicitly quantized, including
    # steep cap derivatives. Scientific component accuracy is10%; same-precision
    # CUDA/JAX retains its separate1e-6 gate below.
    for key in got:np.testing.assert_allclose(got[key],reference[key],rtol=.1,atol=2e-4 if key!='liquid' else 2e-6)
    assert np.all(got['liquid']>=0.) and np.all(got['liquid']<=a[:,0])
    print('explicit-suction freezing',str(dtype),len(a),maximum,flush=True)

def test_native_freezing_same_precision():
    require_jax();cp=pytest.importorskip('cupy')
    from noahmp_clm.gpu.cuda.clm_freezing import run_freezing_cuda
    a=cases(np.float32);cuda=run_freezing_cuda(a);jax=run_freezing_jax(a);maximum={}
    for key in cuda:
        left,right=cp.asnumpy(cuda[key]),np.asarray(jax[key]);maximum[key]=float(np.max(abs(left-right)))
        np.testing.assert_allclose(left,right,rtol=1e-6,atol=1e-6,err_msg=key)
    print('explicit-suction same precision',maximum,flush=True)
