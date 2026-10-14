"""MOST native backends against NumPy and unchanged 1e-6 same-precision gate."""
from dataclasses import astuple
import numpy as np
import pytest
from noahmp_clm.physics.canopy_most import MOSTGeometry,canopy_exchange,flux_stability
from noahmp_clm.gpu.canopy_most import run_canopy_most_jax,MOST_OUTPUTS


def cases(dtype):
    rng=np.random.default_rng(71129);geometries=[];atmosphere=[];iteration=[]
    for k,zeta in enumerate(np.r_[[-100.,-10.,-1.574,-.465,-.01,0.,.01,1.,2.],rng.uniform(-100.,2.,91)]):
        d=float(rng.uniform(0.,15.));heights=d+rng.uniform(5.,40.,3);rough=rng.uniform(.001,1.,3)
        g=MOSTGeometry(*heights,d,*rough,1000.,.01)
        geometries.append(astuple(g))
        atmosphere.append([rng.uniform(250.,315.),rng.uniform(.0001,.025),rng.uniform(70000.,102000.),
            rng.uniform(0.,10.),rng.uniform(-100.,500.),rng.uniform(-1e-5,2e-4),rng.uniform(.05,1.)])
        iteration.append([zeta,rng.uniform(1.,10.),rng.uniform(.01,.1),rng.uniform(0.,8.)])
    return tuple(np.asarray(v,dtype) for v in (geometries,atmosphere,iteration))


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_most_gpu_reference(dtype):
    jax=pytest.importorskip('jax')
    if dtype is np.float64 and not jax.config.x64_enabled:pytest.skip('JAX x64 required')
    arrays=cases(dtype);got={k:np.asarray(v) for k,v in run_canopy_most_jax(*arrays).items()}
    reference=[]
    for g,a,s in zip(*arrays):
        g=MOSTGeometry(*g.astype(float));a=a.astype(float);s=s.astype(float)
        result=flux_stability(*a,g)
        result.update(canopy_exchange(s[0],s[1],g,leaf_dimension=s[2],plant_area=s[3]))
        reference.append([result[k] for k in MOST_OUTPUTS])
    reference=np.asarray(reference);maximum={}
    for k,name in enumerate(MOST_OUTPUTS):
        actual=got[name];expected=reference[:,k]
        np.testing.assert_allclose(actual,expected,rtol=1e-4 if dtype is np.float32 else 1e-11,atol=1e-6 if dtype is np.float32 else 1e-11)
        maximum[name]=float(abs(actual-expected).max())
    print('MOST native vs NumPy',str(dtype),maximum,flush=True)


def test_most_native_same_precision():
    cp=pytest.importorskip('cupy');pytest.importorskip('jax')
    from noahmp_clm.gpu.cuda.canopy_most import run_canopy_most_cuda
    arrays=cases(np.float32);a=run_canopy_most_cuda(*arrays);b=run_canopy_most_jax(*arrays);maximum={}
    for name in MOST_OUTPUTS:
        left,right=cp.asnumpy(a[name]),np.asarray(b[name]);maximum[name]=float(abs(left-right).max())
        np.testing.assert_allclose(left,right,rtol=1e-6,atol=1e-6,err_msg=name)
    print('MOST same precision',maximum,flush=True)
