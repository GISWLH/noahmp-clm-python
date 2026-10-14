"""All seasonal phases, trigger boundaries and native backend equivalence."""
import numpy as np
import pytest
from noahmp_clm.physics.phenology import PhenologyState,PhenologyParameters,step_phenology
from noahmp_clm.gpu.phenology import pack_phenology,validate_phenology


def cases(dtype):
    rows=[];rng=np.random.default_rng(95042)
    for i in range(512):
        phase=i%4;remaining=(rng.uniform(1800.,86400.*(30. if phase==1 else 15.)) if phase in (1,3) else 0.)
        fraction=0. if phase==0 else 1. if phase==2 else 1.-remaining/(30.*86400.) if phase==1 else remaining/(15.*86400.)
        state=PhenologyState(rng.uniform(0.,450.) if phase==0 else 0.,fraction,remaining,phase,bool(phase==0 and i%3))
        rows.append(pack_phenology(state,rng.uniform(260.,300.),rng.uniform(30000.,50000.),bool(i%3),bool(i%9==0),
            (60.,1800.,86400.)[i%3],dtype=dtype))
    return tuple(np.stack([r[k] for r in rows]) for k in range(5))


def reference(row):
    s,o,u,c,p=row
    old=PhenologyState(*s.astype(float),int(o[0]),bool(o[1]))
    after,d=step_phenology(old,float(u[0]),float(u[1]),bool(c[0]),bool(c[1]),float(u[2]),PhenologyParameters(*p.astype(float)))
    return dict(state=np.array([after.gdd,after.leaf_fraction,after.remaining_seconds]),
        flags=np.array([after.phase,after.armed],np.int32),**d)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_native_phenology_reference(dtype):
    jax=pytest.importorskip('jax')
    if dtype is np.float64 and not jax.config.x64_enabled:pytest.skip('JAX x64 required')
    from noahmp_clm.gpu.phenology import run_phenology_jax
    arrays=cases(dtype);result={k:np.asarray(v) for k,v in run_phenology_jax(*arrays).items()};maximum={}
    for i,row in enumerate(zip(*arrays)):
        ref=reference(row)
        for key,value in ref.items():
            got=result[key][i]
            if key in ('flags','onset_triggered','offset_triggered'):np.testing.assert_array_equal(got,value)
            else:
                if key=='state':
                    maximum[key]=np.maximum(maximum.get(key,np.zeros(3)),abs(got-value)).tolist()
                    for k,bound in enumerate((.0001,1e-6,.2)):
                        np.testing.assert_allclose(got[k],value[k],rtol=0.,atol=bound if dtype is np.float32 else 1e-9)
                else:
                    error=np.max(abs(got-value));maximum[key]=max(maximum.get(key,0.),float(error))
                    np.testing.assert_allclose(got,value,rtol=.1 if dtype is np.float32 else 1e-10,atol=1e-5 if dtype is np.float32 else 1e-10)
    print('Phenology native',str(dtype),maximum,flush=True)


def test_native_phenology_same_precision():
    cp=pytest.importorskip('cupy');pytest.importorskip('jax')
    from noahmp_clm.gpu.phenology import run_phenology_jax
    from noahmp_clm.gpu.cuda.phenology import run_phenology_cuda
    arrays=cases(np.float32);a=run_phenology_cuda(*arrays);b=run_phenology_jax(*arrays)
    for key in a:
        x,y=cp.asnumpy(a[key]),np.asarray(b[key])
        print('Phenology same precision',key,np.max(x!=y) if x.dtype.kind in 'biu' else np.max(abs(x-y)),flush=True)
        if x.dtype.kind in 'biu':np.testing.assert_array_equal(x,y)
        else:np.testing.assert_allclose(x,y,rtol=1e-6,atol=1e-6)


def test_phenology_host_contract_and_reference():
    arrays=cases(np.float64);validate_phenology(*arrays,np)
    for row in zip(*arrays):reference(row)
    bad=[v.copy() for v in arrays];bad[3][0,0]=2
    with pytest.raises(ValueError):validate_phenology(*bad,np)
