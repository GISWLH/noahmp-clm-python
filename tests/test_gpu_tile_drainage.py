"""Native tile outlet, independent NumPy reference and real soil trajectory."""
from dataclasses import astuple
import numpy as np
import pytest
from noahmp_clm.physics.tile_drainage import TileDrainageParameters,tile_drainage_demand,step_tile_drainage
from noahmp_clm.physics.soil_enthalpy import SoilEnthalpyState
from noahmp_clm.gpu.tile_drainage import pack_tile_drainage,validate_tile_drainage
from test_gpu_snicar import assert_science_close


def batch(dtype,n=512):
    rng=np.random.default_rng(17016);rows=[]
    for k in range(n):
        dz=np.array([.125,.375,.5,1.]);water=rng.uniform(.02,.44,4)*1000.*dz;ice=np.zeros(4);t=rng.uniform(274.,310.,4)
        if k%5==0:ice=.2*water;water*=.8;t[:]=270.
        p=TileDrainageParameters(rng.uniform(.01,2.),rng.uniform(.1,.35),rng.uniform(0.,50.))
        rows.append(pack_tile_drainage(water,ice,dz,t,rng.uniform(1.,86400.),p,dtype))
    return tuple(np.stack([r[k] for r in rows]) for k in range(3))


def reference(row):
    layers,u,p=[a.astype(float) for a in row]
    demand,d=tile_drainage_demand(layers[:,0],layers[:,1],layers[:,2],layers[:,3],u[0],TileDrainageParameters(*p))
    return dict(demand_kgm2=demand,**d)


def compare_same(ja,cu,cp):
    maximum=0.
    for key,value in ja.items():
        a=np.asarray(value);b=cp.asnumpy(cu[key])
        if a.dtype.kind in 'biu':np.testing.assert_array_equal(a,b)
        else:
            np.testing.assert_allclose(a,b,rtol=1e-6,atol=1e-6,err_msg=key)
            maximum=max(maximum,float(abs(a-b).max()))
    return maximum


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_tile_native_reference(dtype):
    jax=pytest.importorskip('jax')
    if dtype==np.float64 and not jax.config.x64_enabled:pytest.skip('x64 required')
    from noahmp_clm.gpu.tile_drainage import run_tile_drainage_jax
    arrays=batch(dtype);got={k:np.asarray(v) for k,v in run_tile_drainage_jax(*arrays).items()};maximum={}
    for k,row in enumerate(zip(*arrays)):
        expected=reference(row)
        for key,value in got.items():
            if value.dtype.kind in 'biu':np.testing.assert_array_equal(value[k],expected[key]);continue
            maximum[key]=max(maximum.get(key,0.),float(np.max(abs(value[k]-expected[key]))))
            floor,absolute=(10.,1.) if key=='drained_enthalpy_jm2' else (.001,.0001)
            assert_science_close(value[k],expected[key],relative=.1,zero_floor=floor,absolute=absolute)
    print('tile native reference',dtype,maximum,flush=True)


def test_tile_cuda_demand_same_precision():
    cp=pytest.importorskip('cupy');pytest.importorskip('jax')
    from noahmp_clm.gpu.tile_drainage import run_tile_drainage_jax
    from noahmp_clm.gpu.cuda.tile_drainage import run_tile_drainage_cuda
    a=batch(np.float32)
    print('tile demand same precision max',compare_same(run_tile_drainage_jax(*a),run_tile_drainage_cuda(*a),cp),flush=True)


def test_actual_soil_native_drainage_continuous_mass_heat_and_numpy_reference():
    cp=pytest.importorskip('cupy');pytest.importorskip('jax')
    from noahmp_clm.gpu.tile_drainage import step_tile_soil_jax
    from noahmp_clm.gpu.cuda.tile_drainage import step_tile_soil_cuda
    from noahmp_clm.gpu.soil_transport import pack_soil_transport
    from noahmp_clm.column import ColumnState,SoilParams
    states=[SoilEnthalpyState.from_column(ColumnState(tsoil=np.full(4,t),smc=np.full(4,theta)))
        for t,theta in ((290.,.4),(270.,.4),(285.,.2),(300.,.35))]
    packed=[pack_soil_transport(s,1,dtype=np.float32) for s in states]
    c,p,o,_=(np.stack([row[k] for row in packed]) for k in range(4))
    # Rebuild the NumPy reference from the same quantized extensive inputs.
    states=[SoilEnthalpyState(c[k,:,0],c[k,:,1],c[k,:,2],SoilParams(smcmax=float(p[k,0]),
        bexp=float(p[k,1]),smcref=float(p[k,2]),satdk=float(p[k,3])),int(o[k,1])) for k in range(4)]
    params=np.tile(np.asarray(astuple(TileDrainageParameters(.8,.28,10.)),np.float32),(4,1));dt=np.full((4,1),1800.,np.float32)
    a=b=c;initial=c.astype(float);total=np.zeros((4,2));maxstep=np.zeros(2);maxsame=0.;maxscience=np.zeros(3)
    for step in range(480):
        ja=step_tile_soil_jax(a,p,o,dt,params);cu=step_tile_soil_cuda(b,p,o,dt,params)
        maxsame=max(maxsame,compare_same(ja,cu,cp));out=cp.asnumpy(cu['cells']).astype(float);old=cp.asnumpy(b).astype(float)
        flux=np.column_stack((cp.asnumpy(cu['drained_kgm2']),cp.asnumpy(cu['drained_enthalpy_jm2']))).astype(float)
        residual=out[:,:,:2].sum(axis=1)-old[:,:,:2].sum(axis=1)+flux
        bound=64.*np.finfo(np.float32).eps*np.maximum(abs(out[:,:,:2]).sum(axis=1)+abs(old[:,:,:2]).sum(axis=1)+abs(flux),1.)
        assert np.all(abs(residual)<=bound);maxstep=np.maximum(maxstep,abs(residual).max(axis=0));total+=flux
        for k in range(4):
            states[k],d=step_tile_drainage(states[k],1800.,TileDrainageParameters(*params[k].astype(float)))
            expected=np.array([d['drained_kgm2'],d['drained_enthalpy_jm2']])
            assert_science_close(flux[k,0],expected[0],relative=.1,zero_floor=.001,absolute=.0001)
            assert_science_close(flux[k,1],expected[1],relative=.1,zero_floor=10.,absolute=1.)
            np.testing.assert_allclose(out[k,:,0],states[k].water,rtol=0,atol=.01)
            maxscience=np.maximum(maxscience,[abs(flux[k,0]-expected[0]),abs(flux[k,1]-expected[1]),abs(out[k,:,0]-states[k].water).max()])
        a=ja['cells'];b=cu['cells'];assert isinstance(b,cp.ndarray)
    residual=out[:,:,:2].sum(axis=1)-initial[:,:,:2].sum(axis=1)+total
    bound=64.*np.finfo(np.float32).eps*np.maximum(abs(out[:,:,:2]).sum(axis=1)+abs(initial[:,:,:2]).sum(axis=1)+abs(total),1.)
    assert np.all(abs(residual)<=bound)
    print('tile 4x480 two native actual-soil paths same/max step/cumulative/bounds/science',maxsame,maxstep.tolist(),abs(residual).max(axis=0).tolist(),bound.max(axis=0).tolist(),maxscience.tolist(),flush=True)


def test_native_tile_contract():
    a=batch(np.float64);validate_tile_drainage(*a,np)
    b=[v.copy() for v in a];b[2][0,0]=2.1
    with pytest.raises(ValueError):validate_tile_drainage(*b,np)
    b=[v.copy() for v in a];b[0][0,0,0]=-1.
    with pytest.raises(ValueError):validate_tile_drainage(*b,np)
