"""Native methane reference, independent C accounting and carried device pool."""
from dataclasses import replace
import numpy as np
import pytest
from noahmp_clm.physics.wetland import WetlandState,WetlandParameters,step_wetland
from noahmp_clm.gpu.wetland import pack_wetland,validate_wetland
from test_gpu_snicar import assert_science_close


def batch(dtype,n=512):
    rng=np.random.default_rng(51816);rows=[]
    for k in range(n):
        porosity=rng.uniform(.25,.7);theta=porosity*rng.uniform(0.,1.);ice=0.;t=rng.uniform(260.,315.)
        if k%7==0:theta=porosity
        if k%11==0:theta=0.;ice=porosity;t=270.
        p=replace(WetlandParameters(),active_depth_m=rng.uniform(.05,1.),methane_co2_ratio=rng.uniform(0.,1.),
            oxidation_rate_s=0. if k%5==0 else rng.uniform(1e-8,1e-4))
        rows.append(pack_wetland(WetlandState(10.**rng.uniform(-8.,2.)),10.**rng.uniform(-7.,1.),t,theta,ice,porosity,
            rng.uniform(70000.,110000.),rng.uniform(1.,86400.),parameters=p,dtype=dtype))
    return tuple(np.stack([r[k] for r in rows]) for k in range(3))


def reference(row):
    old,u,p=[np.asarray(v,float) for v in row]
    after,d=step_wetland(WetlandState(old[0]),*u,parameters=WetlandParameters(*p))
    return dict(state=np.array([after.methane_gc_m2]),**d)


def budget(arrays,got):
    old=arrays[0][:,0].astype(float);new=got['state'][:,0].astype(float);donor=arrays[1][:,0].astype(float)
    co2=got['co2_gc_m2'].astype(float);ch4=got['methane_emission_gc_m2'].astype(float)
    residual=new-old+co2+ch4-donor
    bound=128.*np.finfo(arrays[0].dtype).eps*np.maximum(new+old+co2+ch4+donor,1.)
    assert np.all(abs(residual)<=bound),(residual,bound)
    mr=new-old+got['oxidized_gc_m2'].astype(float)+ch4-got['methane_production_gc_m2'].astype(float)
    assert np.all(abs(mr)<=bound)
    return dict(carbon=float(abs(residual).max()),methane=float(abs(mr).max()),bound=float(bound.max()))


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_wetland_native_reference_and_independent_budget(dtype):
    jax=pytest.importorskip('jax')
    if dtype is np.float64 and not jax.config.x64_enabled:pytest.skip('JAX x64 required')
    from noahmp_clm.gpu.wetland import run_wetland_jax
    arrays=batch(dtype);got={k:np.asarray(v) for k,v in run_wetland_jax(*arrays).items()};maximum={}
    saturated=arrays[1][:,2]>=arrays[1][:,4]
    np.testing.assert_array_equal(got['anaerobic_fraction'][saturated],1.)
    np.testing.assert_array_equal(got['oxidation_rate_s'][saturated],0.)
    for k,row in enumerate(zip(*arrays)):
        expected=reference(row)
        for name,value in got.items():
            if value.dtype.kind=='b':np.testing.assert_array_equal(value[k],expected[name]);continue
            maximum[name]=max(maximum.get(name,0.),float(np.max(abs(value[k]-expected[name]))))
            if name in ('carbon_residual_gc_m2','methane_residual_gc_m2','carbon_bound_gc_m2'):continue
            floor,absolute=(.1,.01) if name=='methane_emission_mgch4_m2' else (.0001,.00001)
            if name.endswith('_rate_s'):floor,absolute=1e-10,1e-11
            assert_science_close(value[k],expected[name],relative=.1,zero_floor=floor,absolute=absolute)
    print('wetland native',dtype,maximum,'budget',budget(arrays,got),flush=True)


def test_wetland_native_same_precision_and_year_state_carry():
    cp=pytest.importorskip('cupy');pytest.importorskip('jax')
    from noahmp_clm.gpu.wetland import run_wetland_jax
    from noahmp_clm.gpu.cuda.wetland import run_wetland_cuda
    from test_gpu_clm_enthalpy_column import compare_same_precision,host
    arrays=batch(np.float32);a=host(run_wetland_cuda(*arrays),cp.asnumpy);b=host(run_wetland_jax(*arrays))
    print('wetland same precision',compare_same_precision(a,b),'budget',budget(arrays,a),flush=True)
    arrays=list(batch(np.float32,8));arrays[0][:]=.1;initial=arrays[0][:,0].astype(float).copy();aa=bb=arrays[0]
    arrays[1][:,4]=.5;arrays[1][:,5]=101325.;arrays[1][:,6]=86400.;co2=np.zeros(8);ch4=np.zeros(8);donor=np.zeros(8)
    max_residual=0.;max_bound=0.
    for day in range(365):
        arrays[1][:,0]=.1+.05*np.sin(2.*np.pi*day/365.)
        arrays[1][:,1]=285.+15.*np.sin(2.*np.pi*day/365.)
        arrays[1][:,2]=.4+.1*np.sin(np.pi*day/365.);arrays[1][:,3]=0.
        if 250<=day<300:arrays[1][:,2]=0.;arrays[1][:,3]=.5;arrays[1][:,1]=270.
        a=run_wetland_cuda(aa,*arrays[1:]);b=run_wetland_jax(bb,*arrays[1:]);ah=host(a,cp.asnumpy);bh=host(b)
        compare_same_precision(ah,bh)
        co2+=ah['co2_gc_m2'].astype(float);ch4+=ah['methane_emission_gc_m2'].astype(float);donor+=arrays[1][:,0].astype(float)
        residual=ah['state'][:,0].astype(float)-initial+co2+ch4-donor
        bound=128.*np.finfo(np.float32).eps*np.maximum(initial+ah['state'][:,0].astype(float)+co2+ch4+donor,1.)
        assert np.all(abs(residual)<=bound)
        max_residual=max(max_residual,float(abs(residual).max()));max_bound=max(max_bound,float(bound.max()))
        aa=a['state'];bb=b['state'];assert isinstance(aa,cp.ndarray)
    print('wetland 8x365 two native paths cumulative gC/m2',max_residual,'bound',max_bound,flush=True)


def test_wetland_native_contract():
    arrays=batch(np.float64);validate_wetland(*arrays,np)
    bad=[v.copy() for v in arrays];bad[1][0,2]=2.
    with pytest.raises(ValueError):validate_wetland(*bad,np)
    bad=[v.copy() for v in arrays];bad[2][0,6]=0.
    with pytest.raises(ValueError):validate_wetland(*bad,np)
