"""Device condensate state, CLM phase choice and independent parcel budgets."""
import numpy as np
import pytest
from noahmp_clm.physics.canopy_interception import CanopyCondensate,intercept_precipitation,exchange_vapor
from noahmp_clm.gpu.canopy_interception import pack_canopy_interception,pack_canopy_vapor,validate_canopy_water,LAYER_FIELDS,LEDGER
from test_gpu_snicar import assert_science_close


def cases(dtype=np.float64,random_count=40):
    rows=[]
    for fv in (0.,.2,.8,1.):
        for air,rain,snow in ((295.,8.,0.),(265.,0.,8.),(273.15,.2,.3),(280.,1.,1.)):
            state=CanopyCondensate.from_temperature(np.linspace(0.,2.,10),np.linspace(260.,290.,10))
            rows.append(pack_canopy_interception(state,rain,snow,air,1800.,np.full(10,.3),stem_area=.05,vegetated_fraction=fv,wind=3.,dtype=dtype))
    for leaf in (260.,273.15,285.):
        for amount in (-.1,0.,.1,5.):
            state=CanopyCondensate.from_temperature(np.linspace(0.,2.,10),np.full(10,leaf))
            rows.append(pack_canopy_vapor(state,amount,leaf,dtype=dtype))
    rng=np.random.default_rng(67081)
    for k in range(random_count):
        state=CanopyCondensate(rng.uniform(.01,3.,10),rng.uniform(0.,1.,10)) if k%3==0 else CanopyCondensate.from_temperature(rng.uniform(0.,3.,10),rng.uniform(250.,300.,10))
        if k%2:
            rows.append(pack_canopy_vapor(state,rng.uniform(-.1,4.,10),rng.uniform(250.,300.,10),dtype=dtype))
        else:
            rows.append(pack_canopy_interception(state,*rng.uniform(0.,20.,2),rng.uniform(250.,300.),rng.uniform(60.,3600.),
                rng.uniform(0.,1.,10),vegetated_fraction=rng.uniform(0.,1.),wind=rng.uniform(0.,20.),dtype=dtype))
    return tuple(np.stack([row[k] for row in rows]) for k in range(5))


def reference(row):
    c,a,u,v,mode=row;c,a,u,v=[np.asarray(x,float) for x in (c,a,u,v)]
    state=CanopyCondensate(c[:,0],c[:,1]);result={key:np.zeros(10) for key in LAYER_FIELDS};ledger={key:0. for key in LEDGER}
    if mode==0:
        new,d=intercept_precipitation(state,*u[:4],a[:,0],stem_area=a[:,1],vegetated_fraction=u[4],wind=u[5],unloading_threshold=u[6])
        for key in result:
            if key in d:result[key]=d[key]
        for key in ledger:
            if key in d:ledger[key]=d[key]
    else:
        new,d=exchange_vapor(state,v[:,0],leaf_temperature=v[:,1])
        result.update(d)
        ledger['water_residual_kgm2']=float(new.mass.sum()-state.mass.sum()+d['mass_kgm2'].sum())
        ledger['energy_residual_jm2']=float(new.enthalpy.sum()-state.enthalpy.sum()+d['advected_enthalpy_jm2'].sum())
    ledger['valid']=1.;result.update(cells=np.column_stack((new.mass,new.enthalpy)),temperature=new.temperature,ice=new.ice,liquid=new.liquid,ledger=ledger)
    return result


def science(got,ref,dtype):
    maximum={}
    for key in got:
        if key=='ledger':
            maximum[key]={}
            for name,actual in got[key].items():
                if name in ('valid','water_residual_kgm2','energy_residual_jm2'):continue
                value=np.asarray(actual,float);expected=np.asarray(ref[key][name],float);error=float(np.max(abs(value-expected)))
                maximum[key][name]=error;heat='enthalpy' in name
                assert_science_close(value,expected,relative=.1,zero_floor=100. if heat else .001,absolute=10. if heat else 1e-4)
            continue
        value=np.asarray(got[key],float);expected=np.asarray(ref[key],float);maximum[key]=np.max(abs(value-expected),axis=0).tolist()
        if key=='temperature':np.testing.assert_allclose(value,expected,rtol=0.,atol=.001 if dtype==np.float32 else 1e-9)
        elif key=='cells':
            assert_science_close(value[:,0],expected[:,0],relative=.1,zero_floor=.001,absolute=1e-4)
            assert_science_close(value[:,1],expected[:,1],relative=.1,zero_floor=100.,absolute=10.)
        elif 'enthalpy' in key or 'energy' in key:assert_science_close(value,expected,relative=.1,zero_floor=100.,absolute=10.)
        elif 'fraction' in key:np.testing.assert_allclose(value,expected,rtol=.1,atol=.001)
        else:assert_science_close(value,expected,relative=.1,zero_floor=.001,absolute=1e-4)
    return maximum


def budgets(arrays,result):
    c,a,u,v,mode=arrays;old=c.astype(float);new=np.asarray(result['cells'],float);d=result['ledger']
    vapor=mode==1
    rain=u[:,0].astype(float);snow=u[:,1].astype(float);air=u[:,2].astype(float)
    incoming_mass=np.where(vapor,0.,rain+snow)
    incoming_h=np.where(vapor,0.,rain*(334000.+4180.*(air-273.15))+snow*2100.*(np.minimum(air,273.15)-273.15))
    out_mass=np.where(vapor,np.asarray(result['mass_kgm2'],float).sum(1),np.asarray(d['ground_liquid_kgm2'],float)+np.asarray(d['ground_ice_kgm2'],float))
    out_heat=np.where(vapor,np.asarray(result['advected_enthalpy_jm2'],float).sum(1),np.asarray(d['ground_liquid_enthalpy_jm2'],float)+np.asarray(d['ground_ice_enthalpy_jm2'],float))
    error=np.stack((new[:,:,0].sum(1)-old[:,:,0].sum(1)-incoming_mass+out_mass,
        new[:,:,1].sum(1)-old[:,:,1].sum(1)-incoming_h+out_heat),axis=1)
    magnitude=np.stack((abs(old[:,:,0]).sum(1)+abs(new[:,:,0]).sum(1)+abs(incoming_mass)+abs(out_mass),
        abs(old[:,:,1]).sum(1)+abs(new[:,:,1]).sum(1)+abs(incoming_h)+abs(out_heat)),axis=1)
    bounds=64.*np.finfo(c.dtype).eps*np.maximum(magnitude,1.)+np.array([1e-12,1e-6])
    assert np.all(abs(error)<=bounds),(error,bounds)
    return dict(max_residual=np.max(abs(error),axis=0).tolist(),max_numerical_bound=np.max(bounds,axis=0).tolist())


def host(value,convert=np.asarray):return {k:host(v,convert) if isinstance(v,dict) else convert(v) for k,v in value.items()}


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_canopy_water_reference(dtype):
    jax=pytest.importorskip('jax')
    if dtype==np.float64 and not jax.config.x64_enabled:pytest.skip('JAX x64 required')
    from noahmp_clm.gpu.canopy_interception import run_canopy_water_jax
    a=cases(dtype);result=host(run_canopy_water_jax(*a));maximum={}
    for k,row in enumerate(zip(*a)):
        one={key:({x:y[k] for x,y in val.items()} if isinstance(val,dict) else val[k]) for key,val in result.items()}
        science(one,reference(row),dtype)
    print('canopy water reference',dtype,budgets(a,result),flush=True)


def test_canopy_water_same_precision():
    cp=pytest.importorskip('cupy');pytest.importorskip('jax')
    from noahmp_clm.gpu.canopy_interception import run_canopy_water_jax
    from noahmp_clm.gpu.cuda.canopy_interception import run_canopy_water_cuda
    arrays=cases(np.float32);a=host(run_canopy_water_cuda(*arrays),cp.asnumpy);b=host(run_canopy_water_jax(*arrays));max_error=0.
    for key in b:
        for field in b[key] if isinstance(b[key],dict) else [None]:
            aa,bb=(a[key][field],b[key][field]) if field else (a[key],b[key])
            np.testing.assert_allclose(aa,bb,rtol=1e-6,atol=1e-6,err_msg=f'{key}/{field}')
            max_error=max(max_error,float(np.max(abs(aa-bb))))
    print('canopy water same precision',max_error,budgets(arrays,a),flush=True)


def test_canopy_water_contract():
    arrays=cases();validate_canopy_water(*arrays,np)
    bad=[x.copy() for x in arrays];bad[0][0,0]=[0.,1.]
    with pytest.raises(ValueError):validate_canopy_water(*bad,np)
    bad=[x.copy() for x in arrays];bad[2][0,3]=0.
    with pytest.raises(ValueError):validate_canopy_water(*bad,np)
