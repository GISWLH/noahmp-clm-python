"""Complete native winter surface, with independent extensive budget sums."""
import numpy as np
import pytest
from noahmp_clm.column import ColumnState,SoilParams
from noahmp_clm.physics.soil_enthalpy import SoilEnthalpyState,enthalpy_and_capacity
from noahmp_clm.physics.snowpack import Snowpack,TF
from noahmp_clm.physics.canopy_energy import CanopyExchange
from noahmp_clm.physics.canopy_column import surface_step
from noahmp_clm.gpu.canopy_surface import pack_canopy_surface,validate_surface
from test_gpu_snicar import assert_science_close


def cases(dtype=np.float64,random_count=0):
    rows=[]
    for mass,air,humidity,solar in ((0.,295.,.008,700.),(0.,268.,.003,0.),(0.,270.,.0001,200.),
        (20.,260.,.0002,300.),(20.,270.,.004,0.),(20.,280.,.004,800.),(.05,285.,.001,900.),
        (.001,295.,.0001,800.),(0.,295.,.02,0.),(.1,285.,.01,800.)):
        snow=Snowpack.from_ice([mass],min(air,TF)) if mass else Snowpack.empty()
        soil=SoilEnthalpyState.from_column(ColumnState(tsoil=np.full(4,min(air+2.,290.)),smc=np.full(4,.3)))
        e=CanopyExchange(air,humidity,101325.,.02,.004,.03,float(soil.temperature[0]),15.)
        bare=mass==.1
        rows.append(pack_canopy_surface(snow,soil,1800.,[0. if bare else .3]*10,solar,300.,e,
            canopy_water_kgm2=0. if bare else .3,wet_fraction=.3,root_weights=[.6,.25,.1,.05],vegetated_fraction=.8,dtype=dtype))
    rng=np.random.default_rng(71127)
    for k in range(random_count):
        ns=k%6;mass=10.**rng.uniform(-2.,1.5,ns);temperature=rng.uniform(255.,273.15)
        snow=Snowpack(mass,mass*2100.*(temperature-TF),mass/rng.uniform(100.,400.,ns),
            rng.uniform(54.5,1000.,ns),mass[:,None]*10.**rng.uniform(-10.,-6.,(ns,5)))
        soil=SoilEnthalpyState.from_column(ColumnState(tsoil=rng.uniform(260.,295.,4),smc=rng.uniform(.15,.4,4)))
        if k>=6 and k%2:
            h=enthalpy_and_capacity(soil.water,soil.temperature,soil.dz,soil.params,2)[0]
            soil=SoilEnthalpyState(soil.water,h,soil.dz,soil.params,2)
        air=rng.uniform(258.,300.)
        e=CanopyExchange(air,rng.uniform(.0001,.008),90000.,.02,.004,.03,float(soil.temperature[0]),15.)
        rows.append(pack_canopy_surface(snow,soil,1800.,rng.uniform(.05,.5,10),rng.uniform(0.,800.),rng.uniform(220.,360.),e,
            canopy_water_kgm2=.2,wet_fraction=.2,root_weights=[.6,.25,.1,.05],vegetated_fraction=.8,dtype=dtype))
    return tuple(np.stack([row[k] for row in rows]) for k in range(8))


def reference(row):
    c,g,p,o,a,u,f,v=row;n=int(o[0]);c,g,p,a,u,f,v=[np.asarray(x,float) for x in (c,g,p,a,u,f,v)]
    pack=Snowpack(c[:n,0],c[:n,1],c[:n,2],c[:n,3],c[:n,4:])
    props=SoilParams(smcmax=p[0],bexp=p[1],smcref=p[2],satdk=p[3],quartz=p[4],smcwlt=p[5])
    soil=SoilEnthalpyState(g[:,0],g[:,1],g[:,2],props,int(o[1]))
    pack,soil,water,d=surface_step(pack,soil,v[0],a,u[11],u[12],CanopyExchange(*u[:11]),
        canopy_water_kgm2=v[1],canopy_enthalpy_jm2=v[2],root_weights=v[3:],wet_fraction=u[14],
        vegetated_fraction=u[13],transpiration_limit=u[16],ground_emissivity=u[18],ground_albedo=u[19:21],
        diffuse_fraction=u[21],cosine_zenith=u[22],nitrogen_factor=f[:,0],water_stress=f[:,1])
    d['snow_cells']=np.zeros((5,9));d['snow_cells'][:,3]=54.5;n=len(pack.mass)
    d['snow_cells'][:n]=np.column_stack((pack.mass,pack.enthalpy,pack.thickness,pack.radius_um,pack.impurities))
    d['snow_counts']=n;d['soil_cells']=np.column_stack((soil.water,soil.enthalpy,soil.dz))
    d['soil_temperature']=soil.temperature;d['soil_parameters']=props;d['freezing_option']=int(o[1])
    return d


def science(got,ref,dtype):
    maximum={}
    for key in ('leaf_temperature','canopy_temperature','surface_temperature','h','le','ground_heat','gpp',
        'et_kgm2','runoff_kgm2','snow_drainage_kgm2','canopy_water_after_kgm2','canopy_enthalpy_after_jm2'):
        a=np.asarray(got[key]);e=np.asarray(ref[key]);maximum[key]=float(np.max(abs(a-e)))
        if 'temperature' in key:np.testing.assert_allclose(a,e,rtol=0.,atol=.05 if dtype==np.float32 else 1e-5)
        elif 'kgm2' in key:assert_science_close(a,e,relative=.1,zero_floor=.001,absolute=.0001)
        elif 'enthalpy' in key:assert_science_close(a,e,relative=.1,zero_floor=18000.,absolute=1800.)
        else:assert_science_close(a,e,relative=.1,zero_floor=10.,absolute=1.)
    assert int(got['snow_counts'])==ref['snow_counts']
    for key in ('snow_cells','soil_cells'):
        actual=np.asarray(got[key]);expected=ref[key]
        for k in range(actual.shape[1]):
            if k==1:floor,atol=18000.,1800.
            elif k==3:floor,atol=1.,.1
            elif k>=4:floor,atol=1e-10,1e-11
            else:floor,atol=.001,.0001
            assert_science_close(actual[:,k],expected[:,k],relative=.1,zero_floor=floor,absolute=atol)
        maximum[key]=np.max(abs(actual-expected),axis=0).tolist()
    # Temperature is an absolute difference; never divide Kelvin by Kelvin.
    cc=np.asarray(got['soil_cells'],float)
    temp=SoilEnthalpyState(cc[:,0],cc[:,1],cc[:,2],ref['soil_parameters'],ref['freezing_option']).temperature
    np.testing.assert_allclose(temp,ref['soil_temperature'],rtol=0.,atol=.05 if dtype==np.float32 else 1e-5)
    maximum['soil_temperature']=float(np.max(abs(temp-ref['soil_temperature'])))
    return maximum


def budgets(arrays,got):
    c,g,p,o,a,u,f,v=arrays;dt=v[:,0].astype(float)
    snow=np.asarray(got['snow_cells'],float);soil=np.asarray(got['soil_cells'],float)
    old_m=c[:,:,0].astype(float).sum(1)+g[:,:,0].astype(float).sum(1)+v[:,1]
    new_m=snow[:,:,0].sum(1)+soil[:,:,0].sum(1)+np.asarray(got['canopy_water_after_kgm2'],float)
    old_h=c[:,:,1].astype(float).sum(1)+g[:,:,1].astype(float).sum(1)+v[:,2]
    new_h=snow[:,:,1].sum(1)+soil[:,:,1].sum(1)+np.asarray(got['canopy_enthalpy_after_jm2'],float)
    et=np.asarray(got['et_kgm2'],float);runoff=np.asarray(got['runoff_kgm2'],float)
    # Reconstruct atmospheric input from radiation and measured sensible/latent,
    # not the device's cancellation-prone summed budget diagnostic.
    net=(u[:,11].astype(float)+u[:,12]-np.asarray(got['outgoing_shortwave'],float)-np.asarray(got['outgoing_longwave'],float))
    atmosphere=(net-np.asarray(got['h'],float)-np.asarray(got['le'],float))*dt
    adv=np.asarray(got['advected_enthalpy_jm2'],float);runh=np.asarray(got['runoff_enthalpy_jm2'],float)
    water=new_m-old_m+et+runoff;heat=new_h-old_h-atmosphere+adv+runh
    eps=np.finfo(c.dtype).eps
    massbound=32.*eps*np.maximum(abs(old_m)+abs(new_m)+abs(et)+abs(runoff),1.)
    # Component heat convergence + leaf residuals + finite storage arithmetic.
    # This bound is numerical, not the 10% physical comparison criterion.
    storage=(abs(c[:,:,1].astype(float)).sum(1)+abs(g[:,:,1].astype(float)).sum(1)+abs(v[:,2])+
        abs(snow[:,:,1]).sum(1)+abs(soil[:,:,1]).sum(1)+abs(np.asarray(got['canopy_enthalpy_after_jm2'],float)))
    bound=np.asarray(got['thermal_layer_tolerance'],float).sum(1)+dt*np.abs(np.asarray(got['layer_residual'],float)).sum(1)
    bound+=32.*eps*np.maximum(storage+abs(atmosphere)+abs(adv)+abs(runh),1.)+1e-5
    assert np.all(abs(water)<=massbound),(water,massbound)
    assert np.all(abs(heat)<=bound),(heat,bound)
    tracer= snow[:,:,4:].sum(1)-c[:,:,4:].astype(float).sum(1)+np.asarray(got['impurity_output_kgm2'],float)
    tracerbound=64.*eps*np.maximum(c[:,:,4:].astype(float).sum(1),1e-20)
    assert np.all(abs(tracer)<=tracerbound),(tracer,tracerbound)
    return dict(water_kgm2=float(np.max(abs(water))),heat_jm2=float(np.max(abs(heat))),
        impurity_kgm2=float(np.max(abs(tracer))),heat_bound_max_jm2=float(np.max(bound)),
        water_bound_max_kgm2=float(np.max(massbound)))


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_surface_reference_and_budgets(dtype):
    jax=pytest.importorskip('jax')
    if dtype is np.float64 and not jax.config.x64_enabled:pytest.skip('JAX x64 required')
    from noahmp_clm.gpu.canopy_surface import run_canopy_surface_jax
    arrays=cases(dtype,15);got=run_canopy_surface_jax(*arrays)
    for i,row in enumerate(zip(*arrays)):
        ref=reference(row);one={k:np.asarray(val[i]) for k,val in got.items() if not isinstance(val,dict)}
        print('surface',str(dtype),i,science(one,ref,dtype),flush=True)
    print('surface budgets',budgets(arrays,got),flush=True)


def test_surface_same_precision():
    cp=pytest.importorskip('cupy');pytest.importorskip('jax')
    from noahmp_clm.gpu.canopy_surface import run_canopy_surface_jax
    from noahmp_clm.gpu.cuda.canopy_surface import run_canopy_surface_cuda
    arrays=cases(np.float32,15);a=run_canopy_surface_cuda(*arrays);b=run_canopy_surface_jax(*arrays)
    for key in b:
        for field in (b[key] if isinstance(b[key],dict) else [None]):
            aa,bb=(a[key][field],b[key][field]) if field else (a[key],b[key])
            aa,bb=cp.asnumpy(aa),np.asarray(bb)
            print('surface CUDA/JAX32',key,field,np.max(aa!=bb) if aa.dtype.kind in 'ib' else np.max(abs(aa-bb)),flush=True)
            if aa.dtype.kind in 'ib':np.testing.assert_array_equal(aa,bb)
            else:np.testing.assert_allclose(aa,bb,rtol=1e-6,atol=1e-6)
    print('CUDA budgets',budgets(arrays,{k:cp.asnumpy(v) for k,v in a.items() if not isinstance(v,dict)}),flush=True)


def test_surface_contract():
    arrays=cases();validate_surface(arrays,np)
    bad=[v.copy() for v in arrays];bad[7][0,0]=0.
    with pytest.raises(ValueError):validate_surface(bad,np)
    bad=[v.copy() for v in arrays];bad[2][0,5]=1.
    with pytest.raises(ValueError):validate_surface(bad,np)


def test_mixed_snow_plateau_deposition_is_dew():
    arrays=cases(np.float32,6);ref=reference(tuple(x[12] for x in arrays))
    assert ref['ground_evaporation']<0.
    assert ref['thermal_snow_temperature'][0]==TF
    assert ref['surface_temperature']==TF
    assert ref['ground_latent_heat']==2.5e6
    assert abs(ref['energy_residual_jm2'])<.01
