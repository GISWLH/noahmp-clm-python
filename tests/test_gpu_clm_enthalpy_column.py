"""Native extensive column: complete precipitation, storage and runoff ledger."""
from dataclasses import replace
import numpy as np
import pytest
from noahmp_clm.column import ColumnState,Forcing,SoilParams
from noahmp_clm.clm_state import CLMColumnState,clm_options
from noahmp_clm.clm_enthalpy_driver import step_clm_enthalpy_column
from noahmp_clm.physics.snowpack import Snowpack,TF
from noahmp_clm.physics.soil_enthalpy import SoilEnthalpyState
from noahmp_clm.gpu.clm_enthalpy_column import pack_clm_enthalpy_columns,validate_column,OPTION_NAMES,SITE_NAMES
from test_gpu_snicar import assert_science_close


def cases(dtype=np.float64):
    names=('summer','rain','snow','rain_on_snow','melt','frost','dew','drip','bare','five_layers',
        'uneven_soil','mixed_precip','runoff_3','runoff_4','phase_2','weak_lai')
    states=[];forcing=[]
    for k,name in enumerate(names):
        air,qa,sw,lw,prcp,water=295.,.008,700.,350.,0.,.3
        old=ColumnState(ts=295.,tsoil=np.full(4,290.),smc=np.full(4,.3))
        snow=Snowpack.empty()
        if name.startswith('rain') or name.startswith('runoff'):
            prcp=5e-6;old.smc[:]=.44
        if name in ('snow','frost','phase_2'):
            air,qa,sw,lw=265.,.002,0.,240.;old.ts=265.;old.tsoil[:]=268.
            prcp=2e-6 if name=='snow' else 0.
        if name=='melt':snow=Snowpack.from_ice([.1],TF)
        if name=='rain_on_snow':snow=Snowpack.from_ice([10.,20.],270.,300.)
        if name=='five_layers':
            snow=Snowpack.from_ice([10.,20.,30.,50.,80.],265.,300.)
            air,qa,sw,lw=268.,.001,300.,250.;old.ts=265.;old.tsoil[:]=270.
        if name=='dew':qa=.025;sw=0.;lw=270.
        if name=='drip':water=3.
        if name=='bare':old.fveg=0.;water=2.
        if name=='uneven_soil':old.dz=np.array([.07,.23,.5,1.2]);prcp=1e-6
        if name=='mixed_precip':
            air,qa,sw,lw,prcp=274.15,.004,200.,300.,2e-6
            old.ts=273.;old.tsoil[:]=272.
        if name=='weak_lai':old.lai=.003;water=.001
        old.canw=water/1000.
        options=replace(clm_options(),OptRainSnowPartition=1+k%3,OptGroundResistanceEvap=1+k%4,
            OptSoilWaterTranspiration=1+k%3,OptSurfaceDrag=1+k%2,
            OptRunoffSurface=1+k%4,OptRunoffSubsurface=1+(k+1)%4,OptSoilPermeabilityFrozen=1+k%2,
            OptSoilSupercoolWater=2 if name=='phase_2' else 1)
        if name=='uneven_soil':options=replace(options,OptGroundResistanceEvap=1,OptRunoffSurface=1,OptRunoffSubsurface=2)
        if name=='mixed_precip':options=replace(options,OptRainSnowPartition=1)
        states.append(CLMColumnState.from_column(old,snow=snow,options=options))
        forcing.append([1800.,air,qa,3.,101325.,sw,lw,prcp])
    arrays=pack_clm_enthalpy_columns(states,dtype=dtype)
    dep=np.broadcast_to(np.array([1e-12,2e-11,1e-11,5e-12,2e-12],dtype),(len(states),5)).copy()
    return arrays,np.asarray(forcing,dtype),dep


def state_from_row(row):
    c,g,p,o,site,store,roots=row;n=int(o[0]);c,g,p,site,store,roots=[np.asarray(x,float) for x in (c,g,p,site,store,roots)]
    params=SoilParams(smcmax=p[0],bexp=p[1],smcref=p[2],satdk=p[3],quartz=p[4],smcwlt=p[5])
    ground=SoilEnthalpyState(g[:,0],g[:,1],g[:,2],params,int(o[1]))
    snow=Snowpack(c[:n,0],c[:n,1],c[:n,2],c[:n,3],c[:n,4:])
    options=replace(clm_options(),**{key:int(o[k+1]) for k,key in enumerate(OPTION_NAMES)})
    # Preserve the quantized root inputs; CLMColumnState normally normalizes
    # at initialization, but this is a reconstruction of an existing state.
    state=CLMColumnState(ground,snow,store[0],store[1],**dict(zip(SITE_NAMES[:7],site[:7])),
        surface_temperature=store[2],root_frac=roots,options=options)
    roots=roots.copy();roots.setflags(write=False);object.__setattr__(state,'root_frac',roots)
    return state


def reference(row,forcing,deposition):
    state=state_from_row(row);site=np.asarray(row[4],float)
    final,d=step_clm_enthalpy_column(state,Forcing(*np.asarray(forcing,float)),
        **dict(zip(SITE_NAMES[7:],site[7:])),deposition_rates=np.asarray(deposition,float))
    cells=np.zeros((5,9));cells[:,3]=54.5;n=len(final.snow.mass)
    cells[:n]=np.column_stack((final.snow.mass,final.snow.enthalpy,final.snow.thickness,final.snow.radius_um,final.snow.impurities))
    options=row[3].copy();options[0]=n
    d['leaf_respiration_umol_m2s']=d['leaf_respiration'];d['valid']=1.
    d['thermal_layer_tolerance']=np.zeros(9)
    d['thermal_layer_tolerance'][:len(d['thermal']['layer_residual_jm2'])]=d['thermal']['convergence_tolerance_jm2']
    return dict(snow_cells=cells,soil_cells=np.column_stack((final.ground.water,final.ground.enthalpy,final.ground.dz)),
        options=options,storage=np.array([final.canopy_water_kgm2,final.canopy_enthalpy_jm2,final.surface_temperature]),
        diagnostics=d,impurity_output_kgm2=d['impurity_output_kgm2'],surface=d,
        soil_temperature=final.ground.temperature,soil_parameters=final.ground.params,freezing_option=final.ground.freezing_option)


def soil_temperature(cells,params,phase,dtype):
    """Independent inversion without changing a quantized saturated water mass.

    A float32 saturated mass can exceed the double product of quantized
    porosity and thickness by one ULP. Check that numerical capacity bound;
    keep mass/H unchanged in the thermodynamic equation and budget audit.
    """
    from scipy.optimize import brentq
    from noahmp_clm.physics.soil_enthalpy import enthalpy_and_capacity
    cells=np.asarray(cells,float);capacity=1000.*params.smcmax*cells[:,2]
    assert np.all(cells[:,0]<=capacity+8.*np.finfo(dtype).eps*np.maximum(1.,cells[:,0]))
    return np.array([brentq(lambda t:float(enthalpy_and_capacity(w,t,z,params,phase)[0])-h,
        150.,400.,xtol=2e-12) for w,h,z in cells])


def science(got,ref,dtype):
    maximum={}
    for key,value in got['diagnostics'].items():
        if key in ('energy_residual_jm2','water_residual_kgm2','valid'):continue
        value=np.asarray(value,float);expected=np.asarray(ref['diagnostics'][key],float)
        maximum[key]=float(np.max(abs(value-expected)))
        if 'temperature' in key or key=='ts':
            np.testing.assert_allclose(value,expected,rtol=0.,atol=.05 if dtype==np.float32 else 1e-5)
        elif key.endswith('_jm2'):assert_science_close(value,expected,relative=.1,zero_floor=18000.,absolute=1800.)
        elif key.endswith('_mm'):assert_science_close(value,expected,relative=.1,zero_floor=.001,absolute=.0001)
        elif key in ('swe','canw','canopy_humidity'):assert_science_close(value,expected,relative=.1,zero_floor=1e-6,absolute=1e-7)
        else:assert_science_close(value,expected,relative=.1,zero_floor=10.,absolute=1.)
    np.testing.assert_array_equal(got['options'],ref['options'])
    for key in ('snow_cells','soil_cells'):
        actual=np.asarray(got[key]);expected=ref[key]
        for k in range(actual.shape[1]):
            if k==1:floor,absolute=18000.,1800.
            elif k==3:floor,absolute=1.,.1
            elif k>=4:floor,absolute=1e-10,1e-11
            else:floor,absolute=.001,.0001
            assert_science_close(actual[:,k],expected[:,k],relative=.1,zero_floor=floor,absolute=absolute)
        maximum[key]=np.max(abs(actual-expected),axis=0).tolist()
    cells=np.asarray(got['soil_cells'],float)
    temperature=soil_temperature(cells,ref['soil_parameters'],ref['freezing_option'],dtype)
    np.testing.assert_allclose(temperature,ref['soil_temperature'],rtol=0.,atol=.05 if dtype==np.float32 else 1e-5)
    maximum['soil_temperature']=float(np.max(abs(temperature-ref['soil_temperature'])))
    assert_science_close(got['impurity_output_kgm2'],ref['impurity_output_kgm2'],relative=.1,zero_floor=1e-10,absolute=1e-11)
    return maximum


def budgets(arrays,forcing,deposition,got):
    from noahmp_clm.physics.precip import partition_precipitation
    c,g,p,o,site,store,roots=arrays;dt=forcing[:,0].astype(float);d=got['diagnostics'];sf=got['surface']
    snow=np.asarray(got['snow_cells'],float);soil=np.asarray(got['soil_cells'],float);storage=np.asarray(got['storage'],float)
    oldm=c[:,:,0].astype(float).sum(1)+g[:,:,0].astype(float).sum(1)+store[:,0]
    newm=snow[:,:,0].sum(1)+soil[:,:,0].sum(1)+storage[:,0]
    oldh=c[:,:,1].astype(float).sum(1)+g[:,:,1].astype(float).sum(1)+store[:,1]
    newh=snow[:,:,1].sum(1)+soil[:,:,1].sum(1)+storage[:,1]
    precipitation=[];ph=[]
    for f,opt in zip(forcing.astype(float),o):
        rain,ice=partition_precipitation(f[7],f[0],f[1],int(opt[2]),qair=f[2],pressure=f[4]);rain*=1000.;ice*=1000.
        precipitation.append(rain+ice);ph.append(rain*(334000.+4180.*(f[1]-TF))+ice*2100.*(min(f[1],TF)-TF))
    precipitation=np.array(precipitation);ph=np.array(ph)
    atmo=(forcing[:,5].astype(float)+forcing[:,6]-np.asarray(sf['outgoing_shortwave'],float)-
        np.asarray(sf['outgoing_longwave'],float)-np.asarray(d['h'],float)-np.asarray(d['le'],float))*dt
    et=np.asarray(d['et_mm'],float);runoff=np.asarray(d['runoff_mm'],float)
    adv=np.asarray(d['advected_enthalpy_jm2'],float);runh=np.asarray(d['runoff_enthalpy_jm2'],float)
    water=newm-oldm-precipitation+et+runoff;heat=newh-oldh-ph-atmo+adv+runh
    eps=np.finfo(c.dtype).eps
    massbound=32.*eps*np.maximum(abs(oldm)+abs(newm)+abs(precipitation)+abs(et)+abs(runoff),1.)
    magnitudes=(abs(c[:,:,1].astype(float)).sum(1)+abs(g[:,:,1].astype(float)).sum(1)+abs(store[:,1])+
        abs(snow[:,:,1]).sum(1)+abs(soil[:,:,1]).sum(1)+abs(storage[:,1]))
    bound=np.asarray(sf['thermal_layer_tolerance'],float).sum(1)+dt*np.abs(np.asarray(sf['layer_residual'],float)).sum(1)
    bound+=32.*eps*np.maximum(magnitudes+abs(ph)+abs(atmo)+abs(adv)+abs(runh),1.)+1e-5
    assert np.all(abs(water)<=massbound),(water,massbound)
    assert np.all(abs(heat)<=bound),(heat,bound)
    tracer=snow[:,:,4:].sum(1)-c[:,:,4:].astype(float).sum(1)-deposition.astype(float)*dt[:,None]+np.asarray(got['impurity_output_kgm2'],float)
    tracerbound=64.*eps*np.maximum(c[:,:,4:].astype(float).sum(1)+deposition.astype(float)*dt[:,None],1e-20)
    assert np.all(abs(tracer)<=tracerbound),(tracer,tracerbound)
    return dict(water_kgm2=float(np.max(abs(water))),heat_jm2=float(np.max(abs(heat))),impurity_kgm2=float(np.max(abs(tracer))),
        water_bound_max_kgm2=float(np.max(massbound)),heat_bound_max_jm2=float(np.max(bound)))


def host(value,convert=np.asarray):
    return {k:host(v,convert) if isinstance(v,dict) else convert(v) for k,v in value.items()}


def compare_same_precision(left,right,prefix=''):
    maximum=0.
    for key in right:
        name=f'{prefix}/{key}'
        if isinstance(right[key],dict):maximum=max(maximum,compare_same_precision(left[key],right[key],name));continue
        a,b=left[key],right[key]
        if b.dtype.kind in 'ib':np.testing.assert_array_equal(a,b,err_msg=name)
        else:
            error=float(np.max(abs(a-b),initial=0.));maximum=max(maximum,error)
            np.testing.assert_allclose(a,b,rtol=1e-6,atol=1e-6,err_msg=name)
    return maximum


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_full_column_reference_and_conservation(dtype):
    jax=pytest.importorskip('jax')
    if dtype is np.float64 and not jax.config.x64_enabled:pytest.skip('JAX x64 required')
    from noahmp_clm.gpu.clm_enthalpy_column import step_clm_enthalpy_columns_jax
    arrays,f,dep=cases(dtype);got=host(step_clm_enthalpy_columns_jax(*arrays,f,deposition=dep,return_surface=True))
    for k,row in enumerate(zip(*arrays)):
        one={key:({a:b[k] for a,b in v.items()} if isinstance(v,dict) else v[k]) for key,v in got.items() if key!='surface'}
        print('extensive column',str(dtype),k,science(one,reference(row,f[k],dep[k]),dtype),flush=True)
    print('extensive budgets',budgets(arrays,f,dep,got),flush=True)


def test_full_column_same_precision():
    cp=pytest.importorskip('cupy');pytest.importorskip('jax')
    from noahmp_clm.gpu.clm_enthalpy_column import step_clm_enthalpy_columns_jax
    from noahmp_clm.gpu.cuda.clm_enthalpy_column import step_clm_enthalpy_columns_cuda
    arrays,f,dep=cases(np.float32)
    a=host(step_clm_enthalpy_columns_cuda(*arrays,f,deposition=dep,return_surface=True),cp.asnumpy)
    b=host(step_clm_enthalpy_columns_jax(*arrays,f,deposition=dep,return_surface=True))
    print('extensive CUDA/JAX32',compare_same_precision(a,b),flush=True)
    print('extensive CUDA budgets',budgets(arrays,f,dep,a),flush=True)


def test_full_column_contract():
    arrays,f,dep=cases();validate_column(arrays,f,dep,np)
    accepted=[x.copy() for x in arrays];accepted[3][0,2]=5
    validate_column(accepted,f,dep,np) # Wet-bulb option 5 is now implemented.
    bad=[x.copy() for x in arrays];bad[3][0,2]=4
    with pytest.raises(NotImplementedError):validate_column(bad,f,dep,np)
    bad=[x.copy() for x in arrays];bad[5][0,0]=-1.
    with pytest.raises(ValueError):validate_column(bad,f,dep,np)
    with pytest.raises(ValueError):validate_column(arrays,f,-dep,np)
    state=CLMColumnState.from_column(ColumnState())
    state=replace(state,options=replace(state.options,OptSoilTemperatureBottom=2))
    with pytest.raises(NotImplementedError):pack_clm_enthalpy_columns([state])
