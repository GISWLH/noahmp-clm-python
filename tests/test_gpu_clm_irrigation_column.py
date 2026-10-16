"""Actual device irrigation, water/heat routing and managed vegetation feedback."""
from dataclasses import replace
import numpy as np
import pytest
from test_clm_irrigation_column import case
from test_clm_phenology_column import case as seasonal_case,Phase
from test_gpu_clm_layered_column import reference as physical_reference,budgets,science,host,compare_same_precision
from test_gpu_clm_bgc_column import reference as bgc_reference,cn_budgets as bgc_budgets
from test_gpu_clm_crop_column import cn_budgets as crop_budgets
from test_gpu_snicar import assert_science_close
from noahmp_clm.column import ColumnState
from noahmp_clm.physics.soil_enthalpy import SoilEnthalpyState
from noahmp_clm.physics.irrigation import IrrigationInput,IrrigationState
from noahmp_clm.physics.crop import CropManagement
from noahmp_clm.physics.phenology import PhenologyEnvironment
from noahmp_clm.gpu.clm_irrigation_column import (pack_clm_irrigated_columns,
    pack_irrigation_inputs,next_irrigated_arrays)
from noahmp_clm.gpu.clm_crop_column import pack_crop_management
from noahmp_clm.gpu.clm_phenology_column import pack_phenology_environment


def cases(kind,dtype):
    states=[];forcing=[];requests=[];events=[];environments=[]
    names={'layered':['plain','snow','limited','frozen','rain','saturated'],
        'bgc':['bgc'],'crop':['crop','harvest'],'seasonal':['active','offset']}[kind]
    for name in names:
        if kind=='seasonal':
            s,f,e=seasonal_case(Phase.ACTIVE if name=='active' else Phase.OFFSET)
            ground=SoilEnthalpyState.from_column(ColumnState(tsoil=np.full(4,285.),smc=np.full(4,.18),soil=s.ground.params))
            s=replace(s,ground=ground).with_irrigation(initial=IrrigationState(20.,3600.) if name=='offset' else None)
            request=IrrigationInput(check=name=='active',supply_kgm2=100.,supply_temperature_k=290.)
            environments.append(e)
        else:
            s,f,request=case(name if name not in ('rain','saturated') else 'plain')
            if name=='rain':f=replace(f,prcp=2e-6)
            if name=='saturated':
                # Binary-exact capacity avoids a float32 saturated mass lying
                # above the quantized double-precision reference porosity.
                params=replace(s.ground.params,smcmax=.5)
                ground=SoilEnthalpyState.from_column(ColumnState(tsoil=np.full(4,285.),smc=np.full(4,.5),
                    dz=np.array([.125,.375,.5,1.]),soil=params))
                s=replace(s,ground=ground,irrigation_state=IrrigationState(12.,3600.))
                request=replace(request,check=False)
        s=replace(s,options=replace(s.options,OptRainSnowPartition=5,OptRunoffSurface=6,OptRunoffSubsurface=6))
        states.append(s);forcing.append(list(vars(f).values()));requests.append(request)
        events.append(CropManagement(force_harvest=name=='harvest'))
    arrays=pack_clm_irrigated_columns(states,dtype=dtype)
    inputs,checks=pack_irrigation_inputs(requests,dtype=dtype)
    kwargs={}
    if kind=='crop':kwargs['management']=pack_crop_management(events)
    if kind=='seasonal':kwargs['environment'],kwargs['calendar']=pack_phenology_environment(environments,dtype=dtype)
    return arrays,np.asarray(forcing,dtype),inputs,checks,kwargs


def check_science(arrays,f,inputs,checks,kwargs,got,dtype):
    maximum={};count=len(arrays)-2
    for k,row in enumerate(zip(*arrays)):
        irrigation_data=(*row[-2:],inputs[k],checks[k]);dep=np.zeros(5)
        if count in (16,19):
            crop_data=None if count==16 else (*row[16:19],kwargs['management'][k])
            ref=bgc_reference(row[:16],f[k],dep,np.zeros(2),np.zeros(2,int),0.,1.,
                crop_data=crop_data,irrigation_data=irrigation_data)
        else:
            seasonal={}
            if count==11:
                e=kwargs['environment'][k];cal=kwargs['calendar'][k]
                seasonal=dict(phenology=row[8:11],phenology_environment=PhenologyEnvironment(float(e[0]),bool(cal[0]),bool(cal[1]),float(e[1])))
            ref=physical_reference(row[:8],f[k],dep,irrigation_data=irrigation_data,**seasonal)
        one={key:({a:b[k] for a,b in value.items()} if key=='diagnostics' else value[k]) for key,value in got.items()
            if key in ('snow_cells','soil_cells','options','storage','diagnostics','impurity_output_kgm2','condensate')}
        error=science(one,ref,dtype)
        np.testing.assert_allclose(one['condensate'][:,0],ref['condensate'][:,0],rtol=.1,atol=1e-4)
        np.testing.assert_allclose(one['condensate'][:,1],ref['condensate'][:,1],rtol=.1,atol=1800.)
        for key in ('available_kgm2','deficit_kgm2','requested_kgm2','applied_kgm2','unmet_kgm2','cancelled_kgm2'):
            a=got['irrigation'][key][k];b=ref['irrigation'][key]
            assert_science_close(a,b,relative=.1,zero_floor=.001,absolute=.0001)
            error['irrigation_'+key]=float(abs(a-b))
        for key in ('eligible','triggered'):np.testing.assert_array_equal(got['irrigation'][key][k],ref['irrigation'][key])
        assert_science_close(got['irrigation_state'][k,0],ref['irrigation_state'][0],relative=.1,zero_floor=.001,absolute=.0001)
        np.testing.assert_array_equal(got['irrigation_state'][k,1],ref['irrigation_state'][1])
        for key in ('plant_state','soil_cn_state','crop_state'):
            if key not in ref:continue
            a=got[key][k];b=ref[key];error[key]=float(abs(a-b).max())
            ci=[0,1,2,6] if key=='plant_state' else list(range(7)) if key=='soil_cn_state' else [3]
            ni=[3,4,5,7] if key=='plant_state' else list(range(7,15)) if key=='soil_cn_state' else [4]
            assert_science_close(a[ci],b[ci],relative=.1,zero_floor=.001,absolute=.0001)
            assert_science_close(a[ni],b[ni],relative=.1,zero_floor=.0001,absolute=.00001)
        for key in ('gpp_gc_m2','nee_gc_m2'):
            if key in ref['diagnostics']:
                assert_science_close(got['diagnostics'][key][k],ref['diagnostics'][key],relative=.1,zero_floor=.001,absolute=.0001)
        for key,value in error.items():
            if np.ndim(value)==0:maximum[key]=max(maximum.get(key,0.),value)
    return maximum


def independent_budgets(arrays,f,inputs,got):
    # External supply stays separate from independently partitioned rain/snow.
    # Recompute its enthalpy from measured mass and specified temperature.
    mass=got['irrigation']['applied_kgm2'].astype(float)
    heat=mass*(334000.+4180.*(inputs[:,1].astype(float)-273.15))
    result=budgets(arrays[:8],f,np.zeros((len(f),5)),got,external_input=(mass,heat))
    old=arrays[-2][:,0].astype(float);new=got['irrigation_state'][:,0].astype(float)
    d=got['irrigation'];unmet=d['unmet_kgm2'].astype(float);cancel=d['cancelled_kgm2'].astype(float);request=d['requested_kgm2'].astype(float)
    residual=new-old+mass+unmet+cancel-request
    bound=128.*np.finfo(arrays[0].dtype).eps*np.maximum(old+new+mass+unmet+cancel+request,1.)
    assert np.all(abs(residual)<=bound),(residual,bound)
    result.update(request_kgm2=float(abs(residual).max()),request_bound_max_kgm2=float(bound.max()))
    if len(arrays)==18:result.update(bgc_budgets(arrays[:-2],got,np.zeros(len(f))))
    if len(arrays)==21:result.update(crop_budgets(arrays[:-2],got,np.zeros(len(f))))
    # No irrigation may appear in measured precipitation or canopy interception.
    np.testing.assert_allclose(got['diagnostics']['prcp_mm'],f[:,7].astype(float)*f[:,0]*1000.,rtol=8.*np.finfo(arrays[0].dtype).eps,atol=1e-12)
    return result


@pytest.mark.parametrize('kind',['layered','seasonal','bgc','crop'])
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_irrigation_native_whole_reference_and_conservation(kind,dtype):
    jax=pytest.importorskip('jax')
    if dtype is np.float64 and not jax.config.x64_enabled:pytest.skip('JAX x64 required')
    from noahmp_clm.gpu.clm_irrigation_column import step_clm_irrigated_columns_jax
    arrays,f,inputs,checks,kwargs=cases(kind,dtype)
    for step in range(2):
        got=host(step_clm_irrigated_columns_jax(arrays,f,inputs=inputs,checks=checks,return_surface=True,**kwargs))
        print('irrigation whole reference',kind,str(dtype),step,check_science(arrays,f,inputs,checks,kwargs,got,dtype),flush=True)
        print('irrigation whole budgets',kind,str(dtype),step,independent_budgets(arrays,f,inputs,got),flush=True)
        arrays=next_irrigated_arrays(arrays,got);checks=np.zeros_like(checks)
        if 'management' in kwargs:kwargs['management']=np.zeros_like(kwargs['management'])


@pytest.mark.parametrize('kind',['layered','seasonal','bgc','crop'])
def test_irrigation_native_whole_same_precision_and_device_carry(kind):
    cp=pytest.importorskip('cupy');pytest.importorskip('jax')
    from noahmp_clm.gpu.clm_irrigation_column import step_clm_irrigated_columns_jax
    from noahmp_clm.gpu.cuda.clm_irrigation_column import step_clm_irrigated_columns_cuda
    arrays,f,inputs,checks,kwargs=cases(kind,np.float32);aa=bb=arrays
    for step in range(2):
        a=step_clm_irrigated_columns_cuda(aa,f,inputs=inputs,checks=checks,return_surface=True,**kwargs)
        b=step_clm_irrigated_columns_jax(bb,f,inputs=inputs,checks=checks,return_surface=True,**kwargs)
        ah=host(a,cp.asnumpy);bh=host(b)
        print('irrigation whole same precision',kind,step,compare_same_precision(ah,bh),flush=True)
        before=tuple(cp.asnumpy(v) if isinstance(v,cp.ndarray) else v for v in aa)
        print('irrigation whole CUDA budgets',kind,step,independent_budgets(before,f,inputs,ah),flush=True)
        aa=next_irrigated_arrays(aa,a);bb=next_irrigated_arrays(bb,b);checks=np.zeros_like(checks)
        assert all(isinstance(aa[k],cp.ndarray) for k in (0,1,3,5,7,len(aa)-2))
        if 'management' in kwargs:kwargs['management']=np.zeros_like(kwargs['management'])


def test_irrigation_extended_contract():
    for kind,count in (('layered',10),('seasonal',13),('bgc',18),('crop',21)):
        arrays,f,inputs,checks,kwargs=cases(kind,np.float64)
        assert len(arrays)==count and arrays[-2].shape==(len(f),2) and arrays[-1].shape==(len(f),5)
        assert checks.dtype==np.int32
    a,_,_=case();b,_,_=case('bgc')
    with pytest.raises(ValueError,match='base state ABI'):pack_clm_irrigated_columns([a,b])
    with pytest.raises(ValueError,match='Every irrigated'):pack_clm_irrigated_columns([])
    with pytest.raises(ValueError,match='IrrigationInput'):pack_irrigation_inputs([True])
