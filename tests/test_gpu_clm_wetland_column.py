"""Native wetland uses funded BGC, actual soil phases and separate CO2/CH4."""
from dataclasses import replace
import numpy as np
import pytest
from test_clm_wetland_column import case
from test_gpu_clm_bgc_column import reference
from test_gpu_clm_layered_column import budgets,science,host,compare_same_precision
from test_gpu_snicar import assert_science_close
from noahmp_clm.gpu.clm_wetland_column import pack_clm_wetland_columns,next_wetland_arrays
from noahmp_clm.gpu.clm_irrigation_column import pack_irrigation_inputs
from noahmp_clm.gpu.clm_crop_column import pack_crop_management
from noahmp_clm.gpu.clm_bgc_column import pack_bgc_environment


def cases(kind,dtype):
    names={'bgc':['wet','dry','frozen','snow','seasonal'],'crop':['crop','harvest'],
        'irrigated':['irrigated'],'bgc_irrigated':['bgc_irrigated']}[kind]
    states=[];forcing=[];controls=[]
    for name in names:
        state,f,kw=case(name)
        state=replace(state,options=replace(state.options,OptRainSnowPartition=5,OptRunoffSurface=6,OptRunoffSubsurface=6))
        states.append(state);forcing.append(list(vars(f).values()));controls.append(kw)
    kwargs={}
    if kind in ('crop','irrigated'):kwargs['management']=pack_crop_management([k.get('crop_management') for k in controls])
    if 'irrigated' in kind:kwargs['inputs'],kwargs['checks']=pack_irrigation_inputs([k['irrigation_input'] for k in controls],dtype=dtype)
    if kind=='bgc':kwargs['environment'],kwargs['calendar']=pack_bgc_environment([k.get('phenology_environment') for k in controls],dtype=dtype)
    return pack_clm_wetland_columns(states,dtype=dtype),np.asarray(forcing,dtype),kwargs


def check_reference(arrays,f,kwargs,got,dtype):
    maximum={};count=len(arrays)-2
    for k,row in enumerate(zip(*arrays)):
        crop_data=(*row[16:19],kwargs['management'][k]) if count in (19,21) else None
        irrigation_data=(*row[count-2:count],kwargs['inputs'][k],kwargs['checks'][k]) if count in (18,21) else None
        env=kwargs['environment'][k] if 'environment' in kwargs else np.zeros(2)
        cal=kwargs['calendar'][k] if 'calendar' in kwargs else np.zeros(2,int)
        ref=reference(row[:16],f[k],np.zeros(5),env,cal,.0001,1.,crop_data=crop_data,
            irrigation_data=irrigation_data,wetland_data=row[-2:])
        one={key:({a:b[k] for a,b in value.items()} if key=='diagnostics' else value[k]) for key,value in got.items()
            if key in ('snow_cells','soil_cells','options','storage','diagnostics','impurity_output_kgm2','condensate')}
        err=science(one,ref,dtype)
        for key in ('plant_state','soil_cn_state','crop_state','wetland_state','irrigation_state'):
            if key not in ref:continue
            a=got[key][k];b=ref[key];err[key]=float(abs(a-b).max())
            if key=='plant_state':ci=[0,1,2,6];ni=[3,4,5,7]
            elif key=='soil_cn_state':ci=list(range(7));ni=list(range(7,15))
            elif key=='crop_state':ci=[3];ni=[4]
            else:ci=list(range(len(a)));ni=[]
            assert_science_close(a[ci],b[ci],relative=.1,zero_floor=.0001,absolute=.00001)
            if ni:assert_science_close(a[ni],b[ni],relative=.1,zero_floor=.0001,absolute=.00001)
        for key in ('methane_production_gc_m2','oxidized_gc_m2','diffused_gc_m2','ebullition_gc_m2','co2_gc_m2','methane_emission_gc_m2'):
            a=got['wetland'][key][k];b=ref['wetland'][key];err[key]=float(abs(a-b))
            assert_science_close(a,b,relative=.1,zero_floor=.0001,absolute=.00001)
        for key in ('liquid_theta','ice_theta','anaerobic_fraction','oxygen_factor'):
            assert_science_close(got['wetland']['environment'][key][k],ref['wetland']['environment'][key],relative=.1,zero_floor=.0001,absolute=.00001)
        np.testing.assert_allclose(got['wetland']['environment']['temperature_k'][k],ref['wetland']['environment']['temperature_k'],rtol=0.,atol=.05)
        for key in ('nee_gc_m2','gpp_gc_m2','total_atmospheric_carbon_gc_m2'):
            assert_science_close(got['diagnostics'][key][k],ref['diagnostics'][key],relative=.1,zero_floor=.001,absolute=.0001)
        np.testing.assert_array_equal(got['phenology_flags'][k],ref['phenology_flags'])
        if crop_data is not None:np.testing.assert_array_equal(got['crop_phase'][k],ref['crop_phase'])
        for key,value in err.items():
            if np.ndim(value)==0:maximum[key]=max(maximum.get(key,0.),value)
    return maximum


def independent_budgets(arrays,f,kwargs,got):
    count=len(arrays)-2;d=got['diagnostics'];external=None
    if 'inputs' in kwargs:
        mass=got['irrigation']['applied_kgm2'].astype(float)
        heat=mass*(334000.+4180.*(kwargs['inputs'][:,1].astype(float)-273.15));external=(mass,heat)
    result=budgets(arrays[:8],f,np.zeros((len(f),5)),got,external_input=external)
    op=arrays[11].astype(float);os=arrays[13].astype(float);p=got['plant_state'].astype(float);s=got['soil_cn_state'].astype(float)
    oldc=op[:,:3].sum(1)+op[:,6]+os[:,:7].sum(1)+arrays[-2][:,0].astype(float)
    newc=p[:,:3].sum(1)+p[:,6]+s[:,:7].sum(1)+got['wetland_state'][:,0].astype(float)
    oldn=op[:,3:6].sum(1)+op[:,7]+os[:,7:].sum(1);newn=p[:,3:6].sum(1)+p[:,7]+s[:,7:].sum(1)
    ec=en=ic=inn=0.
    if count in (19,21):
        oldc+=arrays[16][:,3].astype(float);oldn+=arrays[16][:,4].astype(float)
        newc+=got['crop_state'][:,3].astype(float);newn+=got['crop_state'][:,4].astype(float)
        ec=got['crop']['harvest_gc_m2'].astype(float);en=got['crop']['harvest_gn_m2'].astype(float)
        ic=got['crop']['seed_input_gc_m2'].astype(float);inn=got['crop']['seed_input_gn_m2'].astype(float)
    gpp=d['gpp_gc_m2'].astype(float);ra=d['autotrophic_respiration_gc_m2'].astype(float);co2=d['heterotrophic_respiration_gc_m2'].astype(float);ch4=d['methane_emission_gc_m2'].astype(float)
    cr=newc-oldc-gpp+ra+co2+ch4+ec-ic;nr=newn-oldn-.0001+en-inn
    eps=np.finfo(arrays[0].dtype).eps
    cb=128.*eps*np.maximum(oldc+newc+gpp+ra+co2+ch4+ec+ic,1.);nb=128.*eps*np.maximum(oldn+newn+.0001+en+inn,1.)
    assert np.all(abs(cr)<=cb),(cr,cb)
    assert np.all(abs(nr)<=nb),(nr,nb)
    result.update(carbon=float(abs(cr).max()),nitrogen=float(abs(nr).max()),carbon_bound_max=float(cb.max()),nitrogen_bound_max=float(nb.max()))
    return result


def next_controls(kwargs):
    return {k:np.zeros_like(v) if k in ('management','checks') else v for k,v in kwargs.items()}


@pytest.mark.parametrize('kind',['bgc','crop','irrigated','bgc_irrigated'])
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_wetland_native_whole_reference_and_conservation(kind,dtype):
    jax=pytest.importorskip('jax')
    if dtype is np.float64 and not jax.config.x64_enabled:pytest.skip('JAX x64 required')
    from noahmp_clm.gpu.clm_wetland_column import step_clm_wetland_columns_jax
    arrays,f,kwargs=cases(kind,dtype)
    for step in range(2):
        got=host(step_clm_wetland_columns_jax(arrays,f,return_surface=True,nitrogen_deposition_gn_m2=.0001,**kwargs))
        print('wetland whole reference',kind,dtype,step,check_reference(arrays,f,kwargs,got,dtype),flush=True)
        print('wetland whole budgets',kind,dtype,step,independent_budgets(arrays,f,kwargs,got),flush=True)
        arrays=next_wetland_arrays(arrays,got);kwargs=next_controls(kwargs)


@pytest.mark.parametrize('kind',['bgc','crop','irrigated','bgc_irrigated'])
def test_wetland_native_whole_same_precision_and_device_carry(kind):
    cp=pytest.importorskip('cupy');pytest.importorskip('jax')
    from noahmp_clm.gpu.clm_wetland_column import step_clm_wetland_columns_jax
    from noahmp_clm.gpu.cuda.clm_wetland_column import step_clm_wetland_columns_cuda
    arrays,f,kwargs=cases(kind,np.float32);aa=bb=arrays
    for step in range(2):
        a=step_clm_wetland_columns_cuda(aa,f,return_surface=True,nitrogen_deposition_gn_m2=.0001,**kwargs)
        b=step_clm_wetland_columns_jax(bb,f,return_surface=True,nitrogen_deposition_gn_m2=.0001,**kwargs)
        ah=host(a,cp.asnumpy);bh=host(b)
        print('wetland whole same precision',kind,step,compare_same_precision(ah,bh),flush=True)
        before=tuple(cp.asnumpy(v) if isinstance(v,cp.ndarray) else v for v in aa)
        print('wetland whole CUDA budgets',kind,step,independent_budgets(before,f,kwargs,ah),flush=True)
        aa=next_wetland_arrays(aa,a);bb=next_wetland_arrays(bb,b);kwargs=next_controls(kwargs)
        assert all(isinstance(aa[k],cp.ndarray) for k in (0,1,5,7,11,13,len(aa)-2))


def test_wetland_extended_contract_retains_every_carbon_store():
    for kind,count in (('bgc',18),('crop',21),('irrigated',23),('bgc_irrigated',20)):
        arrays,f,_=cases(kind,np.float64)
        assert len(arrays)==count and arrays[-2].shape==(len(f),1) and arrays[-1].shape==(len(f),13)
    a,_,_=case();b,_,_=case('crop')
    with pytest.raises(ValueError,match='base state ABI'):pack_clm_wetland_columns([a,b])
    with pytest.raises(ValueError,match='Every wetland'):pack_clm_wetland_columns([])
