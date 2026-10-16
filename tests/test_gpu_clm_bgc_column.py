"""Complete device BGC, real canopy feedback, state carry and four budgets."""
from dataclasses import replace
import numpy as np
import pytest
from test_clm_bgc_column import case,Phase
from test_gpu_clm_layered_column import budgets,science,host,compare_same_precision
from test_gpu_clm_enthalpy_column import state_from_row,SITE_NAMES
from test_gpu_plant_cn import parameters as plant_parameters
from test_gpu_snicar import assert_science_close
from noahmp_clm.column import Forcing
from noahmp_clm.physics.canopy_interception import CanopyCondensate
from noahmp_clm.physics.plant_cn import PlantState
from noahmp_clm.physics.decomposition import DecompositionState,DecompositionParameters
from noahmp_clm.physics.biogeochemistry import BGCState,BGCParameters
from noahmp_clm.physics.phenology import PhenologyState,PhenologyParameters,PhenologyEnvironment
from noahmp_clm.clm_enthalpy_driver import step_clm_enthalpy_column
from noahmp_clm.gpu.clm_bgc_column import (pack_clm_bgc_columns,pack_bgc_environment,validate_bgc_column,next_bgc_arrays)


def cases(dtype):
    states=[];forcings=[];environments=[]
    spec=[(None,285.,False),(Phase.ACTIVE,285.,False),(Phase.ACTIVE,285.,True),
        (Phase.ONSET,285.,False),(Phase.OFFSET,270.,False),(Phase.DORMANT,270.,True),(None,285.,True),(None,285.,False)]
    for k,config in enumerate(spec):
        state,f,e=case(*config)
        state=replace(state,options=replace(state.options,OptRainSnowPartition=5,OptRunoffSurface=6,OptRunoffSubsurface=6))
        if k==6:f=replace(f,swdown=0.,tair=278.,lwdown=280.)
        if k==7:state=replace(state,fveg=0.,bgc_state=replace(state.bgc_state,plant=PlantState(np.zeros(3),np.zeros(3),20.,.1)))
        states.append(state);forcings.append(list(vars(f).values()));environments.append(e)
    env,cal=pack_bgc_environment(environments,dtype=dtype)
    return pack_clm_bgc_columns(states,dtype=dtype),np.asarray(forcings,dtype),np.zeros((8,5),dtype),env,cal


def reference(row,forcing,deposition,env,cal,nitrogen,oxygen,*,crop_data=None,irrigation_data=None,wetland_data=None,tile_parameters=None):
    state=state_from_row(row[:7]);cc=np.asarray(row[7],float)
    state=replace(state,canopy_condensate=CanopyCondensate(cc[:,0],cc[:,1]))
    ps,flags,pp,plant,plant_p,soil,soil_p,bp=row[8:]
    environment=None
    if flags[2]==5:
        state=state.with_phenology(PhenologyParameters(*np.asarray(pp,float)),
            PhenologyState(*np.asarray(ps,float),int(flags[0]),bool(flags[1])))
        environment=PhenologyEnvironment(float(env[0]),bool(cal[0]),bool(cal[1]),float(env[1]))
    plant=np.asarray(plant,float);soil=np.asarray(soil,float);soil_p=np.asarray(soil_p,float)
    plant=PlantState(plant[:3],plant[3:6],plant[6],plant[7]);soil=DecompositionState(soil[:7],soil[7:14],soil[14])
    parameters=BGCParameters(plant_parameters(plant_p),DecompositionParameters(tuple(soil_p[:7]),tuple(soil_p[7:10]),*soil_p[10:]),*map(float,bp))
    state=state.with_biogeochemistry(BGCState(plant,soil),parameters)
    crop_management=None
    if crop_data is not None:
        from noahmp_clm.physics.crop import CropState,CropParameters,CropManagement
        cs,phase,cp,events=crop_data;cs=np.asarray(cs,float)
        state=state.with_crop(CropParameters(*np.asarray(cp,float)),CropState(*cs[:3],int(phase[0]),*cs[3:]))
        crop_management=CropManagement(bool(events[0]),bool(events[1]))
    irrigation_input=None
    if irrigation_data is not None:
        from test_gpu_clm_layered_column import irrigated_reference_state
        state,irrigation_input=irrigated_reference_state(state,irrigation_data)
    if wetland_data is not None:
        from noahmp_clm.physics.wetland import WetlandState,WetlandParameters
        ws,wp=wetland_data
        state=state.with_wetland(WetlandParameters(*np.asarray(wp,float)),WetlandState(float(ws[0])))
    if tile_parameters is not None:
        from noahmp_clm.physics.tile_drainage import TileDrainageParameters
        state=state.with_tile_drainage(TileDrainageParameters(*np.asarray(tile_parameters,float)))
    # CLMColumnState normalizes the quantized root weights on reconstruction;
    # no reference host tendency is supplied to the native implementation.
    site=np.asarray(row[4],float);kwargs=dict(zip(SITE_NAMES[7:-1],site[7:-1]))
    final,d=step_clm_enthalpy_column(state,Forcing(*np.asarray(forcing,float)),**kwargs,deposition_rates=np.asarray(deposition,float),
        phenology_environment=environment,nitrogen_deposition_gn_m2=float(nitrogen),oxygen_factor=float(oxygen),crop_management=crop_management,irrigation_input=irrigation_input)
    cells=np.zeros((5,9));cells[:,3]=54.5;n=len(final.snow.mass)
    cells[:n]=np.column_stack((final.snow.mass,final.snow.enthalpy,final.snow.thickness,final.snow.radius_um,final.snow.impurities))
    options=row[3].copy();options[0]=n;d['leaf_respiration_umol_m2s']=d['leaf_respiration'];d['valid']=1.
    d['thermal_layer_tolerance']=np.zeros(9)
    d['thermal_layer_tolerance'][:len(d['thermal']['layer_residual_jm2'])]=d['thermal']['convergence_tolerance_jm2']
    plant=final.bgc_state.plant;soil=final.bgc_state.soil
    final_site=site.copy();final_site[0]=final.lai;final_site[11]=d['biogeochemistry']['nitrogen_factor_after']
    result=dict(snow_cells=cells,soil_cells=np.column_stack((final.ground.water,final.ground.enthalpy,final.ground.dz)),
        options=options,storage=np.array([final.canopy_water_kgm2,final.canopy_enthalpy_jm2,final.surface_temperature]),
        diagnostics=d,surface=d,site=final_site,impurity_output_kgm2=d['impurity_output_kgm2'],
        condensate=np.column_stack((final.canopy_condensate.mass,final.canopy_condensate.enthalpy)),
        soil_temperature=final.ground.temperature,soil_parameters=final.ground.params,freezing_option=final.ground.freezing_option,
        plant_state=np.r_[plant.carbon,plant.nitrogen,plant.reserve_carbon,plant.reserve_nitrogen],
        soil_cn_state=np.r_[soil.carbon,soil.nitrogen,soil.mineral_nitrogen],biogeochemistry=d['biogeochemistry'],
        phenology_state=(row[8] if environment is None else np.array([final.phenology_state.gdd,final.phenology_state.leaf_fraction,final.phenology_state.remaining_seconds])),
        phenology_flags=(flags if environment is None else np.array([final.phenology_state.phase,final.phenology_state.armed,5])))
    if crop_data is not None:
        cs=final.crop_state
        result.update(crop_state=np.array([cs.air_degree_days,cs.soil_degree_days,cs.age_days,cs.seed_carbon,cs.seed_nitrogen]),
            crop_phase=np.array([cs.phase]),crop=d['crop'])
    if irrigation_data is not None:
        result.update(irrigation=d['irrigation'],irrigation_state=np.array([final.irrigation_state.pending_kgm2,final.irrigation_state.remaining_seconds]))
    if wetland_data is not None:
        result.update(wetland=d['wetland'],wetland_state=np.array([final.wetland_state.methane_gc_m2]))
    return result


def cn_budgets(arrays,got,nitrogen):
    old_plant=arrays[11].astype(float);old_soil=arrays[13].astype(float)
    plant=got['plant_state'].astype(float);soil=got['soil_cn_state'].astype(float);d=got['diagnostics']
    oldc=old_plant[:,:3].sum(1)+old_plant[:,6]+old_soil[:,:7].sum(1)
    oldn=old_plant[:,3:6].sum(1)+old_plant[:,7]+old_soil[:,7:].sum(1)
    newc=plant[:,:3].sum(1)+plant[:,6]+soil[:,:7].sum(1);newn=plant[:,3:6].sum(1)+plant[:,7]+soil[:,7:].sum(1)
    gpp=d['gpp_gc_m2'].astype(float);ra=d['autotrophic_respiration_gc_m2'].astype(float);rh=d['heterotrophic_respiration_gc_m2'].astype(float)
    cr=newc-oldc-gpp+ra+rh;nr=newn-oldn-nitrogen.astype(float);eps=np.finfo(arrays[0].dtype).eps
    cb=128.*eps*np.maximum(newc+oldc+gpp+ra+rh,1.);nb=128.*eps*np.maximum(newn+oldn+nitrogen,1.)
    assert np.all(abs(cr)<=cb),(cr,cb)
    assert np.all(abs(nr)<=nb),(nr,nb)
    return dict(carbon=float(abs(cr).max()),nitrogen=float(abs(nr).max()),carbon_bound_max=float(cb.max()),nitrogen_bound_max=float(nb.max()))


def check_science(arrays,f,dep,env,cal,nitrogen,oxygen,got,dtype):
    maximum={}
    for k,row in enumerate(zip(*arrays)):
        ref=reference(row,f[k],dep[k],env[k],cal[k],nitrogen[k],oxygen[k])
        one={key:({a:b[k] for a,b in value.items()} if key=='diagnostics' else value[k]) for key,value in got.items()
            if key in ('snow_cells','soil_cells','options','storage','diagnostics','impurity_output_kgm2','condensate')}
        error=science(one,ref,dtype)
        for key in ('plant_state','soil_cn_state'):
            value=got[key][k];expected=ref[key];error[key]=float(np.max(abs(value-expected)))
            if key=='plant_state':ci=[0,1,2,6];ni=[3,4,5,7]
            else:ci=list(range(7));ni=list(range(7,15))
            assert_science_close(value[ci],expected[ci],relative=.1,zero_floor=.001,absolute=.0001)
            assert_science_close(value[ni],expected[ni],relative=.1,zero_floor=.0001,absolute=.00001)
        for key in ('gpp_gc_m2','npp_gc_m2','nee_gc_m2','autotrophic_respiration_gc_m2','heterotrophic_respiration_gc_m2','soil_respiration_gc_m2'):
            assert_science_close(got['diagnostics'][key][k],ref['diagnostics'][key],relative=.1,zero_floor=.001,absolute=.0001)
        for key in ('lai_used','lai_after','nitrogen_factor_used','nitrogen_factor_after'):
            a=got['biogeochemistry'][key][k];b=ref['biogeochemistry'][key];error[key]=float(abs(a-b))
            assert_science_close(a,b,relative=.1,zero_floor=.001,absolute=.0001)
        np.testing.assert_array_equal(got['phenology_flags'][k],ref['phenology_flags'])
        np.testing.assert_allclose(got['phenology_state'][k],ref['phenology_state'],rtol=.1,atol=1e-4)
        for name in ('condensate',):
            np.testing.assert_allclose(got[name][k,:,0],ref[name][:,0],rtol=.1,atol=1e-4)
            np.testing.assert_allclose(got[name][k,:,1],ref[name][:,1],rtol=.1,atol=1800.)
        for key,value in error.items():
            if np.ndim(value)==0:maximum[key]=max(maximum.get(key,0.),value)
    return maximum


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_bgc_native_full_column_reference_and_independent_budgets(dtype):
    jax=pytest.importorskip('jax')
    if dtype is np.float64 and not jax.config.x64_enabled:pytest.skip('JAX x64 required')
    from noahmp_clm.gpu.clm_bgc_column import step_clm_bgc_columns_jax
    arrays,f,dep,env,cal=cases(dtype);nitrogen=np.full(len(f),.00001,dtype);oxygen=np.ones(len(f),dtype)
    for step in range(2):
        out=step_clm_bgc_columns_jax(*arrays,f,environment=env,calendar=cal,nitrogen_deposition_gn_m2=nitrogen,
            oxygen_factor=oxygen,deposition=dep,return_surface=True);got=host(out)
        print('BGC full native',str(dtype),step,check_science(arrays,f,dep,env,cal,nitrogen,oxygen,got,dtype),flush=True)
        print('BGC full budgets',str(dtype),step,dict(**budgets(arrays[:8],f,dep,got),**cn_budgets(arrays,got,nitrogen)),flush=True)
        arrays=next_bgc_arrays(arrays,got)


def test_bgc_native_full_column_same_precision_and_device_state_carry():
    cp=pytest.importorskip('cupy');pytest.importorskip('jax')
    from noahmp_clm.gpu.clm_bgc_column import step_clm_bgc_columns_jax
    from noahmp_clm.gpu.cuda.clm_bgc_column import step_clm_bgc_columns_cuda
    arrays,f,dep,env,cal=cases(np.float32);a_arrays=b_arrays=arrays;nitrogen=np.full(len(f),.00001,np.float32)
    for step in range(2):
        a=step_clm_bgc_columns_cuda(*a_arrays,f,environment=env,calendar=cal,nitrogen_deposition_gn_m2=nitrogen,deposition=dep,return_surface=True)
        b=step_clm_bgc_columns_jax(*b_arrays,f,environment=env,calendar=cal,nitrogen_deposition_gn_m2=nitrogen,deposition=dep,return_surface=True)
        ah=host(a,cp.asnumpy);bh=host(b)
        print('BGC full same precision',step,compare_same_precision(ah,bh),flush=True)
        before=tuple(cp.asnumpy(v) if isinstance(v,cp.ndarray) else v for v in a_arrays)
        print('BGC full CUDA budgets',step,dict(**budgets(before[:8],f,dep,ah),**cn_budgets(before,ah,nitrogen)),flush=True)
        a_arrays=next_bgc_arrays(a_arrays,a);b_arrays=next_bgc_arrays(b_arrays,b)
        assert all(isinstance(v,cp.ndarray) for k,v in enumerate(a_arrays) if k in (0,1,4,5,7,8,9,11,13))


def test_bgc_extended_contract_retains_every_cn_pool():
    arrays,f,dep,e,c=cases(np.float64);n=len(f)
    validate_bgc_column(arrays,f,e,c,np.zeros(n),np.ones(n),np)
    bad=[v.copy() for v in arrays];bad[11][0,0]*=2.
    with pytest.raises(ValueError,match='aliases'):validate_bgc_column(bad,f,e,c,np.zeros(n),np.ones(n),np)
    bad=[v.copy() for v in arrays];bad[9]=bad[9].astype(float)
    with pytest.raises(ValueError,match='integer'):validate_bgc_column(bad,f,e,c,np.zeros(n),np.ones(n),np)
    with pytest.raises(ValueError,match='sixteen'):validate_bgc_column(arrays[:-1],f,e,c,np.zeros(n),np.ones(n),np)
    with pytest.raises(ValueError,match='extratropical'):validate_bgc_column(arrays,f,e*0.,c,np.zeros(n),np.ones(n),np)
