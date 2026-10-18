"""Actual native stem geometry through the physical and managed C/N columns."""
from dataclasses import astuple,replace
import json
import numpy as np
import pytest
from noahmp_clm.gpu.clm_structure_column import (
    pack_clm_structure_columns,next_structure_arrays)
from noahmp_clm.physics.vegetation_structure import CanopyStructureParameters as Parameters
from test_gpu_clm_layered_column import host,science,budgets,compare_same_precision,reference as physical_reference
from test_gpu_clm_bgc_column import reference as bgc_reference,cn_budgets as bgc_budgets
from test_gpu_clm_crop_column import cn_budgets as crop_budgets


def cases(kind,dtype):
    from test_clm_stem_column import leafless_case
    from test_clm_bgc_column import case as bgc_case,Phase
    from test_clm_phenology_column import case as seasonal_case
    from test_clm_crop_column import case as crop_case
    from test_clm_wetland_column import case as wetland_case
    from test_clm_irrigation_column import case as irrigation_case
    from noahmp_clm.gpu.clm_bgc_column import pack_bgc_environment
    from noahmp_clm.gpu.clm_phenology_column import pack_phenology_environment
    from noahmp_clm.gpu.clm_irrigation_column import pack_irrigation_inputs
    states=[];forcing=[];kwargs={};environments=[];controls=[]
    for k in range(2):
        if kind=='layered':
            state,f=leafless_case()
            if k:
                from noahmp_clm.physics.snowpack import Snowpack
                from noahmp_clm.physics.canopy_interception import CanopyCondensate
                state=replace(state,snow=Snowpack.from_ice([30.],270.,density=100.),
                    canopy_structure_parameters=Parameters(canopy_top_m=.2,short_vegetation=True),
                    canopy_condensate=CanopyCondensate.from_temperature(np.full(10,.01),270.))
                f=replace(f,tair=270.,qair=.002,swdown=100.,lwdown=280.,prcp=0.)
        elif kind=='seasonal':state,f,e=seasonal_case(Phase.ACTIVE if k==0 else Phase.OFFSET);environments.append(e)
        elif kind=='bgc':state,f,e=bgc_case(None if k==0 else Phase.OFFSET,285.);environments.append(e)
        elif kind=='crop':state,f=crop_case('vegetative' if k==0 else 'mature',285.)
        elif kind=='wetland':state,f,kw=wetland_case('wet' if k==0 else 'dry');controls.append(kw)
        else:state,f,request=irrigation_case('bgc');controls.append(request)
        if kind!='layered':
            scheme=3 if kind=='crop' else 1
            state=state.with_vegetation_structure(Parameters(scheme,3.,.5,kind=='crop'),sai=.8)
        states.append(state);forcing.append(list(vars(f).values()))
    if kind=='seasonal':kwargs['environment'],kwargs['calendar']=pack_phenology_environment(environments,dtype=dtype)
    if kind=='bgc':kwargs['environment'],kwargs['calendar']=pack_bgc_environment(environments,dtype=dtype)
    if kind=='irrigated':kwargs['inputs'],kwargs['checks']=pack_irrigation_inputs(controls,dtype=dtype)
    if kind in ('bgc','crop','wetland','irrigated'):kwargs['nitrogen_deposition_gn_m2']=np.full(2,1e-5,dtype)
    return pack_clm_structure_columns(states,kind=kind,dtype=dtype),np.asarray(forcing,dtype),kwargs


def audit(arrays,f,kwargs,got,kind,dtype):
    base=arrays[:-3];maximum={};external=None
    if kind=='irrigated':
        water=got['irrigation']['applied_kgm2'].astype(float)
        external=(water,water*(334000.+4180.*(kwargs['inputs'][:,1].astype(float)-273.15)))
    budget=budgets(base[:8],f,np.zeros((len(f),5),dtype),got,external_input=external)
    if kind in ('bgc','irrigated'):budget.update(bgc_budgets(base[:16],got,kwargs['nitrogen_deposition_gn_m2']))
    elif kind=='crop':budget.update(crop_budgets(base,got,kwargs['nitrogen_deposition_gn_m2']))
    elif kind=='wetland':
        # Include methane storage/emissions in the independent whole C ledger.
        from test_gpu_clm_wetland_column import independent_budgets
        # Existing wetland audit has a fixed actual deposition of0.0001.
        # This campaign uses exactly that value for its wetland subcase.
        budget.update(independent_budgets(base,f,kwargs,got))
    for k,row in enumerate(zip(*arrays)):
        controls=dict(canopy_structure=row[-3:]);dep=np.zeros(5)
        if 'most_geometry' in kwargs:controls['most_geometry']=kwargs['most_geometry']
        if kind in ('layered','seasonal'):
            if kind=='seasonal':
                from noahmp_clm.physics.phenology import PhenologyEnvironment
                env=kwargs['environment'][k];cal=kwargs['calendar'][k]
                controls.update(phenology=row[8:11],phenology_environment=PhenologyEnvironment(float(env[0]),bool(cal[0]),bool(cal[1]),float(env[1])))
            ref=physical_reference(row[:8],f[k],dep,**controls)
        else:
            if kind=='crop':controls['crop_data']=(*row[16:19],np.zeros(2,int))
            if kind=='irrigated':controls['irrigation_data']=(*row[16:18],kwargs['inputs'][k],kwargs['checks'][k])
            if kind=='wetland':controls['wetland_data']=row[16:18]
            env=kwargs['environment'][k] if 'environment' in kwargs else np.zeros(2)
            cal=kwargs['calendar'][k] if 'calendar' in kwargs else np.zeros(2,int)
            ref=bgc_reference(row[:16],f[k],dep,env,cal,kwargs['nitrogen_deposition_gn_m2'][k],1.,**controls)
        one={key:({a:b[k] for a,b in value.items()} if key=='diagnostics' else value[k]) for key,value in got.items()
            if key in ('snow_cells','soil_cells','options','storage','diagnostics','impurity_output_kgm2','condensate')}
        error=science(one,ref,dtype)
        np.testing.assert_allclose(got['structure_state'][k],ref['structure_state'],rtol=.1,atol=1e-6)
        for key in ('total_lai','total_sai','exposed_lai','exposed_sai','exposed_fraction','exposed_plant_area'):
            np.testing.assert_allclose(got['canopy_geometry'][key][k],ref['canopy_geometry'][key],rtol=.1,atol=1e-6)
        for key in ('plant_state','soil_cn_state','wetland_state','crop_state','irrigation_state'):
            if key in ref:np.testing.assert_allclose(got[key][k],ref[key],rtol=.1,atol=1e-4)
        if kind not in ('layered','seasonal'):
            from test_gpu_snicar import assert_science_close
            for key in ('gpp_gc_m2','nee_gc_m2','autotrophic_respiration_gc_m2','heterotrophic_respiration_gc_m2'):
                assert_science_close(got['diagnostics'][key][k],ref['diagnostics'][key],relative=.1,zero_floor=.001,absolute=.0001)
            for key,indices in [('plant_state',[3,4,5,7]),('soil_cn_state',list(range(7,15))),('crop_state',[4])]:
                if key in ref:assert_science_close(got[key][k,indices],ref[key][indices],relative=.1,zero_floor=.0001,absolute=.00001)
        for key in ('h','le'):maximum[key]=max(maximum.get(key,0.),error[key])
        maximum['sai']=max(maximum.get('sai',0.),float(abs(got['structure_state'][k]-ref['structure_state']).max()))
    if kind=='layered':
        assert got['diagnostics']['gpp_gcm2'][0]==0. and np.all(got['surface']['transpiration'][0]==0.)
        assert got['canopy_geometry']['exposed_plant_area'][1]==0. and got['storage'][1,0]==0.
    print('native structure whole',kind,str(dtype),json.dumps(dict(maximum_absolute=maximum,independent_budgets=budget)),flush=True)


def run_cases(run,dtype,most=False,convert=np.asarray):
    from noahmp_clm.physics.canopy_most import MOSTGeometry
    kinds=['layered'] if most else ['layered','seasonal','bgc','crop','wetland','irrigated']
    for kind in kinds:
        arrays,f,kwargs=cases(kind,dtype)
        if kind=='wetland':kwargs['nitrogen_deposition_gn_m2']=np.full(2,.0001,dtype)
        if most:
            geometry=MOSTGeometry();kwargs['most_geometry']=np.asarray([astuple(geometry)]*2,dtype)
        for step in range(2):
            out=run(arrays,f,kind=kind,return_surface=True,**kwargs);got=host(out,convert)
            refkwargs=dict(kwargs)
            if most:refkwargs['most_geometry']=geometry
            audit(tuple(convert(x) for x in arrays),f,refkwargs,got,kind,dtype)
            before=arrays;arrays=next_structure_arrays(arrays,out,kind=kind)
            np.testing.assert_array_equal(convert(arrays[-3]),got['structure_state'])
            assert arrays[-2] is before[-2] and arrays[-1] is before[-1]
            if kind=='irrigated':kwargs['checks']=np.zeros_like(kwargs['checks'])
        yield kind,got


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('most',[False,True])
def test_structure_native_full_column_reference(dtype,most):
    jax=pytest.importorskip('jax')
    if dtype==np.float64 and not jax.config.x64_enabled:pytest.skip('JAX x64 required')
    from noahmp_clm.gpu.clm_structure_column import step_clm_structure_columns_jax
    list(run_cases(step_clm_structure_columns_jax,dtype,most))


@pytest.mark.parametrize('most',[False,True])
def test_structure_full_cuda_jax_same_precision(most):
    cp=pytest.importorskip('cupy');pytest.importorskip('jax')
    from noahmp_clm.gpu.clm_structure_column import step_clm_structure_columns_jax,step_clm_structure_columns_cuda
    def cuda(*args,**kwargs):return step_clm_structure_columns_cuda(*args,**kwargs)
    # Independent two-step native trajectories for each backend, rather than
    # copying the primary state into its shadow between steps.
    a=dict(run_cases(cuda,np.float32,most,convert=lambda x:cp.asnumpy(x) if isinstance(x,cp.ndarray) else np.asarray(x)))
    b=dict(run_cases(step_clm_structure_columns_jax,np.float32,most))
    for kind in b:print('structure whole same precision',kind,compare_same_precision(a[kind],b[kind]),flush=True)
