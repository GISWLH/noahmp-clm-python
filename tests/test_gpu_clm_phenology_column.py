"""Native seasonal state through real interception/radiation/water/heat steps."""
from dataclasses import replace
import numpy as np
import pytest
from test_clm_phenology_column import case,Phase
from test_gpu_clm_layered_column import reference,budgets,science,host,compare_same_precision
from noahmp_clm.physics.phenology import PhenologyEnvironment,step_phenology,PhenologyState,PhenologyParameters
from noahmp_clm.gpu.clm_phenology_column import (pack_clm_phenology_columns,pack_phenology_environment,
    validate_seasonal_column,next_phenology_arrays)


def cases(dtype):
    states=[];forcings=[];environments=[]
    for t,snow in ((285.,0.),(270.,.01)):
        for phase in Phase:
            state,f,e=case(phase,t,snow)
            state=replace(state,options=replace(state.options,OptRainSnowPartition=5,
                OptRunoffSurface=6,OptRunoffSubsurface=6))
            states.append(state);forcings.append(list(vars(f).values()));environments.append(e)
    arrays=pack_clm_phenology_columns(states,dtype=dtype)
    f=np.asarray(forcings,dtype);dep=np.zeros((len(f),5),dtype)
    environment,calendar=pack_phenology_environment(environments,dtype=dtype)
    return arrays,f,dep,environment,calendar


def check_reference(arrays,f,dep,environment,calendar,got,dtype):
    for k,row in enumerate(zip(*arrays)):
        e=PhenologyEnvironment(float(environment[k,0]),bool(calendar[k,0]),bool(calendar[k,1]),float(environment[k,1]))
        ref=reference(row[:8],f[k],dep[k],phenology=row[8:],phenology_environment=e)
        one={key:({a:b[k] for a,b in v.items()} if key=='diagnostics' else v[k]) for key,v in got.items()
            if key not in ('surface','canopy_interception','canopy_post_interception','phenology','site','phenology_parameters')}
        error=science(one,ref,dtype)
        for key in ('lai','leaf_gain_fraction','leaf_loss_fraction','gdd_increment'):
            delta=abs(got['phenology'][key][k]-ref['diagnostics']['phenology'][key])
            assert delta<1e-4 if key=='gdd_increment' else delta<1e-6
            error[key]=float(delta)
        for key in ('onset_triggered','offset_triggered'):
            assert got['phenology'][key][k]==ref['diagnostics']['phenology'][key]
        np.testing.assert_allclose(one['condensate'][:,0],ref['condensate'][:,0],rtol=.1,atol=1e-4)
        np.testing.assert_allclose(one['condensate'][:,1],ref['condensate'][:,1],rtol=.1,atol=1800.)
        if got['site'][k,0]==0.:
            assert got['condensate'][k,:,0].sum()==0. and got['diagnostics']['gpp_umol_m2s'][k]==0.
        print('Phenology full native',str(dtype),k,error,flush=True)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_seasonal_full_native_reference_and_conservation(dtype):
    jax=pytest.importorskip('jax')
    if dtype is np.float64 and not jax.config.x64_enabled:pytest.skip('JAX x64 required')
    from noahmp_clm.gpu.clm_phenology_column import step_clm_phenology_columns_jax
    arrays,f,dep,env,cal=cases(dtype)
    for step in range(2):
        got=host(step_clm_phenology_columns_jax(*arrays,f,environment=env,calendar=cal,deposition=dep,return_surface=True))
        check_reference(arrays,f,dep,env,cal,got,dtype)
        print('Phenology full budgets',str(dtype),step,budgets(arrays[:8],f,dep,got),flush=True)
        arrays=next_phenology_arrays(arrays,got)


def test_seasonal_full_native_same_precision():
    cp=pytest.importorskip('cupy');pytest.importorskip('jax')
    from noahmp_clm.gpu.clm_phenology_column import step_clm_phenology_columns_jax as jax_step
    from noahmp_clm.gpu.cuda.clm_phenology_column import step_clm_phenology_columns_cuda as cuda_step
    arrays,f,dep,env,cal=cases(np.float32);a_arrays=b_arrays=arrays
    for step in range(2):
        a=cuda_step(*a_arrays,f,environment=env,calendar=cal,deposition=dep,return_surface=True)
        b=jax_step(*b_arrays,f,environment=env,calendar=cal,deposition=dep,return_surface=True)
        ah=host(a,cp.asnumpy);bh=host(b)
        print('Phenology full same precision',step,compare_same_precision(ah,bh),flush=True)
        print('Phenology full CUDA budgets',step,budgets(tuple(cp.asnumpy(v) for v in a_arrays[:8]),f,dep,ah),flush=True)
        a_arrays=next_phenology_arrays(a_arrays,a);b_arrays=next_phenology_arrays(b_arrays,b)


def test_seasonal_column_contract_and_state_carry():
    arrays,f,dep,e,c=cases(np.float64);validate_seasonal_column(arrays,f,e,c,np)
    bad=[v.copy() for v in arrays];bad[4][0,0]=2.
    with pytest.raises(ValueError,match='alias'):validate_seasonal_column(bad,f,e,c,np)
    with pytest.raises(ValueError,match='extratropical'):validate_seasonal_column(arrays,f,e*0.,c,np)
    with pytest.raises(ValueError,match='shapes'):validate_seasonal_column(arrays,f,e,c.astype(float),np)
    from noahmp_clm.gpu.clm_layered_column import pack_clm_layered_columns
    from noahmp_clm.gpu.pack import column_state_to_vec
    state,_,_=case(Phase.ACTIVE)
    with pytest.raises(NotImplementedError,match='phenology'):pack_clm_layered_columns([state])
    with pytest.raises(NotImplementedError,match='phenology'):column_state_to_vec(state.parameter_view())
    with pytest.raises(ValueError,match='OptDynamicVeg'):pack_clm_phenology_columns([replace(state,
        options=replace(state.options,OptDynamicVeg=4),phenology_state=None,phenology_parameters=None)])
