"""NumPy column coupling before native capacity-state integration."""
from dataclasses import replace
from pathlib import Path
import pickle
import sys
import numpy as np
import pytest
from noahmp_clm.physics.canopy_capacity import (
    TemperatureMemory,CanopyCapacityParameters,CanopyCapacityEnvironment,canopy_capacity_profile)
from noahmp_clm.physics.photosynthesis import LeafParameters,leaf_gas_exchange
from noahmp_clm.clm_enthalpy_driver import step_clm_enthalpy_column
from noahmp_clm.physics.canopy_most import MOSTGeometry
from test_clm_bgc_column import case
from test_canopy_phase_energy import cases,independent_reference
from noahmp_clm.physics.canopy_phase_energy import solve_canopy_phase_energy

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
from run_fluxnet_column import audit_step

PFT=CanopyCapacityParameters(.01,58.,.0509)
ENV=CanopyCapacityEnvironment(30000.,45000.)


def test_vectorized_layer_coefficients_equal_independent_scalar_leaf_calls():
    profile=canopy_capacity_profile(np.linspace(.1,.5,10),30000.,45000.,290.,PFT)
    rows=profile['leaf_parameters'];parameters=LeafParameters.from_layer_rows(rows)
    par=np.linspace(10.,1500.,10);temperature=np.linspace(275.,310.,10)
    got=leaf_gas_exchange(par,temperature,parameters=parameters,nitrogen_factor=.6)
    expected=[leaf_gas_exchange(a,t,parameters=LeafParameters(*p),nitrogen_factor=.6)
              for a,t,p in zip(par,temperature,rows)]
    for name,value in got.items():
        np.testing.assert_allclose(value,[one[name] for one in expected],rtol=1e-10,atol=1e-10)


@pytest.mark.parametrize('name',['liquid','dry'])
def test_layer_capacity_phase_equilibrium_against_independent_network(name):
    _,area,sw,lw,e,old,kwargs=next(row for row in cases() if row[0]==name)
    profile=canopy_capacity_profile(area,30000.,45000.,290.,PFT)
    kwargs=dict(kwargs,leaf_parameters=LeafParameters.from_layer_rows(profile['leaf_parameters']))
    got=solve_canopy_phase_energy(area,sw,lw,e,old,1800.,tolerance=1e-7,max_iterations=80,**kwargs)
    reference=independent_reference(area,sw,lw,e,old,1800.,**kwargs)
    error=float(np.max(abs(np.r_[got['leaf_temperature'],got['ground_temperature']]-reference['temperature'])))
    assert error<.05 and np.max(abs(got['solver_residual']))<=1e-7
    for key in ['h','le','gpp']:
        absolute,near=(.01,.1) if key=='gpp' else (1.,10.)
        assert abs(got[key]-reference[key])<=(absolute if abs(reference[key])<=near else .1*abs(reference[key]))
    print('capacity independent network',name,dict(temperature_error_K=error,
          flux_absolute_error={k:float(abs(got[k]-reference[k])) for k in ['h','le','gpp']}),flush=True)


@pytest.mark.parametrize('poor',[False,True])
@pytest.mark.parametrize('most',[False,True])
def test_bgc_capacity_memory_advances_once_and_original_four_budgets_hold(poor,most):
    old,forcing,_=case(None,285.,poor)
    old=old.with_canopy_capacity(PFT,TemperatureMemory.cold(forcing.dt))
    previous=old;options=dict(capacity_environment=ENV)
    if most:options['most_geometry']=MOSTGeometry()
    for index in range(2):
        memory=previous.canopy_capacity_state
        after,diagnostic=step_clm_enthalpy_column(previous,forcing,**options)
        incoming,residual,bound=audit_step(previous,after,forcing,diagnostic,0.)
        assert np.all(abs(residual)<=bound)
        capacity=diagnostic['canopy_capacity']
        assert capacity['temperature_count_before']==index and capacity['temperature_count_after']==index+1
        assert capacity['mean_temperature_k']==memory.mean_temperature_k
        assert after.canopy_capacity_state.mean_temperature_k==forcing.tair
        assert old.canopy_capacity_state.count==0
        assert diagnostic['biogeochemistry']['nitrogen_factor_used']==pytest.approx(.25 if poor else 1.,abs=1e-14)
        assert diagnostic['gpp_gcm2']>0.
        print('whole capacity four budgets',dict(poor=poor,most=most,index=index,
              residual_m_h_c_n=residual.tolist(),original_bounds=bound.tolist(),
              gpp_gc_m2=diagnostic['gpp_gcm2'],nee_gc_m2=diagnostic['nee_gc_m2'],
              memory_count=after.canopy_capacity_state.count,mean_temperature_used_K=capacity['mean_temperature_k']),flush=True)
        if index==0:
            restored=pickle.loads(pickle.dumps(after))
            resumed,rd=step_clm_enthalpy_column(restored,forcing,**options)
            continuous,cd=step_clm_enthalpy_column(after,forcing,**options)
            np.testing.assert_array_equal(resumed.ground.water,continuous.ground.water)
            np.testing.assert_array_equal(resumed.ground.enthalpy,continuous.ground.enthalpy)
            np.testing.assert_array_equal(resumed.canopy_capacity_state.values,continuous.canopy_capacity_state.values)
            for key in ['h','le','gpp_gcm2','nee_gc_m2']:assert rd[key]==cd[key]
        previous=after


def test_capacity_state_cannot_be_dropped_or_advanced_without_explicit_calendar():
    from noahmp_clm.gpu.clm_bgc_column import pack_clm_bgc_columns
    old,forcing,_=case(None)
    with pytest.raises(ValueError,match='retained'):
        step_clm_enthalpy_column(old,forcing,capacity_environment=ENV)
    old=old.with_canopy_capacity(PFT,TemperatureMemory.cold(forcing.dt))
    with pytest.raises(ValueError,match='photoperiod'):step_clm_enthalpy_column(old,forcing)
    with pytest.raises(ValueError,match='interval'):
        step_clm_enthalpy_column(old,replace(forcing,dt=3600.),capacity_environment=ENV)
    with pytest.raises(NotImplementedError,match='temperature history'):pack_clm_bgc_columns([old])
    with pytest.raises(ValueError,match='Canopy capacity'):replace(old,canopy_capacity_parameters=None)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('structured',[False,True])
def test_native_capacity_encoding_retains_all_samples_and_unchanged_physical_prefix(dtype,structured):
    from noahmp_clm.gpu.clm_capacity_column import pack_clm_capacity_columns,pack_capacity_environment
    from noahmp_clm.gpu.clm_bgc_column import pack_clm_bgc_columns
    from noahmp_clm.gpu.clm_structure_column import pack_clm_structure_columns
    from noahmp_clm.physics.vegetation_structure import CanopyStructureParameters
    old,forcing,_=case(None)
    if structured:old=old.with_vegetation_structure(CanopyStructureParameters(canopy_top_m=8.,canopy_bottom_m=1.),sai=.55)
    base=pack_clm_structure_columns([old],kind='bgc',dtype=dtype) if structured else pack_clm_bgc_columns([old],dtype=dtype)
    memory=TemperatureMemory.cold(forcing.dt).append(280.,forcing.dt)
    packed=pack_clm_capacity_columns([old.with_canopy_capacity(PFT,memory)],kind='bgc',structured=structured,dtype=dtype)
    assert len(packed)==(23 if structured else 20)
    for original,actual in zip(base,packed[:-4]):np.testing.assert_array_equal(actual,original)
    np.testing.assert_array_equal(packed[-4][0],memory.values.astype(dtype))
    np.testing.assert_array_equal(packed[-3],np.asarray([[1,1]],np.int32))
    np.testing.assert_array_equal(packed[-2],np.asarray([[forcing.dt,293.15]],dtype))
    np.testing.assert_array_equal(packed[-1],np.asarray([[.01,58.,.0509,1.]],dtype))
    assert pack_capacity_environment([ENV],dtype=dtype).dtype==np.dtype(dtype)
