"""Actual column coupling of closed Richards transport and its heat ledger."""
from dataclasses import replace
import json
import numpy as np
import pytest
from noahmp_clm.column import ColumnState,Forcing
from noahmp_clm.clm_state import CLMColumnState
from noahmp_clm.clm_driver import step_clm_column
from noahmp_clm.physics.richards import RichardsHydrology
from noahmp_clm.physics.soil_enthalpy import SoilEnthalpyState
from noahmp_clm.physics.snowpack import Snowpack,TF
from test_richards import independent_thermal_ode


@pytest.mark.parametrize('kind',['summer','rain','snow','melt','layered','bgc','dry','five_layers'])
def test_actual_closed_column_mass_heat_and_independent_postsurface_transport(kind,monkeypatch):
    from noahmp_clm import clm_enthalpy_driver as driver
    old=ColumnState(ts=295.,tsoil=np.array([290.,289.,288.,287.]),smc=np.array([.35,.24,.28,.25]),canw=.0003)
    f=Forcing(1800.,295.,.008,3.,101325.,700.,350.,0.);snow=Snowpack.empty()
    if kind=='rain':f=replace(f,prcp=5e-6)
    if kind=='snow':
        old.ts=265.;old.tsoil[:]=268.;f=Forcing(1800.,265.,.002,3.,101325.,0.,240.,2e-6)
    if kind=='melt':snow=Snowpack.from_ice([.1],TF)
    if kind=='dry':old.smc[:]=.08;old.canw=0.
    if kind=='five_layers':
        snow=Snowpack.from_ice([10.,20.,30.,50.,80.],265.,300.);old.ts=265.;old.tsoil[:]=270.
        f=Forcing(1800.,268.,.001,3.,101325.,300.,250.,0.)
    state=CLMColumnState.from_column(old,snow=snow)
    kwargs={}
    if kind=='layered':state=state.with_layered_canopy()
    if kind=='bgc':
        from test_clm_bgc_column import case
        state,f,_=case(None,285.,False);kwargs['nitrogen_deposition_gn_m2']=1e-5
    initial=(state.ground.water.copy(),state.ground.enthalpy.copy())
    def forbidden(*args,**kw):raise AssertionError('Closed Richards must replace legacy gravity drainage')
    monkeypatch.setattr(driver,'drain_soil',forbidden)
    after,d=step_clm_column(state,f,soil_hydrology=RichardsHydrology(),**kwargs)
    h=d['soil_hydrology'];assert h['lower_boundary_condition']==2 and h['method']=='richards_zero_flux'
    assert d['bottom_outflow_kgm2']==d['bottom_outflow_enthalpy_jm2']==d['runoff_deep_mm']==0.
    assert d['runoff_surface_mm']==pytest.approx(d['runoff_mm'],abs=1e-12,rel=0.)
    before=SoilEnthalpyState(h['water_before_kg_m2'],h['enthalpy_before_j_m2'],state.ground.dz,state.ground.params,state.ground.freezing_option)
    expected=independent_thermal_ode(before,f.dt)
    water_error=float(np.max(abs(after.ground.water-expected.water)/(1000.*state.ground.dz)))
    temperature_error=float(np.max(abs(after.ground.temperature-expected.temperature)))
    assert water_error<=.002 and temperature_error<=.05
    assert np.all(abs(h['residual_mass_heat'])<=h['bound_mass_heat'])
    assert abs(d['water_residual_kgm2'])<3e-10 and abs(d['energy_residual_jm2'])<.02
    np.testing.assert_array_equal(state.ground.water,initial[0]);np.testing.assert_array_equal(state.ground.enthalpy,initial[1])
    if kind=='bgc':
        bg=d['biogeochemistry'];carbon=after.bgc_state.total_carbon-state.bgc_state.total_carbon+d['nee_gc_m2']
        nitrogen=after.bgc_state.total_nitrogen-state.bgc_state.total_nitrogen-kwargs['nitrogen_deposition_gn_m2']
        assert abs(carbon)<=bg['carbon_bound_gc_m2'] and abs(nitrogen)<=bg['nitrogen_bound_gn_m2']
    print('Actual Richards column',json.dumps(dict(kind=kind,max_soil_water_error_m3m3=water_error,max_temperature_error_k=temperature_error,
        column_water_residual_kg_m2=d['water_residual_kgm2'],column_heat_residual_j_m2=d['energy_residual_jm2'],
        transport_residual_mass_heat=h['residual_mass_heat'].tolist(),transport_original_bounds=h['bound_mass_heat'].tolist(),runoff_mm=d['runoff_mm'])),flush=True)


@pytest.mark.parametrize('bad',['type','arno','legacy'])
def test_closed_boundary_cannot_discard_requested_outlets_or_legacy_state(bad):
    old=ColumnState();state=CLMColumnState.from_column(old);f=Forcing(1800.,295.,.008,3.,101325.,700.,350.,0.)
    if bad=='type':
        with pytest.raises(TypeError,match='RichardsHydrology'):step_clm_column(state,f,soil_hydrology={})
    elif bad=='arno':
        state=replace(state,options=replace(state.options,OptRunoffSubsurface=6))
        with pytest.raises(ValueError,match='bottom outlets'):step_clm_column(state,f,soil_hydrology=RichardsHydrology())
    else:
        with pytest.raises(NotImplementedError,match='CLMColumnState'):step_clm_column(old,f,soil_hydrology=RichardsHydrology())


def test_closed_bottom_keeps_real_surface_overflow_and_its_enthalpy():
    old=ColumnState(ts=290.,tsoil=np.full(4,290.),smc=np.full(4,.44),fveg=0.,lai=0.,canw=0.)
    state=CLMColumnState.from_column(old);f=Forcing(1800.,295.,.008,3.,101325.,600.,350.,1e-4)
    after,d=step_clm_column(state,f,soil_hydrology=RichardsHydrology())
    assert d['runoff_mm']>100. and d['runoff_enthalpy_jm2']>0.
    assert d['runoff_surface_mm']==pytest.approx(d['runoff_mm'],abs=1e-12,rel=0.)
    assert d['bottom_outflow_kgm2']==d['runoff_deep_mm']==0.
    assert abs(d['water_residual_kgm2'])<3e-10 and abs(d['energy_residual_jm2'])<.02
    assert np.all(after.ground.water<=1000.*after.ground.dz*after.ground.params.smcmax+1e-9)
    print('Richards actual overflow',json.dumps(dict(runoff_mm=d['runoff_mm'],runoff_enthalpy_j_m2=d['runoff_enthalpy_jm2'],
        water_residual_kg_m2=d['water_residual_kgm2'],heat_residual_j_m2=d['energy_residual_jm2'])),flush=True)
