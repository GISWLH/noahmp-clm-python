from dataclasses import replace
import numpy as np
import pytest
from noahmp_clm.column import ColumnState,Forcing
from noahmp_clm.clm_state import CLMColumnState
from noahmp_clm.clm_driver import step_clm_column
from noahmp_clm.physics.snowpack import Snowpack,TF


@pytest.mark.parametrize('case',['summer','rain','snow','melt','frost','dew','drip','bare','five_layers'])
def test_extensive_column_precipitation_evaporation_and_runoff(case):
    air,qa,sw,lw,precip,water=295.,.008,700.,350.,0.,.3
    old=ColumnState(ts=295.,tsoil=np.full(4,290.),smc=np.full(4,.3))
    snow=Snowpack.empty()
    if case=='rain': precip=5e-6;old.smc[:]=.44
    if case in ('snow','frost'):
        air,qa,sw,lw=265.,.002,0.,240.;old.ts=265.;old.tsoil[:]=268.
        precip=2e-6 if case=='snow' else 0.
    if case=='melt': snow=Snowpack.from_ice([.1],TF)
    if case=='five_layers':
        snow=Snowpack.from_ice([10.,20.,30.,50.,80.],265.,300.)
        air,qa,sw,lw=268.,.001,300.,250.;old.ts=265.;old.tsoil[:]=270.
    if case=='dew': qa=.025;sw=0.;lw=270.
    if case=='drip': water=3.
    if case=='bare': old.fveg=0.;water=2.
    old.canw=water/1000.
    state=CLMColumnState.from_column(old,snow=snow)
    before=(state.ground.water.copy(),state.ground.enthalpy.copy(),state.snow.mass.copy(),state.enthalpy_jm2)
    final,d=step_clm_column(state,Forcing(1800.,air,qa,3.,101325.,sw,lw,precip))
    assert isinstance(final,CLMColumnState)
    assert abs(d['water_residual_kgm2'])<3e-10
    assert abs(d['energy_residual_jm2'])<.02
    assert final.canopy_water_kgm2<=.5*final.lai*final.fveg+1e-13
    assert np.all(final.ground.water>=0.)
    assert final.surface_temperature==float(final.snow.temperature[0] if len(final.snow.mass) else final.ground.temperature[0])
    assert d['runoff_mm']==d['runoff_kgm2']
    assert d['canopy_water_after_kgm2']==final.canopy_water_kgm2
    assert d['canopy_enthalpy_after_jm2']==final.canopy_enthalpy_jm2
    np.testing.assert_array_equal(state.ground.water,before[0])
    np.testing.assert_array_equal(state.ground.enthalpy,before[1])
    np.testing.assert_array_equal(state.snow.mass,before[2])
    assert state.enthalpy_jm2==before[3]
    if case=='rain': assert d['runoff_mm']>0.
    if case=='snow': assert final.swe>0.
    if case=='melt': assert final.swe==0.
    if case=='five_layers': assert len(final.snow.mass)==5


def test_extensive_state_retains_five_layers_and_does_not_share_parameters():
    old=ColumnState()
    snow=Snowpack.from_ice([10.,20.,30.,50.,80.],265.,300.)
    first=CLMColumnState.from_column(old,snow=snow)
    second=CLMColumnState.from_column(old,snow=snow)
    old.soil.satdk=9.
    first.ground.params.satdk=8.
    assert second.ground.params.satdk==1e-5
    assert len(first.snow.mass)==5
    with pytest.raises(ValueError): first.snow.mass[0]=0.
    with pytest.raises(ValueError): first.root_frac[0]=0.
    with pytest.raises(ValueError,match='phase curve'):
        replace(second,options=replace(second.options,OptSoilSupercoolWater=2))


def test_unsupported_options_are_not_silently_ignored():
    state=CLMColumnState.from_column(ColumnState())
    state=replace(state,options=replace(state.options,OptSoilTemperatureBottom=2))
    with pytest.raises(NotImplementedError,match='OptSoilTemperatureBottom'):
        step_clm_column(state,Forcing(1800.,295.,.008,3.,101325.,700.,350.,0.))


def test_bare_soil_frost_does_not_create_a_discontinuous_snow_boundary():
    import json
    from pathlib import Path
    from noahmp_clm.column import SoilParams
    from noahmp_clm.options import PhysicsOptions
    from noahmp_clm.physics.soil_enthalpy import SoilEnthalpyState
    data=json.loads((Path(__file__).parent/'fixtures/surface_soil_frost.json').read_text())
    values=data['state'];ground=values.pop('ground');ground['params']=SoilParams(**ground['params'])
    values['ground']=SoilEnthalpyState(**ground);values['snow']=Snowpack.empty()
    values['options']=PhysicsOptions(**values['options'])
    state=CLMColumnState(**values)
    after,d=step_clm_column(state,Forcing(**data['forcing']),cosine_zenith=np.sin(np.pi*(14.5-6.)/12.))
    assert not len(after.snow.mass) and d['ground_evaporation']<0.
    assert after.surface_temperature<TF
    assert d['surface_evaluations']<20
    assert abs(d['energy_residual_jm2'])<.02
    assert abs(d['water_residual_kgm2'])<3e-10
