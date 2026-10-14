"""Precipitation-to-drainage column with authoritative canopy layer M/H."""
from dataclasses import replace
import numpy as np
import pytest
from noahmp_clm.column import ColumnState,Forcing
from noahmp_clm.clm_state import CLMColumnState
from noahmp_clm.clm_driver import step_clm_column
from noahmp_clm.physics.snowpack import Snowpack,TF


@pytest.mark.parametrize('case',['summer','rain','snow','melt','frost','dew','drip','bare','five_layers','drought'])
def test_layered_column_routes_all_canopy_phase_outlets(case):
    air,qa,sw,lw,precip,water=295.,.008,700.,350.,0.,.3
    old=ColumnState(ts=295.,tsoil=np.full(4,290.),smc=np.full(4,.3));snow=Snowpack.empty()
    if case=='rain':precip=5e-6;old.smc[:]=.44
    if case in ('snow','frost'):
        air,qa,sw,lw=265.,.002,0.,240.;old.ts=265.;old.tsoil[:]=268.;precip=2e-6 if case=='snow' else 0.
    if case=='melt':snow=Snowpack.from_ice([.1],TF)
    if case=='five_layers':
        snow=Snowpack.from_ice([10.,20.,30.,50.,80.],265.,300.);air,qa,sw,lw=268.,.001,300.,250.;old.ts=265.;old.tsoil[:]=270.
    if case=='dew':qa=.025;sw=0.;lw=270.;water=0.
    if case=='drip':water=3.
    if case=='bare':old.fveg=0.;water=2.
    if case=='drought':old.smc[:]=.08;water=0.
    old.canw=water/1000.;state=CLMColumnState.from_column(old,snow=snow).with_layered_canopy()
    initial=state.canopy_condensate.mass.copy();initial_h=state.canopy_condensate.enthalpy.copy()
    final,d=step_clm_column(state,Forcing(1800.,air,qa,3.,101325.,sw,lw,precip))
    m=final.ground.water.sum()+final.snow.mass.sum()+final.canopy_condensate.mass.sum()
    h=final.ground.enthalpy.sum()+final.snow.enthalpy.sum()+final.canopy_condensate.enthalpy.sum()
    water_error=m-state.water_kgm2-d['prcp_mm']+d['et_mm']+d['runoff_mm']
    heat_error=h-state.enthalpy_jm2-d['precipitation_enthalpy_jm2']-d['atmospheric_energy_jm2']+d['advected_enthalpy_jm2']+d['runoff_enthalpy_jm2']
    assert abs(water_error)<3e-10
    assert abs(heat_error)<.02
    np.testing.assert_array_equal(state.canopy_condensate.mass,initial);np.testing.assert_array_equal(state.canopy_condensate.enthalpy,initial_h)
    assert final.canopy_water_kgm2==float(final.canopy_condensate.mass.sum())
    assert final.canopy_enthalpy_jm2==float(final.canopy_condensate.enthalpy.sum())
    assert final.surface_temperature==float(final.snow.temperature[0] if len(final.snow.mass) else final.ground.temperature[0])
    assert np.all(final.canopy_condensate.liquid<=.1*final.fveg*final.lai/10.+1e-12)
    assert np.all(final.canopy_condensate.ice<=6.*final.fveg*final.lai/10.+1e-12)
    if case=='snow':assert final.snow.mass.sum()>0. and final.canopy_condensate.ice.sum()>0.
    if case=='bare':assert final.canopy_water_kgm2==0.
    if case=='drought':assert d['root_uptake'].sum()==0.
    print(case,dict(water=water_error,heat=heat_error,ice=final.canopy_condensate.ice.sum(),liquid=final.canopy_condensate.liquid.sum(),runoff=d['runoff_mm']),flush=True)


def test_layered_state_preserves_energy_and_cannot_use_aggregate_gpu_abi():
    from noahmp_clm.gpu.clm_enthalpy_column import pack_clm_enthalpy_columns
    old=ColumnState(canw=.001,ts=260.);base=CLMColumnState.from_column(old);state=base.with_layered_canopy()
    assert state.water_kgm2==pytest.approx(base.water_kgm2,abs=1e-12)
    assert state.enthalpy_jm2==pytest.approx(base.enthalpy_jm2,abs=1e-7)
    assert state.with_layered_canopy() is state
    with pytest.raises(NotImplementedError,match='phase profile'):pack_clm_enthalpy_columns([state])
    with pytest.raises(ValueError,match='remapping'):state.with_layered_canopy(5)
