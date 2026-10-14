"""Seasonal leaf display must affect the real column without deleting M/H."""
from dataclasses import replace
import numpy as np
import pytest
from noahmp_clm.clm_state import CLMColumnState
from noahmp_clm.column import ColumnState,Forcing
from noahmp_clm.clm_enthalpy_driver import step_clm_enthalpy_column
from noahmp_clm.physics.phenology import (PhenologyState,PhenologyParameters,PhenologyEnvironment,
    PhenologyPhase as Phase,step_phenology)


def case(phase,temperature=285.,snow=0.):
    old=ColumnState(ts=temperature,tsoil=np.full(4,temperature),smc=np.full(4,.3),
        canw=.001,swe=snow,lai=3.,fveg=.8)
    state=CLMColumnState.from_column(old).with_layered_canopy()
    p=PhenologyParameters(onset_days=1.,offset_days=1.)
    initial={Phase.DORMANT:PhenologyState(gdd=399.99),
        Phase.ONSET:PhenologyState(0.,.5,43200.,Phase.ONSET,False),
        Phase.ACTIVE:PhenologyState(0.,1.,0.,Phase.ACTIVE,False),
        Phase.OFFSET:PhenologyState(0.,1./48.,1800.,Phase.OFFSET,False)}[phase]
    env=PhenologyEnvironment(45000. if phase in (Phase.DORMANT,Phase.ONSET) else 36000.,
        phase in (Phase.DORMANT,Phase.ONSET),False,45.)
    return state.with_phenology(p,initial),Forcing(1800.,280.,.006,2.5,101325.,350.,320.,0.),env


@pytest.mark.parametrize('phase',list(Phase))
@pytest.mark.parametrize('temperature,snow',[(285.,0.),(270.,.01)])
def test_full_phenology_matches_explicit_leaf_area_and_conserves(phase,temperature,snow):
    old,f,e=case(phase,temperature,snow)
    next_state,expected=step_phenology(old.phenology_state,old.tsoil[2],e.daylength_seconds,
        e.increasing_daylength,e.winter_solstice,f.dt,old.phenology_parameters)
    fixed=replace(old,options=replace(old.options,OptDynamicVeg=4),phenology_state=None,
        phenology_parameters=None,lai=expected['lai'])
    manual,md=step_clm_enthalpy_column(fixed,f)
    after,d=step_clm_enthalpy_column(old,f,phenology_environment=e)
    assert after.phenology_state==next_state and after.lai==expected['lai']
    np.testing.assert_array_equal(after.ground.water,manual.ground.water)
    np.testing.assert_array_equal(after.ground.enthalpy,manual.ground.enthalpy)
    np.testing.assert_array_equal(after.canopy_condensate.mass,manual.canopy_condensate.mass)
    for key in ('gpp','h','le','runoff_mm'):assert d[key]==md[key]
    water=after.water_kgm2-old.water_kgm2-f.prcp*f.dt*1000.+d['et_mm']+d['runoff_mm']
    heat=after.enthalpy_jm2-old.enthalpy_jm2-d['precipitation_enthalpy_jm2']+d['advected_enthalpy_jm2']+d['runoff_enthalpy_jm2']
    heat-=(f.swdown+f.lwdown-d['outgoing_shortwave']-d['outgoing_longwave']-d['h']-d['le'])*f.dt
    assert abs(water)<1e-9 and abs(heat)<.02
    if phase==Phase.OFFSET:
        assert after.lai==0. and after.canopy_water_kgm2==0.
        flow=d['canopy_interception']
        assert flow['ground_liquid_kgm2']+flow['ground_ice_kgm2']==pytest.approx(old.canopy_water_kgm2,abs=1e-12)
        assert flow['ground_liquid_enthalpy_jm2']+flow['ground_ice_enthalpy_jm2']==pytest.approx(old.canopy_enthalpy_jm2,abs=1e-7)
        assert d['gpp']==0.
    print('Phenology full NumPy',int(phase),temperature,dict(lai_before=old.lai,lai=after.lai,
        water_kgm2=water,heat_jm2=heat,gpp=d['gpp']),flush=True)


def test_dynamic_vegetation_requires_explicit_state_and_calendar():
    s,f,e=case(Phase.DORMANT)
    with pytest.raises(ValueError,match='PhenologyEnvironment'):step_clm_enthalpy_column(s,f)
    with pytest.raises(ValueError,match='third soil'):replace(s,phenology_state=None)
    fixed=replace(s,options=replace(s.options,OptDynamicVeg=4),phenology_state=None,phenology_parameters=None)
    with pytest.raises(ValueError,match='OptDynamicVeg=5'):step_clm_enthalpy_column(fixed,f,phenology_environment=e)
    with pytest.raises(ValueError,match='extratropical'):replace(e,latitude_degrees=0.)
    # Southern sites use their own explicit trend, not a hard-coded calendar.
    south=replace(e,latitude_degrees=-45.)
    a,_=step_clm_enthalpy_column(s,f,phenology_environment=e)
    b,_=step_clm_enthalpy_column(s,f,phenology_environment=south)
    assert a.phenology_state==b.phenology_state


def test_legacy_drivers_cannot_silently_hold_dynamic_lai_fixed():
    from noahmp_clm.driver import step_column
    from noahmp_clm.clm_driver import step_clm_column
    old=ColumnState();old.options=replace(old.options,OptDynamicVeg=5)
    _,f,_=case(Phase.DORMANT)
    for driver in (step_column,step_clm_column):
        with pytest.raises(NotImplementedError,match='Dynamic vegetation'):driver(old,f)
