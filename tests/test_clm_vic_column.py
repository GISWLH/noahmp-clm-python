"""VIC must alter admission and bottom storage while retaining water and heat."""
from dataclasses import replace
import numpy as np
import pytest
from noahmp_clm.column import ColumnState,Forcing
from noahmp_clm.clm_state import CLMColumnState
from noahmp_clm.clm_enthalpy_driver import step_clm_enthalpy_column
from noahmp_clm.physics.vic import VICParameters


def case(temperature=280.,snow=0.,options=(6,6)):
    old=ColumnState(ts=temperature,tsoil=np.full(4,temperature),smc=np.full(4,.35),
                    swe=snow,lai=1.8,fveg=.7)
    old.options=replace(old.options,OptRunoffSurface=options[0],OptRunoffSubsurface=options[1],
                        OptSoilPermeabilityFrozen=2)
    state=CLMColumnState.from_column(old).with_layered_canopy()
    forcing=Forcing(1800.,280.,.006,2.5,101325.,350.,320.,8e-6)
    return state,forcing


@pytest.mark.parametrize('temperature,snow',[(280.,0.),(272.,0.),(270.,.01),(273.15,.003)])
@pytest.mark.parametrize('options',[(6,6),(6,3),(3,6)])
def test_vic_column_physical_outlets_and_independent_conservation(temperature,snow,options):
    old,f=case(temperature,snow,options)
    after,d=step_clm_enthalpy_column(old,f)
    water=after.water_kgm2-old.water_kgm2-f.prcp*f.dt*1000.+d['et_mm']+d['runoff_mm']
    heat=after.enthalpy_jm2-old.enthalpy_jm2-d['precipitation_enthalpy_jm2']+d['advected_enthalpy_jm2']+d['runoff_enthalpy_jm2']
    heat-=(f.swdown+f.lwdown-d['outgoing_shortwave']-d['outgoing_longwave']-d['h']-d['le'])*f.dt
    assert abs(water)<1e-9 and abs(heat)<.02
    assert d['runoff_deep_mm']==d['bottom_outflow_kgm2']
    assert d['runoff_mm']==pytest.approx(d['runoff_surface_mm']+d['runoff_deep_mm'],abs=1e-12)
    assert d['runoff_surface_mm']>=0. and d['runoff_deep_mm']>=0.
    print('VIC column',temperature,snow,options,dict(surface_kgm2=d['runoff_surface_mm'],
        bottom_kgm2=d['runoff_deep_mm'],water_kgm2=water,heat_jm2=heat),flush=True)


def test_vic_changes_actual_upper_and_bottom_reservoirs():
    state,f=case()
    p=VICParameters(infiltration_shape=0.,maximum_baseflow_mm_day=0.)
    uniform,a=step_clm_enthalpy_column(state,f,vic_parameters=p)
    variable,b=step_clm_enthalpy_column(state,f,vic_parameters=replace(p,infiltration_shape=.4))
    drained,c=step_clm_enthalpy_column(state,f,vic_parameters=replace(p,maximum_baseflow_mm_day=50.))
    assert b['runoff_surface_mm']>a['runoff_surface_mm']+1.
    assert variable.ground.water[:2].sum()<uniform.ground.water[:2].sum()
    assert uniform.water_kgm2-variable.water_kgm2==pytest.approx(
        b['runoff_mm']-a['runoff_mm']+b['et_mm']-a['et_mm'],abs=1e-9)
    assert c['runoff_deep_mm']>0. and a['runoff_deep_mm']==0.
    np.testing.assert_allclose(uniform.ground.water[:3],drained.ground.water[:3],atol=1e-12)
    assert uniform.ground.water[-1]-drained.ground.water[-1]==pytest.approx(c['runoff_deep_mm'],abs=1e-12)
    assert uniform.ground.enthalpy[-1]-drained.ground.enthalpy[-1]==pytest.approx(c['bottom_outflow_enthalpy_jm2'],abs=1e-7)


def test_vic_configuration_must_be_used_and_mixed_legacy_routing_is_explicit():
    state,f=case(options=(3,3))
    with pytest.raises(ValueError,match='require runoff option 6'):
        step_clm_enthalpy_column(state,f,vic_parameters=VICParameters())
    state,f=case(options=(6,1))
    with pytest.raises(NotImplementedError,match='combinations'):
        step_clm_enthalpy_column(state,f)
