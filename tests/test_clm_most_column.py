"""Actual H/E feedback, tower heights and full-column conservative trials."""
from dataclasses import replace
import numpy as np
import pytest
from noahmp_clm.column import ColumnState,Forcing
from noahmp_clm.clm_state import CLMColumnState
from noahmp_clm.clm_enthalpy_driver import step_clm_enthalpy_column
from noahmp_clm.physics.snowpack import Snowpack,TF
from noahmp_clm.physics.canopy_most import MOSTGeometry
from noahmp_clm.physics.energy import CP,RD


@pytest.mark.parametrize('case',['summer','cold_snow','dew','calm'])
def test_full_column_most_feedback_and_independent_budgets(case):
    air,q,sw,lw,wind,rain=295.,.008,700.,350.,3.,2e-6
    old=ColumnState(ts=292.,tsoil=np.full(4,290.),smc=np.full(4,.3),canw=.0001)
    snow=Snowpack.empty()
    if case=='cold_snow':
        air,q,sw,lw,rain=260.,.0004,0.,220.,2e-6
        old.ts=265.;old.tsoil[:]=268.;snow=Snowpack.from_ice([5.,10.,20.,30.,50.],265.)
    if case=='dew':q=.025;sw=0.;lw=280.;rain=0.
    if case=='calm':wind=0.;rain=0.
    state=CLMColumnState.from_column(old,snow=snow).with_layered_canopy()
    geometry=MOSTGeometry(wind_height=15.,temperature_height=12.,humidity_height=10.,displacement=2.)
    forcing=Forcing(1800.,air,q,wind,101325.,sw,lw,rain)
    initial=state.canopy_condensate.mass.copy()
    final,d=step_clm_enthalpy_column(state,forcing,most_geometry=geometry)
    m=final.ground.water.sum()+final.snow.mass.sum()+final.canopy_condensate.mass.sum()
    heat=final.ground.enthalpy.sum()+final.snow.enthalpy.sum()+final.canopy_condensate.enthalpy.sum()
    vapor=float(d['wet_evaporation'].sum()+d['transpiration'].sum()+d['ground_evaporation'])
    water_error=m-state.water_kgm2-forcing.dt*rain*1000.+forcing.dt*vapor+d['runoff_mm']
    atm=(sw+lw-d['outgoing_shortwave']-d['outgoing_longwave']-d['h']-d['le'])*forcing.dt
    heat_error=heat-state.enthalpy_jm2-d['precipitation_enthalpy_jm2']-atm+d['advected_enthalpy_jm2']+d['runoff_enthalpy_jm2']
    assert abs(water_error)<5e-10
    assert abs(heat_error)<.02
    np.testing.assert_array_equal(state.canopy_condensate.mass,initial)
    # Reconstruct buoyancy directly from independently reported whole-column
    # H/E, not from the iteration's returned target or a latent-energy proxy.
    theta=air+.0098*geometry.temperature_height
    rho=forcing.ps/(RD*air*(1.+(.378/.622)*q))
    buoyancy=9.81*((1.+.61*q)*d['h']/CP+.61*theta*vapor)/(rho*theta*(1.+.61*q))
    most=d['most'];target=-(geometry.wind_height-geometry.displacement)*.4*buoyancy/most['friction_velocity']**3
    zeta=max(-100.,min(2.,target))
    assert most['zeta']==pytest.approx(zeta,rel=2e-13,abs=2e-13)
    assert abs(most['used_zeta']-zeta)<.003
    effective=max(1.,np.sqrt(wind**2+max(0.,buoyancy*1000.)**(2/3)))
    assert abs(most['used_effective_wind']-effective)<.001
    assert most['coefficient_residual']<=1e-4 and 2<=most['iterations']<=50
    assert most['heat_velocity']!=most['water_velocity']
    print('MOST full column',case,dict(iterations=most['iterations'],zeta=zeta,raw_zeta=target,
        coefficient_residual=most['coefficient_residual'],h=d['h'],le=d['le'],gpp=d['gpp'],
        heat_change_wm2=most['heat_change_wm2'],water_residual_kgm2=water_error,heat_residual_jm2=heat_error),flush=True)


def test_extended_exchange_cannot_silently_enter_old_gpu_contract():
    from noahmp_clm.physics.canopy_energy import CanopyExchange
    from noahmp_clm.gpu.canopy_energy import pack_canopy_case
    exchange=CanopyExchange(295.,.008,101325.,.02,.003,.03,290.,15.)
    for changed in (replace(exchange,atmosphere_water_velocity=.03),replace(exchange,potential_temperature=295.2)):
        with pytest.raises(NotImplementedError,match='MOST'):
            pack_canopy_case(np.full(10,.3),700.,350.,changed)


def stable_trajectory_case():
    import json
    from pathlib import Path
    from noahmp_clm.column import SoilParams
    from noahmp_clm.options import PhysicsOptions
    from noahmp_clm.physics.soil_enthalpy import SoilEnthalpyState
    from noahmp_clm.physics.canopy_interception import CanopyCondensate
    data=json.loads((Path(__file__).parent/'fixtures/canopy_most_stable.json').read_text())
    values=data['state'];ground=values['ground'];ground['params']=SoilParams(**ground['params'])
    values['ground']=SoilEnthalpyState(**ground)
    values['snow']=Snowpack(**values['snow']) if values['snow']['mass'] else Snowpack.empty()
    values['options']=PhysicsOptions(**values['options'])
    values['canopy_condensate']=CanopyCondensate(**values['canopy_condensate'])
    return CLMColumnState(**values),Forcing(**data['forcing'])


def test_stable_trajectory_full_state_regression():
    state,forcing=stable_trajectory_case()
    geometry=MOSTGeometry(wind_height=15.,temperature_height=12.,humidity_height=10.,displacement=2.)
    after,d=step_clm_enthalpy_column(state,forcing,cosine_zenith=.05,
        deposition_rates=np.array([1.,2.,1.,.5,.25])*1e-12,most_geometry=geometry)
    water=after.water_kgm2-state.water_kgm2+d['et_mm']+d['runoff_mm']
    heat=after.enthalpy_jm2-state.enthalpy_jm2-d['atmospheric_energy_jm2']+d['advected_enthalpy_jm2']+d['runoff_enthalpy_jm2']
    assert abs(water)<1e-9 and abs(heat)<.02
    assert d['most']['iterations']<30 and d['most']['coefficient_residual']<=1e-4
    print('MOST stable trajectory regression',dict(iterations=d['most']['iterations'],
        used_zeta=d['most']['used_zeta'],h=d['h'],le=d['le'],
        coefficient_residual=d['most']['coefficient_residual'],water_kgm2=water,heat_jm2=heat),flush=True)
