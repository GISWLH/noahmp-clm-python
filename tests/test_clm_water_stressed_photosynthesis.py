"""Pinned, independently generated Decimal80 soil-beta equations (CTSM CLM5)."""
import json
from pathlib import Path

import numpy as np
import pytest

from noahmp_clm.physics.photosynthesis import (
    ball_berry_conductance, farquhar_rates, leaf_gas_exchange)


def oracle():
    return json.loads((Path(__file__).parent/'fixtures/leaf_clm5_water_stress.json').read_text())


def test_ball_berry_all675_independent_decimal80_cases():
    rows=oracle()['ball_berry']
    result=ball_berry_conductance(*[np.asarray([q[k] for q in rows])
        for k in ['net','cs','rh','gb']],water_stress=np.asarray([q['beta'] for q in rows]))
    expected=np.asarray([float(q['reference_decimal']) for q in rows])
    np.testing.assert_allclose(result,expected,rtol=2e-12,atol=1e-14)


def test_coupled_all540_independent_decimal80_cases():
    rows=oracle()['leaf']
    result=leaf_gas_exchange(np.asarray([q['par'] for q in rows]),298.15,
        co2_ppm=np.asarray([q['ca'] for q in rows]),relative_humidity=np.asarray([q['rh'] for q in rows]),
        boundary_conductance=np.asarray([q['gb'] for q in rows]),water_stress=np.asarray([q['beta'] for q in rows]),nitrogen_factor=1.)
    for name in ['ci_ppm','gs_mol','gross','net','respiration','rubisco','rubp','tpu']:
        expected=np.asarray([float(q['decimal80'][name]) for q in rows])
        np.testing.assert_allclose(result[name],expected,rtol=2e-12,atol=1e-10 if name=='ci_ppm' else 1e-12,err_msg=name)
    assert np.max(abs(result['diffusion_residual']))<1e-10


def test_water_and_nitrogen_are_independent_at_three_temperatures():
    temperature,par,nitrogen,water=np.meshgrid([280.,298.15,315.],[0.,50.,1500.],[0.,.2,1.],[0.,.1,.7,1.],indexing='ij')
    r=leaf_gas_exchange(par,temperature,nitrogen_factor=nitrogen,water_stress=water)
    base=farquhar_rates(r['ci_ppm'],par,temperature,nitrogen_factor=nitrogen)
    expected=dict(rubisco=base['rubisco']*water,rubp=base['rubp'],tpu=base['tpu'],respiration=base['respiration']*water)
    expected['gross']=np.minimum(np.minimum(expected['rubisco'],expected['rubp']),expected['tpu'])
    expected['net']=expected['gross']-expected['respiration']
    for key,value in expected.items():
        np.testing.assert_allclose(r[key],value,rtol=2e-12,atol=1e-12,err_msg=key)
    assert np.max(abs(r['diffusion_residual']))<1e-9


def test_zero_light_uses_stressed_minimum_and_respiration():
    stress=np.asarray([0.,1e-8,.01,.5,1.])
    r=leaf_gas_exchange(np.zeros(5),298.15,water_stress=stress,boundary_conductance=.5)
    minimum=np.maximum(.01*stress,1e-6)
    np.testing.assert_allclose(r['gs_mol'],minimum,rtol=1e-12,atol=1e-15)
    np.testing.assert_allclose(r['net'],-.9*stress,rtol=1e-12,atol=1e-15)
    np.testing.assert_allclose(r['ci_ppm'],420.+.9*stress*(1.4/.5+1.6/minimum),rtol=1e-12,atol=1e-10)
    assert np.max(abs(r['diffusion_residual']))<1e-10


@pytest.mark.parametrize('phase',[False,True])
@pytest.mark.parametrize('rooted',[False,True])
def test_energy_callers_keep_nitrogen_separate_and_disable_inaccessible_roots(phase,rooted):
    from noahmp_clm.physics.canopy_energy import CanopyExchange,solve_canopy_energy
    from noahmp_clm.physics.canopy_phase_energy import solve_canopy_phase_energy
    from noahmp_clm.physics.canopy_interception import CanopyCondensate
    from noahmp_clm.physics.canopy_radiation import canopy_shortwave
    from noahmp_clm.physics.soil_uptake import SoilWaterSupply
    area=np.full(10,.3)
    e=CanopyExchange(298.15,.009,101325.,.02,.003,.03,296.,15.)
    supply=SoilWaterSupply(np.full(4,1e-4 if rooted else 0.),np.full(4,.25),4e-4)
    kwargs=dict(soil_supply=supply,nitrogen_factor=.6,water_stress=.35)
    if phase:
        result=solve_canopy_phase_energy(area,550.,350.,e,CanopyCondensate.empty(),1800.,
            root_enthalpy=np.full(4,334000.+4180.*(296.-273.15)),**kwargs)
    else:
        result=solve_canopy_energy(area,550.,350.,e,**kwargs)
    t=result['leaf_temperature'];q=result['canopy_humidity']
    es=611.2*np.exp(17.67*(t-273.15)/(t-29.65))
    gb=e.leaf_velocity*e.pressure/(8.314462618*e.air_temperature)
    gas=result['gas_exchange']
    # The liquid solver updates q after its last gas evaluation. Recover the
    # humidity actually used from the independent boundary mixing identity;
    # also bound that lag by the solver's original reported q correction.
    used_rh=(gas['surface_relative_humidity']*(gb+gas['gs_mol'])-gas['gs_mol'])/gb
    final_rh=np.minimum(q*e.pressure/(.622+.378*q)/es,1.)
    lag=result.get('canopy_humidity_change',0.)
    assert np.all(abs(used_rh-final_rh)<=e.pressure/(.622*es)*lag+2e-14)
    expected=leaf_gas_exchange(canopy_shortwave(area,550.)['absorbed_par'],t,
        relative_humidity=used_rh,boundary_conductance=gb,
        nitrogen_factor=.6,water_stress=.35 if rooted else 0.)
    for key in ['gross','net','respiration','gs_mol']:
        np.testing.assert_allclose(result['gas_exchange'][key],expected[key],rtol=2e-10,atol=1e-11,err_msg=key)
    if not rooted:
        np.testing.assert_array_equal(result['gas_exchange']['gross'],np.zeros(10))
        np.testing.assert_array_equal(result['gas_exchange']['respiration'],np.zeros(10))
