"""CLM5 exposed wet-area bound, storage ownership and vapor-law reference."""
import numpy as np
import pytest
from noahmp_clm.physics.canopy_interception import CanopyCondensate,coverage
from noahmp_clm.physics.canopy_energy import CanopyExchange
from noahmp_clm.physics.canopy_phase_energy import solve_canopy_phase_energy
from noahmp_clm.physics.soil_uptake import SoilWaterSupply


@pytest.mark.parametrize('temperature',[260.,273.15,290.])
@pytest.mark.parametrize('unlimited,expected',[(0.,0.),(.001,.001),(.01,.01),(.05,.05),(.2,.05),(1.,.05)])
def test_exposed_area_limit_leaves_stored_mass_heat_and_snow_coverage_owned(temperature,unlimited,expected):
    # Prescribe covered surface area first, then construct the corresponding
    # store. The official0.05 input is independent of production constants.
    leaves=np.full(10,.3);stems=np.full(10,.05);mass=np.full(10,.035*unlimited**1.5)
    old=CanopyCondensate.from_temperature(mass,temperature,liquid_fraction=.5 if temperature==273.15 else None)
    before=(old.mass.tobytes(),old.enthalpy.tobytes());f=coverage(old,leaves,stem_area=stems)
    np.testing.assert_allclose(f['wet_fraction'],expected,rtol=0.,atol=2e-16)
    np.testing.assert_allclose(f['dry_fraction'],(1.-expected)*6./7.,rtol=0.,atol=3e-16)
    assert before==(old.mass.tobytes(),old.enthalpy.tobytes())
    if temperature==290.:np.testing.assert_array_equal(f['snow_fraction'],0.)
    elif unlimited>0.:assert np.all(f['snow_fraction']>0.)


def test_wet_canopy_keeps_transpiring_leaf_area_and_matches_independent_vapor_law():
    leaf=np.full(10,.3);stem=np.full(10,.05);fv=.8;dt=300.
    old=CanopyCondensate.from_temperature(np.full(10,.1*fv*.35),295.)
    exchange=CanopyExchange(300.,.006,101325.,.02,.003,.02,295.,10.)
    supply=SoilWaterSupply(np.full(4,.01),np.full(4,.25),.01)
    out=solve_canopy_phase_energy(leaf,500.,370.,exchange,old,dt,stem_area=stem,
        vegetated_fraction=fv,soil_supply=supply,root_enthalpy=np.full(4,334000.+4180.*(295.-273.15)),
        fixed_ground_temperature=298.,ground_shares_soil=True)
    assert np.all(out['leaf_temperature']>273.15) and np.all(out['condensate'].mass>0.)
    assert out['transpiration'].sum()>0. and np.all(out['wet_evaporation']>0.)
    # Buck's independent liquid saturation-pressure relation. Its small
    # difference from the production vapor-pressure law is included in1%.
    tc=out['leaf_temperature']-273.15
    vapor_pressure=611.21*np.exp((18.678-tc/234.5)*tc/(257.14+tc))
    qsat=.622*vapor_pressure/(exchange.pressure-.378*vapor_pressure)
    density=exchange.pressure/(287.*exchange.air_temperature*(1.+.61*exchange.specific_humidity))
    expected=density*fv*(leaf+stem)*exchange.leaf_velocity*.05*(qsat-out['canopy_humidity'])
    np.testing.assert_allclose(out['wet_evaporation'],expected,rtol=.01,atol=1e-12)
    np.testing.assert_allclose(out['condensate'].mass+dt*out['wet_evaporation'],old.mass,rtol=0.,atol=1e-14)
    energy=500.+370.-out['outgoing_shortwave']-out['outgoing_longwave']-out['h']-out['le']-out['ground_heat']
    energy-=out['canopy_storage_change_wm2']+out['wet_advected_heat_wm2']+out['transpiration_advected_heat_wm2']-out['root_advected_heat_wm2']
    assert abs(energy)<1e-5 and abs(out['air_water_residual'])<1e-11
    print('CLM5 wet area independent vapor law',dict(maximum_relative_flux_difference=float(abs(out['wet_evaporation']-expected).max()/abs(expected).max()),energy_residual_wm2=float(energy),water_residual_kgm2=float(out['canopy_water_residual'])),flush=True)
