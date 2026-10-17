"""Layered canopy phase coupled to real snow/soil, with independent budgets."""
import numpy as np
import pytest
from noahmp_clm.column import ColumnState
from noahmp_clm.physics.snowpack import Snowpack,TF
from noahmp_clm.physics.soil_enthalpy import SoilEnthalpyState
from noahmp_clm.physics.canopy_interception import CanopyCondensate
from noahmp_clm.physics.canopy_energy import CanopyExchange
from noahmp_clm.physics.canopy_column import surface_step


@pytest.mark.parametrize('snow_mass,air,humidity,solar,water,old_t',[
    (0.,295.,.008,700.,.02,290.),(0.,268.,.003,0.,.1,265.),
    (20.,260.,.0002,300.,.2,260.),(20.,270.,.004,0.,.1,270.),
    (20.,280.,.004,800.,.1,270.),(.05,285.,.001,900.,.1,270.),
    (.001,295.,.0001,800.,1e-5,280.),(0.,295.,.02,0.,0.,290.),
    (0.,265.,.004,0.,0.,265.),(0.,273.15,.0035,80.,.2,273.15)])
def test_layered_phase_surface_conserves_extensive_stores(snow_mass,air,humidity,solar,water,old_t):
    snow=Snowpack.from_ice([snow_mass],min(air,TF)) if snow_mass else Snowpack.empty()
    soil=SoilEnthalpyState.from_column(ColumnState(tsoil=np.full(4,min(air+2.,290.)),smc=np.full(4,.3)))
    canopy=CanopyCondensate.from_temperature(np.full(10,water),old_t,liquid_fraction=.5 if old_t==TF else None)
    exchange=CanopyExchange(air,humidity,101325.,.02,.004,.03,float(soil.temperature[0]),15.)
    pack,ground,total,d=surface_step(snow,soil,1800.,np.full(10,.3),solar,300.,exchange,
        canopy_condensate=canopy,root_weights=[.6,.25,.1,.05],vegetated_fraction=.8)
    final=d['canopy_condensate'];dt=1800.
    assert total==pytest.approx(final.mass.sum(),abs=1e-14)
    old_mass=snow.mass.sum()+soil.water.sum()+canopy.mass.sum()
    new_mass=pack.mass.sum()+ground.water.sum()+final.mass.sum()
    et=dt*(d['wet_evaporation'].sum()+d['transpiration'].sum()+d['ground_evaporation'])
    water_error=new_mass-old_mass+et+d['runoff_kgm2']
    wet_latent=d['vapor_parcels']['latent_energy_jm2'].sum()
    tr_latent=2.5e6*dt*d['transpiration'].sum()
    ground_latent=d['ground_latent_energy_jm2']
    assert d['le']*dt==pytest.approx(wet_latent+tr_latent+ground_latent,abs=1e-7)
    atm=dt*(solar+300.-d['outgoing_shortwave']-d['outgoing_longwave']-d['h'])-wet_latent-tr_latent-ground_latent
    old_heat=snow.enthalpy.sum()+soil.enthalpy.sum()+canopy.enthalpy.sum()
    new_heat=pack.enthalpy.sum()+ground.enthalpy.sum()+final.enthalpy.sum()
    ground_mass=dt*d['ground_evaporation'];surface=d['surface_temperature']
    if ground_mass>0.:
        ground_adv=ground_mass*(snow.enthalpy[0]/snow.mass[0] if snow_mass else 334000.+4180.*(soil.temperature[0]-TF))
    else:ground_adv=ground_mass*(2100.*(surface-TF) if surface<TF else 334000.+4180.*(surface-TF))
    wet_adv=(d['thermal_condensate'].enthalpy-final.enthalpy).sum()
    transp_adv=dt*float(d['transpiration']@(334000.+4180.*(d['leaf_temperature']-TF)))
    advected=ground_adv+wet_adv+transp_adv
    assert d['advected_enthalpy_jm2']==pytest.approx(advected,abs=1e-7)
    energy_error=new_heat-old_heat-atm+advected+d['runoff_enthalpy_jm2']
    assert abs(water_error)<2e-10
    assert abs(energy_error)<.01
    assert abs(d['water_residual_kgm2']-water_error)<2e-12
    assert abs(d['energy_residual_jm2']-energy_error)<1e-7
    assert d['transpiration_advected_heat_wm2']==pytest.approx(float(d['transpiration']@(334000.+4180.*(d['leaf_temperature']-273.15))),abs=1e-12)
    assert d['surface_evaluations']<45
    print('layered surface',snow_mass,air,dict(water=water_error,heat=energy_error,evaluations=d['surface_evaluations'],canopy_ice=final.ice.sum(),canopy_liquid=final.liquid.sum()),flush=True)


def test_surface_rejects_ambiguous_canopy_stores():
    soil=SoilEnthalpyState.from_column(ColumnState());e=CanopyExchange(285.,.005,101325.,.02,.003,.03,285.,15.)
    with pytest.raises(ValueError,match='not both'):
        surface_step(Snowpack.empty(),soil,1800.,np.full(10,.3),300.,300.,e,canopy_water_kgm2=.1,canopy_condensate=CanopyCondensate.empty())


def test_phase_edge_backtracking_reaches_conservative_surface_root():
    import json
    from pathlib import Path
    from test_gpu_canopy_phase_surface import reference
    data=json.loads(Path(__file__).with_name('fixtures').joinpath('canopy_phase_surface_regression.json').read_text())
    row=tuple(np.asarray(x,np.int32 if k==3 else float) for k,x in enumerate(data))
    from clm_soil_oracle import fixture_suction
    row=(*row[:2],fixture_suction(row[2],.355),*row[3:])
    result=reference(row)
    assert abs(result['energy_residual_jm2'])<.01
    assert abs(result['water_residual_kgm2'])<2e-10
    assert np.max(abs(result['layer_residual']))<1.05e-7
