"""Initially dry leaves admit a mass/enthalpy-consistent mixed condensate."""
import numpy as np
import pytest
import json
from pathlib import Path
from noahmp_clm.physics.canopy_phase_energy import solve_canopy_phase_energy
from noahmp_clm.physics.canopy_energy import CanopyExchange, _saturation, _ice_saturation, _longwave_operator
from noahmp_clm.physics.canopy_interception import CanopyCondensate
from noahmp_clm.physics.soil_uptake import SoilWaterSupply
from noahmp_clm.physics.snowpack import TF,LF


def analytic_mixed_deposition(fraction, layers=1):
    # Prescribe the exact equilibrium T=Tf and liquid fraction. Solve the
    # independent linear air-water balance, then choose downwelling longwave
    # to balance latent heat at the leaf. One layer is the analytic fixture.
    assert layers==1
    dt=1800.;leaf=np.array([1.]);pressure=100000.;qair=.005
    ga,gl,gg=.01,.02,.005
    ql=float(_saturation(np.array([TF]),pressure)[0][0]);qi=float(_ice_saturation(np.array([TF]),pressure)[0][0])
    qs=fraction*ql+(1-fraction)*qi
    qc=(ga*qair+gl*qs+gg*ql)/(ga+gl+gg)
    rho=pressure/(287.*TF*(1+.61*qair))
    amount=rho*gl*(qc-qs)*dt
    latent=-amount/dt*(2.5e6+(1-fraction)*LF)
    lw=_longwave_operator(leaf,.97,1.)
    incoming=(latent-(lw[0,0]+lw[0,1])*5.670374419e-8*TF**4)/lw[0,2]
    exchange=CanopyExchange(TF,qair,pressure,ga,gg,gl,TF,0.,ground_resistance=0.)
    return (leaf,0.,incoming,exchange,CanopyCondensate(np.zeros(1),np.zeros(1)),dt),dict(
        fixed_ground_temperature=TF,soil_supply=SoilWaterSupply(np.zeros(4),np.zeros(4),0.),
        root_enthalpy=np.zeros(4),tolerance=1e-8),dict(mass=amount,enthalpy=amount*fraction*LF,humidity=qc,fraction=fraction)


@pytest.mark.parametrize('fraction',[.1,.5,.9])
def test_mixed_new_condensation_matches_analytic_water_and_heat(fraction):
    args,kw,ref=analytic_mixed_deposition(fraction)
    d=solve_canopy_phase_energy(*args,**kw)
    assert d['leaf_temperature'][0]==TF
    np.testing.assert_allclose(d['condensate'].mass,[ref['mass']],rtol=1e-9,atol=0)
    np.testing.assert_allclose(d['condensate'].enthalpy,[ref['enthalpy']],rtol=1e-8,atol=1e-8)
    assert d['canopy_humidity']==pytest.approx(ref['humidity'],rel=1e-10)
    assert d['condensate'].liquid[0]/d['condensate'].mass[0]==pytest.approx(fraction,abs=1e-8)
    assert abs(d['canopy_water_residual'])<1e-15
    assert max(abs(d['layer_residual']))<=1e-8
    # No initial condensed mass was introduced by changing coordinates.
    assert d['condensate'].mass[0]==pytest.approx(-d['wet_evaporation'][0]*args[-1],abs=1e-15)
    assert abs(d['canopy_storage_change_wm2']+d['wet_advected_heat_wm2'])<1e-12


def test_actual_winter_measurement_height_trial_closes_original_energy():
    from noahmp_clm.physics.snowpack import Snowpack
    from noahmp_clm.physics.photosynthesis import LeafParameters
    raw=json.loads((Path(__file__).parent/'fixtures/qia_winter_dry_deposition.json').read_text())
    kw=raw['kwargs'];kw['soil_supply']=SoilWaterSupply(**{k:np.asarray(v) if isinstance(v,list) else v for k,v in kw['soil_supply'].items()})
    snow={k:np.asarray(v) for k,v in kw['snowpack'].items()}
    snow['impurities']=snow['impurities'].reshape(-1,5)
    kw['snowpack']=Snowpack(**snow)
    kw['root_enthalpy']=np.asarray(kw['root_enthalpy'])
    if kw.get('leaf_parameters') is not None:kw['leaf_parameters']=LeafParameters(**kw['leaf_parameters'])
    d=solve_canopy_phase_energy(np.asarray(raw['leaf']),raw['swdown'],raw['lwdown'],CanopyExchange(**raw['exchange']),
        CanopyCondensate(**{k:np.asarray(v) for k,v in raw['condensate'].items()}),raw['dt'],**kw)
    assert d['leaf_temperature'][1]==TF
    fraction=d['condensate'].liquid[1]/d['condensate'].mass[1]
    assert 0.<fraction<1.
    assert max(abs(d['layer_residual']))<=kw['tolerance']*1.01
    assert abs(d['canopy_water_residual'])<1e-14
    assert abs(d['canopy_storage_change_wm2']+d['wet_advected_heat_wm2'])<1e-12
    print('actual Qia dry deposition',dict(iterations=d['iterations'],fraction=float(fraction),
        heat_residual=float(abs(d['layer_residual']).max()),water_residual=d['canopy_water_residual']),flush=True)
