"""Independent analytic reference, crop cycle, burial and non-green stems."""
from dataclasses import replace
import json
import numpy as np
import pytest
from noahmp_clm.physics.vegetation_structure import (
    CanopyStructureParameters,StemAreaScheme,step_stem_area,exposed_canopy_geometry,DAY)


@pytest.mark.parametrize('kind,monthly_loss',[(StemAreaScheme.CN_NONCROP,.5),(StemAreaScheme.CN_GENERIC_CROP,1.)])
@pytest.mark.parametrize('dt',[1800.,3600.,DAY])
@pytest.mark.parametrize('leaf_loss_rate',[0.,.1])
def test_cn_area_matches_independent_continuous_monthly_relaxation(kind,monthly_loss,dt,leaf_loss_rate):
    p=CanopyStructureParameters(kind);sai=2.;lai=6.;days=20.;count=round(days*DAY/dt)
    for _ in range(count):
        next_lai=lai-leaf_loss_rate*dt/DAY
        sai,_=step_stem_area(sai,lai,next_lai,dt,p);lai=next_lai
    # Continuous-rate solution with constant leaf-loss source, independent
    # of the component's timestep recurrence and its branch implementation.
    rate=monthly_loss/30.;decay=np.exp(-rate*days)
    reference=2.*decay+leaf_loss_rate/rate*(1.-decay)
    relative=abs(sai-reference)/reference
    assert relative<=.1
    print('structure analytic reference',json.dumps(dict(scheme=int(kind),dt=dt,leaf_loss_per_day=leaf_loss_rate,
        actual_sai=sai,continuous_reference_sai=float(reference),relative_error_percent=100.*relative)),flush=True)


@pytest.mark.parametrize('kind,minimum',[(StemAreaScheme.CN_NONCROP,.5),(StemAreaScheme.CN_GENERIC_CROP,.05)])
def test_minimum_and_right_endpoint_leaf_fall_impulse(kind,minimum):
    p=CanopyStructureParameters(kind)
    assert step_stem_area(0.,3.,3.,1800.,p)[0]==minimum
    decayed=step_stem_area(2.,3.,3.,1800.,p)[0]
    fallen=step_stem_area(2.,3.,0.,1800.,p)[0]
    assert fallen-decayed==pytest.approx(3.,abs=1e-14)


def test_prescribed_geometry_does_not_depend_on_leaf_senescence():
    p=CanopyStructureParameters();assert step_stem_area(.8,5.,0.,1800.,p)[0]==.8
    assert step_stem_area(.8,5.,0.,1800.,p,prescribed_sai=.6)[0]==.6
    with pytest.raises(ValueError,match='bypass'):
        step_stem_area(.8,5.,0.,1800.,replace(p,scheme=StemAreaScheme.CN_NONCROP),prescribed_sai=.6)


def test_annual_c3_harvest_stubble_and_replanting_cycle():
    p=CanopyStructureParameters(StemAreaScheme.ANNUAL_C3_CROP)
    sai,_=step_stem_area(0.,0.,5.,DAY,p);assert sai==1.
    sai,_=step_stem_area(sai,5.,0.,DAY,p,harvested=True);assert sai==.25
    sai,_=step_stem_area(sai,0.,0.,DAY,p,fallow=True);assert sai==.25
    sai,_=step_stem_area(sai,0.,0.,DAY,p,newly_planted=True);assert sai==0.
    sai,_=step_stem_area(sai,0.,.4,DAY,p);assert sai==pytest.approx(.08)


@pytest.mark.parametrize('depth,fraction',[(0.,1.),(1.,1.),(2.,.5),(3.,0.),(5.,0.)])
def test_tall_canopy_burial_uses_physical_bottom_and_top(depth,fraction):
    p=CanopyStructureParameters(canopy_top_m=3.,canopy_bottom_m=1.)
    d=exposed_canopy_geometry(4.,1.,depth,p)
    assert d['total_lai']==4. and d['total_sai']==1.
    assert d['exposed_lai']==4.*fraction and d['exposed_sai']==fraction
    assert d['exposed_plant_area']==5.*fraction


@pytest.mark.parametrize('depth,fraction',[(0.,1.),(.05,.75),(.1,.5),(.2,0.),(1.,0.)])
def test_short_canopy_burial_uses_independent_twenty_cm_reference(depth,fraction):
    d=exposed_canopy_geometry(0.,1.,depth,CanopyStructureParameters(short_vegetation=True))
    assert d['exposed_lai']==0. and d['exposed_sai']==fraction


def test_non_green_stems_exchange_energy_without_gpp_or_transpiration():
    from noahmp_clm.physics.canopy_energy import CanopyExchange
    from noahmp_clm.physics.canopy_interception import CanopyCondensate
    from noahmp_clm.physics.canopy_phase_energy import solve_canopy_phase_energy
    from noahmp_clm.physics.soil_uptake import SoilWaterSupply
    area=exposed_canopy_geometry(0.,1.,0.)
    exchange=CanopyExchange(285.,.006,101325.,.02,.003,.03,283.,0.)
    result=solve_canopy_phase_energy(np.zeros(10),300.,320.,exchange,CanopyCondensate.empty(),1800.,
        stem_area=np.full(10,area['exposed_sai']/10.),soil_supply=SoilWaterSupply(np.zeros(4),np.zeros(4),0.),
        root_enthalpy=np.zeros(4),fixed_ground_temperature=283.,ground_evaporation_limit=0.,tolerance=1e-7)
    assert result['gpp']==0. and result['leaf_respiration']==0. and np.all(result['transpiration']==0.)
    assert np.max(abs(result['layer_residual']))<=1.01e-7
    assert abs(result['canopy_water_residual'])<1e-12
    print('leafless stem energy',result['h'],result['le'],float(np.max(abs(result['layer_residual']))),flush=True)


@pytest.mark.parametrize('args',[(np.nan,1.,1.,1800.),(-1.,1.,1.,1800.),(1.,-1.,1.,1800.),(1.,1.,-1.,1800.),(1.,1.,1.,0.),(1.,1.,1.,DAY+1.)])
def test_invalid_area_and_time_cannot_enter_geometry_state(args):
    with pytest.raises(ValueError):step_stem_area(*args)
