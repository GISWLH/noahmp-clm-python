"""Independent deficit arithmetic, application integration and physical M/H."""
from dataclasses import replace
from decimal import Decimal,localcontext
import numpy as np
import pytest
from noahmp_clm.physics.irrigation import IrrigationParameters,IrrigationState,step_irrigation
from noahmp_clm.physics.soil_enthalpy import SoilEnthalpyState,infiltrate
from noahmp_clm.column import ColumnState


def soil(theta=.2,temperature=285.):
    return SoilEnthalpyState.from_column(ColumnState(tsoil=np.full(4,temperature),smc=np.full(4,theta)))


def test_independent_decimal_root_zone_deficits_and_partial_layers():
    errors=[]
    with localcontext() as context:
        context.prec=45
        for depth in (.05,.1,.25,.6,2.,3.):
            for theta in (.01,.12,.2,.28,.4):
                for fraction in (.25,.5,1.):
                    s=soil(theta);p=IrrigationParameters(root_zone_depth=depth,threshold_fraction=fraction)
                    _,d=step_irrigation(IrrigationState(),s.liquid,s.ice,s.dz,1800.,3.,check=True,parameters=p)
                    D=Decimal;root=min(D(str(depth)),D(2))
                    expected=max(D(0),(D(str(fraction))*(D('.28')-D('.12'))+D('.12')-D(str(theta)))*root*1000)
                    errors.append(abs(d['requested_kgm2']-float(expected)))
    assert max(errors)<1e-12
    print('irrigation Decimal deficit max kg/m2',max(errors),flush=True)


def test_six_hour_application_has_no_invented_water_and_closes_real_soil_heat():
    initial=ground=soil();request=IrrigationState();applied=heat=runoff=runoff_heat=0.;last=None
    for step in range(12):
        request,d=step_irrigation(request,ground.liquid,ground.ice,ground.dz,1800.,3.,check=step==0,
            supply_kgm2=100.,supply_temperature_k=290.)
        assert d['applied_kgm2']==pytest.approx(4.,abs=2e-14)
        ground,r,h=infiltrate(ground,d['applied_kgm2'],d['applied_enthalpy_jm2'])
        applied+=d['applied_kgm2'];heat+=d['applied_enthalpy_jm2'];runoff+=r;runoff_heat+=h;last=d
    assert request==IrrigationState() and applied==pytest.approx(48.,abs=1e-12)
    water=float(ground.water.sum()-initial.water.sum())+runoff-applied
    energy=float(ground.enthalpy.sum()-initial.enthalpy.sum())+runoff_heat-heat
    assert abs(water)<1e-10 and abs(energy)<1e-7
    assert last['unmet_kgm2']==0.
    print('irrigation integrated applied/residual M/H',applied,water,energy,flush=True)


def test_supply_shortage_is_diagnosed_and_request_is_not_a_water_store():
    ground=soil();request=IrrigationState();actual=unmet=0.
    for k in range(12):
        request,d=step_irrigation(request,ground.liquid,ground.ice,ground.dz,1800.,3.,check=k==0,supply_kgm2=1.)
        actual+=d['applied_kgm2'];unmet+=d['unmet_kgm2']
        assert abs(d['request_residual_kgm2'])<=d['request_bound_kgm2']
    assert request==IrrigationState() and actual==12. and unmet==pytest.approx(36.)


@pytest.mark.parametrize('temperature,lai',[(270.,3.),(285.,0.)])
def test_frozen_or_harvested_crop_cancels_only_unfilled_request(temperature,lai):
    ground=soil(temperature=temperature)
    after,d=step_irrigation(IrrigationState(20.,3600.),ground.liquid,ground.ice,ground.dz,1800.,lai,supply_kgm2=100.)
    assert after==IrrigationState() and d['cancelled_kgm2']==20. and d['applied_kgm2']==d['applied_enthalpy_jm2']==0.
    assert d['request_residual_kgm2']==0.


def test_no_check_does_not_automatically_irrigate_and_invalid_events_reject():
    ground=soil();args=(ground.liquid,ground.ice,ground.dz,1800.,3.)
    _,d=step_irrigation(IrrigationState(),*args,supply_kgm2=100.)
    assert d['deficit_kgm2']==pytest.approx(48.) and d['applied_kgm2']==0.
    with pytest.raises(ValueError,match='overwrite'):step_irrigation(IrrigationState(10.,3600.),*args,check=True)
    with pytest.raises(ValueError,match='Invalid'):step_irrigation(IrrigationState(),*args,check=.5)
    with pytest.raises(ValueError,match='Invalid'):step_irrigation(IrrigationState(),*args,supply_temperature_k=270.)
    with pytest.raises(ValueError,match='pending'):IrrigationState(10.,0.)
    with pytest.raises(ValueError,match='Invalid'):replace(IrrigationParameters(),target_theta=.1)
