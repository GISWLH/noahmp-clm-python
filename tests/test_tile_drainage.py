"""Independent drainage arithmetic, exact reservoirs and actual soil heat."""
from dataclasses import replace
from decimal import Decimal,localcontext
import numpy as np
import pytest
from noahmp_clm.physics.tile_drainage import TileDrainageParameters,tile_drainage_demand,step_tile_drainage
from noahmp_clm.physics.soil_enthalpy import SoilEnthalpyState
from noahmp_clm.column import ColumnState


def ground(theta=.4,t=290.):
    return SoilEnthalpyState.from_column(ColumnState(tsoil=np.full(4,t),smc=np.full(4,theta)))


def test_decimal_independent_selected_reservoir_and_rate_units():
    errors=[];count=0
    with localcontext() as ctx:
        ctx.prec=45;D=Decimal
        for index,depth in enumerate((.05,.2,.8,1.5)):
            for theta in (.05,.28,.4):
                for dt in (60.,1800.,86400.):
                    for rate in (0.,10.,1000.):
                        s=ground(theta);p=TileDrainageParameters(depth,.28,rate)
                        demand,d=tile_drainage_demand(s.water,np.zeros(4),s.dz,np.full(4,290.),dt,p)
                        excess=max(D(0),(D(str(theta))-D('.28'))*D(str(s.dz[index]))*1000)
                        expected=float(min(excess,D(str(rate))*D(str(dt))/86400))
                        errors.append(abs(d['drained_kgm2']-expected));count+=1
                        assert d['layer_index']==index
                        assert np.count_nonzero(demand)<=1
                        np.testing.assert_allclose(demand.sum(),expected,rtol=0,atol=1e-12)
    print('tile Decimal cases/max kg/m2',count,max(errors),flush=True)


def test_daily_vs_half_hour_actual_soil_drains_to_field_capacity_with_heat():
    initial=ground();a=b=initial;p=TileDrainageParameters(.8,.28,10.);totals=np.zeros(2)
    max_budget=np.zeros(2)
    for _ in range(10):
        a,d=step_tile_drainage(a,86400.,p);totals+=[d['drained_kgm2'],d['drained_enthalpy_jm2']]
        max_budget=np.maximum(max_budget,abs(np.array([d['water_residual_kgm2'],d['heat_residual_jm2']])))
        for __ in range(48):b,_=step_tile_drainage(b,1800.,p)
    np.testing.assert_allclose(a.water,b.water,rtol=0,atol=1e-10)
    np.testing.assert_allclose(a.enthalpy,b.enthalpy,rtol=0,atol=1e-5)
    assert totals[0]==pytest.approx(72.,abs=1e-10)
    assert a.water[2]==pytest.approx(.28*1000.*.6,abs=1e-12)
    np.testing.assert_allclose(a.temperature,initial.temperature,rtol=0,atol=1e-10)
    residual=np.array([a.water.sum()-initial.water.sum(),a.enthalpy.sum()-initial.enthalpy.sum()])+totals
    assert abs(residual[0])<1e-10 and abs(residual[1])<1e-6
    print('tile 10 days mass/heat/step budgets/cumulative',totals.tolist(),max_budget.tolist(),residual.tolist(),flush=True)


def test_frozen_drain_is_closed_and_existing_water_heat_preserved():
    s=ground(t=270.);out,d=step_tile_drainage(s,86400.)
    assert not d['eligible'] and d['drained_kgm2']==d['drained_enthalpy_jm2']==0.
    np.testing.assert_array_equal(out.water,s.water);np.testing.assert_array_equal(out.enthalpy,s.enthalpy)


def test_tube_interface_uses_shallower_layer_and_below_field_capacity_stays_dry():
    s=ground(.2)
    for depth,expected in ((.1,0),(.4,1),(1.,2),(2.,3)):
        out,d=step_tile_drainage(s,1800.,TileDrainageParameters(depth))
        assert d['layer_index']==expected and d['drained_kgm2']==0.
        np.testing.assert_array_equal(out.water,s.water)


def test_reject_unmodeled_depth_invalid_rate_and_field_capacity():
    s=ground()
    with pytest.raises(ValueError):step_tile_drainage(s,1800.,TileDrainageParameters(2.1))
    with pytest.raises(ValueError):step_tile_drainage(s,0.)
    with pytest.raises(ValueError):step_tile_drainage(s,1800.,TileDrainageParameters(field_capacity=.9))
    with pytest.raises(ValueError):replace(TileDrainageParameters(),maximum_mm_day=-1.)
