"""Surface irrigation must enter actual snow/soil with external M/H ledgers."""
from dataclasses import replace
import numpy as np
import pytest
from noahmp_clm.clm_driver import step_clm_column
from noahmp_clm.clm_state import CLMColumnState
from noahmp_clm.column import ColumnState,Forcing
from noahmp_clm.physics.irrigation import IrrigationState,IrrigationInput
from noahmp_clm.physics.crop import CropManagement
from noahmp_clm.physics.canopy_interception import CanopyCondensate
from noahmp_clm.physics.soil_enthalpy import SoilEnthalpyState,infiltrate
from noahmp_clm.physics.snowpack import Snowpack
from test_clm_crop_column import case as crop_case
from test_clm_bgc_column import case as bgc_case


def case(kind='plain'):
    temperature=270. if kind=='frozen' else 285.
    if kind in ('crop','harvest'):
        state,f=crop_case('vegetative',temperature)
    elif kind=='bgc':state,f,_=bgc_case(None,temperature)
    else:
        state=CLMColumnState.from_column(ColumnState(tsoil=np.full(4,temperature),smc=np.full(4,.18),ts=temperature)).with_layered_canopy()
        f=Forcing(1800.,temperature,.006 if temperature>273.15 else .002,3.,101325.,350.,320.,0.)
    ground=SoilEnthalpyState.from_column(ColumnState(tsoil=np.full(4,temperature),smc=np.full(4,.18),soil=state.ground.params))
    state=replace(state,ground=ground,canopy_condensate=CanopyCondensate.empty())
    if kind=='snow':state=replace(state,snow=Snowpack.from_ice([1.,2.],270.))
    state=state.with_irrigation(initial=IrrigationState(20.,3600.) if kind in ('frozen','harvest') else None)
    request=IrrigationInput(check=kind not in ('frozen','harvest'),supply_kgm2=1. if kind=='limited' else 100.,supply_temperature_k=290.)
    return state,f,request


def budgets(old,new,f,d):
    water=new.water_kgm2-old.water_kgm2-f.prcp*f.dt*1000.-d['irrigation_mm']+d['et_mm']+d['runoff_mm']
    heat=new.enthalpy_jm2-old.enthalpy_jm2-d['precipitation_enthalpy_jm2']-d['irrigation_enthalpy_jm2']+d['advected_enthalpy_jm2']+d['runoff_enthalpy_jm2']
    heat-=(f.swdown+f.lwdown-d['outgoing_shortwave']-d['outgoing_longwave']-d['h']-d['le'])*f.dt
    assert abs(water)<1e-9 and abs(heat)<.02
    if old.bgc_state is not None:
        crop=d.get('crop',{})
        cr=new.carbon_gc_m2-old.carbon_gc_m2+d['nee_gc_m2']+crop.get('harvest_gc_m2',0.)-crop.get('seed_input_gc_m2',0.)
        nr=new.nitrogen_gn_m2-old.nitrogen_gn_m2+crop.get('harvest_gn_m2',0.)-crop.get('seed_input_gn_m2',0.)
        assert abs(cr)<=d['biogeochemistry']['carbon_bound_gc_m2'] and abs(nr)<=d['biogeochemistry']['nitrogen_bound_gn_m2']
    return dict(water=water,heat=heat)


@pytest.mark.parametrize('kind',['plain','bgc','crop','snow','limited','frozen','harvest'])
def test_irrigation_routes_actual_water_and_heat_then_preserves_request(kind):
    state,f,request=case(kind)
    event=CropManagement(force_harvest=True) if kind=='harvest' else None
    new,d=step_clm_column(state,f,irrigation_input=request,crop_management=event)
    expected=0. if kind in ('frozen','harvest') else 1. if kind=='limited' else 5.
    assert d['irrigation_mm']==pytest.approx(expected,abs=1e-12)
    assert d['prcp_mm']==d['precipitation_enthalpy_jm2']==0.
    assert d['canopy_interception']['ground_liquid_kgm2']==d['canopy_interception']['ground_ice_kgm2']==0.
    if kind in ('frozen','harvest'):
        assert new.irrigation_state==IrrigationState() and d['irrigation']['cancelled_kgm2']==20.
    else:assert new.irrigation_state.pending_kgm2==pytest.approx(55.) and new.irrigation_state.remaining_seconds==19800.
    print('irrigation full NumPy',kind,dict(**budgets(state,new,f,d),applied=d['irrigation_mm'],
        supplied_heat=d['irrigation_enthalpy_jm2'],pending=new.irrigation_state.pending_kgm2),flush=True)
    after,dd=step_clm_column(new,f,irrigation_input=IrrigationInput(supply_kgm2=100.,supply_temperature_k=290.))
    assert dd['irrigation']['requested_kgm2']==0.
    budgets(new,after,f,dd)


def test_surface_application_matches_explicit_ground_transfer_without_interception():
    old,f,request=case();new,d=step_clm_column(old,f,irrigation_input=request)
    mass=5.;heat=mass*(334000.+4180.*(290.-273.15))
    ground,runoff,out_heat=infiltrate(old.ground,mass,heat)
    manual=replace(old,ground=ground,irrigation_state=None,irrigation_parameters=None,
        options=replace(old.options,OptIrrigation=0,OptIrrigationMethod=0))
    expected,ed=step_clm_column(manual,f)
    assert runoff==out_heat==0.
    np.testing.assert_allclose(new.ground.water,expected.ground.water,rtol=0.,atol=1e-12)
    np.testing.assert_allclose(new.ground.enthalpy,expected.ground.enthalpy,rtol=0.,atol=1e-7)
    assert d['h']==pytest.approx(ed['h'],abs=1e-9) and d['le']==pytest.approx(ed['le'],abs=1e-9)


def test_irrigation_state_and_native_packing_cannot_drop_pending_requests():
    from noahmp_clm.gpu.clm_layered_column import pack_clm_layered_columns
    from noahmp_clm.gpu.clm_crop_column import pack_clm_crop_columns
    state,f,request=case()
    with pytest.raises(NotImplementedError,match='Irrigation'):pack_clm_layered_columns([state])
    crop,_,_=case('crop')
    with pytest.raises(NotImplementedError,match='Irrigation'):pack_clm_crop_columns([crop])
    with pytest.raises(ValueError,match='Irrigation'):replace(state,irrigation_parameters=None)
    with pytest.raises(ValueError,match='method'):replace(state,options=replace(state.options,OptIrrigationMethod=1))
    with pytest.raises(ValueError,match='IrrigationInput'):step_clm_column(state,f,irrigation_input=True)
    dry=replace(state,irrigation_state=None,irrigation_parameters=None,options=replace(state.options,OptIrrigation=0,OptIrrigationMethod=0))
    with pytest.raises(ValueError,match='OptIrrigation'):step_clm_column(dry,f,irrigation_input=request)
