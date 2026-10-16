"""Wetland uses actual soil phases and funded BGC losses, retaining all stores."""
from dataclasses import replace
import numpy as np
import pytest
from noahmp_clm.clm_driver import step_clm_column
from noahmp_clm.clm_state import CLMColumnState
from noahmp_clm.column import ColumnState
from noahmp_clm.physics.soil_enthalpy import SoilEnthalpyState
from noahmp_clm.physics.snowpack import Snowpack
from noahmp_clm.physics.wetland import WetlandState,WetlandParameters,column_wetland_environment
from noahmp_clm.physics.irrigation import IrrigationInput
from noahmp_clm.physics.crop import CropManagement
from test_clm_bgc_column import case as bgc_case,Phase
from test_clm_crop_column import case as crop_case


def case(kind='wet'):
    temperature=270. if kind in ('frozen','snow') else 285.
    kwargs={}
    if kind in ('crop','harvest','irrigated'):
        s,f=crop_case('vegetative',temperature)
        if kind=='harvest':kwargs['crop_management']=CropManagement(force_harvest=True)
        if kind=='irrigated':
            s=s.with_irrigation();kwargs['irrigation_input']=IrrigationInput(check=True,supply_kgm2=100.,supply_temperature_k=290.)
    else:
        s,f,e=bgc_case(Phase.ACTIVE if kind=='seasonal' else None,temperature)
        if e is not None:kwargs['phenology_environment']=e
    if kind=='bgc_irrigated':
        s=s.with_irrigation();kwargs['irrigation_input']=IrrigationInput(check=True,supply_kgm2=100.,supply_temperature_k=290.)
    theta=.18 if kind in ('dry','irrigated','bgc_irrigated') else .449
    ground=SoilEnthalpyState.from_column(ColumnState(tsoil=np.full(4,temperature),smc=np.full(4,theta),soil=s.ground.params))
    s=replace(s,ground=ground)
    if kind=='snow':s=replace(s,snow=Snowpack.from_ice([3.,7.],270.))
    return s.with_wetland(initial=WetlandState(5.)),f,kwargs


def budgets(old,new,f,d,nitrogen):
    crop=d.get('crop',{})
    cr=new.carbon_gc_m2-old.carbon_gc_m2+d['nee_gc_m2']+d['methane_emission_gc_m2']+crop.get('harvest_gc_m2',0.)-crop.get('seed_input_gc_m2',0.)
    nr=new.nitrogen_gn_m2-old.nitrogen_gn_m2-nitrogen+crop.get('harvest_gn_m2',0.)-crop.get('seed_input_gn_m2',0.)
    m=new.water_kgm2-old.water_kgm2-f.prcp*f.dt*1000.-d.get('irrigation_mm',0.)+d['et_mm']+d['runoff_mm']
    h=new.enthalpy_jm2-old.enthalpy_jm2-d['precipitation_enthalpy_jm2']-d.get('irrigation_enthalpy_jm2',0.)+d['advected_enthalpy_jm2']+d['runoff_enthalpy_jm2']
    h-=(f.swdown+f.lwdown-d['outgoing_shortwave']-d['outgoing_longwave']-d['h']-d['le'])*f.dt
    assert abs(cr)<=d['biogeochemistry']['carbon_bound_gc_m2']
    assert abs(nr)<=d['biogeochemistry']['nitrogen_bound_gn_m2']
    assert abs(m)<1e-9 and abs(h)<.02
    return dict(carbon=cr,nitrogen=nr,water=m,heat=h)


@pytest.mark.parametrize('kind',['wet','dry','frozen','snow','crop','harvest','irrigated','bgc_irrigated','seasonal'])
def test_wetland_actual_column_carries_methane_and_closes_four_budgets(kind):
    old,f,kwargs=case(kind)
    for step in range(2):
        new,d=step_clm_column(old,f,nitrogen_deposition_gn_m2=.0001,**kwargs)
        print('wetland full NumPy',kind,step,dict(**budgets(old,new,f,d,.0001),
            methane_stored=new.wetland_state.methane_gc_m2,methane_emitted=d['methane_emission_gc_m2'],
            oxygen=d['wetland']['environment']['oxygen_factor'],nee_co2=d['nee_gc_m2']),flush=True)
        w=d['wetland'];raw=d['biogeochemistry']['decomposition']['heterotrophic_respiration_gc_m2']
        assert d['decomposed_carbon_gc_m2']==raw
        assert d['heterotrophic_respiration_gc_m2']==raw-w['methane_production_gc_m2']+w['oxidized_gc_m2']
        assert d['total_atmospheric_carbon_gc_m2']==d['nee_gc_m2']+d['methane_emission_gc_m2']
        assert d['wetland']['environment']==column_wetland_environment(new.ground,f.ps,old.wetland_parameters)
        if kind in ('frozen','snow'):assert w['methane_production_gc_m2']==0.
        if step==0 and kind=='wet':assert w['ebullition_gc_m2']>4.
        old=new
        if 'crop_management' in kwargs:kwargs['crop_management']=None
        if 'irrigation_input' in kwargs:kwargs['irrigation_input']=replace(kwargs['irrigation_input'],check=False)


def test_wetland_partitions_same_actual_donor_and_keeps_water_heat_unchanged():
    old,f,kwargs=case();new,d=step_clm_column(old,f,**kwargs)
    plain=replace(old,wetland_state=None,wetland_parameters=None,options=replace(old.options,OptWetlandModel=0))
    matched,md=step_clm_column(plain,f,oxygen_factor=d['wetland']['environment']['oxygen_factor'],**kwargs)
    for key in ('water','enthalpy'):
        np.testing.assert_array_equal(getattr(new.ground,key),getattr(matched.ground,key))
    np.testing.assert_array_equal(new.bgc_state.soil.carbon,matched.bgc_state.soil.carbon)
    np.testing.assert_array_equal(new.bgc_state.plant.carbon,matched.bgc_state.plant.carbon)
    w=d['wetland']
    assert d['nee_gc_m2']==pytest.approx(md['nee_gc_m2']-w['methane_production_gc_m2']+w['oxidized_gc_m2'],abs=1e-15)
    assert d['decomposed_carbon_gc_m2']==md['heterotrophic_respiration_gc_m2']


def test_wetland_column_contract_never_drops_the_carbon_store():
    from noahmp_clm.gpu.clm_bgc_column import pack_clm_bgc_columns
    from noahmp_clm.gpu.clm_crop_column import pack_clm_crop_columns
    old,f,kwargs=case()
    with pytest.raises(NotImplementedError,match='Wetland'):pack_clm_bgc_columns([old])
    crop,_,_=case('crop')
    with pytest.raises(NotImplementedError,match='Wetland'):pack_clm_crop_columns([crop])
    with pytest.raises(ValueError,match='Wetland'):replace(old,wetland_state=None)
    with pytest.raises(ValueError,match='depth'):old.with_wetland(WetlandParameters(active_depth_m=3.))
    with pytest.raises(ValueError,match='Q10'):old.with_wetland(WetlandParameters(donor_q10=3.))
    with pytest.raises(ValueError,match='oxygen'):step_clm_column(old,f,oxygen_factor=.5)
    with pytest.raises(ValueError,match='BGC'):CLMColumnState.from_column(ColumnState()).with_wetland()
