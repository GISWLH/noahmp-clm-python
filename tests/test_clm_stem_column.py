"""SAI must reach actual interception, radiation/MOST and carried C/N state."""
from dataclasses import replace
import json,pickle
import numpy as np
import pytest
from noahmp_clm.clm_state import CLMColumnState
from noahmp_clm.column import ColumnState,Forcing
from noahmp_clm.driver import step_column
from noahmp_clm.physics.snowpack import Snowpack
from noahmp_clm.physics.vegetation_structure import CanopyStructureParameters,StemAreaScheme


def extensive_budgets(before,after,f,d):
    water=after.water_kgm2-before.water_kgm2-f.prcp*f.dt*1000.+d['et_mm']+d['runoff_mm']
    heat=after.enthalpy_jm2-before.enthalpy_jm2-d['precipitation_enthalpy_jm2']+d['advected_enthalpy_jm2']+d['runoff_enthalpy_jm2']
    heat-=(f.swdown+f.lwdown-d['outgoing_shortwave']-d['outgoing_longwave']-d['h']-d['le'])*f.dt
    assert abs(water)<1e-9 and abs(heat)<.02
    return dict(water_kgm2=water,heat_jm2=heat)


def leafless_case():
    c=ColumnState(lai=0.,canw=0.,tsoil=np.full(4,285.),smc=np.full(4,.3),ts=285.)
    p=CanopyStructureParameters(canopy_top_m=3.,canopy_bottom_m=.5)
    old=CLMColumnState.from_column(c).with_vegetation_structure(p,sai=1.)
    return old,Forcing(1800.,285.,.006,2.5,101325.,350.,320.,5e-7)


@pytest.mark.parametrize('use_most',[False,True])
def test_leafless_stems_retain_rain_and_exchange_without_photosynthetic_leaf(use_most):
    from noahmp_clm.physics.canopy_most import MOSTGeometry
    old,f=leafless_case();geometry=MOSTGeometry() if use_most else None
    new,d=step_column(old,f,most_geometry=geometry)
    bare=replace(old,sai=0.,canopy_structure_parameters=None)
    _,bare_diag=step_column(bare,f,most_geometry=geometry)
    assert old.canopy_water_kgm2==0. and new.canopy_water_kgm2>0.
    assert new.sai==1. and new.lai==0. and d['gpp_gcm2']==0. and np.all(d['transpiration']==0.)
    assert d['canopy_geometry']['exposed_plant_area']==1.
    assert abs(d['outgoing_shortwave']-bare_diag['outgoing_shortwave'])>1.
    assert d['canopy_interception']['ground_liquid_kgm2']<f.prcp*f.dt*1000.
    print('whole leafless stems',use_most,json.dumps(dict(**extensive_budgets(old,new,f,d),h=d['h'],le=d['le'],stored_canopy_kgm2=new.canopy_water_kgm2)),flush=True)


def test_complete_snow_burial_routes_actual_stores_without_erasing_total_geometry():
    from noahmp_clm.physics.canopy_interception import CanopyCondensate
    old,f=leafless_case();p=CanopyStructureParameters(canopy_top_m=.2,short_vegetation=True)
    old=replace(old,snow=Snowpack.from_ice([30.],270.,density=100.),canopy_structure_parameters=p,
        canopy_condensate=CanopyCondensate.from_temperature(np.full(10,.01),270.))
    f=replace(f,tair=270.,qair=.002,swdown=100.,lwdown=280.,prcp=0.)
    new,d=step_column(old,f)
    assert new.sai==1. and d['canopy_geometry']['exposed_sai']==0.
    assert new.canopy_water_kgm2==0. and d['gpp_gcm2']==0.
    assert d['canopy_interception']['ground_ice_kgm2']==pytest.approx(.1,abs=1e-14)
    print('buried stem geometry',json.dumps(dict(**extensive_budgets(old,new,f,d),total_sai=new.sai,exposed_sai=d['canopy_geometry']['exposed_sai'])),flush=True)


def test_prescribed_input_and_checkpoint_keep_explicit_stem_state():
    old,f=leafless_case();new,d=step_column(old,f,prescribed_sai=.6)
    restored=pickle.loads(pickle.dumps(new))
    assert restored.sai==.6 and restored.canopy_structure_parameters==old.canopy_structure_parameters
    assert d['canopy_geometry']['total_sai']==.6
    extensive_budgets(old,new,f,d)
    with pytest.raises(ValueError,match='explicit'):replace(old,canopy_structure_parameters=None)


@pytest.mark.parametrize('seasonal',[False,True])
def test_cn_structure_uses_funded_final_lai_and_does_not_decay_twice(seasonal):
    from test_clm_bgc_column import case,budgets
    from noahmp_clm.physics.phenology import PhenologyPhase
    phase=PhenologyPhase.OFFSET if seasonal else None
    old,f,env=case(phase,285.)
    old=old.with_vegetation_structure(CanopyStructureParameters(StemAreaScheme.CN_NONCROP,canopy_top_m=8.,canopy_bottom_m=1.),sai=1.)
    new,d=step_column(old,f,phenology_environment=env,nitrogen_deposition_gn_m2=1e-5)
    # Independent pinned reference: one30-day decay and net displayed-leaf
    # loss using funded final leaf C. Using the provisional LAI again or a
    # second decay here would silently alter the carried structure state.
    expected=max(.5,(1.-.5*f.dt/(30.*86400.))*old.sai+max(old.lai-new.lai,0.))
    assert new.sai==pytest.approx(expected,abs=1e-14)
    assert d['vegetation_structure']['sai_used']==d['canopy_geometry']['total_sai']
    assert d['vegetation_structure']['sai_after']==new.sai
    assert new.canopy_structure_parameters==old.canopy_structure_parameters
    with pytest.raises(ValueError,match='bypass'):step_column(old,f,phenology_environment=env,prescribed_sai=.1)
    print('CN stem full column',seasonal,json.dumps(dict(**budgets(old,new,f,d,1e-5),sai_before=old.sai,sai_after=new.sai,sai_used=d['vegetation_structure']['sai_used'])),flush=True)


def test_harvested_stubble_and_new_planting_follow_actual_crop_carbon_transfers():
    from test_clm_crop_column import case,budgets
    from noahmp_clm.physics.crop import CropManagement
    old,f=case('mature',285.)
    old=old.with_vegetation_structure(CanopyStructureParameters(StemAreaScheme.ANNUAL_C3_CROP,canopy_top_m=1.,short_vegetation=True),sai=.8)
    harvested,d=step_column(old,f)
    assert d['crop']['harvested'] and harvested.lai==0. and harvested.sai==.25
    budgets(old,harvested,f,d,0.)
    fallow,fd=step_column(harvested,f);assert fallow.sai==.25
    budgets(harvested,fallow,f,fd,0.)
    planted,pd=step_column(fallow,f,crop_management=CropManagement(planting=True))
    assert pd['crop']['planted'] and planted.sai==0.
    budgets(fallow,planted,f,pd,0.)
    print('crop stem harvest/replant',harvested.sai,fallow.sai,planted.sai,flush=True)
