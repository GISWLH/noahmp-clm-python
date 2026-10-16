"""Tile outflow reaches whole-column M/H and post-drainage C/N feedback."""
from dataclasses import replace
import numpy as np
import pytest
from noahmp_clm.clm_driver import step_clm_column
from noahmp_clm.clm_state import CLMColumnState
from noahmp_clm.column import ColumnState,Forcing
from noahmp_clm.physics.soil_enthalpy import SoilEnthalpyState
from noahmp_clm.physics.tile_drainage import TileDrainageParameters,step_tile_drainage
from test_clm_wetland_column import case as wetland_case,budgets as wetland_budgets


def case(kind):
    if kind in ('wetland','managed'):
        s,f,kwargs=wetland_case('wet' if kind=='wetland' else 'irrigated')
        if kind=='managed':
            # A wet drain layer and a dry root zone exercise simultaneous
            # external irrigation and subsurface tile export.
            ground=SoilEnthalpyState.from_column(ColumnState(tsoil=np.full(4,285.),smc=np.array([.18,.18,.4,.4]),soil=s.ground.params))
            s=replace(s,ground=ground)
    else:
        t=270. if kind=='frozen' else 285.;theta=.2 if kind=='dry' else .4
        s=CLMColumnState.from_column(ColumnState(tsoil=np.full(4,t),smc=np.full(4,theta),ts=t)).with_layered_canopy()
        f=Forcing(1800.,t,.002 if t<273.15 else .006,3.,101325.,350.,320.,0.);kwargs={}
    return s.with_tile_drainage(TileDrainageParameters(.8,.28,10.)),f,kwargs


@pytest.mark.parametrize('kind',['warm','dry','frozen','wetland','managed'])
def test_two_steps_tile_outlet_closes_water_heat_and_bgc_uses_drained_soil(kind):
    old,f,kwargs=case(kind)
    for step in range(2):
        new,d=step_clm_column(old,f,**kwargs)
        mass=new.water_kgm2-old.water_kgm2-f.prcp*f.dt*1000.-d.get('irrigation_mm',0.)+d['et_mm']+d['runoff_mm']
        heat=new.enthalpy_jm2-old.enthalpy_jm2-d['precipitation_enthalpy_jm2']-d.get('irrigation_enthalpy_jm2',0.)+d['advected_enthalpy_jm2']+d['runoff_enthalpy_jm2']
        heat-=(f.swdown+f.lwdown-d['outgoing_shortwave']-d['outgoing_longwave']-d['h']-d['le'])*f.dt
        assert abs(mass)<1e-9 and abs(heat)<.02
        assert d['tile_drainage_mm']==pytest.approx(0. if kind in ('dry','frozen') else 10./48.,abs=1e-12)
        assert d['runoff_mm']==pytest.approx(d['runoff_surface_mm']+d['runoff_deep_mm'],abs=1e-12)
        if old.bgc_state is not None:
            wetland_budgets(old,new,f,d,0.)
            from noahmp_clm.physics.wetland import column_wetland_environment
            assert d['wetland']['environment']==column_wetland_environment(new.ground,f.ps,new.wetland_parameters)
        print('tile whole NumPy',kind,step,dict(water=mass,heat=heat,tile=d['tile_drainage_mm'],tile_heat=d['tile_drainage_enthalpy_jm2']),flush=True)
        old=new
        if 'irrigation_input' in kwargs:kwargs['irrigation_input']=replace(kwargs['irrigation_input'],check=False)


def test_tile_whole_physical_result_matches_explicit_post_column_withdrawal():
    old,f,kwargs=case('warm');new,d=step_clm_column(old,f,**kwargs)
    plain=replace(old,tile_drainage_parameters=None,options=replace(old.options,OptTileDrainage=0))
    base,bd=step_clm_column(plain,f,**kwargs);earth,td=step_tile_drainage(base.ground,f.dt,old.tile_drainage_parameters)
    np.testing.assert_array_equal(new.ground.water,earth.water);np.testing.assert_array_equal(new.ground.enthalpy,earth.enthalpy)
    assert d['runoff_mm']==pytest.approx(bd['runoff_mm']+td['drained_kgm2'],abs=1e-12)
    assert d['runoff_enthalpy_jm2']==pytest.approx(bd['runoff_enthalpy_jm2']+td['drained_enthalpy_jm2'],abs=1e-9)


def test_tile_option_requires_parameters_and_old_packer_cannot_drop_it():
    from noahmp_clm.gpu.clm_layered_column import pack_clm_layered_columns
    s,_,_=case('warm')
    with pytest.raises(ValueError):replace(s,tile_drainage_parameters=None)
    with pytest.raises(ValueError):replace(s,options=replace(s.options,OptTileDrainage=2))
    with pytest.raises(ValueError):s.with_tile_drainage(TileDrainageParameters(3.))
    with pytest.raises(NotImplementedError,match='OptTileDrainage'):pack_clm_layered_columns([s])
