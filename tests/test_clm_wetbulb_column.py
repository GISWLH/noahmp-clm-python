"""Wang partition in the conservative full single-column driver."""
from dataclasses import replace
import numpy as np
import pytest
from noahmp_clm.column import ColumnState,Forcing
from noahmp_clm.clm_state import CLMColumnState
from noahmp_clm.clm_enthalpy_driver import step_clm_enthalpy_column
from noahmp_clm.physics.wetbulb import wetbulb_snow_fraction


@pytest.mark.parametrize('tc,rh',[(1.,.5),(3.,.9),(8.,.2),(-10.,.8)])
def test_wetbulb_full_column_mass_and_precipitation_enthalpy(tc,rh):
    from test_wetbulb import humidity
    old=ColumnState(ts=273.,tsoil=np.full(4,272.),smc=np.full(4,.3))
    old.options=replace(old.options,OptRainSnowPartition=5)
    state=CLMColumnState.from_column(old).with_layered_canopy()
    f=Forcing(1800.,273.15+tc,humidity(tc,rh,101325.),2.5,101325.,150.,280.,2e-6)
    after,d=step_clm_enthalpy_column(state,f)
    snow=wetbulb_snow_fraction(f.tair,f.qair,f.ps)*f.prcp*f.dt*1000.
    rain=f.prcp*f.dt*1000.-snow
    ph=rain*(334000.+4180.*tc)+snow*2100.*min(tc,0.)
    assert d['precipitation_enthalpy_jm2']==pytest.approx(ph,rel=1e-13,abs=1e-8)
    water=after.water_kgm2-state.water_kgm2-rain-snow+d['et_mm']+d['runoff_mm']
    atmo=(f.swdown+f.lwdown-d['outgoing_shortwave']-d['outgoing_longwave']-d['h']-d['le'])*f.dt
    heat=after.enthalpy_jm2-state.enthalpy_jm2-ph-atmo+d['advected_enthalpy_jm2']+d['runoff_enthalpy_jm2']
    assert abs(water)<1e-9 and abs(heat)<.02
    print('wetbulb full column',tc,rh,dict(snow_kgm2=snow,rain_kgm2=rain,water_kgm2=water,heat_jm2=heat),flush=True)
