"""Independent flux/energy roots and glacier atmospheric mass/heat budgets."""
from dataclasses import replace
import numpy as np
import pytest
from noahmp_clm.column import Forcing
from noahmp_clm.clm_glacier import GlacierColumnState,step_clm_glacier_column
from noahmp_clm.physics.glacier import GlacierIce,GlacierParameters,exchange_glacier_vapor,add_glacier_surface_water
from noahmp_clm.physics.snowpack import Snowpack,TF


@pytest.mark.parametrize('air,sw,wind',[(250.,0.,2.),(260.,80.,3.),(268.,150.,5.)])
def test_cold_bare_ice_independent_scalar_energy_root(air,sw,wind):
    root=pytest.importorskip('scipy.optimize').brentq
    m=500.;old_t=263.15;dt=1800.;p=75000.;qair=.0002
    state=GlacierColumnState(GlacierIce([m],[m*2100.*(old_t-TF)]))
    f=Forcing(dt,air,qair,wind,p,sw,230.,0.)
    velocity=.16*wind/(np.log(10./.002)*np.log(2./.0002));rho=p/(287.*air*(1.+.61*qair))
    def flux(t):
        es=np.exp(9.550426-5723.265/t+3.53068*np.log(t)-.00728332*t)
        e=rho*velocity*(.622*es/(p-.378*es)-qair)
        sensible=rho*1004.*velocity*(t-air)
        radiation=.45*sw+.97*(230.-5.670374419e-8*t**4)
        return e,sensible,radiation
    def equation(t):
        e,h,r=flux(t)
        return (m-dt*e)*2100.*(t-old_t)/dt-(r-h-2834000.*e+.05)
    expected=root(equation,220.,273.14,xtol=1e-12);e,h,r=flux(expected)
    new,d=step_clm_glacier_column(state,f)
    assert d['flux_surface_temperature']==pytest.approx(expected,abs=3e-7)
    assert d['h']==pytest.approx(h,abs=1e-5)
    assert d['le']==pytest.approx(2834000.*e,abs=1e-5)
    assert d['runoff_mm']==0.
    print('glacier cold independent',air,'T/H/LE errors',d['flux_surface_temperature']-expected,d['h']-h,d['le']-2834000.*e,flush=True)


def independent(state,final,d,f,deposition):
    water=final.mass_kgm2-state.mass_kgm2-f.prcp*f.dt*1000.+d['et_mm']+d['runoff_mm']
    heat=(final.enthalpy_jm2-state.enthalpy_jm2-d['precipitation_enthalpy_jm2']-
        (d['rnet']-d['h']-d['le'])*f.dt-state.parameters.geothermal_wm2*f.dt+d['advected_enthalpy_jm2']+d['runoff_enthalpy_jm2'])
    old=state.snow.impurities.sum(0)+state.ice.impurities.sum(0)
    new=final.snow.impurities.sum(0)+final.ice.impurities.sum(0)
    assert abs(water)<=d['water_bound_kgm2']
    assert abs(heat)<=d['energy_bound_jm2']
    np.testing.assert_allclose(new-old+d['impurity_out_kgm2'],deposition*f.dt,atol=1e-18,rtol=1e-10)
    return np.array([water,heat])


@pytest.mark.parametrize('kind',['bare_cold','bare_melt','snow_cold','snow_melt','rain','snowfall','frost','dirty'])
def test_glacier_whole_column_two_steps(kind):
    temp=273.15 if 'melt' in kind or kind=='rain' else 263.15
    ice=GlacierIce.from_thickness([.1,.3,.6,9.],temp)
    snow=Snowpack.from_ice([2.,5.,11.,23.,30.],temp,300.) if kind.startswith('snow_') or kind=='dirty' else Snowpack.empty()
    state=GlacierColumnState(ice,snow)
    air=285. if 'melt' in kind or kind=='rain' else 260.
    humidity=.008 if kind=='frost' else .001
    rate=2e-6 if kind in ('rain','snowfall') else 0.
    f=Forcing(1800.,air,humidity,3.,80000.,600. if air>TF else 150.,350. if air>TF else 240.,rate)
    dep=np.arange(1,6)*1e-12 if kind=='dirty' else np.zeros(5);maxima=np.zeros(2)
    for _ in range(2):
        final,d=step_clm_glacier_column(state,f,deposition_rates=dep)
        maxima=np.maximum(maxima,abs(independent(state,final,d,f,dep)));state=final
    print('glacier full',kind,'M/H residual',maxima.tolist(),'ET/runoff/T',d['et_mm'],d['runoff_mm'],d['ts'],flush=True)
    if kind=='frost':assert d['et_mm']<0.
    if kind=='bare_melt':assert d['runoff_mm']>0.
    if kind=='snowfall':assert len(state.snow.mass)>0


def test_glacier_vapor_and_supercooled_rain_have_closed_ledgers():
    original=GlacierIce.from_thickness([.1,.5],268.15)
    original=GlacierIce(original.mass,original.enthalpy,np.ones((2,5))*1e-8)
    for amount,t in ((.1,260.),(-.1,260.),(-.1,280.),(91.7,270.)):
        new,d=exchange_glacier_vapor(original,amount,t)
        assert new.mass.sum()-original.mass.sum()+d['mass_kgm2']+d['runoff_kgm2']==pytest.approx(0.,abs=1e-12)
        assert new.enthalpy.sum()-original.enthalpy.sum()+d['advected_enthalpy_jm2']+d['runoff_enthalpy_jm2']==pytest.approx(0.,abs=1e-8)
        np.testing.assert_allclose(new.impurities.sum(0)+d['residue_kgm2'],original.impurities.sum(0),rtol=0,atol=1e-22)
    for t in (200.,260.,273.15,350.,400.):
        amount=2.;heat=amount*(334000.+4180.*(t-TF));new,d=add_glacier_surface_water(original,amount,heat)
        assert new.enthalpy.sum()-original.enthalpy.sum()+d['runoff_enthalpy_jm2']==pytest.approx(heat,abs=1e-8)
    with pytest.raises(ValueError):exchange_glacier_vapor(original,1000.,270.)


def test_glacier_profile_is_explicit_and_rejects_unsupported_options():
    s=GlacierColumnState(GlacierIce.from_thickness([1.]))
    with pytest.raises(NotImplementedError,match='OptGlacierTreatment'):replace(s,options=replace(s.options,OptGlacierTreatment=2))
    with pytest.raises(NotImplementedError,match='OptCropModel'):replace(s,options=replace(s.options,OptCropModel=1))
    with pytest.raises(ValueError):step_clm_glacier_column(s,Forcing(0.,260.,.001,3.,80000.,100.,250.,0.))


@pytest.mark.parametrize('air,sw',[(250.,0.),(268.,400.)])
def test_glacier_most_uses_actual_signed_vapor_and_separate_heights(air,sw):
    state=GlacierColumnState(GlacierIce.from_thickness([.1,.3,.6,9.],263.15),temperature_height=2.,humidity_height=4.)
    state=replace(state,options=replace(state.options,OptSurfaceDrag=1))
    f=Forcing(1800.,air,.001,3.,75000.,sw,240.,0.)
    new,d=step_clm_glacier_column(state,f);independent(state,new,d,f,np.zeros(5))
    most=d['most'];assert most['coefficient_residual']<=1e-4
    assert d['heat_velocity']!=d['water_velocity']
    assert most['history'][-1]['water_vapor']==pytest.approx(d['et_mm']/f.dt,abs=1e-15)
    print('glacier MOST',air,'iterations/residual/H/E',most['iterations'],most['coefficient_residual'],d['h'],d['et_mm']/f.dt,flush=True)
