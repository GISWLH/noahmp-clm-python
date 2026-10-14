"""Joint surface/thermal reference using MINPACK, independent of nested solvers."""
import numpy as np
import pytest
from noahmp_clm.column import ColumnState,default_soil_params
from noahmp_clm.physics.snowpack import Snowpack,TF
from noahmp_clm.physics.soil_enthalpy import SoilEnthalpyState
from noahmp_clm.physics.frozen_soil import unfrozen_water_content
from noahmp_clm.physics.canopy_energy import CanopyExchange
from noahmp_clm.physics.canopy_column import surface_step
from noahmp_clm.physics.snicar import snow_shortwave


def independent_bare_surface(snow,soil,dt,sw,lw,e,albedo=(.2,.3),emissivity=.97):
    """Solve all snow enthalpies, soil temperatures and the surface together.

    Mass exchange, advected heat, thermal coefficients and layer balances are
    written here; no production surface iteration, heat solve, phase inverse
    or vapor-transfer routine is used. Previously independently validated
    SNICAR radiation and the retained soil liquid curve are component inputs.
    This reference covers bare ground without complete vapor exhaustion.
    """
    root=pytest.importorskip('scipy.optimize').root
    ns=len(snow.mass);nz=len(soil.water);p=soil.params;zsoil=soil.dz
    optics=snow_shortwave(snow,[sw*.25]*2,[sw*.25]*2,ground_albedo=albedo,cosine_zenith=.5)
    absorbed=np.r_[optics['absorbed'],optics['ground_absorbed'],np.zeros(nz-1)]
    rho=e.pressure/(287.*e.air_temperature*(1.+.61*e.specific_humidity))
    old_liquid=unfrozen_water_content(soil.water/(1000*zsoil),soil.temperature,p,soil.freezing_option)*1000*zsoil
    snow_scale=snow.mass*334000.
    def evaluate(x,details=False):
        tsurface=x[-1]*10.+273.15
        if tsurface<273.15:
            es=np.exp(9.550426-5723.265/tsurface+3.53068*np.log(tsurface)-.00728332*tsurface)
        else:
            offset=tsurface-273.15;es=611.2*np.exp(17.67*offset/(offset+243.5))
        q=.622*es/(e.pressure-.378*es)
        evaporated=dt*rho*(q-e.specific_humidity)/(1./e.atmosphere_velocity+e.ground_resistance)
        evaporated=min(evaporated,float(snow.mass[0]) if ns else float(old_liquid[0]))
        m=snow.mass.copy();hs0=snow.enthalpy.copy();zs=snow.thickness.copy()
        water=soil.water.copy();hg0=soil.enthalpy.copy()
        if ns:
            if evaporated>=0.:
                ice=evaporated*float(snow.ice[0]/snow.mass[0]);advected=evaporated*snow.enthalpy[0]/snow.mass[0]
                zs[0]*=1.-evaporated/snow.mass[0]
            else:
                ice=evaporated if tsurface<273.15 else 0.
                advected=ice*2100.*(tsurface-273.15)+(evaporated-ice)*(334000.+4180.*(tsurface-273.15))
                zs[0]-=ice/200.
            m[0]-=evaporated;hs0[0]-=advected
            assert m[0]>0.,'Reference excludes complete top-layer vapor exhaustion'
            liquid_star=np.clip(hs0[0]/334000.,0.,m[0])
            zs[0]=max(zs[0],(m[0]-liquid_star)/917.)
        else:
            source_t=soil.temperature[0] if evaporated>=0. else tsurface
            ice=evaporated if evaporated<0. and tsurface<273.15 else 0.
            advected=(ice*2100.*(source_t-273.15)+
                (evaporated-ice)*(334000.+4180.*(source_t-273.15)))
            water[0]-=evaporated;hg0[0]-=advected
        hs=x[:ns]*snow_scale
        ts=np.empty(ns)
        for j in range(ns):
            ts[j]=hs[j]/(2100*m[j]) if hs[j]<0 else ((hs[j]-334000*m[j])/(4180*m[j]) if hs[j]>334000*m[j] else 0.)
        tg=x[ns:ns+nz]*10.
        liquid=unfrozen_water_content(water/(1000*zsoil),tg+273.15,p,soil.freezing_option)*1000*zsoil
        hg=((1-p.smcmax)*2e6*zsoil+2100.*(water-liquid)+4180.*liquid)*tg+334000.*liquid
        density=np.minimum(m/zs,917.)
        ks=.023+(7.75e-5*density+1.105e-6*density**2)*(2.29-.023)
        saturation=np.clip(water/(1000*zsoil*p.smcmax),0.,1.)
        kersten=np.where(saturation>.1,np.clip(np.log10(np.maximum(saturation,1e-6))+1.,0.,1.),0.)
        kg=.2+(.6+1.6*p.quartz-.2)*kersten
        k=np.r_[ks,kg];z=np.r_[zs,zsoil];temperature=np.r_[ts,tg]
        sensible=rho*1004.*e.atmosphere_velocity*(tsurface-e.air_temperature)
        latent=(2.5e6*evaporated+334000.*ice)/dt
        net_lw=emissivity*(lw-5.670374419e-8*tsurface**4)
        flux=[net_lw-sensible-latent]
        for j in range(ns+nz-1):
            flux.append((temperature[j]-temperature[j+1])/(z[j]/(2*k[j])+z[j+1]/(2*k[j+1])))
        flux.append(0.)
        energy=np.r_[hs,hg]
        residual=(energy-np.r_[hs0,hg0])/dt-absorbed-np.diff(-np.asarray(flux))
        residual=np.r_[residual,100.*(temperature[0]+273.15-tsurface)]
        if details: return energy,temperature+273.15,water,m,evaporated,sensible,latent,residual
        return residual
    initial=np.r_[snow.enthalpy/snow_scale,(soil.temperature-273.15)/10.,
        ((snow.temperature[0] if ns else soil.temperature[0])-273.15)/10.]
    sol=root(evaluate,initial,tol=1e-10)
    result=evaluate(sol.x,True)
    assert np.max(np.abs(result[-1]))<3e-6,(sol.message,result[-1])
    return result


@pytest.mark.parametrize('ns,air,soil_t,snow_h,rh,solar',[
    (0,290.,288.,0.,.5,600.),(0,290.,288.,0.,.95,0.),
    (0,265.,268.,0.,.1,200.),(0,269.,266.,0.,.99,0.),(1,265.,268.,-12000.,.4,200.),
    (3,265.,270.,-6000.,.99,0.),(5,280.,271.,100000.,.4,800.),
    (1,285.,280.,320000.,.4,600.),(3,275.,274.,200000.,.99,0.)])
def test_joint_reference(ns,air,soil_t,snow_h,rh,solar):
    mass=np.arange(1,ns+1)*3.
    snow=Snowpack(mass,mass*snow_h,mass/250.) if ns else Snowpack.empty()
    ground=SoilEnthalpyState.from_column(ColumnState(tsoil=np.full(4,soil_t),smc=np.full(4,.3),soil=default_soil_params('loam')))
    es=611.2*np.exp(17.67*(air-TF)/(air-TF+243.5));q=rh*.622*es/(101325.-.378*es)
    e=CanopyExchange(air,q,101325.,.02,.004,.03,soil_t,0.)
    _,_,_,d=surface_step(snow,ground,1800.,[0.]*10,solar,300.,e,vegetated_fraction=0.,ground_albedo=(.2,.3))
    reference=independent_bare_surface(snow,ground,1800.,solar,300.,e)
    h,t,water,mass,evap,hflux,le,residual=reference
    np.testing.assert_allclose(np.r_[d['thermal_snow_temperature'],d['thermal_soil_temperature']],t,rtol=0,atol=2e-6)
    np.testing.assert_allclose(np.r_[d['thermal_snow_enthalpy'],d['thermal_soil_enthalpy']],h,rtol=0,atol=.01)
    np.testing.assert_allclose(d['thermal_soil_water'],water,rtol=0,atol=1e-7)
    np.testing.assert_allclose(d['thermal_snow_mass'],mass,rtol=0,atol=1e-7)
    assert abs(d['et_kgm2']-evap)<1e-7
    assert abs(d['h']-hflux)<1e-4
    assert abs(d['le']-le)<1e-4
