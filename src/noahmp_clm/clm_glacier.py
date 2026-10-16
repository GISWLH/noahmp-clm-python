"""Meteorology-driven finite glacier land unit, separate from vegetated soil.

Surface fluxes and the snow/ice thermal state share one temperature root.
Snow optics/supply are explicit over a step, as in the canopy/soil column.
No ice dynamics, replacement ice, plant stores or porous ice infiltration.
"""
from dataclasses import dataclass,field,replace,fields
import numpy as np
from .options import PhysicsOptions
from .physics.glacier import GlacierIce,GlacierParameters,glacier_heat_step,add_glacier_surface_water,exchange_glacier_vapor
from .physics.snowpack import Snowpack,TF,CI,CW,LF,add_snowfall,add_rain
from .physics.snow_vapor import exchange_vapor
from .physics.snow_evolution import fresh_snow_properties,compact,grain_growth,deposit_impurities
from .physics.snicar import snow_shortwave
from .physics.precip import partition_precipitation
from .physics.canopy_energy import _saturation,_ice_saturation,CanopyExchange
from .physics.canopy_most import MOSTGeometry,solve_surface_most
from .physics.energy import SIGMA,CP,air_density


def glacier_options():
    return PhysicsOptions(OptGlacierTreatment=1,OptSnowAlbedo=3,OptRainSnowPartition=5,OptSurfaceDrag=2)


@dataclass(frozen=True)
class GlacierColumnState:
    ice: GlacierIce
    snow: Snowpack=field(default_factory=Snowpack.empty)
    parameters: GlacierParameters=field(default_factory=GlacierParameters)
    options: PhysicsOptions=field(default_factory=glacier_options)
    z0m: float=.002
    wind_height: float=10.
    temperature_height: float=2.
    humidity_height: float=2.

    def __post_init__(self):
        if not isinstance(self.ice,GlacierIce) or not isinstance(self.snow,Snowpack):raise ValueError('Glacier requires snow/ice states')
        if not np.isfinite(self.z0m) or not 1e-5<=self.z0m<=.5:raise ValueError('Invalid glacier roughness')
        self.exchange_geometry()
        baseline=glacier_options();allowed={'OptSurfaceDrag','OptRainSnowPartition'}
        for f in fields(self.options):
            if f.name not in allowed and getattr(self.options,f.name)!=getattr(baseline,f.name):
                raise NotImplementedError(f'{f.name} is outside the finite glacier profile')

    def exchange_geometry(self):
        return MOSTGeometry(wind_height=self.wind_height,temperature_height=self.temperature_height,
            humidity_height=self.humidity_height,momentum_roughness=self.z0m,heat_roughness=self.z0m/10.,
            water_roughness=self.z0m/10.,ground_roughness=self.z0m)

    @property
    def mass_kgm2(self):return float(self.ice.mass.sum()+self.snow.mass.sum())

    @property
    def enthalpy_jm2(self):return float(self.ice.enthalpy.sum()+self.snow.enthalpy.sum())

    @property
    def surface_temperature(self):
        if len(self.snow.mass):return float(self.snow.temperature[0])
        if len(self.ice.mass):return float(self.ice.temperature[0])
        raise ValueError('Exhausted finite glacier has no surface temperature')


def glacier_surface_step(state,forcing,*,cosine_zenith=.5,diffuse_fraction=.5,temperature_tolerance=2e-7,_exchange=None):
    """Atmospheric exchange with conservative vapor and implicit shared heat.

    Returns temperature before meltwater drainage as the flux temperature;
    surviving exposed material supplies the next-step state temperature.
    Drag option 1 iterates MOST with the actual H and signed vapor mass.
    Default 2 is neutral bulk exchange. Both use independent measurement
    heights; defaults follow Forcing's 10 m wind and 2 m temperature/humidity.
    """
    f=forcing;p=state.parameters;snow0,ice0=state.snow,state.ice
    if not len(ice0.mass):raise ValueError('Exhausted finite glacier needs an explicit new land-unit state')
    if not 0.<=diffuse_fraction<=1. or not 0.<cosine_zenith<=1. or not np.isfinite(temperature_tolerance) or temperature_tolerance<=0.:
        raise ValueError('Invalid glacier optics or temperature tolerance')
    rad=snow_shortwave(snow0,np.full(2,.5*f.swdown*(1.-diffuse_fraction)),np.full(2,.5*f.swdown*diffuse_fraction),
        cosine_zenith=cosine_zenith,ground_albedo=(p.ice_albedo,p.ice_albedo))
    geometry=state.exchange_geometry();momentum=np.log(state.wind_height/state.z0m)
    heat_velocity=.16*max(f.wind,.3)/(momentum*np.log(state.temperature_height/(state.z0m/10.)))
    water_velocity=.16*max(f.wind,.3)/(momentum*np.log(state.humidity_height/(state.z0m/10.)))
    heat_temperature=f.tair
    if state.options.OptSurfaceDrag==1 and _exchange is None:
        boundary=CanopyExchange(f.tair,f.qair,f.ps,heat_velocity,heat_velocity,.01,state.surface_temperature,0.,
            atmosphere_water_velocity=water_velocity)
        def solve(exchange):
            new,d=glacier_surface_step(state,f,cosine_zenith=cosine_zenith,diffuse_fraction=diffuse_fraction,
                temperature_tolerance=temperature_tolerance,_exchange=exchange)
            d.update(et_kgm2=d['et_mm'],leaf_temperature=np.empty(0))
            return new,d
        final,d=solve_surface_most(solve,boundary,f.wind,geometry,plant_area=0.,time_step=f.dt)
        d.pop('leaf_temperature');return final,d
    if _exchange is not None:
        heat_velocity=_exchange.atmosphere_velocity;water_velocity=_exchange.water_velocity
        heat_temperature=_exchange.heat_temperature
    rho=air_density(f)
    supply=float(snow0.mass[0] if len(snow0.mass) else ice0.mass[0]);evaluations=0
    def evaluate(t):
        nonlocal evaluations
        evaluations+=1
        saturation=float((_ice_saturation if t<TF else _saturation)(t,f.ps)[0])
        mass=min(rho*water_velocity*(saturation-f.qair)*f.dt,supply)
        snow,ice=snow0,ice0;qs=rad['absorbed'].copy();qi=np.zeros(len(ice.mass));qi[0]=rad['ground_absorbed']
        if len(snow.mass):
            snow,v=exchange_vapor(snow,mass,deposition_temperature=t,remap=False)
            if len(snow.mass)<len(snow0.mass):
                removed=qs[0];qs=qs[1:]
                if len(qs):qs[0]+=removed
                else:qi[0]+=removed
            out=oh=0.;tracer=v['residue_kgm2']
        else:
            ice,v=exchange_glacier_vapor(ice,mass,t)
            out,oh=v['runoff_kgm2'],v['runoff_enthalpy_jm2'];tracer=v['residue_kgm2']
            if len(ice.mass)<len(ice0.mass):
                removed=qi[0];qi=qi[1:]
                if len(qi):qi[0]+=removed
            if not len(ice.mass):raise RuntimeError('Glacier exhausted by vapor exchange; reduce the forcing time step')
        sensible=rho*CP*heat_velocity*(t-heat_temperature);latent=v['latent_energy_jm2']/f.dt
        longwave=p.emissivity*(f.lwdown-SIGMA*t**4);top=longwave-sensible-latent
        news,newi,d=glacier_heat_step(snow,ice,f.dt,top,snow_absorbed=qs,ice_absorbed=qi,parameters=p)
        actual=float(d['thermal_temperature_k'][0])
        return actual-t,dict(snow=news,ice=newi,heat=d,vapor=v,h=sensible,le=latent,temperature=t,
            rnet=f.swdown-rad['reflected']+longwave,runoff=out+d['runoff_kgm2'],
            runoff_heat=oh+d['runoff_enthalpy_jm2'],tracer=tracer+d['runoff_impurities_kgm2'])
    center=float(np.clip(state.surface_temperature,200.,350.));fc,result=evaluate(center)
    if abs(fc)>temperature_tolerance:
        left=right=center;fl=fr=fc;width=2.
        for _ in range(8):
            if fl<0. and left>200.:left=max(200.,center-width);fl,_=evaluate(left)
            if fr>0. and right<350.:right=min(350.,center+width);fr,_=evaluate(right)
            if fl>=0. and fr<=0.:break
            width*=2.
        else:raise RuntimeError('Glacier surface root not bracketed in 200--350 K')
        for _ in range(60):
            trial=(left*fr-right*fl)/(fr-fl) if fr!=fl else .5*(left+right)
            trial=float(np.clip(trial,left+.05*(right-left),right-.05*(right-left)))
            residual,result=evaluate(trial)
            if abs(residual)<=temperature_tolerance:break
            if residual>0.:left,fl=trial,residual
            else:right,fr=trial,residual
        else:raise RuntimeError(f'Glacier surface did not converge: {residual} K')
    if result['heat']['thermal_temperature_k'][0]==TF and result['temperature']!=TF:
        residual,plateau=evaluate(TF)
        if abs(residual)<=temperature_tolerance:result=plateau
    snow,ice=result['snow'],result['ice'];thermal=result['heat'];v=result['vapor']
    # Geometry and optical radius evolve without altering M/H/tracer stores.
    # Aging uses the actual final interface gradient, including exposed ice.
    if len(snow.mass):
        z=np.r_[snow.thickness,ice.thickness[0]] if len(ice.mass) else snow.thickness
        t=np.r_[snow.temperature,ice.temperature[0]] if len(ice.mass) else snow.temperature
        centers=np.cumsum(z)-.5*z
        gradient=np.abs(np.gradient(t,centers))[:len(snow.mass)] if len(t)>1 else np.zeros(1)
        snow=grain_growth(compact(snow,f.dt,remap=False),f.dt,temperature_gradient=gradient)
    final=replace(state,snow=snow,ice=ice);atm=(result['rnet']-result['h']-result['le'])*f.dt
    dm=final.mass_kgm2-state.mass_kgm2+v['mass_kgm2']+result['runoff']
    dh=final.enthalpy_jm2-state.enthalpy_jm2-atm-thermal['geothermal_input_jm2']+v['advected_enthalpy_jm2']+result['runoff_heat']
    scale=abs(state.enthalpy_jm2)+abs(atm)+abs(result['runoff_heat'])+abs(v['advected_enthalpy_jm2'])
    wm=thermal['water_bound_kgm2']+64.*np.finfo(float).eps*max(state.mass_kgm2+abs(v['mass_kgm2']),1.)
    wh=thermal['energy_bound_jm2']+64.*np.finfo(float).eps*max(scale,1.)
    if abs(dm)>wm or abs(dh)>wh:raise RuntimeError('Glacier surface mass/enthalpy conservation failed')
    return final,dict(thermal=thermal,radiation=rad,water_residual_kgm2=dm,energy_residual_jm2=dh,
        water_bound_kgm2=wm,energy_bound_jm2=wh,h=result['h'],le=result['le'],rnet=result['rnet'],
        et_mm=v['mass_kgm2'],runoff_mm=result['runoff'],runoff_enthalpy_jm2=result['runoff_heat'],
        impurity_out_kgm2=result['tracer'],advected_enthalpy_jm2=v['advected_enthalpy_jm2'],
        atmospheric_energy_jm2=atm,geothermal_input_jm2=thermal['geothermal_input_jm2'],
        flux_surface_temperature=result['temperature'],surface_evaluations=evaluations,
        heat_velocity=heat_velocity,water_velocity=water_velocity,
        ts=final.surface_temperature if len(snow.mass)+len(ice.mass) else None,
        g=thermal['thermal']['input_energy_jm2']/f.dt-p.geothermal_wm2,swe=float(snow.mass.sum()/1000.))


def step_clm_glacier_column(state,forcing,*,cosine_zenith=.5,diffuse_fraction=.5,deposition_rates=None):
    """Full precipitation/atmosphere/snow/finite-ice step for glacier option 1."""
    f=forcing
    if (not all(np.isfinite(v) for v in vars(f).values()) or f.dt<=0. or f.prcp<0. or f.wind<0.
        or f.ps<30000. or not 200.<=f.tair<=330. or not 0.<=f.qair<.2 or f.swdown<0. or f.lwdown<0.):
        raise ValueError('Invalid glacier meteorological forcing')
    if not len(state.ice.mass):raise ValueError('Exhausted finite glacier cannot advance')
    rain,snowfall=partition_precipitation(f.prcp,f.dt,f.tair,state.options.OptRainSnowPartition,qair=f.qair,pressure=f.ps)
    rain*=1000.;snowfall*=1000.;precip_h=rain*(LF+CW*(f.tair-TF))+snowfall*CI*(min(f.tair,TF)-TF)
    snow,ice=state.snow,state.ice;out=oh=0.;tracer=np.zeros(5)
    if snowfall:
        density,radius=fresh_snow_properties(f.tair,f.wind)
        snow=add_snowfall(snow,snowfall,min(f.tair,TF),float(density),float(radius))
    if rain:
        if len(snow.mass):snow=add_rain(snow,rain,f.tair)
        else:
            ice,flow=add_glacier_surface_water(ice,rain,rain*(LF+CW*(f.tair-TF)))
            out,oh=flow['runoff_kgm2'],flow['runoff_enthalpy_jm2'];tracer=flow['runoff_impurities_kgm2']
    work=replace(state,snow=snow,ice=ice)
    final,d=glacier_surface_step(work,f,cosine_zenith=cosine_zenith,diffuse_fraction=diffuse_fraction)
    if deposition_rates is not None:
        snow,bare=deposit_impurities(final.snow,deposition_rates,f.dt);ice=final.ice
        if len(ice.mass):
            a=ice.impurities.copy();a[0]+=bare;ice=GlacierIce(ice.mass,ice.enthalpy,a)
        else:tracer+=bare
        final=replace(final,snow=snow,ice=ice)
    d['runoff_mm']+=out;d['runoff_enthalpy_jm2']+=oh;d['impurity_out_kgm2']+=tracer
    d.update(prcp_mm=rain+snowfall,precipitation_enthalpy_jm2=precip_h,
        water_residual_kgm2=final.mass_kgm2-state.mass_kgm2-(rain+snowfall)+d['et_mm']+d['runoff_mm'],
        energy_residual_jm2=final.enthalpy_jm2-state.enthalpy_jm2-precip_h-d['atmospheric_energy_jm2']-
            d['geothermal_input_jm2']+d['advected_enthalpy_jm2']+d['runoff_enthalpy_jm2'])
    d['water_bound_kgm2']+=64.*np.finfo(float).eps*max(rain+snowfall+state.mass_kgm2,1.)
    d['energy_bound_jm2']+=64.*np.finfo(float).eps*max(abs(precip_h)+abs(state.enthalpy_jm2),1.)
    if abs(d['water_residual_kgm2'])>d['water_bound_kgm2'] or abs(d['energy_residual_jm2'])>d['energy_bound_jm2']:
        raise RuntimeError('Full glacier mass/enthalpy conservation failed')
    return final,d
