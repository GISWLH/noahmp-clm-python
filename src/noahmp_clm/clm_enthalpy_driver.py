"""Developing all-season extensive CLM column: precipitation to runoff.

An optional layered canopy carries intercepted ice/liquid and their enthalpy
through the leaf solve, snowfall, drip and unloading. Explicit MOST geometry
enables actual H/E feedback with independent heat and humidity measurement
heights. The released default driver is not switched by this development entry.
"""
from dataclasses import replace,fields
import numpy as np
from .clm_state import clm_options
from .physics.snowpack import TF,CI,CW,LF,add_snowfall,add_rain,drain_liquid
from .physics.snow_evolution import fresh_snow_properties,deposit_impurities
from .physics.soil_enthalpy import infiltrate
from .physics.soil_drainage import drain_soil
from .physics.canopy_column import surface_step
from .physics.canopy_energy import CanopyExchange
from .physics.precip import partition_precipitation
from .physics.surface_drag import exchange_velocity
from .physics.ground_evap import ground_resistance
from .physics.stress import soil_water_stress
from .physics.runoff import route_runoff
from .physics.canopy_interception import intercept_precipitation


def step_clm_enthalpy_column(state,forcing,*,co2_ppm=420.,cosine_zenith=.5,diffuse_fraction=.5,
                            leaf_dimension=.04,canopy_layers=10,leaf_parameters=None,nitrogen_factor=1.,
                            deposition_rates=None,most_geometry=None):
    """Return a new CLMColumnState and actual extensive closure diagnostics."""
    f=forcing;opt=state.options
    if most_geometry is not None and opt.OptSurfaceDrag!=1:
        raise ValueError('Explicit MOST geometry requires OptSurfaceDrag=1')
    supported={'OptRainSnowPartition','OptSoilWaterTranspiration','OptGroundResistanceEvap',
        'OptSurfaceDrag','OptSoilSupercoolWater','OptRunoffSurface','OptRunoffSubsurface','OptSoilPermeabilityFrozen'}
    baseline=clm_options()
    for item in fields(opt):
        if item.name not in supported and getattr(opt,item.name)!=getattr(baseline,item.name):
            raise NotImplementedError(f'{item.name} is outside the current CLM enthalpy profile')
    if (not np.all(np.isfinite(list(vars(f).values()))) or f.dt<=0. or f.prcp<0. or f.wind<0.
            or not isinstance(canopy_layers,int) or canopy_layers<1 or leaf_dimension<=0.):
        raise ValueError('Invalid CLM forcing or canopy discretization')
    rain,snowfall=partition_precipitation(f.prcp,f.dt,f.tair,opt.OptRainSnowPartition,qair=f.qair,pressure=f.ps)
    rain*=1000.;snowfall*=1000.
    rain_heat=LF+CW*(f.tair-TF);ice_heat=CI*(min(f.tair,TF)-TF)
    precipitation_heat=rain*rain_heat+snowfall*ice_heat
    snow,ground=state.snow,state.ground;canopy=state.canopy_water_kgm2;canopy_heat=state.canopy_enthalpy_jm2
    condensate=state.canopy_condensate
    if condensate is not None and len(condensate.mass)!=canopy_layers:raise ValueError('Canopy state and leaf discretization disagree')
    area=np.full(canopy_layers,state.lai/canopy_layers)
    capacity=.5*state.fveg*state.lai
    runoff=runoff_heat=0.
    def route_water(pack,earth,mass,heat):
        if mass==0.: return pack,earth,0.,0.
        if len(pack.mass):
            temperature=TF+(heat/mass-LF)/CW
            pack=add_rain(pack,mass,temperature)
            return pack,earth,0.,0.
        earth,out,out_heat=infiltrate(earth,mass,heat)
        return pack,earth,out,out_heat
    def route_canopy_outlets(pack,earth,flow):
        ice=flow['ground_ice_kgm2']
        if ice:
            density,radius=fresh_snow_properties(f.tair,f.wind)
            temperature=TF+flow['ground_ice_enthalpy_jm2']/(ice*CI)
            pack=add_snowfall(pack,ice,temperature,float(density),float(radius))
        return route_water(pack,earth,flow['ground_liquid_kgm2'],flow['ground_liquid_enthalpy_jm2'])
    if condensate is not None:
        condensate,interception=intercept_precipitation(condensate,rain,snowfall,f.tair,f.dt/2.,area,vegetated_fraction=state.fveg,wind=f.wind)
        snow,ground,out,oh=route_canopy_outlets(snow,ground,interception);runoff+=out;runoff_heat+=oh
        canopy=float(condensate.mass.sum());canopy_heat=float(condensate.enthalpy.sum())
    elif canopy>capacity:
        drip=canopy-capacity;heat=drip*canopy_heat/canopy
        canopy-=drip;canopy_heat-=heat
        snow,ground,out,oh=route_water(snow,ground,drip,heat);runoff+=out;runoff_heat+=oh
    if condensate is None:
        intercepted=min(state.fveg*rain,max(0.,capacity-canopy))
        canopy+=intercepted;canopy_heat+=intercepted*rain_heat
        if snowfall:
            density,radius=fresh_snow_properties(f.tair,f.wind)
            snow=add_snowfall(snow,snowfall,min(f.tair,TF),float(density),float(radius))
        snow,ground,out,oh=route_water(snow,ground,rain-intercepted,(rain-intercepted)*rain_heat)
        runoff+=out;runoff_heat+=oh
    work=replace(state,ground=ground,snow=snow,canopy_water_kgm2=canopy,canopy_enthalpy_jm2=canopy_heat,canopy_condensate=condensate)
    view=work.parameter_view();view.smc=ground.liquid/(1000.*ground.dz)
    stress=soil_water_stress(view,opt.OptSoilWaterTranspiration)
    velocity=exchange_velocity(f,state.z0m,state.surface_temperature,opt.OptSurfaceDrag)
    z0=np.clip(state.z0m,1e-5,5.);ustar=.4*max(f.wind,.3)/np.log(10./z0)
    leaf_velocity=.01*np.sqrt(ustar/leaf_dimension)
    coefficient=.4/.13*(z0*ustar/1.5e-5)**(-.45)
    ground_velocity=(coefficient*np.exp(-state.lai)+.004*(1.-np.exp(-state.lai)))*ustar
    exchange=CanopyExchange(f.tair,f.qair,f.ps,velocity,ground_velocity,leaf_velocity,float(ground.temperature[0]),0.,
        co2_ppm=co2_ppm,ground_resistance=ground_resistance(view,opt.OptGroundResistanceEvap,snow_cover=bool(len(snow.mass))))
    wet=min(1.,(canopy/capacity)**(2./3.)) if capacity>0. else 0.
    storage_args=(dict(canopy_water_kgm2=canopy,canopy_enthalpy_jm2=canopy_heat) if condensate is None else dict(canopy_condensate=condensate))
    def solve_surface(boundary):
        return surface_step(snow,ground,f.dt,area,f.swdown,f.lwdown,
            boundary,**storage_args,root_weights=state.root_frac,
            wet_fraction=wet,water_stress=stress,vegetated_fraction=state.fveg,nitrogen_factor=nitrogen_factor,
            leaf_parameters=leaf_parameters,ground_albedo=(state.albedo,state.albedo),ground_emissivity=state.emiss,
            cosine_zenith=cosine_zenith,diffuse_fraction=diffuse_fraction)
    if most_geometry is None:surface_result=solve_surface(exchange)
    else:
        from .physics.canopy_most import solve_surface_most
        surface_result=solve_surface_most(solve_surface,exchange,f.wind,most_geometry,
            leaf_dimension=leaf_dimension,plant_area=state.lai,time_step=f.dt)
    snow,ground,canopy,d=surface_result
    canopy_heat=d['canopy_enthalpy_after_jm2'];runoff+=d['runoff_kgm2'];runoff_heat+=d['runoff_enthalpy_jm2']
    impurity_out=d['impurity_output_kgm2'].copy()
    if condensate is not None:
        condensate,post=intercept_precipitation(d['canopy_condensate'],0.,0.,f.tair,f.dt/2.,area,vegetated_fraction=state.fveg,wind=f.wind)
        snow,ground,out,oh=route_canopy_outlets(snow,ground,post);runoff+=out;runoff_heat+=oh
        canopy=float(condensate.mass.sum());canopy_heat=float(condensate.enthalpy.sum())
        snow,flow=drain_liquid(snow,return_ledger=True)
        ground,out,oh=infiltrate(ground,flow['drainage_kgm2'],flow['drainage_enthalpy_jm2'])
        runoff+=out+flow['runoff_kgm2'];runoff_heat+=oh+flow['runoff_enthalpy_jm2']
        impurity_out+=flow['runoff_impurities_kgm2']+flow['drainage_impurities_kgm2']
        d.update(canopy_condensate=condensate,canopy_interception=interception,canopy_post_interception=post)
    elif canopy>capacity:
        drip=canopy-capacity;heat=drip*canopy_heat/canopy
        canopy-=drip;canopy_heat-=heat
        snow,ground,out,oh=route_water(snow,ground,drip,heat);runoff+=out;runoff_heat+=oh
        snow,flow=drain_liquid(snow,return_ledger=True)
        ground,out,oh=infiltrate(ground,flow['drainage_kgm2'],flow['drainage_enthalpy_jm2'])
        runoff+=out+flow['runoff_kgm2'];runoff_heat+=oh+flow['runoff_enthalpy_jm2']
        impurity_out+=flow['runoff_impurities_kgm2']+flow['drainage_impurities_kgm2']
    ground,drain=drain_soil(ground,f.dt,permeability_option=opt.OptSoilPermeabilityFrozen)
    runoff+=drain['bottom_mass_kgm2'];runoff_heat+=drain['bottom_enthalpy_jm2']
    if deposition_rates is not None:
        snow,bare=deposit_impurities(snow,deposition_rates,f.dt);impurity_out+=bare
    # Melting/drainage can remove the layer that supplied this step's flux
    # temperature. The next step's drag must see the surviving surface.
    final_temperature=float(snow.temperature[0] if len(snow.mass) else ground.temperature[0])
    final=replace(state,ground=ground,snow=snow,canopy_water_kgm2=canopy,canopy_enthalpy_jm2=canopy_heat,
        surface_temperature=final_temperature,canopy_condensate=condensate)
    surface,deep=route_runoff(runoff/1000.,final.parameter_view(),opt.OptRunoffSurface,opt.OptRunoffSubsurface)
    d.update(ts=final.surface_temperature,h=float(d['h']),le=float(d['le']),g=float(d['ground_heat']),
        et_mm=d['et_kgm2'],prcp_mm=rain+snowfall,runoff_mm=runoff,runoff_kgm2=runoff,
        runoff_surface_mm=surface*1000.,runoff_deep_mm=deep*1000.,
        flux_surface_temperature=d['surface_temperature'],surface_temperature=final_temperature,
        swe=final.swe,canw=final.canw,tsoil=final.tsoil,smc=final.smc,
        gpp_umol_m2s=d['gpp'],gpp_gcm2=d['gpp']*12.0107e-6*f.dt,
        precipitation_enthalpy_jm2=precipitation_heat,runoff_enthalpy_jm2=runoff_heat,
        water_residual_kgm2=final.water_kgm2-state.water_kgm2-rain-snowfall+d['et_kgm2']+runoff,
        energy_residual_jm2=final.enthalpy_jm2-state.enthalpy_jm2-precipitation_heat-d['atmospheric_energy_jm2']+d['advected_enthalpy_jm2']+runoff_heat,
        impurity_output_kgm2=impurity_out,canopy_water_after_kgm2=canopy,canopy_enthalpy_after_jm2=canopy_heat)
    return final,d
