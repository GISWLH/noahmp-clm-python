"""Developing all-season extensive CLM column: precipitation to runoff.

An optional layered canopy carries intercepted ice/liquid and their enthalpy
through the leaf solve, snowfall, drip and unloading. Explicit MOST geometry
enables actual H/E feedback with independent heat and humidity measurement
heights. The released default driver is not switched by this development entry.
"""
from dataclasses import replace,fields
import numpy as np
from .clm_state import clm_options
from .physics.snowpack import TF,CI,CW,LF,Snowpack,add_snowfall,add_rain,drain_liquid
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
from .physics.vegetation_structure import step_stem_area,exposed_canopy_geometry


def step_clm_enthalpy_column(state,forcing,*,co2_ppm=420.,cosine_zenith=.5,diffuse_fraction=.5,
                            leaf_dimension=.04,canopy_layers=10,leaf_parameters=None,nitrogen_factor=1.,
                            deposition_rates=None,most_geometry=None,vic_parameters=None,phenology_environment=None,
                            nitrogen_deposition_gn_m2=0.,oxygen_factor=1.,crop_management=None,irrigation_input=None,
                            prescribed_sai=None):
    """Return a new CLMColumnState and actual extensive closure diagnostics."""
    if state.bgc_state is not None:
        from .clm_bgc_driver import step_clm_bgc_column
        return step_clm_bgc_column(state,forcing,co2_ppm=co2_ppm,cosine_zenith=cosine_zenith,diffuse_fraction=diffuse_fraction,
            leaf_dimension=leaf_dimension,canopy_layers=canopy_layers,leaf_parameters=leaf_parameters,nitrogen_factor=nitrogen_factor,
            deposition_rates=deposition_rates,most_geometry=most_geometry,vic_parameters=vic_parameters,
            phenology_environment=phenology_environment,nitrogen_deposition_gn_m2=nitrogen_deposition_gn_m2,oxygen_factor=oxygen_factor,
            crop_management=crop_management,irrigation_input=irrigation_input,prescribed_sai=prescribed_sai)
    if crop_management is not None:raise ValueError('Crop management requires an annual crop BGC column')
    if nitrogen_deposition_gn_m2!=0. or oxygen_factor!=1.:
        raise ValueError('Nitrogen deposition/oxygen modifiers require a BGC state')
    f=forcing;opt=state.options
    vic_surface=opt.OptRunoffSurface==6;vic_bottom=opt.OptRunoffSubsurface==6
    if vic_surface or vic_bottom:
        if opt.OptRunoffSurface not in (3,6) or opt.OptRunoffSubsurface not in (3,6):
            raise NotImplementedError('VIC supports explicit surface/bottom combinations of options 3 and 6')
        from .physics.vic import VICParameters,infiltrate_vic
        if vic_parameters is None:vic_parameters=VICParameters()
        if not isinstance(vic_parameters,VICParameters):raise TypeError('Expected VICParameters')
    elif vic_parameters is not None:raise ValueError('VIC parameters require runoff option 6')
    def admit(earth,mass,heat):
        return infiltrate_vic(earth,mass,heat,vic_parameters) if vic_surface else infiltrate(earth,mass,heat)
    if most_geometry is not None and opt.OptSurfaceDrag!=1:
        raise ValueError('Explicit MOST geometry requires OptSurfaceDrag=1')
    supported={'OptRainSnowPartition','OptSoilWaterTranspiration','OptGroundResistanceEvap',
        'OptSurfaceDrag','OptSoilSupercoolWater','OptRunoffSurface','OptRunoffSubsurface','OptSoilPermeabilityFrozen','OptDynamicVeg',
        'OptIrrigation','OptIrrigationMethod','OptTileDrainage'}
    baseline=clm_options()
    for item in fields(opt):
        if item.name not in supported and getattr(opt,item.name)!=getattr(baseline,item.name):
            raise NotImplementedError(f'{item.name} is outside the current CLM enthalpy profile')
    if (not np.all(np.isfinite(list(vars(f).values()))) or f.dt<=0. or f.prcp<0. or f.wind<0.
            or not isinstance(canopy_layers,int) or canopy_layers<1 or leaf_dimension<=0.):
        raise ValueError('Invalid CLM forcing or canopy discretization')
    phenology=None;previous_lai=state.lai
    if opt.OptDynamicVeg==5:
        from .physics.phenology import PhenologyEnvironment,step_phenology
        if not isinstance(phenology_environment,PhenologyEnvironment):
            raise ValueError('Dynamic vegetation requires an explicit PhenologyEnvironment')
        e=phenology_environment;old_lai=state.lai
        next_phenology,phenology=step_phenology(state.phenology_state,float(state.ground.temperature[2]),
            e.daylength_seconds,e.increasing_daylength,e.winter_solstice,f.dt,state.phenology_parameters)
        state=replace(state,phenology_state=next_phenology)
        phenology.update(lai_before=old_lai)
    elif phenology_environment is not None:
        raise ValueError('Phenology environment requires OptDynamicVeg=5')
    structure=None;geometry=None
    if state.canopy_structure_parameters is not None:
        sai,structure=step_stem_area(state.sai,previous_lai,state.lai,f.dt,
            state.canopy_structure_parameters,prescribed_sai=prescribed_sai)
        state=replace(state,sai=sai)
        # Geometry is frozen from beginning-of-step snow depth, before the
        # interception/precipitation split. Total LAI/SAI remain state variables.
        geometry=exposed_canopy_geometry(state.lai,state.sai,float(state.snow.thickness.sum()),
            state.canopy_structure_parameters)
    elif prescribed_sai is not None:raise ValueError('Prescribed SAI requires explicit canopy structure')
    irrigation=None;irrigation_state=state.irrigation_state;irrigation_mass=irrigation_heat=0.
    if opt.OptIrrigation==1:
        from .physics.irrigation import IrrigationInput,step_irrigation
        request=IrrigationInput() if irrigation_input is None else irrigation_input
        if not isinstance(request,IrrigationInput):raise ValueError('Expected IrrigationInput')
        # Reconstruct phase diagnostics within the unchanged total store.
        # Volume-to-mass round trips can otherwise report -3e-15 kg ice in
        # a warm layer. Warm soil is exactly liquid; no stored M/H is clipped.
        irrigation_liquid=np.where(state.ground.temperature>=TF,state.ground.water,
            np.clip(state.ground.liquid,0.,state.ground.water))
        irrigation_state,irrigation=step_irrigation(irrigation_state,irrigation_liquid,state.ground.water-irrigation_liquid,state.ground.dz,
            f.dt,state.lai,check=request.check,supply_kgm2=request.supply_kgm2,supply_temperature_k=request.supply_temperature_k,
            parameters=state.irrigation_parameters)
        irrigation_mass=irrigation['applied_kgm2'];irrigation_heat=irrigation['applied_enthalpy_jm2']
    elif irrigation_input is not None:raise ValueError('Irrigation input requires OptIrrigation=1')
    rain,snowfall=partition_precipitation(f.prcp,f.dt,f.tair,opt.OptRainSnowPartition,qair=f.qair,pressure=f.ps)
    rain*=1000.;snowfall*=1000.
    rain_heat=LF+CW*(f.tair-TF);ice_heat=CI*(min(f.tair,TF)-TF)
    precipitation_heat=rain*rain_heat+snowfall*ice_heat
    snow,ground=state.snow,state.ground;canopy=state.canopy_water_kgm2;canopy_heat=state.canopy_enthalpy_jm2
    condensate=state.canopy_condensate
    if condensate is not None and len(condensate.mass)!=canopy_layers:raise ValueError('Canopy state and leaf discretization disagree')
    exposed_lai=state.lai if geometry is None else geometry['exposed_lai']
    exposed_sai=0. if geometry is None else geometry['exposed_sai']
    plant_area=exposed_lai+exposed_sai
    area=np.full(canopy_layers,exposed_lai/canopy_layers)
    stems=np.full(canopy_layers,exposed_sai/canopy_layers)
    capacity=.5*state.fveg*plant_area
    runoff=runoff_heat=0.
    def route_water(pack,earth,mass,heat):
        if mass==0.: return pack,earth,0.,0.
        if len(pack.mass):
            temperature=TF+(heat/mass-LF)/CW
            pack=add_rain(pack,mass,temperature)
            return pack,earth,0.,0.
        earth,out,out_heat=admit(earth,mass,heat)
        return pack,earth,out,out_heat
    def route_canopy_outlets(pack,earth,flow):
        ice=flow['ground_ice_kgm2']
        if ice:
            density,radius=fresh_snow_properties(f.tair,f.wind)
            temperature=TF+flow['ground_ice_enthalpy_jm2']/(ice*CI)
            pack=add_snowfall(pack,ice,temperature,float(density),float(radius))
        return route_water(pack,earth,flow['ground_liquid_kgm2'],flow['ground_liquid_enthalpy_jm2'])
    if condensate is not None:
        condensate,interception=intercept_precipitation(condensate,rain,snowfall,f.tair,f.dt/2.,area,stem_area=stems,vegetated_fraction=state.fveg,wind=f.wind)
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
    # Surface irrigation bypasses leaves but traverses any snow and the
    # actual infiltration/runoff operator, retaining the supplied enthalpy.
    if irrigation_mass:
        snow,ground,out,oh=route_water(snow,ground,irrigation_mass,irrigation_heat)
        runoff+=out;runoff_heat+=oh
    # A fully melted precipitation pack is liquid, not a snow thermal cell.
    # Route its complete M/H and tracers before surface optics/evaporation.
    # Otherwise an arbitrarily thin rain-filled cell disappears at its
    # evaporation limit and makes the MOST surface response discontinuous.
    pre_impurity_out=np.zeros(5)
    if len(snow.mass) and np.all(snow.ice==0.):
        ground,out,oh=admit(ground,float(snow.mass.sum()),float(snow.enthalpy.sum()))
        runoff+=out;runoff_heat+=oh
        pre_impurity_out=snow.impurities.sum(axis=0)
        snow=Snowpack.empty()
    work=replace(state,ground=ground,snow=snow,canopy_water_kgm2=canopy,canopy_enthalpy_jm2=canopy_heat,canopy_condensate=condensate)
    view=work.parameter_view();view.smc=ground.liquid/(1000.*ground.dz)
    stress=soil_water_stress(view,opt.OptSoilWaterTranspiration)
    velocity=exchange_velocity(f,state.z0m,state.surface_temperature,opt.OptSurfaceDrag)
    z0=np.clip(state.z0m,1e-5,5.);ustar=.4*max(f.wind,.3)/np.log(10./z0)
    leaf_velocity=.01*np.sqrt(ustar/leaf_dimension)
    coefficient=.4/.13*(z0*ustar/1.5e-5)**(-.45)
    ground_velocity=(coefficient*np.exp(-plant_area)+.004*(1.-np.exp(-plant_area)))*ustar
    exchange=CanopyExchange(f.tair,f.qair,f.ps,velocity,ground_velocity,leaf_velocity,float(ground.temperature[0]),0.,
        co2_ppm=co2_ppm,ground_resistance=ground_resistance(view,opt.OptGroundResistanceEvap,snow_cover=bool(len(snow.mass))))
    wet=min(1.,(canopy/capacity)**(2./3.)) if capacity>0. else 0.
    storage_args=(dict(canopy_water_kgm2=canopy,canopy_enthalpy_jm2=canopy_heat) if condensate is None else dict(canopy_condensate=condensate,stem_area=stems))
    # Working guesses live only within this physical step. Every MOST trial
    # still solves from the same old extensive stores.
    phase_cache={} if condensate is not None else None
    def solve_surface(boundary):
        return surface_step(snow,ground,f.dt,area,f.swdown,f.lwdown,
            boundary,**storage_args,root_weights=state.root_frac,_phase_cache=phase_cache,
            wet_fraction=wet,water_stress=stress,vegetated_fraction=state.fveg,nitrogen_factor=nitrogen_factor,
            leaf_parameters=leaf_parameters,ground_albedo=(state.albedo,state.albedo),ground_emissivity=state.emiss,
            cosine_zenith=cosine_zenith,diffuse_fraction=diffuse_fraction,
            infiltration_parameters=vic_parameters if vic_surface else None)
    if most_geometry is None:surface_result=solve_surface(exchange)
    else:
        from .physics.canopy_most import solve_surface_most
        surface_result=solve_surface_most(solve_surface,exchange,f.wind,most_geometry,
            leaf_dimension=leaf_dimension,plant_area=plant_area,time_step=f.dt)
    snow,ground,canopy,d=surface_result
    canopy_heat=d['canopy_enthalpy_after_jm2'];runoff+=d['runoff_kgm2'];runoff_heat+=d['runoff_enthalpy_jm2']
    impurity_out=d['impurity_output_kgm2']+pre_impurity_out
    if condensate is not None:
        condensate,post=intercept_precipitation(d['canopy_condensate'],0.,0.,f.tair,f.dt/2.,area,stem_area=stems,vegetated_fraction=state.fveg,wind=f.wind)
        snow,ground,out,oh=route_canopy_outlets(snow,ground,post);runoff+=out;runoff_heat+=oh
        canopy=float(condensate.mass.sum());canopy_heat=float(condensate.enthalpy.sum())
        snow,flow=drain_liquid(snow,return_ledger=True)
        ground,out,oh=admit(ground,flow['drainage_kgm2'],flow['drainage_enthalpy_jm2'])
        runoff+=out+flow['runoff_kgm2'];runoff_heat+=oh+flow['runoff_enthalpy_jm2']
        impurity_out+=flow['runoff_impurities_kgm2']+flow['drainage_impurities_kgm2']
        d.update(canopy_condensate=condensate,canopy_interception=interception,canopy_post_interception=post)
    elif canopy>capacity:
        drip=canopy-capacity;heat=drip*canopy_heat/canopy
        canopy-=drip;canopy_heat-=heat
        snow,ground,out,oh=route_water(snow,ground,drip,heat);runoff+=out;runoff_heat+=oh
        snow,flow=drain_liquid(snow,return_ledger=True)
        ground,out,oh=admit(ground,flow['drainage_kgm2'],flow['drainage_enthalpy_jm2'])
        runoff+=out+flow['runoff_kgm2'];runoff_heat+=oh+flow['runoff_enthalpy_jm2']
        impurity_out+=flow['runoff_impurities_kgm2']+flow['drainage_impurities_kgm2']
    ground,drain=drain_soil(ground,f.dt,permeability_option=opt.OptSoilPermeabilityFrozen,
        bottom_parameters=vic_parameters if vic_bottom else None)
    runoff+=drain['bottom_mass_kgm2'];runoff_heat+=drain['bottom_enthalpy_jm2']
    routing_ground=ground;tile=None;tile_mass=0.
    if state.tile_drainage_parameters is not None:
        from .physics.tile_drainage import step_tile_drainage
        ground,tile=step_tile_drainage(ground,f.dt,state.tile_drainage_parameters)
        tile_mass=tile['drained_kgm2'];runoff_heat+=tile['drained_enthalpy_jm2']
    if deposition_rates is not None:
        snow,bare=deposit_impurities(snow,deposition_rates,f.dt);impurity_out+=bare
    # Melting/drainage can remove the layer that supplied this step's flux
    # temperature. The next step's drag must see the surviving surface.
    final_temperature=float(snow.temperature[0] if len(snow.mass) else ground.temperature[0])
    final=replace(state,ground=ground,snow=snow,canopy_water_kgm2=canopy,canopy_enthalpy_jm2=canopy_heat,
        surface_temperature=final_temperature,canopy_condensate=condensate,irrigation_state=irrigation_state)
    if vic_surface or vic_bottom:
        # These are physical outlets, already removed from the state with heat.
        deep=drain['bottom_mass_kgm2']/1000.;surface=(runoff-drain['bottom_mass_kgm2'])/1000.
    else:surface,deep=route_runoff(runoff/1000.,replace(final,ground=routing_ground).parameter_view(),opt.OptRunoffSurface,opt.OptRunoffSubsurface)
    # Tile flow is a known subsurface outlet, never repartitioned as overland
    # runoff by the legacy diagnostic routing fractions.
    deep+=tile_mass/1000.;runoff+=tile_mass
    d.update(ts=final.surface_temperature,h=float(d['h']),le=float(d['le']),g=float(d['ground_heat']),
        et_mm=d['et_kgm2'],prcp_mm=rain+snowfall,runoff_mm=runoff,runoff_kgm2=runoff,
        runoff_surface_mm=surface*1000.,runoff_deep_mm=deep*1000.,
        bottom_outflow_kgm2=drain['bottom_mass_kgm2'],bottom_outflow_enthalpy_jm2=drain['bottom_enthalpy_jm2'],
        flux_surface_temperature=d['surface_temperature'],surface_temperature=final_temperature,
        swe=final.swe,canw=final.canw,tsoil=final.tsoil,smc=final.smc,
        gpp_umol_m2s=d['gpp'],gpp_gcm2=d['gpp']*12.0107e-6*f.dt,
        precipitation_enthalpy_jm2=precipitation_heat,runoff_enthalpy_jm2=runoff_heat,
        water_residual_kgm2=final.water_kgm2-state.water_kgm2-rain-snowfall-irrigation_mass+d['et_kgm2']+runoff,
        energy_residual_jm2=final.enthalpy_jm2-state.enthalpy_jm2-precipitation_heat-irrigation_heat-d['atmospheric_energy_jm2']+d['advected_enthalpy_jm2']+runoff_heat,
        impurity_output_kgm2=impurity_out,canopy_water_after_kgm2=canopy,canopy_enthalpy_after_jm2=canopy_heat)
    if phenology is not None:d['phenology']=phenology
    if geometry is not None:
        d['canopy_geometry']=geometry
        d['vegetation_structure']=structure
    if irrigation is not None:
        d.update(irrigation=irrigation,irrigation_mm=irrigation_mass,irrigation_enthalpy_jm2=irrigation_heat)
    if tile is not None:
        d.update(tile_drainage=tile,tile_drainage_mm=tile_mass,tile_drainage_enthalpy_jm2=tile['drained_enthalpy_jm2'])
    return final,d
