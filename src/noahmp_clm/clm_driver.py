"""Developing CLM column path: multilayer canopy connected to soil water/heat.

The legacy ColumnState path covers snow-free columns. CLMColumnState dispatches
to the developing extensive canopy/snow/soil driver with five snow layers.
The released driver remains in driver.py until the complete path and both
GPU backends have passed validation.
"""
from __future__ import annotations
import numpy as np

from .physics import soil
from .physics.canopy_energy import CanopyExchange, solve_canopy_energy
from .physics.energy import KARMAN, LV
from .physics.frozen_soil import update_soil_phase
from .physics.ground_evap import ground_resistance
from .physics.precip import partition_precipitation
from .physics.soil_uptake import SoilWaterSupply
from .physics.stress import soil_water_stress
from .physics.surface_drag import exchange_velocity


def step_clm_column(state, forcing, *, co2_ppm=420., cosine_zenith=.5,
                    diffuse_fraction=.5, leaf_dimension=.04, canopy_layers=10,
                    leaf_parameters=None, nitrogen_factor=1., most_geometry=None):
    """Advance one NumPy CLM column, dispatching by its state representation.

    Solar partition, CO2 and leaf dimension are explicit site assumptions until
    the forcing metadata adapter supplies them. With explicit most_geometry,
    the extensive path iterates exchange against actual H/E and measurement
    heights. Otherwise neutral friction velocity sets
    leaf/under-canopy conductance (CLM5 equations 2.5.116--122); above-canopy
    stability uses the previous surface temperature. Full aerodynamic iteration
    and complete winter physics are required before making this the default driver.
    """
    from .clm_state import CLMColumnState
    if isinstance(state,CLMColumnState):
        from .clm_enthalpy_driver import step_clm_enthalpy_column
        return step_clm_enthalpy_column(state,forcing,co2_ppm=co2_ppm,cosine_zenith=cosine_zenith,
            diffuse_fraction=diffuse_fraction,leaf_dimension=leaf_dimension,canopy_layers=canopy_layers,
            leaf_parameters=leaf_parameters,nitrogen_factor=nitrogen_factor,most_geometry=most_geometry)
    if most_geometry is not None:
        raise NotImplementedError('MOST flux feedback requires an extensive CLMColumnState')
    f = forcing
    if not np.isfinite(f.dt) or f.dt <= 0 or not (0 <= state.fveg <= 1) or state.lai < 0:
        raise ValueError("Invalid CLM column time step, vegetation fraction, or LAI")
    if not isinstance(canopy_layers,int) or canopy_layers < 1 or leaf_dimension <= 0:
        raise ValueError("Invalid canopy discretization or leaf dimension")
    if state.options.OptStomataResistance != 1:
        raise NotImplementedError("The CLM canopy currently requires Ball-Berry stomata")
    rain, snowfall = partition_precipitation(f.prcp,f.dt,f.tair,state.options.OptRainSnowPartition,qair=f.qair,pressure=f.ps)
    if state.swe > 0. or snowfall > 0.:
        raise NotImplementedError("CLM winter coupling requires the pending five-layer snow implementation")
    s = state.copy()
    s.tsoil = np.asarray(s.tsoil,dtype=float)
    s.smc = np.asarray(s.smc,dtype=float)
    update_soil_phase(s)
    capacity = .0005*s.fveg*s.lai
    intercepted = min(s.fveg*rain,max(0.,capacity-s.canw)) if s.lai > 0. else 0.
    s.canw += intercepted
    throughfall = rain-intercepted
    wet_fraction = min(1.,(s.canw/capacity)**(2./3.)) if capacity > 0. else 0.
    # Use liquid water consistently for stress, ground resistance, and supply.
    liquid_view = s.copy()
    liquid_view.smc = np.maximum(s.smc-s.soil_ice,0.)
    stress = soil_water_stress(liquid_view,s.options.OptSoilWaterTranspiration)
    supply = SoilWaterSupply.from_column(s,f.dt)
    ce = exchange_velocity(f,s.z0m,s.ts,s.options.OptSurfaceDrag)
    z0 = np.clip(s.z0m,1e-5,5.)
    ustar = KARMAN*max(f.wind,.3)/np.log(10./z0)
    leaf_velocity = .01*np.sqrt(ustar/leaf_dimension)
    bare_coefficient = KARMAN/.13*(z0*ustar/1.5e-5)**(-.45)
    ground_velocity = (bare_coefficient*np.exp(-s.lai)+.004*(1.-np.exp(-s.lai)))*ustar
    exchange = CanopyExchange(f.tair,f.qair,f.ps,ce,ground_velocity,leaf_velocity,
        float(s.tsoil[0]),float(soil.thermal_conductivity(s.smc[:1],s.soil)[0]/(.5*s.dz[0])),
        co2_ppm=co2_ppm,ground_resistance=ground_resistance(liquid_view,s.options.OptGroundResistanceEvap))
    result = solve_canopy_energy(np.full(canopy_layers,s.lai/canopy_layers),f.swdown,f.lwdown,
        exchange,vegetated_fraction=s.fveg,wet_fraction=wet_fraction,
        canopy_evaporation_limit=s.canw*1000./f.dt,soil_supply=supply,
        water_stress=stress,nitrogen_factor=nitrogen_factor,leaf_parameters=leaf_parameters,
        ground_albedo=(s.albedo,s.albedo),ground_emissivity=s.emiss,
        diffuse_fraction=diffuse_fraction,cosine_zenith=cosine_zenith)
    ecan = float(result["wet_evaporation"].sum()*f.dt/1000.)
    transp = float(result["transpiration"].sum()*f.dt/1000.)
    esoil = float(result["ground_evaporation"]*f.dt/1000.)
    s.canw -= ecan
    if s.canw < -1e-12:
        raise RuntimeError("Canopy evaporation exceeds its diagnosed supply")
    s.canw = max(0.,s.canw)
    drip = max(0.,s.canw-capacity)
    s.canw -= drip
    water = soil.step_soil_water(s,f.dt,throughfall+drip,transp,esoil,
                                 root_uptake_m=result["root_uptake"]*f.dt/1000.)
    soil.step_soil_heat(s,f.dt,result["ground_heat"])
    s.ts = float(result["ground_temperature"])
    et = ecan+water["transp_m"]+water["esoil_m"]
    runoff = water["runoff_surf_m"]+water["runoff_deep_m"]
    storage_before = float(np.dot(state.smc,state.dz)+state.canw)
    storage_after = float(np.dot(s.smc,s.dz)+s.canw)
    return s, {"ts":s.ts,"h":float(result["h"]),"le":float(result["le"]),
        "g":float(result["ground_heat"]),"rnet":float(result["rnet"]),
        "residual":float(result["residual"]),"et_mm":et*1000.,"prcp_mm":rain*1000.,
        "runoff_mm":runoff*1000.,"swe":s.swe,"canw":s.canw,"melt_mm":0.,
        "tsoil":s.tsoil.copy(),"smc":s.smc.copy(),"gpp_umol_m2s":result["gpp"],
        "gpp_gcm2":result["gpp"]*12.0107e-6*f.dt,
        "leaf_respiration_umol_m2s":result["leaf_respiration"],
        "leaf_temperature":result["leaf_temperature"],
        "canopy_temperature":result["canopy_temperature"],"canopy_humidity":result["canopy_humidity"],
        "water_residual_m":storage_after-storage_before-rain+et+runoff,
        "latent_water_residual_wm2":float(result["le"])-LV*et*1000./f.dt,
        "canopy_iterations":result["iterations"]}
