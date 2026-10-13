"""Multilayer leaf/ground energy balance with a diagnostic canopy air space.

Original implementation of the resistance network in CLM5 technical note 2.5.
Each leaf layer has its own temperature and gas exchange; canopy air is one
well-mixed, zero-storage box. Supplied conductances are fixed during a solve.
This component is not yet the column driver's default surface scheme.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .canopy_radiation import canopy_shortwave
from .energy import CP, LV, RD, SIGMA
from .photosynthesis import R_GAS, leaf_gas_exchange
from .soil_uptake import _allocate


@dataclass(frozen=True)
class CanopyExchange:
    """Boundary conditions, with velocities in m/s and resistance in s/m.

    Air/soil temperatures are K, pressure Pa, humidity kg/kg, CO2 ppm.
    Ground heat = soil_conductance * (Tground - soil_temperature) + offset,
    in W/m² ground. The caller must supply mutually consistent exchange
    velocities and separate root/ground water budgets for the time step.
    """
    air_temperature: float
    specific_humidity: float
    pressure: float
    atmosphere_velocity: float
    ground_velocity: float
    leaf_velocity: float
    soil_temperature: float
    soil_conductance: float
    co2_ppm: float = 420.
    ground_resistance: float = 0.
    soil_flux_offset: float = 0.

    def __post_init__(self):
        if not np.all(np.isfinite(list(vars(self).values()))):
            raise ValueError("Nonfinite canopy exchange boundary")
        if not (200 <= self.air_temperature <= 350 and 200 <= self.soil_temperature <= 350
                and 0 <= self.specific_humidity < .2 and self.pressure >= 30000
                and self.atmosphere_velocity > 0 and self.ground_velocity > 0
                and self.leaf_velocity > 0 and self.soil_conductance >= 0
                and self.ground_resistance >= 0 and self.co2_ppm > 0):
            raise ValueError("Unphysical canopy exchange boundary")


def _saturation(temperature, pressure):
    tc = np.asarray(temperature)-273.15
    es = 611.2*np.exp(17.67*tc/(tc+243.5))
    denominator = pressure-.378*es
    q = .622*es/denominator
    derivative = .622*pressure*es*17.67*243.5/(denominator**2*(tc+243.5)**2)
    return q, derivative, es


def _longwave_operator(area, emissivity, fraction):
    """Map leaf/ground blackbody exitance and sky flux to net absorption."""
    n = len(area)
    down, up = np.zeros((n+1,n+2)), np.zeros((n+1,n+2))
    transmission = np.exp(-area)
    down[0,-1] = 1.
    for j in range(n):
        down[j+1] = transmission[j]*down[j]
        down[j+1,j] += 1.-transmission[j]
    up[n] = (1.-emissivity)*down[n]
    up[n,n] += emissivity
    for j in range(n-1,-1,-1):
        up[j] = transmission[j]*up[j+1]
        up[j,j] += 1.-transmission[j]
    matrix = fraction*np.vstack((down[:-1]+up[1:]-down[1:]-up[:-1],down[-1]-up[-1]))
    matrix[-1,n] -= (1.-fraction)*emissivity
    matrix[-1,-1] += (1.-fraction)*emissivity
    return matrix


def _positive_scale(flux, limit):
    # Dew on one leaf is available to the shared canopy store. This assumes
    # instantaneous redistribution within the time step, with no air storage.
    positive = np.maximum(flux,0.).sum()
    available = limit-np.minimum(flux,0.).sum()
    return min(1.,available/positive) if positive > 0 else 1.


def _water_flux(qcan, qsat, gs_velocity, area, fraction, wet_fraction,
                rho, exchange, limits, soil_supply=None):
    """Mass fluxes and their effective conductances at a trial canopy q."""
    e = exchange
    leaf_boundary = rho*fraction*area*e.leaf_velocity
    drying = qsat[:-1] > qcan
    kw = leaf_boundary*np.where(drying,wet_fraction,1.)
    kt = leaf_boundary*np.where(drying,1.-wet_fraction,0.)*gs_velocity/(e.leaf_velocity+gs_velocity)
    wet = kw*(qsat[:-1]-qcan)
    kw *= np.where(wet > 0.,_positive_scale(wet,limits[0]),1.)
    transp = kt*(qsat[:-1]-qcan)
    tr_limit = limits[1]
    if soil_supply is not None:
        tr_limit = min(tr_limit,np.where(soil_supply.weights>0.,soil_supply.roots,0.).sum())
    kt *= _positive_scale(transp,tr_limit)
    root_uptake = (np.empty(0) if soil_supply is None else
                   _allocate(np.sum(kt*(qsat[:-1]-qcan)),soil_supply.roots,soil_supply.weights))
    ground_limit = limits[2] if soil_supply is None else min(limits[2],max(0.,soil_supply.surface-root_uptake[0]))
    under = rho*fraction/(1./e.ground_velocity+e.ground_resistance)
    bare = rho*(1.-fraction)/(1./e.atmosphere_velocity+e.ground_resistance)
    ground_parts = np.array([under*(qsat[-1]-qcan),bare*(qsat[-1]-e.specific_humidity)])
    scale = np.where(ground_parts > 0.,_positive_scale(ground_parts,ground_limit),1.)
    under, bare = under*scale[0], bare*scale[1]
    wet, transp = kw*(qsat[:-1]-qcan), kt*(qsat[:-1]-qcan)
    ground_under, ground_bare = under*(qsat[-1]-qcan), bare*(qsat[-1]-e.specific_humidity)
    residual = wet.sum()+transp.sum()+ground_under-rho*fraction*e.atmosphere_velocity*(qcan-e.specific_humidity)
    return wet, transp, ground_under, ground_bare, kw+kt, under, bare, residual, root_uptake


def _humidity(qsat, gs, area, fraction, wet_fraction, rho, exchange, limits, soil_supply=None):
    if fraction == 0.:
        qcan = exchange.specific_humidity
    else:
        left, right = min(qsat.min(),exchange.specific_humidity), max(qsat.max(),exchange.specific_humidity)
        for _ in range(42):
            middle = .5*(left+right)
            flux = _water_flux(middle,qsat,gs,area,fraction,wet_fraction,rho,exchange,limits,soil_supply)
            if flux[7] > 0.:
                left = middle
            else:
                right = middle
        qcan = .5*(left+right)
    return qcan, _water_flux(qcan,qsat,gs,area,fraction,wet_fraction,rho,exchange,limits,soil_supply)


def solve_canopy_energy(leaf_area, swdown, lwdown, exchange, *,
                        vegetated_fraction=1., wet_fraction=0.,
                        canopy_evaporation_limit=np.inf, transpiration_limit=np.inf,
                        ground_evaporation_limit=np.inf, water_stress=1., nitrogen_factor=1.,
                        soil_supply=None,
                        leaf_parameters=None, ground_emissivity=.97, ground_albedo=(.15,.3),
                        diffuse_fraction=.5, cosine_zenith=.5, initial_temperatures=None,
                        tolerance=5e-5, max_iterations=100):
    """Solve n leaf temperatures plus ground temperature (K).

    Areas are one-sided LAI per vegetated ground, ordered top to bottom;
    fluxes and water limits are per TOTAL ground area. Water limits are
    kg/m²/s, positive for loss; negative returned wet/ground flux means dew.
    ``wet_fraction`` is the wetted leaf fraction, not water storage itself.
    ``soil_supply`` links surface evaporation and roots to the same liquid
    store, returning exact root withdrawals for the caller's water update.
    Limits without soil_supply must describe disjoint root/surface budgets.
    Nonconvergence raises, so a nonclosed solution cannot silently advance.
    """
    e = exchange
    area = np.asarray(leaf_area,float)
    fv = float(vegetated_fraction)
    limits = (canopy_evaporation_limit,transpiration_limit,ground_evaporation_limit)
    if area.ndim != 1 or not area.size or not np.all(np.isfinite(area)) or np.any(area<0):
        raise ValueError("Invalid canopy layer areas")
    if not (0 <= fv <= 1 and 0 <= wet_fraction <= 1 and np.isfinite(lwdown) and lwdown >= 0
            and 0 < ground_emissivity <= 1 and tolerance > 0 and np.isfinite(tolerance)
            and isinstance(max_iterations,int) and max_iterations > 0
            and all(x >= 0 and not np.isnan(x) for x in limits)):
        raise ValueError("Invalid canopy surface boundary or solver controls")
    if area.sum() == 0.:
        fv = 0.
    n = len(area)
    for name,value in (("nitrogen_factor",nitrogen_factor),("water_stress",water_stress)):
        factor = np.asarray(value,float)
        if not np.all(np.isfinite(factor)) or np.any((factor<0.)|(factor>1.)):
            raise ValueError(f"Invalid {name}")
        if np.broadcast_shapes(factor.shape,(n,)) != (n,):
            raise ValueError(f"{name} must be scalar or one value per leaf layer")
    active = fv*area > 0.
    sw = canopy_shortwave(area,swdown,ground_albedo=ground_albedo,
                         diffuse_fraction=diffuse_fraction,cosine_zenith=cosine_zenith)
    bare_sw = swdown*(1.-.5*sum(ground_albedo))
    shortwave = np.r_[fv*sw["absorbed"],fv*sw["ground_absorbed"]+(1.-fv)*bare_sw]
    lw = _longwave_operator(area,ground_emissivity,fv)
    rho = e.pressure/(RD*e.air_temperature*(1.+.61*e.specific_humidity))
    hcan = rho*CP*np.r_[fv*area*e.leaf_velocity,fv*e.ground_velocity]
    hatm = rho*CP*fv*e.atmosphere_velocity
    hbare = rho*CP*(1.-fv)*e.atmosphere_velocity
    hden = hcan.sum()+hatm
    hjac = np.diag(hcan)
    if hden > 0:
        hjac -= np.outer(hcan,hcan)/hden
    hjac[-1,-1] += hbare
    temperature = (np.full(n+1,e.air_temperature,dtype=float) if initial_temperatures is None
                   else np.array(initial_temperatures,float,copy=True))
    if temperature.shape != (n+1,) or not np.all(np.isfinite(temperature)) or np.any((temperature<200)|(temperature>350)):
        raise ValueError("Invalid initial leaf/ground temperatures")
    temperature[:-1] = np.where(active,temperature[:-1],e.air_temperature)
    qcan = e.specific_humidity
    molar_density = e.pressure/(R_GAS*e.air_temperature)
    # Simplified soil-beta closure: water stress downregulates biochemical
    # capacity as well as the Ball--Berry slope. This is not CLM5's prognostic
    # plant hydraulics. In particular, an inaccessible root store must not
    # sustain positive daytime uptake through the minimum stomatal leak.
    capacity_stress = np.asarray(water_stress,float)
    if transpiration_limit == 0. or (soil_supply is not None and
            np.where(soil_supply.weights>0.,soil_supply.roots,0.).sum() == 0.):
        capacity_stress = capacity_stress*0.
    for iteration in range(1,max_iterations+1):
        saturation, dq, es = _saturation(temperature,e.pressure)
        vapor = qcan*e.pressure/(.622+.378*qcan)
        gas = leaf_gas_exchange(sw["absorbed_par"],temperature[:-1],co2_ppm=e.co2_ppm,
            relative_humidity=vapor/es[:-1],boundary_conductance=e.leaf_velocity*molar_density,
            parameters=leaf_parameters,water_stress=water_stress,
            nitrogen_factor=np.asarray(nitrogen_factor)*capacity_stress)
        qnew, water = _humidity(saturation,gas["gs_mol"]/molar_density,area,fv,
                               wet_fraction,rho,e,limits,soil_supply)
        wet, transp, ground_under, ground_bare, kleaf, kunder, kbare, vapor_residual, root_uptake = water
        tair_can = (hatm*e.air_temperature+hcan@temperature)/hden if hden > 0 else e.air_temperature
        sensible = hcan*(temperature-tair_can)
        sensible[-1] += hbare*(temperature[-1]-e.air_temperature)
        evaporation = np.r_[wet+transp,ground_under+ground_bare]
        net_radiation = shortwave+lw@np.r_[SIGMA*temperature**4,lwdown]
        ground_heat = e.soil_conductance*(temperature[-1]-e.soil_temperature)+e.soil_flux_offset
        residual = net_radiation-sensible-LV*evaporation
        residual[-1] -= ground_heat
        # q supplied to photosynthesis must also be consistent with the solved
        # air space; testing energy alone could accept one lagged stomatal step.
        q_error = abs(qnew-qcan)
        qcan = qnew
        if np.max(np.abs(residual)) <= tolerance and q_error <= 1e-10:
            break
        kcan = np.r_[kleaf,kunder]
        kden = kcan.sum()+rho*fv*e.atmosphere_velocity
        ejac = np.diag(kcan*dq)
        if kden > 0:
            # The bare-ground contribution does not enter the canopy air box.
            ejac -= np.outer(kcan,kcan*dq)/kden
        ejac[-1,-1] += kbare*dq[-1]
        jac = lw[:,:-1]*(4.*SIGMA*temperature**3)[None,:]-hjac-LV*ejac
        jac[-1,-1] -= e.soil_conductance
        inactive = np.flatnonzero(~active)
        jac[inactive,:] = 0.
        jac[inactive,inactive] = -1.
        residual[inactive] = e.air_temperature-temperature[inactive]
        delta = np.linalg.solve(jac,-residual)
        temperature += .8*np.clip(delta,-5.,5.)
        temperature = np.clip(temperature,200.,350.)
    else:
        raise RuntimeError(f"Canopy energy did not converge in {max_iterations} iterations: "
                           f"max residual={np.max(np.abs(residual)):.6g} W/m², dq={q_error:.6g}")
    leaf_gross = np.where(active,gas["gross"],0.)
    leaf_respiration = np.where(active,gas["respiration"],0.)
    outgoing_sw = fv*sw["reflected"]+(1.-fv)*(swdown-bare_sw)
    outgoing_lw = lwdown-(net_radiation-shortwave).sum()
    atmosphere_h = hatm*(tair_can-e.air_temperature)+hbare*(temperature[-1]-e.air_temperature)
    atmosphere_e = rho*fv*e.atmosphere_velocity*(qcan-e.specific_humidity)+ground_bare
    return {"leaf_temperature":temperature[:-1],"ground_temperature":temperature[-1],
        "canopy_temperature":tair_can,"canopy_humidity":qcan,"gas_exchange":gas,
        "gpp":float(fv*np.dot(area,leaf_gross)),
        "leaf_respiration":float(fv*np.dot(area,leaf_respiration)),
        "leaf_sensible":sensible[:-1],"ground_sensible":sensible[-1],"h":sensible.sum(),
        "le":LV*evaporation.sum(),"ground_heat":ground_heat,"rnet":net_radiation.sum(),
        "wet_evaporation":wet,"transpiration":transp,"ground_evaporation":evaporation[-1],
        "root_uptake":root_uptake,
        "layer_residual":residual,"residual":residual.sum(),"iterations":iteration,
        "canopy_water_residual":vapor_residual,"canopy_humidity_change":q_error,
        "air_heat_residual":sensible.sum()-atmosphere_h,
        "air_water_residual":evaporation.sum()-atmosphere_e,
        "outgoing_shortwave":outgoing_sw,"outgoing_longwave":outgoing_lw}
