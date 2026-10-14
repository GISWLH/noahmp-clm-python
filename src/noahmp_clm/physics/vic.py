"""VIC variable infiltration and ARNO baseflow with extensive water/heat.

Original equations implementation, based on Liang et al. (1994), the VIC
parameter documentation, and the Noah-MP surface-VIC algorithm description.
No upstream source is copied. See docs/CLM_VIC.md for the column mapping.
"""
from dataclasses import dataclass
import math
from .soil_enthalpy import infiltrate


@dataclass(frozen=True)
class VICParameters:
    infiltration_shape: float=.2
    baseflow_fraction: float=.1
    maximum_baseflow_mm_day: float=10.
    baseflow_threshold: float=.9
    baseflow_exponent: float=2.
    residual_pore_fraction: float=0.

    def __post_init__(self):
        if (not all(math.isfinite(v) for v in vars(self).values()) or self.infiltration_shape<0.
            or not 0<self.baseflow_threshold<1.
            or not 0<=self.baseflow_fraction<=self.baseflow_threshold
            or self.maximum_baseflow_mm_day<0. or self.baseflow_exponent<1.
            or not 0<=self.residual_pore_fraction<1.):
            raise ValueError('Invalid VIC infiltration or monotone ARNO baseflow parameters')


def vic_infiltration(incoming,water,capacity,shape=.2):
    """Return infiltration, surface runoff and initial saturated fraction.

    All water quantities have the same extensive units, normally kg/m².
    Integrates available area on the VIC capacity curve over the input pulse.
    log1p/expm1 retain tiny pulses in otherwise wet soil. Shape zero is the
    homogeneous capacity limit, not an approximate small positive exponent.
    """
    if (not all(math.isfinite(v) for v in (incoming,water,capacity,shape)) or incoming<0.
            or capacity<0. or water<0. or water>capacity or shape<0.):
        raise ValueError('Invalid VIC water or capacity')
    if capacity==0. or water==capacity:return 0.,incoming,1.
    deficit=capacity-water
    if shape==0.:
        admitted=min(incoming,deficit)
        return admitted,incoming-admitted,0.
    log_x=math.log(deficit/capacity)/(1.+shape)
    saturated=-math.expm1(shape*log_x)
    pulse_limit=(1.+shape)*capacity*math.exp(log_x)
    if incoming>=pulse_limit:admitted=deficit
    else:admitted=-deficit*math.expm1((1.+shape)*math.log1p(-incoming/pulse_limit))
    admitted=min(incoming,max(0.,admitted))
    return admitted,incoming-admitted,saturated


def arno_baseflow(liquid,capacity,dt,parameters=VICParameters()):
    """Return bottom liquid outflow [kg/m²] from the ARNO curve.

    No ice is extracted. The explicit residual liquid store remains in place.
    Dsmax is mm/day, numerically kg/m²/day; input dt is seconds.
    """
    if (not all(math.isfinite(v) for v in (liquid,capacity,dt)) or liquid<0.
            or capacity<=0. or liquid>capacity or dt<=0.):raise ValueError('Invalid ARNO reservoir')
    p=parameters;residual=p.residual_pore_fraction*capacity
    available=max(0.,liquid-residual);wet=available/(capacity-residual)
    linear=p.baseflow_fraction/p.baseflow_threshold
    fraction=linear*wet
    if wet>p.baseflow_threshold:
        fraction+=(1.-linear)*((wet-p.baseflow_threshold)/(1.-p.baseflow_threshold))**p.baseflow_exponent
    return min(available,p.maximum_baseflow_mm_day*(dt/86400.)*fraction)


def infiltrate_vic(state,mass_kgm2,enthalpy_jm2,parameters=VICParameters()):
    """Partition surface liquid, then admit only the VIC infiltration parcel.

    The upper two physical layer thicknesses define the capacity reservoir.
    Ice occupies that capacity. Precipitation and snowmelt use this transfer;
    frost/dew deposited directly on bare soil need not traverse the liquid
    overland-flow curve. Both outlets retain the incoming specific enthalpy.
    """
    if (not math.isfinite(enthalpy_jm2) or (mass_kgm2==0. and enthalpy_jm2!=0.)):
        raise ValueError('Invalid VIC incoming heat')
    if len(state.water)<2:raise ValueError('VIC requires at least two upper soil layers')
    capacity=1000.*state.params.smcmax*float(state.dz[:2].sum())
    water=min(capacity,float(state.water[:2].sum()))
    admitted,surface,saturated=vic_infiltration(mass_kgm2,water,capacity,parameters.infiltration_shape)
    specific=enthalpy_jm2/mass_kgm2 if mass_kgm2>0. else 0.
    updated,overflow,overflow_heat=infiltrate(state,admitted,admitted*specific)
    # Unexpected downstream pore overflow remains an explicit surface outlet.
    runoff=surface+overflow;runoff_heat=surface*specific+overflow_heat
    return updated,runoff,runoff_heat
