"""Wang et al. (2019), GRL 10.1029/2019GL085722, equations 1 and 2.

The wet-bulb depression equation is solved with fixed dry-bulb temperature,
not by iterating toward dew point. The sigmoid uses Celsius, as specified
by Figure 1 and the published coefficient values (the text's K label is
inconsistent with those values). See docs/CLM_WETBULB.md for reference scope.
"""
import numpy as np
from .energy import CP,LV
from .snowpack import TF,LF


def wetbulb_temperature(tair,qair,pressure):
    """Return wet-bulb K from T [K], specific humidity [kg/kg], p [Pa].

    Tetens liquid saturation is retained at all temperatures, matching the
    Wang/Noah-MP parameterization. The psychrometric latent heat is chosen
    from dry-bulb temperature, as in Noah-MP. Supersaturated input is allowed:
    the equation then yields Tw>Ta instead of silently clipping humidity.
    """
    t,q,p=np.broadcast_arrays(np.asarray(tair,float),np.asarray(qair,float),np.asarray(pressure,float))
    if (not all(np.all(np.isfinite(x)) for x in (t,q,p)) or np.any((t<200.)|(t>350.))
            or np.any((q<0.)|(q>=.2)) or np.any((p<30000.)|(p>120000.))):
        raise ValueError('Wet-bulb input requires 200..350 K, 0<=q<0.2, 30000..120000 Pa')
    tc=t-TF;vapor=q*p/(.622+.378*q)
    gamma=CP*p/(.622*np.where(t>TF,LV,LV+LF))
    lower=np.full(t.shape,-150.);upper=np.full(t.shape,100.)
    # F=es(Tw)-e+gamma*(Tw-Ta) is strictly increasing in this bracket.
    # Fixed bisections are also reproducible in the forthcoming device path.
    for _ in range(60):
        middle=.5*(lower+upper)
        residual=610.8*np.exp(17.27*middle/(237.3+middle))-vapor+gamma*(middle-tc)
        lower=np.where(residual<0.,middle,lower)
        upper=np.where(residual>=0.,middle,upper)
    result=TF+.5*(lower+upper)
    return float(result) if result.ndim==0 else result


def wetbulb_snow_fraction(tair,qair,pressure):
    """Wang snowfall probability, 0..1, without a warm-temperature cutoff."""
    tw=wetbulb_temperature(tair,qair,pressure)
    argument=np.log(6.99e-5)+2.*(np.asarray(tw)-TF+3.97)
    tail=np.exp(-np.abs(argument))
    result=np.where(argument>=0.,tail/(1.+tail),1./(1.+tail))
    return float(result) if result.ndim==0 else result
