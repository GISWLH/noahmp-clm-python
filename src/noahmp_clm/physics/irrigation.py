"""Soil-moisture-triggered surface irrigation with an explicit external supply.

CLM5 equations 2.26.13--17 provide the root-zone deficit. This simplified
version accepts site target/wilting contents directly. The caller supplies
the daily check event, available water and its temperature. A pending amount
is an unfilled request, never stored physical water. See CLM_IRRIGATION.md.
"""
from dataclasses import dataclass
import math
import numpy as np
from .snowpack import TF,LF,CW


@dataclass(frozen=True)
class IrrigationParameters:
    root_zone_depth: float=.6
    target_theta: float=.28
    wilting_theta: float=.12
    threshold_fraction: float=1.
    application_seconds: float=21600.

    def __post_init__(self):
        if (not all(math.isfinite(v) for v in vars(self).values()) or self.root_zone_depth<=0.
            or not 0.<=self.wilting_theta<self.target_theta<=1. or not 0.<self.threshold_fraction<=1.
            or not 0.<self.application_seconds<=86400.):
            raise ValueError('Invalid irrigation depth, moisture threshold or duration')


@dataclass(frozen=True)
class IrrigationState:
    pending_kgm2: float=0.
    remaining_seconds: float=0.

    def __post_init__(self):
        if (not math.isfinite(self.pending_kgm2) or not math.isfinite(self.remaining_seconds)
            or self.pending_kgm2<0. or self.remaining_seconds<0.
            or (self.pending_kgm2==0.)!=(self.remaining_seconds==0.)):
            raise ValueError('Invalid pending irrigation request')


@dataclass(frozen=True)
class IrrigationInput:
    """One-step available external water and an explicit daily check event."""
    check: bool=False
    supply_kgm2: float=0.
    supply_temperature_k: float=285.

    def __post_init__(self):
        if (not isinstance(self.check,(bool,np.bool_)) or not math.isfinite(self.supply_kgm2) or self.supply_kgm2<0.
            or not math.isfinite(self.supply_temperature_k) or not TF<=self.supply_temperature_k<=350.):
            raise ValueError('Invalid irrigation input')


def step_irrigation(state,liquid_kgm2,ice_kgm2,dz,dt,lai,*,check=False,supply_kgm2=0.,
                    supply_temperature_k=285.,parameters=IrrigationParameters()):
    """Return a pending request and liquid mass/enthalpy to the ground surface.

    A check starts a constant-rate application over the configured duration.
    Supply shortages are reported as unmet; no unavailable water is invented.
    Bare/frozen root zones cancel unfilled requests. Surface routing and
    rejection/runoff are handled by the actual column infiltration operator.
    """
    liquid,ice,z=[np.asarray(v,float) for v in (liquid_kgm2,ice_kgm2,dz)]
    if (not isinstance(state,IrrigationState) or liquid.ndim!=1 or liquid.size==0 or ice.shape!=liquid.shape or z.shape!=liquid.shape
        or not all(np.all(np.isfinite(v)) for v in (liquid,ice,z)) or np.any(liquid<0.) or np.any(ice<0.) or np.any(z<=0.)
        or np.any(liquid+ice>1000.*z) or not all(math.isfinite(v) for v in (dt,lai,supply_kgm2,supply_temperature_k))
        or not 0.<dt<=86400. or lai<0. or supply_kgm2<0. or not TF<=supply_temperature_k<=350.
        or not isinstance(check,(bool,np.bool_))):
        raise ValueError('Invalid irrigation soil, timestep, leaf area or water supply')
    if check and state.pending_kgm2>0.:raise ValueError('A new irrigation check cannot overwrite a pending request')
    p=parameters;top=np.r_[0.,np.cumsum(z)[:-1]]
    thickness=np.minimum(z,np.maximum(p.root_zone_depth-top,0.));fraction=thickness/z
    depth=float(thickness.sum());available=float(np.sum(liquid*fraction));frozen=float(np.sum(ice*fraction))
    target=1000.*p.target_theta*depth;wilt=1000.*p.wilting_theta*depth
    threshold=p.threshold_fraction*(target-wilt)+wilt;deficit=max(0.,threshold-available)
    eligible=lai>0. and frozen<=1e-9
    cancelled=state.pending_kgm2 if not eligible else 0.
    pending=state.pending_kgm2 if eligible else 0.;seconds=state.remaining_seconds if eligible else 0.
    requested=deficit if check and eligible else 0.
    if requested>0.:pending=requested;seconds=p.application_seconds
    quota=0.
    if pending>0.:
        if dt>=seconds:quota=pending;pending=seconds=0.
        else:
            quota=pending*(dt/seconds);pending-=quota;seconds-=dt
    applied=min(quota,supply_kgm2);unmet=quota-applied
    heat=applied*(LF+CW*(supply_temperature_k-TF))
    residual=pending-state.pending_kgm2+applied+unmet+cancelled-requested
    bound=128.*np.finfo(float).eps*max(pending+state.pending_kgm2+applied+unmet+cancelled+requested,1.)
    if abs(residual)>bound:raise RuntimeError('Irrigation request accounting exceeds numerical bound')
    return IrrigationState(pending,seconds),dict(available_kgm2=available,threshold_kgm2=threshold,
        deficit_kgm2=deficit,frozen_kgm2=frozen,eligible=bool(eligible),triggered=requested>0.,requested_kgm2=requested,
        applied_kgm2=applied,applied_enthalpy_jm2=heat,unmet_kgm2=unmet,cancelled_kgm2=cancelled,
        request_residual_kgm2=residual,request_bound_kgm2=bound)
