"""Simplified seasonal-deciduous GDD phenology, independently implemented.

CLM5 technical note 2.20.3 supplies the onset/offset triggers. This controller
returns leaf-area availability; carbon storage/allocation is a separate budget.
Daylength, its seasonal trend and a winter-solstice crossing are explicit
calendar/site inputs, so no northern-hemisphere calendar is silently assumed.
"""
from dataclasses import dataclass
from enum import IntEnum
import math

DAY=86400.
TF=273.15


class PhenologyPhase(IntEnum):
    DORMANT=0
    ONSET=1
    ACTIVE=2
    OFFSET=3


@dataclass(frozen=True)
class PhenologyParameters:
    gdd_threshold: float=400.
    onset_days: float=30.
    offset_days: float=15.
    offset_daylength_seconds: float=39300.
    lai_minimum: float=0.
    lai_maximum: float=6.

    def __post_init__(self):
        if (not all(math.isfinite(v) for v in vars(self).values()) or self.gdd_threshold<=0.
            or self.onset_days<1. or self.offset_days<1. or not 0.<self.offset_daylength_seconds<DAY
            or self.lai_minimum<0. or self.lai_maximum<self.lai_minimum):
            raise ValueError('Invalid seasonal phenology parameters')


@dataclass(frozen=True)
class PhenologyState:
    gdd: float=0.
    leaf_fraction: float=0.
    remaining_seconds: float=0.
    phase: int=PhenologyPhase.DORMANT
    armed: bool=True

    def __post_init__(self):
        if (not all(math.isfinite(v) for v in (self.gdd,self.leaf_fraction,self.remaining_seconds))
            or self.gdd<0. or not 0.<=self.leaf_fraction<=1. or self.remaining_seconds<0.
            or self.phase not in tuple(PhenologyPhase) or not isinstance(self.armed,bool)):
            raise ValueError('Invalid phenology state')
        if self.phase==PhenologyPhase.DORMANT and (self.leaf_fraction!=0. or self.remaining_seconds!=0.):
            raise ValueError('Dormant phenology cannot retain displayed leaves or an active counter')
        if self.phase==PhenologyPhase.ACTIVE and (self.leaf_fraction!=1. or self.remaining_seconds!=0.):
            raise ValueError('Active phenology requires a full leaf fraction and no counter')
        if self.phase!=PhenologyPhase.DORMANT and self.armed:
            raise ValueError('An active growing season cannot be armed for another onset')


def critical_gdd(annual_mean_air_temperature):
    """CLM seasonal-deciduous threshold [K days] from annual mean air T [K]."""
    if not math.isfinite(annual_mean_air_temperature) or not 200.<=annual_mean_air_temperature<=330.:
        raise ValueError('Invalid annual mean air temperature')
    return math.exp(4.8+.13*(annual_mean_air_temperature-TF))


def degree_days(soil_temperature,dt):
    if not math.isfinite(soil_temperature) or not 150.<=soil_temperature<=400. or not math.isfinite(dt) or not 0.<dt<=DAY:
        raise ValueError('Invalid phenology temperature or timestep (0 < dt <= one day)')
    return max(0.,soil_temperature-TF)*(dt/DAY)


def step_phenology(state,soil_temperature,daylength_seconds,increasing_daylength,winter_solstice,dt,
                   parameters=PhenologyParameters()):
    """Advance an extratropical seasonal-deciduous controller.

    GDD uses the third physical soil layer, as in the cited CLM onset rule.
    One onset is allowed between winter solstices; failure to reach the
    threshold by summer disables accumulation until the next winter reset.
    Onset/offset display is linear over the specified periods (a simplification
    of the carbon-transfer-controlled CLM leaf display). Triggering occurs at
    timestep boundaries, with the current step counted in its new phase.
    """
    addition=degree_days(soil_temperature,dt)
    if (not math.isfinite(daylength_seconds) or not 0.<=daylength_seconds<=DAY
        or not isinstance(increasing_daylength,bool) or not isinstance(winter_solstice,bool)):
        raise ValueError('Explicit daylength/trend/solstice inputs are required')
    p=parameters;gdd=state.gdd;fraction=state.leaf_fraction;remaining=state.remaining_seconds
    phase=PhenologyPhase(state.phase);armed=state.armed;onset=False;offset=False
    if phase==PhenologyPhase.ONSET and remaining>p.onset_days*DAY:raise ValueError('Onset counter exceeds its duration')
    if phase==PhenologyPhase.OFFSET and remaining>p.offset_days*DAY:raise ValueError('Offset counter exceeds its duration')
    if phase==PhenologyPhase.DORMANT:
        if winter_solstice:gdd=0.;armed=True
        if not increasing_daylength:gdd=0.;armed=False
        if armed:
            gdd+=addition
            if gdd>p.gdd_threshold:
                phase=PhenologyPhase.ONSET;remaining=p.onset_days*DAY;gdd=0.;armed=False;onset=True
    if phase==PhenologyPhase.ACTIVE and not increasing_daylength and daylength_seconds<p.offset_daylength_seconds:
        phase=PhenologyPhase.OFFSET;remaining=p.offset_days*DAY;offset=True
    if phase==PhenologyPhase.ONSET:
        remaining=max(0.,remaining-dt);fraction=1.-remaining/(p.onset_days*DAY)
        if remaining==0.:phase=PhenologyPhase.ACTIVE
    elif phase==PhenologyPhase.OFFSET:
        remaining=max(0.,remaining-dt);fraction=remaining/(p.offset_days*DAY)
        if remaining==0.:phase=PhenologyPhase.DORMANT
    result=PhenologyState(gdd,fraction,remaining,phase,armed)
    return result,dict(lai=p.lai_minimum+(p.lai_maximum-p.lai_minimum)*fraction,
        leaf_gain_fraction=max(0.,fraction-state.leaf_fraction),leaf_loss_fraction=max(0.,state.leaf_fraction-fraction),
        onset_triggered=onset,offset_triggered=offset,gdd_increment=addition if state.phase==PhenologyPhase.DORMANT and (state.armed or winter_solstice) and increasing_daylength else 0.)
