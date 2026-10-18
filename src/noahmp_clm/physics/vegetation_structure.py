"""Reference total/exposed leaf and stem geometry for one CLM column.

SAI is an empirical area diagnostic, not an additional carbon reservoir.
The CN relaxation/leaf-loss rule and crop ratios follow the definitions in
the pinned CTSM CN vegetation-structure module; code is independently written.
Prescribed geometry and dynamic CN geometry remain explicit alternatives.
"""
from dataclasses import dataclass
from enum import IntEnum
import math

DAY=86400.
MONTH=30.*DAY


class StemAreaScheme(IntEnum):
    PRESCRIBED=0
    CN_NONCROP=1
    CN_GENERIC_CROP=2
    ANNUAL_C3_CROP=3


@dataclass(frozen=True)
class CanopyStructureParameters:
    scheme: int=StemAreaScheme.PRESCRIBED
    canopy_top_m: float=1.
    canopy_bottom_m: float=0.
    short_vegetation: bool=False

    def __post_init__(self):
        try:scheme=StemAreaScheme(self.scheme)
        except (ValueError,TypeError):raise ValueError('Unknown stem-area scheme') from None
        if (isinstance(self.scheme,bool) or not isinstance(self.short_vegetation,bool)
                or not all(math.isfinite(v) for v in (self.canopy_top_m,self.canopy_bottom_m))
                or not 0.<=self.canopy_bottom_m<self.canopy_top_m):
            raise ValueError('Explicit finite canopy top/bottom and vegetation form required')
        object.__setattr__(self,'scheme',scheme)


def _area(value):
    if not math.isfinite(value) or value<0.:
        raise ValueError('Leaf and stem area indices must be finite and nonnegative')
    return float(value)


def step_stem_area(stem_area,previous_lai,lai,dt,parameters=CanopyStructureParameters(),*,
                   prescribed_sai=None,harvested=False,fallow=False,newly_planted=False):
    """Return next total SAI and the actual geometric update [m2/m2].

    CN noncrop: monthly loss0.5, minimum0.5; generic crop: loss1, minimum0.05.
    Positive loss of displayed LAI contributes to standing area. A prescribed
    SAI cannot override a dynamic rule. The current simplified annual crop is
    C3: SAI/LAI0.2, harvested stubble0.25 until the next planting.
    Use0<dt<=one day, retaining the reference explicit monthly-rate scaling.
    No C/N, intercepted water or enthalpy is added/removed by this diagnostic.
    """
    old=_area(stem_area);before=_area(previous_lai);after=_area(lai)
    if not isinstance(parameters,CanopyStructureParameters):raise TypeError('Expected canopy structure parameters')
    if not math.isfinite(dt) or not 0.<dt<=DAY:raise ValueError('Structure timestep requires0<dt<=one day')
    if not all(isinstance(v,bool) for v in (harvested,fallow,newly_planted)) or (harvested and newly_planted):
        raise ValueError('Invalid structure management flags')
    kind=parameters.scheme;loss=max(0.,before-after);factor=1.;minimum=0.
    if kind==StemAreaScheme.PRESCRIBED:
        if any((harvested,fallow,newly_planted)):raise ValueError('Crop flags require annual crop structure')
        result=old if prescribed_sai is None else _area(prescribed_sai)
    else:
        if prescribed_sai is not None:raise ValueError('Prescribed SAI cannot bypass a dynamic structure rule')
        if kind==StemAreaScheme.ANNUAL_C3_CROP:
            if (harvested or fallow) and after!=0.:raise ValueError('Harvested/fallow crop must have zero live LAI')
            result=.2*after
            if harvested or (fallow and old>0.):result=.25
            if newly_planted and after==0.:result=0.
        else:
            if any((harvested,fallow,newly_planted)):raise ValueError('Crop flags require annual crop structure')
            rate,minimum=((.5,.5) if kind==StemAreaScheme.CN_NONCROP else (1.,.05))
            factor=1.-rate*dt/MONTH
            result=max(minimum,factor*old+loss)
    return float(result),dict(sai_before=old,sai_after=float(result),lai_before=before,lai_after=after,
        leaf_area_loss=loss,decay_factor=factor,minimum_sai=minimum,scheme=int(kind))


def exposed_canopy_geometry(lai,sai,snow_depth_m,parameters=CanopyStructureParameters()):
    """Total and exposed geometry per vegetated area, before ground-area fveg.

    Tall vegetation is buried between its explicit canopy bottom and top.
    Short vegetation follows the reference0.2m complete-burial scale.
    Burial changes exposed geometry; it does not delete stores or total area.
    """
    leaf=_area(lai);stem=_area(sai)
    if not isinstance(parameters,CanopyStructureParameters):raise TypeError('Expected canopy structure parameters')
    if not math.isfinite(snow_depth_m) or snow_depth_m<0.:raise ValueError('Invalid snow depth for canopy exposure')
    if parameters.short_vegetation:
        visible=1.-min(float(snow_depth_m),.2)/.2
    else:
        height=parameters.canopy_top_m-parameters.canopy_bottom_m
        hidden=min(max(float(snow_depth_m)-parameters.canopy_bottom_m,0.),height)
        visible=1.-hidden/height
    return dict(total_lai=leaf,total_sai=stem,exposed_lai=leaf*visible,exposed_sai=stem*visible,
        exposed_plant_area=(leaf+stem)*visible,exposed_fraction=visible,
        snow_depth_m=float(snow_depth_m))
