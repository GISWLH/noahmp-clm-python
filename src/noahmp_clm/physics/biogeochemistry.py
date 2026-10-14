"""Persistent plant/soil C-N stores and explicit column environmental mapping."""
from dataclasses import dataclass,field
import math
import numpy as np
from .plant_cn import PlantState,PlantParameters
from .decomposition import DecompositionState,DecompositionParameters


@dataclass(frozen=True)
class BGCParameters:
    plant: PlantParameters=field(default_factory=PlantParameters)
    decomposition: DecompositionParameters=field(default_factory=DecompositionParameters)
    maximum_lai: float=5.
    saturated_potential_mpa: float=-.0001

    def __post_init__(self):
        if (not isinstance(self.plant,PlantParameters) or not isinstance(self.decomposition,DecompositionParameters)
            or not math.isfinite(self.maximum_lai) or self.maximum_lai<0.
            or not math.isfinite(self.saturated_potential_mpa) or self.saturated_potential_mpa>=0.):
            raise ValueError('Invalid column biogeochemistry parameters')


@dataclass(frozen=True)
class BGCState:
    plant: PlantState
    soil: DecompositionState

    def __post_init__(self):
        if not isinstance(self.plant,PlantState) or not isinstance(self.soil,DecompositionState):
            raise ValueError('Biogeochemistry requires explicit plant and soil C/N stores')

    @property
    def total_carbon(self):return self.plant.total_carbon+float(self.soil.carbon.sum())

    @property
    def total_nitrogen(self):return self.plant.total_nitrogen+float(self.soil.nitrogen.sum())+self.soil.mineral_nitrogen


def soil_environment(ground,root_weights,parameters):
    """Post-physical-step topsoil climate and root-weighted maintenance Q10.

    The aggregate soil C/N cascade sees the top soil layer, explicitly a
    single-column simplification. Matric potential uses liquid saturation
    and the existing Campbell exponent. The saturated suction is explicit;
    default -0.1 kPa matches this repository's stress parameterization.
    """
    t=ground.temperature;roots=np.asarray(root_weights,float)
    if roots.shape!=t.shape or np.any(roots<0.) or not np.all(np.isfinite(roots)):
        raise ValueError('Invalid root distribution for maintenance respiration')
    if abs(float(roots.sum())-1.)>1e-12:raise ValueError('Root fractions must sum to one for BGC climate mapping')
    q10=parameters.plant.maintenance_q10
    factor=float(roots@np.exp(np.log(q10)*(t-293.15)/10.))
    root_temperature=float(roots@t) if q10==1. else 293.15+10.*math.log(factor)/math.log(q10)
    saturation=float(np.clip(ground.liquid[0]/(1000.*ground.dz[0]*ground.params.smcmax),1e-4,1.))
    potential=parameters.saturated_potential_mpa*saturation**(-ground.params.bexp)
    return dict(soil_temperature=float(t[0]),water_potential_mpa=potential,
        root_temperature=root_temperature,root_q10_factor=factor)
