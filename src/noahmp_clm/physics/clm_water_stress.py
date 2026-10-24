"""Water availability for the extensive development column.

Options 2/3 use the Noah-MP soil-potential wetness equations in metres,
including the soil's actual saturated suction. They are independent
implementations of the equations described by SoilWaterTranspirationMod
(NCAR/noahmp 17751dc), rather than copies of that implementation.

The column supplies liquid water and its explicit root profile. This is a
soil-beta approximation, not prognostic CLM5 plant hydraulics. The released
legacy stress module and its GPU ABI remain separate.
"""
from __future__ import annotations

import math

import numpy as np

from ..column import ColumnState
from .stress import soil_water_stress as legacy_soil_water_stress


def soil_water_stress(
    state: ColumnState,
    option: int,
    layer: int | None = None,
    *,
    wilting_potential_m: float = -150.0,
) -> float:
    """Return a layer or root-weighted factor in [0, 1].

    The default wilting head is the pinned Noah-MP parameter-table value.
    Saturated suction is a positive head; matric/wilting potentials are
    negative. Option 1 retains the original Noah arithmetic exactly.
    Zero root fractions mean no accessible roots and give factor zero.
    """
    if option == 1:
        return legacy_soil_water_stress(state, option, layer)
    if option not in (2, 3):
        raise ValueError('Soil water stress option must be 1, 2 or 3')
    p = state.soil
    head = p.saturated_suction_m
    if (not math.isfinite(wilting_potential_m)
            or not math.isfinite(head)
            or not 0.0 < head < -wilting_potential_m
            or not math.isfinite(p.smcmax) or p.smcmax <= 0.0
            or not math.isfinite(p.bexp) or p.bexp <= 0.0):
        raise ValueError('Invalid saturated/wilting potential or soil parameters')
    liquid = np.asarray(state.smc, dtype=float)
    roots = np.asarray(state.root_frac, dtype=float)
    if (liquid.ndim != 1 or roots.shape != liquid.shape
            or not np.all(np.isfinite(liquid)) or np.any(liquid < 0.0)
            or not np.all(np.isfinite(roots)) or np.any(roots < 0.0)
            or (roots.sum() != 0.0 and not np.isclose(roots.sum(), 1.0,
                                                     rtol=0.0, atol=8*np.finfo(np.float32).eps))):
        raise ValueError('Expected finite liquid water and normalized root fractions')
    # Work in log potential to avoid overflow for very dry/high-B soils.
    log_head = math.log(head) - p.bexp*np.log(np.maximum(0.01, liquid)/p.smcmax)
    log_wilt = math.log(-wilting_potential_m)
    potential = np.where(log_head >= log_wilt, wilting_potential_m,
                         -np.exp(np.minimum(log_head, log_wilt)))
    if option == 2:
        available = ((1.0-potential/wilting_potential_m)
                     / (1.0+head/wilting_potential_m))
    else:
        available = -np.expm1(-5.8*np.log(wilting_potential_m/potential))
    available = np.clip(available, 0.0, 1.0)
    if layer is not None:
        return float(available[layer])
    return float(np.clip(np.sum(available*roots), 0.0, 1.0))
