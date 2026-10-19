"""C3 trait capacity, photoperiod, vertical leaf N and ten-day acclimation.

This is the NumPy development reference. It does not yet select a new column
default. Published equations and the remaining integration are documented in
docs/CLM_CANOPY_CAPACITY.md; CTSM source was read for interpretation only.
"""
from dataclasses import astuple, dataclass
import math
import numpy as np
from .photosynthesis import LeafParameters

TEN_DAYS = 10. * 86400.


@dataclass(frozen=True)
class TemperatureMemory:
    """Exact equal-interval rolling mean, with an explicit cold-start fallback.

    Samples cover completed intervals only. The first physical interval uses
    initial_temperature_k; each successful interval appends its air temperature
    for the following interval. No future forcing enters this memory. The full
    buffer, count and cursor must be carried through a restart.
    """
    values: np.ndarray
    time_step_seconds: float
    count: int = 0
    cursor: int = 0
    initial_temperature_k: float = 293.15

    def __post_init__(self):
        dt = self.time_step_seconds
        if not math.isfinite(dt) or dt < 60. or dt > TEN_DAYS:
            raise ValueError('Temperature memory requires a fixed interval of 60 seconds to ten days')
        size = round(TEN_DAYS / dt)
        if abs(size * dt - TEN_DAYS) > 1e-7:
            raise ValueError('The fixed interval must divide ten days exactly')
        values = np.asarray(self.values, float)
        if values.shape != (size,) or not np.all(np.isfinite(values)) or np.any((values < 200.) | (values > 350.)):
            raise ValueError('Invalid ten-day temperature buffer')
        if (type(self.count) is not int or type(self.cursor) is not int
                or not 0 <= self.count <= size or not 0 <= self.cursor < size
                or (self.count < size and self.cursor != self.count)):
            raise ValueError('Invalid temperature-memory count or cursor')
        if not math.isfinite(self.initial_temperature_k) or not 200. <= self.initial_temperature_k <= 350.:
            raise ValueError('Invalid initial temperature')
        values = values.copy(); values.setflags(write=False)
        object.__setattr__(self, 'values', values)

    @classmethod
    def cold(cls, time_step_seconds, initial_temperature_k=293.15):
        if (not math.isfinite(time_step_seconds) or not 60. <= time_step_seconds <= TEN_DAYS):
            raise ValueError('Invalid temperature-memory interval')
        size = round(TEN_DAYS / time_step_seconds)
        return cls(np.full(size, initial_temperature_k), time_step_seconds,
                   initial_temperature_k=initial_temperature_k)

    @property
    def mean_temperature_k(self):
        if self.count == 0:
            return self.initial_temperature_k
        # Recompute from the retained samples: no long-run subtract/add drift.
        return math.fsum(self.values[:self.count]) / self.count

    def append(self, air_temperature_k, time_step_seconds):
        if time_step_seconds != self.time_step_seconds:
            raise ValueError('Temperature-memory interval cannot change across a restart')
        if not math.isfinite(air_temperature_k) or not 200. <= air_temperature_k <= 350.:
            raise ValueError('Invalid air temperature for acclimation')
        values = self.values.copy(); values[self.cursor] = air_temperature_k
        return TemperatureMemory(values, self.time_step_seconds, min(self.count + 1, len(values)),
                                 (self.cursor + 1) % len(values), self.initial_temperature_k)

    def __reduce__(self):
        # Reconstruct through validation, including the read-only buffer, when
        # an internal station checkpoint is loaded. NumPy pickle otherwise
        # restores a writable array without calling dataclass __post_init__.
        return (type(self), (self.values, self.time_step_seconds, self.count,
                            self.cursor, self.initial_temperature_k))


@dataclass(frozen=True)
class CanopyCapacityParameters:
    """Prescribed C3 PFT traits, in m2 leaf/gC and gC/gN; fractions unitless.

    This is the non-LUNA trait relation. Existing prognostic BGC leaf-N
    limitation remains a separate factor; this object creates no C/N stores.
    """
    specific_leaf_area: float
    leaf_cn: float
    rubisco_nitrogen_fraction: float
    foliage_nitrogen_factor: float = 1.

    def __post_init__(self):
        values = np.asarray(astuple(self), float)
        if (not np.all(np.isfinite(values)) or self.specific_leaf_area <= 0. or self.leaf_cn <= 0.
                or not 0. <= self.rubisco_nitrogen_fraction <= 1.
                or not 0. <= self.foliage_nitrogen_factor <= 1.):
            raise ValueError('Invalid C3 leaf-capacity traits')


@dataclass(frozen=True)
class CanopyCapacityEnvironment:
    """Current and annual-maximum photoperiod, from explicit station geometry."""
    daylength_seconds: float
    maximum_daylength_seconds: float

    def __post_init__(self):
        day,maximum=self.daylength_seconds,self.maximum_daylength_seconds
        if not (math.isfinite(day) and math.isfinite(maximum) and 0.<=day<=maximum<=86400. and maximum>0.):
            raise ValueError('Invalid canopy-capacity photoperiod')


def canopy_capacity_profile(leaf_area, daylength_seconds, maximum_daylength_seconds,
                            mean_temperature_k, parameters, *, leaf_parameters=None):
    """Return the complete per-layer LeafParameters rows and raw diagnostics.

    Leaf N decays with cumulative LEAF area at the layer midpoint; stem area
    never enters this profile. The same profile scales Vcmax, Jmax, TPU and Rd.
    Photoperiod scales top capacity with a floor of 0.01. Jmax/Vcmax and the
    temperature-response entropies use the clipped ten-day temperature.
    """
    if not isinstance(parameters, CanopyCapacityParameters):
        raise ValueError('Explicit CanopyCapacityParameters required')
    p = parameters
    leaf = np.asarray(leaf_area, float)
    if leaf.ndim != 1 or not leaf.size or not np.all(np.isfinite(leaf)) or np.any(leaf < 0.):
        raise ValueError('Leaf area must be a nonempty finite nonnegative vector')
    day, longest, temperature = daylength_seconds, maximum_daylength_seconds, mean_temperature_k
    if (not all(math.isfinite(v) for v in (day, longest, temperature))
            or not 0. <= day <= longest <= 86400. or longest == 0. or not 200. <= temperature <= 350.):
        raise ValueError('Invalid daylength or acclimation temperature')
    template = LeafParameters() if leaf_parameters is None else leaf_parameters
    if not isinstance(template, LeafParameters):
        raise ValueError('Expected LeafParameters for gas-exchange coefficients')
    nitrogen_top = 1. / (p.specific_leaf_area * p.leaf_cn)
    unscaled = nitrogen_top * p.rubisco_nitrogen_fraction * 7.16 * 60. * p.foliage_nitrogen_factor
    exponent = .00963 * unscaled - 2.43
    if not math.isfinite(unscaled) or exponent > math.log(np.finfo(float).max):
        raise ValueError('Leaf traits exceed finite C3 capacity range')
    kn = math.exp(exponent)
    photoperiod = max(.01, (day / longest)**2)
    growth_c = min(max(temperature - 273.15, 11.), 35.)
    midpoint = np.cumsum(leaf) - .5 * leaf
    profile = np.where(leaf > 0., np.exp(-kn * midpoint), 0.)
    vcmax_top = unscaled * photoperiod
    rows = np.broadcast_to(np.asarray(astuple(template), float), (len(leaf), 10)).copy()
    rows[:, 0] = vcmax_top * profile
    rows[:, 1] = (2.59 - .035 * growth_c) * rows[:, 0]
    rows[:, 2] = .167 * rows[:, 0]
    rows[:, 3] = .015 * rows[:, 0]
    rows[:, 8] = growth_c + 273.15
    return dict(leaf_parameters=rows, nitrogen_profile_factor=profile,
                leaf_nitrogen_gn_m2_leaf=nitrogen_top * profile,
                unscaled_vcmax25_top=unscaled, vcmax25_top=vcmax_top,
                nitrogen_extinction=kn, daylength_factor=photoperiod,
                mean_temperature_k=float(temperature), growth_temperature_k=growth_c + 273.15)
