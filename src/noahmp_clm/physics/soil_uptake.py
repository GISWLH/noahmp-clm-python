"""Shared liquid-water supply for root uptake and surface evaporation."""
from dataclasses import dataclass
import numpy as np


def allocate_root_uptake(demand, available, weights):
    """Root-weighted withdrawal with compensation from nondepleted layers.

    Solve take[i] = min(available[i], multiplier * weights[i]), with total
    take = min(demand, accessible water). Inputs share any consistent water
    unit (the energy solver uses kg/m²/s). Zero-root layers cannot supply water.
    """
    available, weights = np.asarray(available,float), np.asarray(weights,float)
    if (available.ndim != 1 or available.shape != weights.shape or not available.size
            or not np.all(np.isfinite(available)) or not np.all(np.isfinite(weights))
            or np.any(available < 0) or np.any(weights < 0)
            or not np.isfinite(demand) or demand < 0):
        raise ValueError("Invalid root water supply")
    return _allocate(demand,available,weights)


def _allocate(demand, available, weights):
    take = np.zeros_like(available)
    target = min(demand,np.where(weights > 0.,available,0.).sum())
    for _ in range(len(available)):
        remaining = max(0.,target-take.sum())
        active_weights = np.where(take < available,weights,0.)
        total_weight = active_weights.sum()
        if total_weight > 0.:
            take += np.minimum(remaining*active_weights/total_weight,available-take)
    return take


@dataclass(frozen=True)
class SoilWaterSupply:
    """Limits in kg/m²/s over one time step; surface includes the root share.

    Root capacities contain only liquid water above the extraction threshold.
    ``surface`` is ALL extractable top-layer liquid, so top root withdrawal
    must be subtracted before evaluating the ground evaporation limit.
    """
    roots: np.ndarray
    weights: np.ndarray
    surface: float

    def __post_init__(self):
        roots, weights = np.array(self.roots,float,copy=True), np.array(self.weights,float,copy=True)
        allocate_root_uptake(0.,roots,weights)
        if not np.isfinite(self.surface) or self.surface < 0 or roots[0] > self.surface:
            raise ValueError("Invalid surface liquid-water supply")
        roots.setflags(write=False)
        weights.setflags(write=False)
        object.__setattr__(self,"roots",roots)
        object.__setattr__(self,"weights",weights)

    @classmethod
    def from_column(cls, state, dt):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Time step must be positive")
        liquid = np.maximum(state.smc-state.soil_ice,0.)
        roots = np.maximum(liquid-state.soil.smcwlt,0.)*state.dz*1000./dt
        surface = liquid[0]*state.dz[0]*1000./dt
        return cls(roots,state.root_frac,surface)
