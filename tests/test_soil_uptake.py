import numpy as np
import pytest
from noahmp_clm.column import ColumnState
from noahmp_clm.physics.soil_uptake import allocate_root_uptake, SoilWaterSupply


def test_compensation_and_no_roots():
    available = np.array([.001,.1,.5,.3])
    actual = allocate_root_uptake(.1,available,[.8,.1,.1,0.])
    np.testing.assert_allclose(actual,[.001,.0495,.0495,0.],atol=1e-15)
    np.testing.assert_array_equal(allocate_root_uptake(1.,available,np.zeros(4)),np.zeros(4))


def test_root_water_fill_matches_independent_multiplier():
    rng = np.random.default_rng(943)
    for _ in range(128):
        available,weights = rng.uniform(0.,1.,(2,4))
        weights[rng.uniform(0.,1.,4)<.2] = 0.
        demand = rng.uniform(0.,4.)
        take = allocate_root_uptake(demand,available,weights)
        rooted = weights>0
        target = min(demand,available[rooted].sum())
        lo,hi = 0.,np.max(available[rooted]/weights[rooted],initial=0.)
        for _ in range(80):
            level = .5*(lo+hi)
            if np.minimum(available,level*weights).sum() < target:
                lo = level
            else:
                hi = level
        np.testing.assert_allclose(take,np.minimum(available,.5*(lo+hi)*weights),atol=1e-14)
        assert abs(take.sum()-target) < 1e-14


def test_supply_uses_liquid_water_and_copies_inputs():
    state = ColumnState(smc=np.array([.3,.3,.3,.3]),soil_ice=np.array([.25,.15,0.,0.]))
    supply = SoilWaterSupply.from_column(state,1800.)
    assert supply.roots[0] == 0.
    assert supply.surface == pytest.approx(.05*.1*1000./1800.)
    np.testing.assert_allclose(supply.roots,np.maximum(state.smc-state.soil_ice-.12,0.)*state.dz*1000./1800.)
    state.root_frac[:] = 0.
    assert supply.weights.sum() == pytest.approx(1.)
    with pytest.raises(ValueError):
        SoilWaterSupply(np.ones(4),np.ones(4),.5)
