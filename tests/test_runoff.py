"""TOPMODEL 与 BATS 径流的物理契约测试。"""

import numpy as np

from noahmp_clm import ColumnState, PhysicsOptions
from noahmp_clm.physics.runoff import route_runoff, topmodel_runoff


def test_topmodel_conserves_candidate_runoff() -> None:
    state = ColumnState(
        terrain_index=1.5,
        groundwater_depth=0.5,
        options=PhysicsOptions(OptRunoffSurface=1),
    )
    state.smc[:] = state.soil.smcmax
    surface, subsurface = topmodel_runoff(0.01, state, equilibrium=False)
    assert surface > 0.0
    assert np.isclose(surface + subsurface, 0.01)


def test_topmodel_terrain_index_increases_surface_fraction() -> None:
    low = ColumnState(terrain_index=-1.0, groundwater_depth=0.5)
    high = ColumnState(terrain_index=2.0, groundwater_depth=0.5)
    low.smc[:] = low.soil.smcmax
    high.smc[:] = high.soil.smcmax
    low_surface, _ = topmodel_runoff(0.01, low, equilibrium=True)
    high_surface, _ = topmodel_runoff(0.01, high, equilibrium=True)
    assert high_surface >= low_surface


def test_option_four_bats_and_option_three_are_available() -> None:
    state = ColumnState()
    state.smc[:] = state.soil.smcmax
    assert np.isclose(sum(route_runoff(0.01, state, 3)), 0.01)
    assert route_runoff(0.01, state, 4)[0] > 0.0


def test_surface_and_subsurface_options_can_be_selected_independently() -> None:
    state = ColumnState(terrain_index=2.0, groundwater_depth=0.5)
    state.smc[:] = state.soil.smcmax
    surface, subsurface = route_runoff(0.01, state, 3, subsurface_option=4)
    assert np.isclose(surface + subsurface, 0.01)

