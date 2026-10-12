"""冻土未冻水与冻结渗透率测试。"""

import numpy as np

from noahmp_clm import ColumnState, PhysicsOptions
from noahmp_clm.physics.frozen_soil import (
    apparent_heat_capacity,
    permeability_factor,
    unfrozen_water_content,
    update_soil_phase,
)


def test_minus_five_celsius_has_less_unfrozen_water() -> None:
    state = ColumnState(options=PhysicsOptions(OptSoilSupercoolWater=1))
    state.smc[:] = 0.30
    state.tsoil[:] = 268.15
    liquid = unfrozen_water_content(
        state.smc, np.full(4, 268.15), state.soil, option=1
    )
    assert np.all(liquid < state.smc)
    update_soil_phase(state)
    assert np.all(state.soil_ice > 0.0)


def test_apparent_heat_capacity_includes_phase_change() -> None:
    state = ColumnState()
    normal = apparent_heat_capacity(
        state.smc, np.full(4, 280.0), state.soil, option=1
    )
    freezing = apparent_heat_capacity(
        state.smc, np.full(4, 272.5), state.soil, option=1
    )
    assert np.all(freezing > normal)


def test_nonlinear_frozen_permeability_is_smaller() -> None:
    state = ColumnState()
    state.smc[:] = 0.30
    state.soil_ice[:] = 0.10
    linear = permeability_factor(state, 1)
    nonlinear = permeability_factor(state, 2)
    assert np.all(nonlinear < linear)

