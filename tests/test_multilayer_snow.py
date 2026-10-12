"""多层积雪的分层、合并和隐式导热测试。"""

import numpy as np

from noahmp_clm import ColumnState
from noahmp_clm.physics.snow import (
    add_snowfall,
    apply_snow_heat,
    combine_snow_layers,
    divide_snow_layers,
)


def test_snow_divide_and_combine_preserve_swe() -> None:
    layers = divide_snow_layers(0.30)
    assert len(layers) == 3 and np.sum(layers) == 0.30
    swe, temp, dens = combine_snow_layers(
        np.array([0.01, 0.02, 0.03, 0.24]),
        np.array([268.0, 269.0, 270.0, 271.0]),
        np.full(4, 300.0),
    )
    assert len(swe) <= 3
    np.testing.assert_allclose(np.sum(swe), 0.30)


def test_multilayer_heat_couples_layers_and_melts_at_zero() -> None:
    state = ColumnState(
        swe=0.30,
        snow_swe_layers=np.array([0.10, 0.10, 0.10]),
        snow_t_layers=np.array([268.0, 269.0, 270.0]),
        snow_density_layers=np.full(3, 300.0),
    )
    before = state.snow_t_layers.copy()
    melt = apply_snow_heat(state, 1800.0, 200.0)
    assert np.all(np.isfinite(state.snow_t_layers))
    assert state.snow_t_layers[1] > before[1]
    assert melt >= 0.0


def test_new_snow_is_added_and_capped_at_three_layers() -> None:
    state = ColumnState()
    for _ in range(10):
        add_snowfall(state, 0.04, 268.0)
    assert len(state.snow_swe_layers) <= 3
    assert state.swe == np.sum(state.snow_swe_layers)

