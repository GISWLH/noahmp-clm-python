"""Anderson 雪压实的物理契约测试。"""

import numpy as np

from noahmp_clm import ColumnState, PhysicsOptions
from noahmp_clm.physics.snow import compact_snow


def test_anderson_compaction_increases_density_and_conserves_swe() -> None:
    state = ColumnState(
        swe=0.30,
        snow_swe_layers=np.array([0.10, 0.10, 0.10]),
        snow_t_layers=np.array([268.0, 270.0, 272.0]),
        snow_density_layers=np.array([120.0, 250.0, 350.0]),
        options=PhysicsOptions(OptSnowCompaction=1),
    )
    before_swe = state.swe
    before_density = state.snow_density_layers.copy()
    compact_snow(state, 3600.0, option=1)
    assert np.all(state.snow_density_layers >= before_density)
    assert np.any(state.snow_density_layers > before_density)
    assert np.isclose(state.swe, before_swe)

