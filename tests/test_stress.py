"""Noah/CLM/SSiB 土壤水胁迫的物理契约测试。"""

import numpy as np

from noahmp_clm import ColumnState, PhysicsOptions
from noahmp_clm.physics.stress import soil_water_stress


def test_all_stress_options_are_wet_dry_bounded() -> None:
    for option in (1, 2, 3):
        wet = ColumnState(options=PhysicsOptions(OptSoilWaterTranspiration=option))
        dry = wet.copy()
        wet.smc[:] = wet.soil.smcref
        dry.smc[:] = dry.soil.smcwlt
        assert 0.0 <= soil_water_stress(wet, option) <= 1.0
        assert 0.0 <= soil_water_stress(dry, option) <= 1.0
        assert soil_water_stress(wet, option) > soil_water_stress(dry, option)


def test_matric_stress_is_monotone_and_layer_aware() -> None:
    state = ColumnState(options=PhysicsOptions(OptSoilWaterTranspiration=2))
    state.smc[:] = np.array([0.14, 0.20, 0.25, 0.28])
    root = soil_water_stress(state, 2)
    surface = soil_water_stress(state, 2, layer=0)
    assert 0.0 <= surface <= 1.0 and 0.0 <= root <= 1.0
    wetter = state.copy()
    wetter.smc[:] = np.minimum(wetter.soil.smcmax, wetter.smc + 0.05)
    assert soil_water_stress(wetter, 2) > root

