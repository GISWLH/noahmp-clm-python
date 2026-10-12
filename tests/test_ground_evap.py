"""Sakaguchi--Zeng 地面蒸发阻抗测试。"""

from noahmp_clm import ColumnState, PhysicsOptions
from noahmp_clm.physics.ground_evap import ground_resistance


def test_sakaguchi_zeng_resistance_rises_for_dry_surface() -> None:
    wet = ColumnState(options=PhysicsOptions(OptGroundResistanceEvap=1))
    dry = wet.copy()
    wet.smc[0] = wet.soil.smcmax
    dry.smc[0] = dry.soil.smcwlt
    assert ground_resistance(wet, 1) < ground_resistance(dry, 1)
    assert ground_resistance(wet, 1) > 0.0


def test_ground_resistance_dispatch_and_snow_value() -> None:
    state = ColumnState()
    assert ground_resistance(state, 2) != ground_resistance(state, 3)
    assert ground_resistance(state, 4, snow_cover=True) == 1.0

