"""BATS/CLASS 雪反照率物理契约测试。"""

from noahmp_clm.physics.snow import albedo_bats, albedo_class, update_snow_albedo


def test_bats_albedo_decays_with_age_and_resets() -> None:
    fresh = albedo_bats(0.0, 268.0, new_snow=True)
    old = albedo_bats(10.0, 268.0)
    assert fresh > old
    reset, age = update_snow_albedo(0.1, old, 10.0, 268.0, 1800.0, 0.01, 1)
    assert age == 0.0 and reset == fresh


def test_class_uses_temperature_regimes() -> None:
    cold = albedo_class(3.0, 268.0)
    wet = albedo_class(3.0, 272.0)
    assert cold > wet
    assert albedo_class(0.0, 268.0, new_snow=True) == 0.84

