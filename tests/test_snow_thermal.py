"""雪热传导选项的物理契约测试。"""

from noahmp_clm.physics.snow import snow_thermal_conductivity


def test_stieglitz_conductivity_increases_with_density() -> None:
    light = snow_thermal_conductivity(150.0, option=1)
    dense = snow_thermal_conductivity(450.0, option=1)
    assert dense > light > 0.0


def test_all_upstream_conductivity_options_are_positive() -> None:
    values = [snow_thermal_conductivity(300.0, option=i) for i in range(1, 6)]
    assert all(value > 0.0 for value in values)
    assert snow_thermal_conductivity(300.0, option=3) == 0.30

