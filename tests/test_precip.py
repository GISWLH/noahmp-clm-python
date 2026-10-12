"""降水相态分配选项的物理契约测试。"""

import pytest

from noahmp_clm.physics.precip import (
    bats_snow_fraction,
    freezing_threshold_snow_fraction,
    jordan91_snow_fraction,
    partition_precipitation,
)


def test_jordan91_piecewise_temperature_response() -> None:
    assert jordan91_snow_fraction(273.65) == 1.0
    assert 0.0 < jordan91_snow_fraction(274.65) < 1.0
    assert jordan91_snow_fraction(276.0) == 0.0


def test_bats_and_freezing_threshold() -> None:
    assert bats_snow_fraction(275.0) == 1.0
    assert bats_snow_fraction(275.35) == 0.0
    assert freezing_threshold_snow_fraction(273.14) == 1.0
    assert freezing_threshold_snow_fraction(273.15) == 0.0


def test_partition_conserves_water_and_rejects_unimplemented_scheme() -> None:
    rain, snow = partition_precipitation(2.0e-5, 600.0, 272.0, option=2)
    assert rain == 0.0
    assert rain + snow == pytest.approx(0.012)
    with pytest.raises(NotImplementedError):
        partition_precipitation(1.0e-5, 600.0, 273.0, option=5)

