"""MOST 地表拖曳方案的物理契约测试。"""

from noahmp_clm import Forcing
from noahmp_clm.physics.energy import bulk_ch
from noahmp_clm.physics.surface_drag import exchange_velocity, most_exchange_velocity


def _forcing(tair: float = 285.0, wind: float = 3.0) -> Forcing:
    return Forcing(1800.0, tair, 0.006, wind, 101325.0, 500.0, 350.0, 0.0)


def test_option_two_matches_neutral_bulk() -> None:
    f = _forcing()
    expected = bulk_ch(f.wind, 0.05) * f.wind
    assert exchange_velocity(f, 0.05, 285.0, option=2) == expected


def test_most_is_finite_and_stability_sensitive() -> None:
    f = _forcing()
    neutral = most_exchange_velocity(f, 0.05, f.tair)
    unstable = most_exchange_velocity(f, 0.05, f.tair + 8.0)
    stable = most_exchange_velocity(f, 0.05, f.tair - 8.0)
    assert neutral > 0.0 and stable > 0.0 and unstable > 0.0
    assert unstable != stable

