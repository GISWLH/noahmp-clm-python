"""雪热传导选项的物理契约测试。"""

from noahmp_clm import ColumnState, PhysicsOptions, default_soil_params, synthetic_forcing
from noahmp_clm.driver import run_column
from noahmp_clm.physics.snow import snow_thermal_conductivity


def test_stieglitz_conductivity_increases_with_density() -> None:
    light = snow_thermal_conductivity(150.0, option=1)
    dense = snow_thermal_conductivity(450.0, option=1)
    assert dense > light > 0.0


def test_all_upstream_conductivity_options_are_positive() -> None:
    values = [snow_thermal_conductivity(300.0, option=i) for i in range(1, 6)]
    assert all(value > 0.0 for value in values)
    assert snow_thermal_conductivity(300.0, option=3) == 0.30


def test_snow_energy_closure_survives_option_switches() -> None:
    """各雪面方案在冷暖转换期间仍满足能量闭合。"""
    forcings = (
        synthetic_forcing(n_days=2, tmean=268.0, tamp=3.0, seed=1)
        + synthetic_forcing(n_days=2, tmean=288.0, tamp=6.0, seed=2)
    )
    fields = {
        "OptSnowAlbedo": (1, 2),
        "OptSnowThermConduct": (1, 2, 3, 4, 5),
        "OptSnowCompaction": (1, 2),
    }
    for field, values in fields.items():
        for value in values:
            options = PhysicsOptions(**{field: value})
            state = ColumnState(soil=default_soil_params("loam"), options=options)
            diagnostics, _ = run_column(state, forcings)
            assert max(abs(item["residual"]) for item in diagnostics) < 0.5

