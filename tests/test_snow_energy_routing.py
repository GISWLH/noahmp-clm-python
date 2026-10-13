"""Independent mass/enthalpy budgets for partial and complete snow melting."""
import numpy as np
import pytest

from noahmp_clm import ColumnState, Forcing
from noahmp_clm.driver import step_column
from noahmp_clm.physics import snow, soil


def sensible(state):
    depth = state.snow_swe_layers * 1000.0 / state.snow_density_layers
    return np.sum(snow.C_SNOW * depth * (state.snow_t_layers - 273.15))


@pytest.mark.parametrize("flux", [-40.0, 20.0, 200.0, 1500.0])
@pytest.mark.parametrize("temperatures", [[273.15], [273.15, 260.0], [269.0, 271.0, 265.0]])
def test_snow_heat_has_no_unrouted_energy(flux, temperatures):
    masses = np.full(len(temperatures), 0.001)
    state = ColumnState(swe=masses.sum(), snow_swe_layers=masses,
        snow_t_layers=np.array(temperatures), snow_density_layers=np.full(len(masses), 300.0))
    old_heat, old_mass = sensible(state), state.swe
    melt, soil_energy = snow.apply_snow_heat(state, 1800.0, flux)
    stored = sensible(state) - old_heat
    np.testing.assert_allclose(state.swe + melt, old_mass, atol=1e-14, rtol=0)
    np.testing.assert_allclose(stored + melt * 1000.0 * snow.LF + soil_energy,
        flux * 1800.0, atol=1e-7, rtol=1e-12)
    assert soil_energy >= 0.0
    assert np.all(state.snow_t_layers <= 273.15)


def test_original_melt_out_gap_reaches_soil(monkeypatch):
    state = ColumnState(swe=0.00011, snow_t=273.15,
        snow_swe_layers=np.array([0.00011]), snow_t_layers=np.array([273.15]),
        snow_density_layers=np.array([300.0]))
    received = []
    original = soil.step_soil_heat
    def capture(s, dt, flux):
        # Soil is unfrozen throughout: this is an independent exact heat budget.
        capacity = soil.heat_capacity(s.smc, s.soil)
        old_temperature = s.tsoil.copy()
        original(s, dt, flux)
        received.append(float(np.sum(capacity * s.dz * (s.tsoil - old_temperature))))
    monkeypatch.setattr(soil, "step_soil_heat", capture)
    final, diag = step_column(state, Forcing(1800., 295., .004, 2., 1e5, 700., 350., 0.))
    assert final.swe == 0.0
    expected = diag["g"] * 1800.0 - state.swe * 1000.0 * snow.LF
    assert expected > 200000.0
    np.testing.assert_allclose(diag["snow_excess_heat_jm2"], expected, atol=1e-7)
    np.testing.assert_allclose(received[0], expected, atol=1e-6)
