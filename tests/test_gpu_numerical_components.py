"""Targeted regressions for numerical discrepancies identified in J2."""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from noahmp_clm.column import ColumnState
from noahmp_clm.gpu import jax_core
from noahmp_clm.gpu.pack import column_state_to_vec
from noahmp_clm.physics import snow


@pytest.mark.parametrize("top_flux", [1.0, 10.0, 100.0])
def test_thin_active_snow_layer_heat_matches_reference(top_flux):
    """A valid thin top layer must retain its actual mass/thermal capacity."""
    jax.config.update("jax_enable_x64", True)
    state = ColumnState(swe=.001001,
                        snow_swe_layers=np.array([1e-6, .001]),
                        snow_t_layers=np.array([273.15, 270.]),
                        snow_density_layers=np.array([300., 300.]))
    f, i = column_state_to_vec(state)
    reference = state.copy()
    expected_melt, expected_soil_heat = snow.apply_snow_heat(reference, 1800., top_flux)
    actual = jax.jit(jax_core._snow_heat)(
        jnp.asarray(f[16:19]), jnp.asarray(f[19:22]), jnp.asarray(f[22:25]),
        jnp.int32(i[32]), jnp.float64(1800.), jnp.float64(top_flux), jnp.int32(1))
    swe, temperature, density, count, melt, soil_heat = map(np.asarray, actual)
    active_count = len(reference.snow_swe_layers)
    print("thin-snow", top_flux, "melt", float(melt), expected_melt,
          "temperature", temperature[:active_count], reference.snow_t_layers, flush=True)
    assert int(count) == active_count
    np.testing.assert_allclose(swe[:active_count], reference.snow_swe_layers, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(temperature[:active_count], reference.snow_t_layers, rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(density[:active_count], reference.snow_density_layers, rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(melt, expected_melt, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(soil_heat, expected_soil_heat, rtol=1e-8, atol=1e-8)
