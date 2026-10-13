"""Regression fixtures from K3, with quantified physical rather than 1e-4 gates."""
import json
from pathlib import Path

import numpy as np
import pytest

from noahmp_clm import ColumnState, default_soil_params
from noahmp_clm.physics.frozen_soil import _smooth_water_cap, unfrozen_water_content
from noahmp_clm.physics.soil import step_soil_heat


def test_smooth_cap_is_local_monotone_and_within_one_percent():
    total = 0.12
    potential = np.linspace(0., .3, 10001)
    smooth = _smooth_water_cap(total, potential)
    old = np.minimum(total, potential)
    assert np.all(np.diff(smooth) >= -1e-15)
    assert np.max(np.abs(smooth - old)) <= .01 * total + 1e-14
    exterior = np.abs(potential - total) >= .04 * total
    np.testing.assert_array_equal(smooth[exterior], old[exterior])


def test_recorded_k3_temperature_perturbation_is_not_amplified():
    path = Path(__file__).parent / "fixtures/k3_phase_causal.json"
    rows = [r for r in json.loads(path.read_text())["heat"] if r["column"] == 49974 and r["step"] == 20]
    final = []
    for row in rows:
        state = ColumnState(soil=default_soil_params("sand"),
            smc=np.array(row["smc"]), tsoil=np.array(row["old_temperature"]))
        step_soil_heat(state, 1800., row["g"])
        final.append(state.tsoil[0])
    # Original independently captured probe: 0.194006536 K. New acceptance
    # requires at least 90% reduction in the same frozen-state perturbation.
    assert abs(final[1] - final[0]) < 0.0194006536, final


@pytest.mark.parametrize("soil_name", ["sand", "loam", "clay"])
def test_zero_flux_uniform_soil_is_exact_equilibrium(soil_name):
    state = ColumnState(soil=default_soil_params(soil_name), tsoil=np.full(4, 268.0))
    initial = state.tsoil.copy()
    for _ in range(48):
        step_soil_heat(state, 1800., 0.)
    np.testing.assert_array_equal(state.tsoil, initial)
    expected_ice = state.smc - unfrozen_water_content(state.smc, initial, state.soil)
    np.testing.assert_allclose(state.soil_ice, expected_ice, atol=1e-15)

