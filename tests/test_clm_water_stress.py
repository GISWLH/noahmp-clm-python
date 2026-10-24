"""Independent pinned-reference and physical contracts for development beta."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from noahmp_clm.column import ColumnState, default_soil_params
from noahmp_clm.physics.clm_water_stress import soil_water_stress
from noahmp_clm.physics.stress import soil_water_stress as legacy_stress


REFERENCE = json.loads((Path(__file__).parent/'fixtures'/'soil_water_stress_reference.json').read_text())


@pytest.mark.parametrize('texture', ['sand', 'loam', 'clay'])
@pytest.mark.parametrize('option', [1, 2, 3])
def test_pinned_equation_reference(texture, option):
    selected = [q for q in REFERENCE['cases']
                if q['texture'] == texture and q['option'] == option]
    assert selected
    for q in selected:
        state = ColumnState(soil=default_soil_params(texture),
                            smc=np.full(4, q['theta_m3m3']),
                            root_frac=np.array([.4, .3, .2, .1]))
        expected = q['reference_beta']
        actual = soil_water_stress(state, option, layer=0)
        # Prospectively fixed diagnostic tolerance; never divide a near-zero beta.
        tolerance = .001 if expected <= .01 else .1*expected
        assert abs(actual-expected) <= tolerance, q


@pytest.mark.parametrize('option', [2, 3])
def test_physical_bounds_and_explicit_root_distribution(option):
    state = ColumnState(smc=np.array([0., .12, .28, .45]),
                        root_frac=np.array([0., .25, .75, 0.]))
    lookup = {q['theta_m3m3']: q['reference_beta'] for q in REFERENCE['cases']
              if q['texture'] == 'loam' and q['option'] == option}
    expected = .25*lookup[.12]+.75*lookup[.28]
    assert soil_water_stress(state, option, 0) == 0.
    assert soil_water_stress(state, option, 3) == pytest.approx(1., abs=1e-12)
    assert abs(soil_water_stress(state, option)-expected) <= .1*expected
    state.root_frac[:] = 0.
    assert soil_water_stress(state, option) == 0.


@pytest.mark.parametrize('option', [2, 3])
def test_saturated_suction_changes_drought_response(option):
    state = ColumnState(smc=np.full(4, .18))
    smaller = replace(state, soil=replace(state.soil, saturated_suction_m=.1))
    larger = replace(state, soil=replace(state.soil, saturated_suction_m=1.))
    assert soil_water_stress(smaller, option) > soil_water_stress(larger, option)


def test_noah_option_retains_legacy_arithmetic():
    state = ColumnState(smc=np.array([.13, .19, .24, .31]))
    assert soil_water_stress(state, 1) == legacy_stress(state, 1)
    for k in range(4):
        assert soil_water_stress(state, 1, k) == legacy_stress(state, 1, k)


def test_quantized_native_root_profile_is_retained():
    state = ColumnState(smc=np.full(4, .28),
                        root_frac=np.array([.4, .3, .2, .1], np.float32).astype(float))
    for option in (2, 3):
        assert 0.0 < soil_water_stress(state, option) <= 1.0


@pytest.mark.parametrize('option', [1, 2, 3])
def test_development_driver_routes_actual_soil_stress(monkeypatch, option):
    import noahmp_clm.clm_enthalpy_driver as driver
    from noahmp_clm.column import Forcing
    from noahmp_clm.clm_state import CLMColumnState, clm_options
    old = ColumnState(smc=np.array([.137, .137, .28, .45]),
                      tsoil=np.full(4, 290.), ts=290.,
                      root_frac=np.array([.4, .3, .2, .1]))
    state = CLMColumnState.from_column(old, options=replace(
        clm_options(), OptSoilWaterTranspiration=option))
    observed = []
    original = driver.surface_step
    def surface(*args, **kwargs):
        observed.append(kwargs['water_stress'])
        return original(*args, **kwargs)
    monkeypatch.setattr(driver, 'surface_step', surface)
    _, diagnostics = driver.step_clm_enthalpy_column(
        state, Forcing(1800., 295., .008, 3., 101325., 500., 350., 0.))
    lookup = {q['theta_m3m3']: q['reference_beta'] for q in REFERENCE['cases']
              if q['texture'] == 'loam' and q['option'] == option}
    expected = sum(root*lookup[theta] for root, theta in zip(old.root_frac, old.smc))
    assert observed and all(abs(beta-expected) <= .1*expected for beta in observed)
    assert abs(diagnostics['water_residual_kgm2']) < 3e-10
    assert abs(diagnostics['energy_residual_jm2']) < .02


@pytest.mark.parametrize('option', [2, 3])
@pytest.mark.parametrize('wilting', [0., 1., -.1, float('nan')])
def test_invalid_potential_rejected(option, wilting):
    with pytest.raises(ValueError):
        soil_water_stress(ColumnState(), option, wilting_potential_m=wilting)


@pytest.mark.parametrize('option', [2, 3])
def test_invalid_root_profile_rejected(option):
    state = ColumnState(root_frac=np.full(4, .5))
    with pytest.raises(ValueError):
        soil_water_stress(state, option)
