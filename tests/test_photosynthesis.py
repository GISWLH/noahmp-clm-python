import json
from pathlib import Path
import numpy as np
import pytest

from noahmp_clm.physics.photosynthesis import LeafParameters, farquhar_rates, leaf_gas_exchange


def test_single_leaf_light_response_against_published_equation_table():
    reference = json.loads((Path(__file__).parent / "fixtures/leaf_clm5_equations.json").read_text())
    rows = reference["rows"]
    result = farquhar_rates([x["ci_ppm"] for x in rows], [x["absorbed_par"] for x in rows], 298.15)
    for key in ("gross", "net", "rubisco", "rubp"):
        expected = np.array([x[key] for x in rows])
        np.testing.assert_allclose(result[key], expected, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("temperature", [280., 298.15, 315.])
@pytest.mark.parametrize("humidity", [0.1, 0.7, 1.0])
def test_coupled_co2_and_humidity_budgets(temperature, humidity):
    r = leaf_gas_exchange(np.array([0., 100., 1000.]), temperature, relative_humidity=humidity)
    assert np.max(np.abs(r["diffusion_residual"])) < 1e-9
    assert r["net"][0] < 0.0 and r["gross"][0] == 0.0
    assert r["ci_ppm"][0] > 420.0  # Respiration diffuses out at night.
    np.testing.assert_allclose(r["gs_mol"][0], .01)
    expected_gs = .01 + 9.0*np.maximum(r["net"], 0.)*r["surface_relative_humidity"]/r["cs_ppm"]
    np.testing.assert_allclose(r["gs_mol"], expected_gs, atol=1e-13)


def test_all_limiting_processes_and_nitrogen_stress():
    limited = farquhar_rates(1000., 2000., 298.15, LeafParameters(tpu25=2.))
    assert limited["gross"] == 6.0
    ordinary = leaf_gas_exchange(1500., 298.15)
    dry = leaf_gas_exchange(1500., 298.15, water_stress=.05)
    low_n = leaf_gas_exchange(1500., 298.15, nitrogen_factor=.2)
    assert dry["gross"] < ordinary["gross"] and low_n["gross"] < ordinary["gross"]
    np.testing.assert_allclose(leaf_gas_exchange(0., 298.15)["net"], -.9)


def test_invalid_leaf_inputs_fail_instead_of_clipping():
    with pytest.raises(ValueError):
        farquhar_rates(280., -1., 298.15)
    with pytest.raises(ValueError):
        LeafParameters(curvature=2.)
    with pytest.raises(ValueError):
        leaf_gas_exchange(100., 298.15, boundary_conductance=0.)
