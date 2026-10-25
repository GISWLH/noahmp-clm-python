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


@pytest.mark.parametrize("minimum", [.001, .01, .2])
def test_dark_closed_form_and_mixed_batch_iterative_reference(minimum):
    rng=np.random.default_rng(1709)
    n=257
    p=LeafParameters(minimum_conductance=minimum)
    temperature=rng.uniform(200.,350.,n)
    ca=rng.uniform(100.,1200.,n)
    gb=np.exp(rng.uniform(np.log(.001),np.log(10.),n))
    rh=rng.uniform(0.,1.5,n)
    stress=rng.uniform(0.,1.,n)
    kw=dict(co2_ppm=ca,relative_humidity=rh,boundary_conductance=gb,water_stress=stress,parameters=p)
    actual=leaf_gas_exchange(np.zeros(n),temperature,**kw)
    # Add a lit leaf to retain the existing iterative path in the same call.
    mixed=leaf_gas_exchange(np.r_[np.zeros(n),500.],np.r_[temperature,298.15],
        co2_ppm=np.r_[ca,420.],relative_humidity=np.r_[rh,.7],boundary_conductance=np.r_[gb,1.],
        water_stress=np.r_[stress,1.],parameters=p)
    np.testing.assert_array_equal(actual['gross'],np.zeros(n))
    effective_minimum=np.maximum(minimum*stress,1e-6)
    np.testing.assert_allclose(actual['gs_mol'],effective_minimum,rtol=1e-14)
    expected_ci=ca+actual['respiration']*(1.4/gb+1.6/effective_minimum)
    np.testing.assert_allclose(actual['ci_ppm'],expected_ci,rtol=1e-14,atol=1e-12)
    for key,value in actual.items():
        np.testing.assert_allclose(value,mixed[key][:-1],rtol=1e-11,atol=1e-10,err_msg=key)
    assert np.max(abs(actual['diffusion_residual']))<1e-10
