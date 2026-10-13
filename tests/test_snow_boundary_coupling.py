"""Regression for the ordinary model's day-three thin-snow overcooling."""
import json
from pathlib import Path
import numpy as np
import pytest

from noahmp_clm import ColumnState, Forcing
from noahmp_clm.driver import step_column
from noahmp_clm.gpu.pack import vec_to_column_state
from noahmp_clm.physics import snow


@pytest.mark.parametrize('layers', [1,2,3])
def test_implicit_snow_boundary_closes_heat_and_flux(layers):
    state = ColumnState(swe=.003*layers,
        snow_swe_layers=np.full(layers,.003),
        snow_t_layers=np.linspace(268.,270.,layers),
        snow_density_layers=np.full(layers,300.))
    old = state.snow_t_layers.copy()
    depth = state.snow_swe_layers*1000./state.snow_density_layers
    surface = 260.
    response = snow.snow_surface_response(state,1800.)
    flux = snow.coupled_snow_flux(surface,response)
    melt, soil_heat = snow.apply_snow_heat(state,1800.,flux)
    assert melt == 0.
    assert soil_heat == 0.
    assert state.snow_t_layers.min() >= surface
    assert state.snow_t_layers.max() <= old.max()
    np.testing.assert_allclose(np.sum(snow.C_SNOW*depth*(state.snow_t_layers-old)),
                               1800.*flux,rtol=1e-11,atol=1e-8)
    np.testing.assert_allclose(response[2]*(surface-state.snow_t_layers[0]),
                               flux,rtol=1e-11,atol=1e-10)


def test_original_day_three_thin_snow_slice_stays_stable():
    fixture = json.loads((Path(__file__).parent/'fixtures/annual_thin_snow.json').read_text())
    state = vec_to_column_state(fixture['f'],fixture['i'],np.array([.1,.3,.6,1.]))
    for values in fixture['forcing']:
        state,diag = step_column(state,Forcing(**values))
        assert abs(diag['residual']) < .5
        assert state.snow_t_layers.min() > 250.
        np.testing.assert_allclose(diag['rnet']-diag['h']-diag['le']-diag['g'],
                                   diag['residual'],atol=1e-12)


def test_uniform_snow_has_exact_zero_flux_reference_temperature():
    temperature = 268.0032043457031
    state = ColumnState(swe=.272,
        snow_swe_layers=np.full(3,.272/3),
        snow_t_layers=np.full(3,temperature),
        snow_density_layers=np.array([258.5,258.3,258.1]))
    reference,_,_ = snow.snow_surface_response(state,1800.)
    assert reference == temperature


def test_melted_out_step_keeps_its_actual_heat_flux():
    state = ColumnState(swe=.00011,snow_t=273.15,
        snow_swe_layers=np.array([.00011]),snow_t_layers=np.array([273.15]),
        snow_density_layers=np.array([300.]))
    final,diag = step_column(state,Forcing(1800.,295.,.004,2.,1.e5,700.,350.,0.))
    assert final.swe <= 1.e-4
    assert diag['g'] > 0.
    np.testing.assert_allclose(diag['rnet']-diag['h']-diag['le']-diag['g'],
                               diag['residual'],atol=1e-12)
