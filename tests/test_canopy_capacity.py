"""Independent time-overlap and continuous canopy-N reference calculations."""
import pickle
import math
import numpy as np
import pytest
from noahmp_clm.physics.canopy_capacity import (
    TemperatureMemory, CanopyCapacityParameters, canopy_capacity_profile)
from noahmp_clm.physics.photosynthesis import LeafParameters, leaf_gas_exchange

PFT1 = CanopyCapacityParameters(.01, 58., .0509, 1.)


@pytest.mark.parametrize('dt', [1800., 3600., 86400.])
def test_ten_day_memory_matches_independent_interval_overlap_and_restart(dt):
    state = TemperatureMemory.cold(dt)
    assert state.mean_temperature_k == 293.15
    intervals = []; maximum = 0.
    count = round(14. * 86400. / dt)
    for step in range(count):
        value = 278. + 9.*math.sin(step*.19) + (12. if step*dt >= 8.*86400. else 0.)
        before = state
        before_values = before.values.copy()
        state = state.append(value, dt)
        np.testing.assert_array_equal(before.values, before_values)
        intervals.append((step*dt, (step+1)*dt, value))
        end = (step+1)*dt; begin = max(0., end - 864000.)
        integral = math.fsum(max(0., min(b, end)-max(a, begin))*v for a,b,v in intervals)
        reference = integral/(end-begin)
        maximum = max(maximum, abs(state.mean_temperature_k-reference))
        assert abs(state.mean_temperature_k-reference) < 1e-10
        if step == count//2:
            restored = pickle.loads(pickle.dumps(state))
            np.testing.assert_array_equal(restored.values, state.values)
            assert (restored.count, restored.cursor, restored.time_step_seconds) == (state.count, state.cursor, dt)
            assert not restored.values.flags.writeable
            state = restored
    assert state.count == len(state.values) and not state.values.flags.writeable
    chronological = np.r_[state.values[state.cursor:], state.values[:state.cursor]]
    np.testing.assert_array_equal(chronological, [entry[2] for entry in intervals[-len(state.values):]])
    print('ten-day independent overlap', dict(dt=dt, steps=count, maximum_temperature_error_K=maximum), flush=True)


@pytest.mark.parametrize('day,expected', [(0., .01), (100., .01), (30000., 4./9.), (45000., 1.)])
def test_photoperiod_and_original_pft_capacity(day, expected):
    got = canopy_capacity_profile(np.full(10, .3), day, 45000., 274., PFT1)
    assert abs(got['daylength_factor']-expected) < 1e-15
    assert abs(got['unscaled_vcmax25_top'] - 37.70110344827586) < 1e-12
    assert abs(got['nitrogen_extinction'] - .1265727074) < 1e-9
    assert got['growth_temperature_k'] == 284.15
    rows = got['leaf_parameters']
    np.testing.assert_allclose(rows[:, 1]/rows[:, 0], 2.205, rtol=0., atol=1e-14)
    np.testing.assert_allclose(rows[:, 2]/rows[:, 0], .167, rtol=0., atol=1e-15)
    np.testing.assert_allclose(rows[:, 3]/rows[:, 0], .015, rtol=0., atol=1e-15)


@pytest.mark.parametrize('lai', [.01, 1., 3., 8.])
@pytest.mark.parametrize('temperature', [270., 298.15, 320.])
def test_layer_capacity_matches_independent_continuous_nitrogen_integral(lai, temperature):
    from scipy.integrate import quad
    leaf = np.full(10, lai/10.)
    got = canopy_capacity_profile(leaf, 30000., 45000., temperature, PFT1)
    # Independent quadrature of the continuous trait/N relation, rather than
    # reusing midpoint factors or the production profile helper.
    top_n = 100./58.
    vmax = top_n * .0509 * 429.6
    attenuation = math.exp(vmax*.00963-2.43)
    scalar = (2./3.)**2
    mean_t = min(35., max(11., temperature-273.15))
    exact = quad(lambda depth: vmax*scalar*math.exp(-attenuation*depth), 0., lai,
                 epsabs=1e-12, epsrel=1e-12)[0]
    actual = float(got['leaf_parameters'][:, 0] @ leaf)
    deviation = abs(actual-exact)/exact
    assert deviation <= .1
    # Also check a complete biochemical capacity integral with independent
    # adaptive quadrature through the unchanged leaf gas-exchange component.
    def assimilation(depth):
        v = vmax*scalar*math.exp(-attenuation*depth)
        p = LeafParameters(v, v*(2.59-.035*mean_t), .167*v, .015*v,
                           growth_temperature=mean_t+273.15)
        return float(leaf_gas_exchange(1200., 298.15, parameters=p)['gross'])
    expected = quad(assimilation, 0., lai, epsabs=1e-7, epsrel=1e-7)[0]
    layer = [float(leaf_gas_exchange(1200., 298.15, parameters=LeafParameters(*row))['gross'])
             for row in got['leaf_parameters']]
    uptake = float(np.asarray(layer) @ leaf)
    uptake_error = abs(uptake-expected)/expected
    assert uptake_error <= .1
    print('continuous capacity reference', dict(lai=lai, temperature_k=temperature,
          integrated_vcmax_relative_error=deviation, integrated_gross_relative_error=uptake_error), flush=True)


def test_leaf_free_layers_have_no_capacity_and_do_not_change_remaining_profile():
    baseline = canopy_capacity_profile([.5, .7], 40000., 45000., 295., PFT1)
    got = canopy_capacity_profile([0., .5, 0., .7, 0.], 40000., 45000., 295., PFT1)
    np.testing.assert_array_equal(got['leaf_parameters'][[0,2,4], :4], np.zeros((3,4)))
    np.testing.assert_array_equal(got['leaf_parameters'][[1,3]], baseline['leaf_parameters'])
    np.testing.assert_array_equal(got['leaf_nitrogen_gn_m2_leaf'][[0,2,4]], np.zeros(3))


@pytest.mark.parametrize('operation', ['interval', 'temperature', 'cursor', 'shape', 'daylength', 'traits'])
def test_invalid_acclimation_state_and_inputs_are_rejected(operation):
    with pytest.raises(ValueError):
        if operation == 'interval': TemperatureMemory.cold(1800.).append(280., 3600.)
        elif operation == 'temperature': TemperatureMemory.cold(1800.).append(np.nan, 1800.)
        elif operation == 'cursor': TemperatureMemory(np.full(480, 280.), 1800., 3, 5)
        elif operation == 'shape': TemperatureMemory(np.full(479, 280.), 1800.)
        elif operation == 'daylength': canopy_capacity_profile([1.], 50000., 45000., 280., PFT1)
        else: CanopyCapacityParameters(.01, 58., 1.1)
