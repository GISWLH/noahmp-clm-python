"""Independent scalar roots across biochemical branches and diffusion regimes."""
from itertools import product
import numpy as np
import pytest
from noahmp_clm.physics.photosynthesis import (
    LeafParameters, farquhar_rates, ball_berry_conductance, leaf_gas_exchange)


def test_leaf_root_against_independent_brent_solver():
    brentq = pytest.importorskip('scipy.optimize').brentq
    rows = np.array(list(product([0., .01, 50., 2000.], [250., 298.15, 320.],
                                 [.02, .8, 1.2], [.002, 2.], [0., .01, .3, 1.])))
    def reference_rates(ci,par,temperature,stress,parameters):
        # Independently compose the already checked unstressed biochemistry.
        # Soil beta scales Rubisco capacity and Rd, not electron transport/TPU.
        rates=farquhar_rates(ci,par,temperature,parameters)
        rates['rubisco']=rates['rubisco']*stress
        rates['respiration']=rates['respiration']*stress
        rates['gross']=np.minimum(np.minimum(rates['rubisco'],rates['rubp']),rates['tpu'])
        rates['net']=rates['gross']-rates['respiration']
        return rates
    for parameters in [LeafParameters(), LeafParameters(tpu25=.5, minimum_conductance=.2),
                       LeafParameters(vcmax25=0., ball_berry_slope=0.)]:
        par, temperature, rh, gb, stress = rows.T
        ca = np.linspace(150., 1000., len(rows))
        actual = leaf_gas_exchange(par, temperature, co2_ppm=ca,
            relative_humidity=rh, boundary_conductance=gb, water_stress=stress,
            parameters=parameters)
        expected_ci = []
        for i in range(len(rows)):
            def residual(ci):
                net = float(reference_rates(ci,par[i],temperature[i],stress[i],parameters)['net'])
                cs = max(ca[i]-1.4*net/gb[i], 1e-9)
                gs = float(ball_berry_conductance(net, cs, rh[i], gb[i], parameters,
                                                water_stress=stress[i]))
                return net-(ca[i]-ci)/(1.4/gb[i]+1.6/gs)
            rd=float(farquhar_rates(0.,par[i],temperature[i],parameters)['respiration'])*stress[i]
            minimum=max(parameters.minimum_conductance*stress[i],1e-6)
            upper = ca[i]+rd*(1.4/gb[i]+1.6/minimum)+1.
            expected_ci.append(brentq(residual, 0., upper, xtol=1e-11, rtol=1e-14))
        np.testing.assert_allclose(actual['ci_ppm'], expected_ci, rtol=1e-12, atol=1e-10)
        expected = reference_rates(expected_ci,par,temperature,stress,parameters)
        for key in expected:
            np.testing.assert_allclose(actual[key], expected[key], rtol=1e-11, atol=1e-10)
        assert np.max(abs(actual['diffusion_residual'])) < 1e-10
        print('independent stressed Brent root',len(rows),'max_ci_ppm',
              float(np.max(abs(actual['ci_ppm']-expected_ci))),'max_diffusion',
              float(np.max(abs(actual['diffusion_residual']))),flush=True)
