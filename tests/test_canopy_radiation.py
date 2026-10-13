import numpy as np
import pytest
from noahmp_clm.physics.canopy_radiation import (
    two_stream_shortwave, canopy_shortwave, canopy_longwave)


def continuous_reference(lai, refl, tran, mu, direct0=400., diffuse0=200., ag=.3):
    """Independent continuous ODE solution using full matrix exponentials.

    No production optical or solver helper is used. Boundary conditions are
    solved simultaneously by dense elimination, rather than layer adding.
    """
    scipy_linalg = pytest.importorskip("scipy.linalg")
    n = len(lai)
    omega = refl+tran
    backscatter = (5.*refl+3.*tran)/8.
    extinction = 1.-omega+backscatter
    k = .5/mu
    beta0 = (1.+k)/(2.*k)*(1.-mu*np.log((1.+mu)/mu))
    generator = np.array([[-extinction, backscatter, omega*k*(1.-beta0)],
        [-backscatter, extinction, -omega*k*beta0], [0.,0.,-k]])
    beam = direct0*np.exp(-k*np.r_[0.,np.cumsum(lai)])
    matrix = np.zeros((2*n+2,2*n+2))
    rhs = np.zeros(2*n+2)
    matrix[0,0], rhs[0] = 1., diffuse0
    matrix[1,n], matrix[1,2*n+1], rhs[1] = -ag, 1., ag*beam[-1]
    for j, width in enumerate(lai):
        transfer = scipy_linalg.expm(generator*width)
        for stream in range(2):
            eq = 2+2*j+stream
            matrix[eq,j+1+stream*(n+1)] = 1.
            matrix[eq,j] -= transfer[stream,0]
            matrix[eq,n+1+j] -= transfer[stream,1]
            rhs[eq] = transfer[stream,2]*beam[j]
    result = np.linalg.solve(matrix,rhs)
    down, up = result[:n+1], result[n+1:]
    absorb = down[:-1]+up[1:]+beam[:-1]-down[1:]-up[:-1]-beam[1:]
    return {"absorbed":absorb,"reflected":up[0],"ground_absorbed":(1.-ag)*(down[-1]+beam[-1])}


@pytest.mark.parametrize("lai", [0.,.1,3.,8.])
@pytest.mark.parametrize("optics", [(.1,.05),(.45,.25),(.6,.4)])
@pytest.mark.parametrize("mu", [.05,.5,1.])
def test_two_stream_against_continuous_clm_equations(lai,optics,mu):
    area = np.full(10,lai/10.)
    actual = two_stream_shortwave(area,400.,200.,leaf_reflectance=optics[0],
        leaf_transmittance=optics[1],ground_albedo=.3,cosine_zenith=mu)
    expected = continuous_reference(area,*optics,mu)
    assert abs(actual["residual"]) < 1e-10
    assert np.min(actual["absorbed"]) > -1e-9
    for key in expected:
        np.testing.assert_allclose(actual[key],expected[key],rtol=.01,atol=1e-8)


def test_black_canopy_beer_law_and_bare_ground():
    r = two_stream_shortwave([.3]*10,600.,0.,leaf_reflectance=0.,leaf_transmittance=0.,ground_albedo=0.)
    np.testing.assert_allclose(r["absorbed"],600.*np.exp(-np.arange(10)*.3)*(1.-np.exp(-.3)),atol=1e-9)
    bare = canopy_shortwave([0.]*10,600.)
    assert np.all(bare["absorbed_par"]==0.)
    np.testing.assert_allclose(bare["ground_absorbed"],600.*(1.-.5*(.15+.3)))
    assert abs(bare["residual"]) < 1e-10


def test_two_bands_partition_energy_and_par():
    r = canopy_shortwave([.3]*10,800.)
    assert np.all(r["absorbed_par"]>0)
    assert r["absorbed_par"][0] > r["absorbed_par"][-1]
    np.testing.assert_allclose(r["absorbed"].sum()+r["ground_absorbed"]+r["reflected"],800.,atol=1e-10)
    np.testing.assert_allclose(r["absorbed_par"]*.3/4.6,r["bands"][0]["absorbed"],atol=1e-12)


def test_longwave_isothermal_and_nonuniform_budgets():
    thermal = 5.670374419e-8*290.**4
    equilibrium = canopy_longwave([.3]*10,[290.]*10,thermal,290.)
    np.testing.assert_allclose(equilibrium["net_absorbed"],0.,atol=3e-13)
    np.testing.assert_allclose(equilibrium["ground_net"],0.,atol=3e-13)
    other = canopy_longwave([.3]*10,np.linspace(280.,298.,10),300.,301.)
    assert abs(other["residual"]) < 1e-10
    assert np.any(other["net_absorbed"]<0.)


def test_unphysical_optics_rejected():
    with pytest.raises(ValueError):
        two_stream_shortwave([1.],300.,100.,leaf_reflectance=.8,leaf_transmittance=.8)
    with pytest.raises(ValueError):
        two_stream_shortwave([1.],300.,100.,cosine_zenith=0.)
