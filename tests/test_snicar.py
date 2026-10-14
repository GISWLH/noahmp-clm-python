"""Snow optics: independent matrix exponential and analytic thick-pack limits."""
import numpy as np
import pytest

from noahmp_clm.physics.snicar import (snow_tables, bulk_optics, snow_shortwave,
    spectral_fluxes)
from noahmp_clm.physics.snowpack import Snowpack,TF


def independent_spectrum(tau,omega,asym,beam,diffuse,albedo,mu,eddington):
    """Solve all interface boundaries at once with SciPy exact transfer matrices.

    Only moderate optical depths are used, avoiding growing-mode conditioning.
    No production delta transform, coefficient/doubling or boundary solver.
    """
    expm=pytest.importorskip('scipy.linalg').expm
    n=len(tau);transfers=[];beam_interfaces=[beam]
    for depth,w,g in zip(tau,omega,asym):
        width=(1-w*g*g)*depth
        w=w*(1-g*g)/(1-w*g*g);g=g/(1+g)
        if eddington: a=(7-w*(4+3*g))/4;b=(w*(4-3*g)-1)/4;beta=(2-3*g*mu)/4
        else: a=2-w*(1+g);b=w*(1-g);beta=(1-g*mu)/2
        system=np.array([[-a,b,w*(1-beta)/mu],[-b,a,-w*beta/mu],[0,0,-1/mu]])
        transfers.append(expm(system*width))
        beam_interfaces.append(beam_interfaces[-1]*np.exp(-width/mu))
    matrix=np.zeros((2*(n+1),2*(n+1)));rhs=np.zeros(2*(n+1))
    matrix[0,0]=1.;rhs[0]=diffuse
    matrix[1,2*n+1]=1.;matrix[1,2*n]=-albedo;rhs[1]=albedo*beam_interfaces[-1]
    for j,transfer in enumerate(transfers):
        matrix[2+2*j:4+2*j,2*j:2*j+2]=-transfer[:2,:2]
        matrix[2+2*j:4+2*j,2*j+2:2*j+4]=np.eye(2)
        rhs[2+2*j:4+2*j]=transfer[:2,2]*beam_interfaces[j]
    flux=np.linalg.solve(matrix,rhs).reshape(n+1,2)
    net=flux[:,0]+beam_interfaces-flux[:,1]
    return dict(absorbed=net[:-1]-net[1:],ground_absorbed=net[-1],
        reflected=flux[0,1],diffuse_down=flux[:,0],diffuse_up=flux[:,1],direct=beam_interfaces)


def test_lookup_identity_and_official_endpoint_properties():
    table=snow_tables()
    assert str(table['optics_source_sha256'])=='a17f1b42616ce0b5c0534ccc0f92c33f0020db574bcc2888d326c7da6b69c305'
    p=Snowpack.from_ice([1.,1.],TF,300.,[30.,1500.])
    b=bulk_optics(p)
    # Rounded physical values in CLM5 technical-note Tables 2.3.5--7, not
    # generated from the production interpolation.
    np.testing.assert_allclose(b['tau'][0],[55.7,56.1,56.3,56.6,57.3],rtol=.006)
    np.testing.assert_allclose(b['tau'][1],[1.09,1.09,1.09,1.09,1.10],rtol=.006)
    np.testing.assert_allclose(b['omega'][1],[.9998,.9960,.9680,.8730,.5500],rtol=.006)


@pytest.mark.parametrize('eddington',[False,True])
def test_against_matrix_exponential(eddington):
    rng=np.random.default_rng(559)
    for _ in range(60):
        n=rng.integers(1,6);tau=rng.uniform(.001,3.,n)
        w=rng.uniform(.9 if eddington else .1,1.,n);g=rng.uniform(.1,.95,n)
        mu=rng.uniform(.05,1.);albedo=rng.uniform(0.,.95)
        beam,diffuse=rng.uniform(0,800,2)
        a=spectral_fluxes(tau,w,g,beam,diffuse,albedo,mu,eddington=eddington)
        b=independent_spectrum(tau,w,g,beam,diffuse,albedo,mu,eddington)
        for field in b:
            np.testing.assert_allclose(a[field],b[field],rtol=2e-9,atol=1e-7)
        assert abs(a['residual'])<1e-10


@pytest.mark.parametrize('eddington',[False,True])
def test_semi_infinite_analytic_limit(eddington):
    # Analytic decaying eigenmode gives both direct and diffuse albedo.
    for w,g,mu in ((.9999,.89,.2),(.999,.85,.7),(.98,.92,.3)):
        f=1-w*g*g;ws=w*(1-g*g)/f;gs=g/(1+g)
        if eddington: a=(7-ws*(4+3*gs))/4;b=(ws*(4-3*gs)-1)/4;beta=(2-3*gs*mu)/4
        else: a=2-ws*(1+gs);b=ws*(1-gs);beta=(1-gs*mu)/2
        lam=np.sqrt(a*a-b*b);rd=b/(a+lam)
        rb=ws*(beta+rd*(1-beta))/(1+lam*mu)
        run=spectral_fluxes([1e6],[w],[g],300.,500.,.2,mu,eddington=eddington)
        assert run['reflected']==pytest.approx(300*rb+500*rd,abs=3e-6)
        assert run['ground_absorbed']<1e-12


def test_pure_scatter_and_direct_resonance():
    # Hemispheric isotropic w=.75 gives eigenvalue=1; mu=1 is the pole of
    # common analytic beam expressions, but not of the physical solution.
    for w,mu in ((.75,1.),(1.,.1),(1.,1.)):
        a=spectral_fluxes([2.,5.],[w,w],[0.,0.],500.,300.,.4,mu)
        assert np.all(np.isfinite(a['absorbed']))
        assert abs(a['residual'])<1e-10
        if w==1.: assert np.max(np.abs(a['absorbed']))<1e-9


def test_clean_dirty_old_thin_bare_snow():
    fresh=Snowpack.from_ice([100.],TF-5,300.,54.5)
    dirty=Snowpack(fresh.mass,fresh.enthalpy,fresh.thickness,fresh.radius_um,
        [[100*1e-6,100*2e-6,100*2e-6,100*2e-6,100*2e-6]])
    aged=Snowpack.from_ice([100.],TF-5,300.,1000.)
    thin=Snowpack.from_ice([.1],TF-5,100.,54.5)
    results=[snow_shortwave(p,[300.,300.],[100.,100.]) for p in (fresh,dirty,aged,thin,Snowpack.empty())]
    assert results[0]['albedo']>results[1]['albedo']
    assert results[0]['albedo']>results[2]['albedo']
    assert results[3]['ground_absorbed']>results[0]['ground_absorbed']
    assert results[4]['albedo']==pytest.approx(.225)
    for r in results:
        assert np.min(r['absorbed'],initial=0)>-1e-7
        assert 0<=r['albedo']<=1
        assert abs(r['residual'])<1e-9


def test_zero_radiation_and_invalid_boundaries():
    p=Snowpack.empty()
    r=snow_shortwave(p,[0.,0.],[0.,0.])
    assert r['albedo'] is None and r['residual']==0.
    with pytest.raises(ValueError): snow_shortwave(p,[-1.,0.],[0.,0.])
