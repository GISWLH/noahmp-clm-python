"""Independent continuous flux BVP and exact tissue absorption limits."""
import numpy as np
import pytest
from noahmp_clm.physics.canopy_optics import CanopyOptics,canopy_shortwave_pft,mixed_optical_properties
from noahmp_clm.physics.canopy_radiation import canopy_shortwave


PFT1=CanopyOptics((.07,.35),(.05,.10),(.16,.39),(.001,.001))


def continuous_mixture_reference(leaf,stem,optics,sw,diff,mu,ag,ag_direct=None):
    # Dense simultaneous boundary solve and matrix exponentials, independent
    # of production mixing, thin slabs, doubling and boundary elimination.
    from scipy.linalg import expm
    n=len(leaf);area=leaf+stem;k=.5/mu;beam0=.5*sw*(1.-diff);sky0=.5*sw*diff
    values=[]
    for band in range(2):
        rho=[];tau=[];fractions=[]
        for l,s in zip(leaf,stem):
            width=l+s
            r=(l*optics.leaf_reflectance[band]+s*optics.stem_reflectance[band])/width if width else 0.
            t=(l*optics.leaf_transmittance[band]+s*optics.stem_transmittance[band])/width if width else 0.
            al=l*(1.-optics.leaf_reflectance[band]-optics.leaf_transmittance[band])
            ast=s*(1.-optics.stem_reflectance[band]-optics.stem_transmittance[band])
            rho.append(r);tau.append(t);fractions.append(al/(al+ast) if al+ast>0. else 0.)
        beam=beam0*np.exp(-k*np.r_[0.,np.cumsum(area)])
        matrix=np.zeros((2*n+2,2*n+2));rhs=np.zeros(2*n+2)
        matrix[0,0]=1.;rhs[0]=sky0
        ad=ag[band] if ag_direct is None else ag_direct[band]
        matrix[1,n]=-ag[band];matrix[1,2*n+1]=1.;rhs[1]=ad*beam[-1]
        beta=(1.+k)/(2.*k)*(1.-mu*np.log((1.+mu)/mu))
        for i,width in enumerate(area):
            omega=rho[i]+tau[i];back=(5.*rho[i]+3.*tau[i])/8.;a=1.-omega+back
            generator=np.array([[-a,back,omega*k*(1.-beta)],[-back,a,-omega*k*beta],[0.,0.,-k]])
            transfer=expm(generator*width)
            for stream in range(2):
                row=2+2*i+stream;matrix[row,i+1+stream*(n+1)]=1.
                matrix[row,i]-=transfer[stream,0];matrix[row,n+1+i]-=transfer[stream,1];rhs[row]=transfer[stream,2]*beam[i]
        solved=np.linalg.solve(matrix,rhs);down,up=solved[:n+1],solved[n+1:]
        absorbed=down[:-1]+up[1:]+beam[:-1]-down[1:]-up[:-1]-beam[1:]
        leaf_abs=absorbed*np.asarray(fractions)
        values.append(dict(absorbed=absorbed,leaf_absorbed=leaf_abs,stem_absorbed=absorbed-leaf_abs,
                           reflected=up[0],ground_absorbed=(1.-ag[band])*down[-1]+(1.-ad)*beam[-1]))
    return values


@pytest.mark.parametrize('mu',[.1,.5,1.])
@pytest.mark.parametrize('geometry',['forest','stem-only','leaf-only','varying'])
def test_pft_mixture_against_independent_continuous_reference(mu,geometry):
    leaf=np.full(10,.3);stem=np.full(10,.055)
    if geometry=='stem-only':leaf[:]=0.
    if geometry=='leaf-only':stem[:]=0.
    if geometry=='varying':leaf=np.linspace(0.,.7,10);stem=np.linspace(.12,0.,10)
    got=canopy_shortwave_pft(leaf,stem,700.,PFT1,diffuse_fraction=.6,cosine_zenith=mu)
    ref=continuous_mixture_reference(leaf,stem,PFT1,700.,.6,mu,(.15,.3))
    maximum={}
    for name in ('absorbed','leaf_absorbed','stem_absorbed'):
        expected=np.asarray([one[name] for one in ref])
        actual=np.asarray([b['absorbed'] for b in got['bands']]) if name=='absorbed' else got[name+'_bands']
        np.testing.assert_allclose(actual,expected,rtol=.1,atol=1e-8)
        maximum[name]=float(abs(actual-expected).max())
    for name in ('reflected','ground_absorbed'):
        expected=sum(one[name] for one in ref);np.testing.assert_allclose(got[name],expected,rtol=.1,atol=1e-8);maximum[name]=float(abs(got[name]-expected))
    np.testing.assert_allclose(got['leaf_absorbed']+got['stem_absorbed'],got['absorbed'],rtol=0.,atol=1e-12)
    np.testing.assert_allclose(got['absorbed_par']*leaf/4.6,got['leaf_absorbed_bands'][0],rtol=0.,atol=1e-12)
    assert abs(got['residual'])<1e-10
    print('PFT optics continuous reference',geometry,mu,maximum,flush=True)


def test_equal_tissue_optics_reproduce_generic_radiation_and_leaf_par():
    leaf=np.linspace(0.,.6,10);stem=np.linspace(.1,.02,10)
    got=canopy_shortwave_pft(leaf,stem,600.,CanopyOptics())
    ref=canopy_shortwave(leaf+stem,600.)
    for name in ('absorbed','absorbed_par','ground_absorbed','reflected'):
        # Leaf-free layers have no photosynthetic PAR, even when stems absorb.
        expected=np.where(leaf>0.,ref[name],0.) if name=='absorbed_par' else ref[name]
        np.testing.assert_allclose(got[name],expected,rtol=0.,atol=2e-10)


def test_nonabsorbing_leaves_never_supply_photosynthetic_par():
    p=CanopyOptics((.6,.8),(.4,.2),(.1,.2),(.1,.1))
    got=canopy_shortwave_pft(np.full(10,.3),np.full(10,.1),600.,p)
    np.testing.assert_array_equal(got['absorbed_par'],np.zeros(10))
    np.testing.assert_array_equal(got['leaf_absorbed'],np.zeros(10))
    assert got['stem_absorbed'].sum()>0. and abs(got['residual'])<1e-10


@pytest.mark.parametrize('geometry',['leaf','stem','mixed'])
def test_fully_scattering_tissues_remain_passive_after_area_mixing(geometry):
    # Includes the original rejected valid leaf-only case, plus independently
    # sampled stem-only and mixed layers over six decades of area.
    rng=np.random.default_rng(391821)
    optics=CanopyOptics((.8,.8),(.2,.2),(.4,.4),(.6,.6))
    for _ in range(32):
        leaf=10.**rng.uniform(-5.,1.,10)
        stem=10.**rng.uniform(-5.,1.,10)
        if geometry=='leaf':stem[:]=0.
        if geometry=='stem':leaf[:]=0.
        _,rho,tau,fraction=mixed_optical_properties(leaf,stem,optics)
        assert np.all(rho+tau<=1.) and np.all(rho>=0.) and np.all(tau>=0.)
        np.testing.assert_array_equal(fraction,np.zeros_like(fraction))
        got=canopy_shortwave_pft(leaf,stem,700.,optics,cosine_zenith=.1)
        assert np.max(abs(got['absorbed']))<1e-9 and abs(got['residual'])<2e-10
        np.testing.assert_array_equal(got['absorbed_par'],np.zeros(10))


def test_bare_ground_is_finite_and_conservative():
    got=canopy_shortwave_pft(np.zeros(10),0.,600.,PFT1)
    np.testing.assert_array_equal(got['absorbed_par'],np.zeros(10))
    assert np.all(np.isfinite(got['stem_absorbed'])) and abs(got['residual'])<1e-10


@pytest.mark.parametrize('kwargs',[dict(leaf_reflectance=(np.nan,.2)),dict(stem_transmittance=(-.1,.2)),
    dict(leaf_reflectance=(.8,.1),leaf_transmittance=(.3,.2)),dict(stem_reflectance=(.2,.3,.4))])
def test_invalid_tissue_optics_fail_before_radiation(kwargs):
    with pytest.raises(ValueError):CanopyOptics(**kwargs)


def test_invalid_tissue_geometry_is_rejected():
    with pytest.raises(ValueError):mixed_optical_properties([.3,-.1],0.,PFT1)
    with pytest.raises(ValueError):mixed_optical_properties([.3,.1],np.nan,PFT1)


@pytest.mark.parametrize('layers',[0,1,5])
def test_pft_optics_snow_multiple_reflections_preserve_joint_energy(layers):
    from noahmp_clm.physics.snowpack import Snowpack
    from noahmp_clm.physics.canopy_snow_radiation import canopy_snow_shortwave
    leaf=np.full(10,.3);stem=np.full(10,.055)
    snow=Snowpack.from_ice(np.full(layers,10.),265.) if layers else Snowpack.empty()
    got=canopy_snow_shortwave(leaf+stem,snow,700.,vegetated_fraction=.65,
        canopy_optics=PFT1,photosynthetic_leaf_area=leaf,diffuse_fraction=.6,cosine_zenith=.4)
    assert abs(got['residual'])<2e-10
    canopy=got['canopy'];np.testing.assert_allclose(canopy['leaf_absorbed']+canopy['stem_absorbed'],
        canopy['absorbed'],rtol=0.,atol=1e-12)
    assert np.all(canopy['absorbed_par']>=0.)
    ref=continuous_mixture_reference(leaf,stem,PFT1,700.,.6,.4,got['snow_diffuse_albedo'],got['snow_direct_albedo'])
    np.testing.assert_allclose(canopy['leaf_absorbed_bands'],[v['leaf_absorbed'] for v in ref],rtol=.1,atol=1e-8)
    for name in ('reflected','ground_absorbed'):
        np.testing.assert_allclose(canopy[name],sum(v[name] for v in ref),rtol=.1,atol=1e-8)


def test_optics_reach_phase_energy_without_stomata_on_stems():
    from noahmp_clm.physics.canopy_energy import CanopyExchange
    from noahmp_clm.physics.canopy_interception import CanopyCondensate
    from noahmp_clm.physics.canopy_phase_energy import solve_canopy_phase_energy
    from noahmp_clm.physics.soil_uptake import SoilWaterSupply
    exchange=CanopyExchange(285.,.006,101325.,.02,.003,.03,283.,0.)
    kw=dict(stem_area=np.full(10,.1),soil_supply=SoilWaterSupply(np.zeros(4),np.zeros(4),0.),
        root_enthalpy=np.zeros(4),fixed_ground_temperature=283.,ground_evaporation_limit=0.,tolerance=1e-7)
    a=solve_canopy_phase_energy(np.zeros(10),300.,320.,exchange,CanopyCondensate.empty(),1800.,**kw)
    b=solve_canopy_phase_energy(np.zeros(10),300.,320.,exchange,CanopyCondensate.empty(),1800.,canopy_optics=PFT1,**kw)
    assert a['h']!=b['h'] and b['gpp']==0. and np.all(b['transpiration']==0.)
    assert np.max(abs(b['layer_residual']))<=1.01e-7 and abs(b['canopy_water_residual'])<1e-12


@pytest.mark.parametrize('use_most',[False,True])
def test_optics_reach_full_structured_bgc_column_with_original_hard_budgets(use_most):
    from test_clm_bgc_column import case
    from noahmp_clm.clm_enthalpy_driver import step_clm_enthalpy_column
    from noahmp_clm.physics.canopy_most import MOSTGeometry
    from noahmp_clm.physics.vegetation_structure import CanopyStructureParameters
    from run_fluxnet_column import audit_step
    old,f,env=case(None,285.)
    old=old.with_vegetation_structure(CanopyStructureParameters(canopy_top_m=8.,canopy_bottom_m=1.),sai=.55)
    controls=dict(phenology_environment=env,nitrogen_deposition_gn_m2=1e-5,
        most_geometry=MOSTGeometry() if use_most else None)
    _,generic=step_clm_enthalpy_column(old,f,**controls)
    new,d=step_clm_enthalpy_column(old,f,canopy_optics=PFT1,**controls)
    _,residual,bound=audit_step(old,new,f,d,1e-5)
    assert d['outgoing_shortwave']!=generic['outgoing_shortwave'] and d['gpp_gc_m2']!=generic['gpp_gc_m2']
    assert new.canopy_structure_parameters==old.canopy_structure_parameters
    print('PFT optics whole BGC',use_most,dict(h=d['h'],le=d['le'],gpp=d['gpp_gc_m2'],nee=d['nee_gc_m2'],
        original_MHCN_residual=residual.tolist(),original_MHCN_bound=bound.tolist()),flush=True)
