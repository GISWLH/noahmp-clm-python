"""Energy-network tests; SciPy is an independent nonlinear solver, not CTSM data."""
from dataclasses import replace
import numpy as np
import pytest

from noahmp_clm.physics.canopy_energy import (
    CanopyExchange, solve_canopy_energy, _longwave_operator)
from noahmp_clm.physics.canopy_radiation import canopy_shortwave, canopy_longwave
from noahmp_clm.physics.photosynthesis import leaf_gas_exchange
from noahmp_clm.physics.soil_uptake import SoilWaterSupply


def boundary(**kwargs):
    return replace(CanopyExchange(298.,.01,101325.,.02,.003,.03,295.,15.),**kwargs)


def independent_energy_reference(area, swdown, lwdown, e, **kwargs):
    """Simultaneous temperatures + air humidity, finite-difference MINPACK.

    Reuse only already-tested radiation and leaf biochemistry components.
    Independently assemble the network and water limits; do not use any of
    the production energy/humidity/linear-operator helpers or their iterates.
    """
    scipy_optimize = pytest.importorskip("scipy.optimize")
    area = np.asarray(area,float)
    n = len(area)
    fv = kwargs.get("vegetated_fraction",1.) if area.sum() > 0 else 0.
    fw = kwargs.get("wet_fraction",0.)
    emiss = kwargs.get("ground_emissivity",.97)
    albedo = kwargs.get("ground_albedo",(.15,.3))
    radiation = canopy_shortwave(area,swdown,ground_albedo=albedo,
        diffuse_fraction=kwargs.get("diffuse_fraction",.5),cosine_zenith=kwargs.get("cosine_zenith",.5))
    coupled=None
    if kwargs.get('snowpack') is not None:
        from noahmp_clm.physics.canopy_snow_radiation import canopy_snow_shortwave
        coupled=canopy_snow_shortwave(area,kwargs['snowpack'],swdown,vegetated_fraction=fv,
            ground_albedo=albedo,diffuse_fraction=kwargs.get('diffuse_fraction',.5),
            cosine_zenith=kwargs.get('cosine_zenith',.5))
        radiation=coupled['canopy']
    rho = e.pressure/(287.*e.air_temperature*(1.+.61*e.specific_humidity))
    mol_density = e.pressure/(8.314462618*e.air_temperature)
    heat_weights = np.r_[area*e.leaf_velocity,e.ground_velocity]
    soil_supply = kwargs.get("soil_supply")
    biochemical_stress = kwargs.get("water_stress",1.)
    if kwargs.get("transpiration_limit",np.inf) == 0. or (soil_supply is not None and
            soil_supply.roots[soil_supply.weights>0].sum()==0.):
        biochemical_stress = 0.
    def cap(flux, name, extra_limit=np.inf):
        pos = np.maximum(flux,0.)
        neg = np.minimum(flux,0.)
        limit = min(kwargs.get(name,np.inf),extra_limit)
        ratio = min(1.,(limit-neg.sum())/pos.sum()) if pos.sum() else 1.
        return pos*ratio+neg
    def evaluate(x, details=False):
        t, qc = x[:-1], x[-1]*.001
        es = 611.2*np.exp(17.67*(t-273.15)/(t-29.65))
        if kwargs.get('ground_is_ice',False):
            es[-1]=np.exp(9.550426-5723.265/t[-1]+3.53068*np.log(t[-1])-.00728332*t[-1])
        qs = .622*es/(e.pressure-.378*es)
        leaf = leaf_gas_exchange(radiation["absorbed_par"],t[:-1],co2_ppm=e.co2_ppm,
            relative_humidity=qc*e.pressure/(.622+.378*qc)/es[:-1],
            boundary_conductance=e.leaf_velocity*mol_density,
            parameters=kwargs.get("leaf_parameters"),nitrogen_factor=kwargs.get("nitrogen_factor",1.)*biochemical_stress,
            water_stress=kwargs.get("water_stress",1.))
        tc = ((heat_weights@t+e.atmosphere_velocity*e.air_temperature)
              /(heat_weights.sum()+e.atmosphere_velocity)) if fv else e.air_temperature
        h = rho*1004.*fv*heat_weights*(t-tc)
        h[-1] += rho*1004.*(1.-fv)*e.atmosphere_velocity*(t[-1]-e.air_temperature)
        wet_area = np.where(qs[:-1] > qc,fw,1.)
        wet = cap(rho*fv*area*e.leaf_velocity*wet_area*(qs[:-1]-qc),"canopy_evaporation_limit")
        supply = kwargs.get("soil_supply")
        root_limit = np.inf if supply is None else supply.roots[supply.weights>0].sum()
        tr = cap(rho*fv*area*(1.-wet_area)*(qs[:-1]-qc)
                 /(1./e.leaf_velocity+mol_density/leaf["gs_mol"]),"transpiration_limit",root_limit)
        surface_limit = np.inf
        if supply is not None:
            # Independent scalar multiplier solve for weighted capped uptake.
            rooted = supply.weights > 0.
            left, right = 0.,np.max(supply.roots[rooted]/supply.weights[rooted],initial=0.)
            for _ in range(60):
                level = .5*(left+right)
                taken = np.minimum(supply.roots,level*supply.weights)
                if taken.sum() < tr.sum():
                    left = level
                else:
                    right = level
            uptake = np.minimum(supply.roots,.5*(left+right)*supply.weights)
            if kwargs.get('ground_shares_soil',True): surface_limit = max(0.,supply.surface-uptake[0])
        ground = cap(rho*np.array([fv*(qs[-1]-qc)/(1./e.ground_velocity+e.ground_resistance),
                     (1.-fv)*(qs[-1]-e.specific_humidity)/(1./e.atmosphere_velocity+e.ground_resistance)]),
                     "ground_evaporation_limit",surface_limit)
        lw = canopy_longwave(area,t[:-1],lwdown,t[-1],ground_emissivity=emiss)
        rn = np.r_[fv*(radiation["absorbed"]+lw["net_absorbed"]),
            fv*(radiation["ground_absorbed"]+lw["ground_net"])+(1.-fv)*
            (swdown*(1.-.5*sum(albedo))+emiss*(lwdown-5.670374419e-8*t[-1]**4))]
        le = 2.5e6*np.r_[wet+tr,ground.sum()]
        if kwargs.get('ground_is_ice',False): le[-1]+=334000.*ground.sum()
        if coupled is not None:
            rn[-1]=coupled['snow_absorbed'].sum()+coupled['soil_absorbed']+fv*lw['ground_net']+(1-fv)*emiss*(lwdown-5.670374419e-8*t[-1]**4)
        g = e.soil_conductance*(t[-1]-e.soil_temperature)+e.soil_flux_offset
        residual = rn-h-le
        if kwargs.get('fixed_ground_temperature') is not None: g=residual[-1]
        residual[-1] -= g
        if kwargs.get('fixed_ground_temperature') is not None: residual[-1]=kwargs['fixed_ground_temperature']-t[-1]
        residual[:-1] = np.where(fv*area>0,residual[:-1],e.air_temperature-t[:-1])
        qres = (wet.sum()+tr.sum()+ground[0]-rho*fv*e.atmosphere_velocity*(qc-e.specific_humidity))*2.5e6
        if fv == 0:
            qres = (qc-e.specific_humidity)*1e5
        if details:
            return {"temperature":t,"canopy_humidity":qc,"h":h.sum(),"le":le.sum(),
                    "ground_heat":g,"gpp":fv*area@leaf["gross"],"residual":np.r_[residual,qres]}
        return np.r_[residual,qres]
    initial = np.r_[np.full(n+1,e.air_temperature),e.specific_humidity*1000.]
    # Bounded trust-region solve avoids the invalid temperatures/humidity that
    # unconstrained trial steps can take even when the physical root is valid.
    fitted = scipy_optimize.least_squares(evaluate,initial,
        bounds=(np.r_[np.full(n+1,200.),0.],np.r_[np.full(n+1,350.),190.]),
        xtol=1e-12,ftol=1e-12,gtol=1e-11,x_scale="jac",max_nfev=100)
    answer = evaluate(fitted.x,True)
    assert fitted.success and np.max(np.abs(answer["residual"])) < 1e-6
    return answer


CASES = [
    ("day",np.full(10,.3),800.,350.,boundary(),{}),
    ("night",np.full(10,.3),0.,300.,boundary(),{}),
    ("dew",np.full(10,.3),0.,280.,boundary(specific_humidity=.019),{}),
    ("wet",np.full(10,.3),800.,350.,boundary(),{"wet_fraction":.6}),
    ("dry",np.full(10,.3),800.,350.,boundary(),{"transpiration_limit":0.,"ground_evaporation_limit":0.}),
    ("limited",np.full(10,.3),800.,350.,boundary(),{"wet_fraction":.7,
        "canopy_evaporation_limit":1e-7,"transpiration_limit":2e-6,"ground_evaporation_limit":3e-6}),
    ("mosaic",np.linspace(.05,.55,10),650.,365.,boundary(ground_resistance=300.),{"vegetated_fraction":.65}),
    ("no_veg",np.full(10,.3),800.,350.,boundary(),{"vegetated_fraction":0.}),
    ("zero_lai",np.zeros(10),800.,350.,boundary(),{}),
    ("empty_layers",np.array([0.,1.,0.,2.,0.]),500.,330.,boundary(),{}),
    ("cold",np.full(10,.3),200.,260.,boundary(air_temperature=272.,soil_temperature=274.,specific_humidity=.003),{}),
    ("dense",np.full(10,.8),1100.,380.,boundary(),{"vegetated_fraction":.9,"cosine_zenith":.2}),
    ("shared_water",np.full(10,.3),800.,350.,boundary(),{"vegetated_fraction":.7,
        "soil_supply":SoilWaterSupply(np.array([1e-5,2e-5,3e-5,4e-5]),np.array([.6,.3,.1,0.]),1.1e-5)}),
    ("frozen_supply",np.full(10,.3),500.,350.,boundary(),{
        "soil_supply":SoilWaterSupply(np.zeros(4),np.array([.6,.3,.1,0.]),0.)}),
]


@pytest.mark.parametrize("name,area,sw,lw,e,kw",CASES,ids=[c[0] for c in CASES])
def test_coupled_energy_against_independent_nonlinear_reference(name,area,sw,lw,e,kw):
    got = solve_canopy_energy(area,sw,lw,e,**kw)
    ref = independent_energy_reference(area,sw,lw,e,**kw)
    np.testing.assert_allclose(np.r_[got["leaf_temperature"],got["ground_temperature"]],
                               ref["temperature"],rtol=0,atol=5e-5)
    for field in ("h","le","ground_heat","gpp"):
        np.testing.assert_allclose(got[field],ref[field],rtol=1e-5,atol=2e-4)
    assert abs(got["canopy_humidity"]-ref["canopy_humidity"]) < 2e-9
    assert np.max(np.abs(got["layer_residual"])) < 5e-5
    assert abs(got["air_heat_residual"]) < 1e-8
    assert abs(got["air_water_residual"]) < 1e-12
    assert abs(sw+lw-got["outgoing_shortwave"]-got["outgoing_longwave"]
               -got["h"]-got["le"]-got["ground_heat"]) < 6e-4
    assert got["wet_evaporation"].sum() <= kw.get("canopy_evaporation_limit",np.inf)+1e-15
    assert got["transpiration"].sum() <= kw.get("transpiration_limit",np.inf)+1e-15
    assert got["ground_evaporation"] <= kw.get("ground_evaporation_limit",np.inf)+1e-15
    if "soil_supply" in kw:
        supply = kw["soil_supply"]
        assert np.all(got["root_uptake"] <= supply.roots+1e-15)
        np.testing.assert_allclose(got["root_uptake"].sum(),got["transpiration"].sum(),atol=1e-15)
        assert got["root_uptake"][0]+got["ground_evaporation"] <= supply.surface+1e-15
    if name == "dew":
        assert got["wet_evaporation"].sum() < 0 and got["le"] < 0
    if name in ("night","zero_lai","no_veg","dry","frozen_supply"):
        assert got["gpp"] == 0.


def test_thermal_equilibrium_and_longwave_linear_operator():
    rng = np.random.default_rng(781)
    for fv in (0.,.3,1.):
        area = rng.uniform(0.,1.,10)
        temp = rng.uniform(270.,310.,11)
        sky = 320.
        operator = _longwave_operator(area,.97,fv)
        net = operator@np.r_[5.670374419e-8*temp**4,sky]
        ref = canopy_longwave(area,temp[:-1],sky,temp[-1])
        np.testing.assert_allclose(net[:-1],fv*ref["net_absorbed"],atol=2e-13)
        np.testing.assert_allclose(net[-1],fv*ref["ground_net"]+(1.-fv)*.97*(sky-5.670374419e-8*temp[-1]**4),atol=2e-13)
    t = 298.
    es = 611.2*np.exp(17.67*(t-273.15)/(t-29.65))
    e = boundary(air_temperature=t,soil_temperature=t,specific_humidity=.622*es/(101325.-.378*es))
    got = solve_canopy_energy([.3]*10,0.,5.670374419e-8*t**4,e,wet_fraction=1.)
    np.testing.assert_allclose(got["leaf_temperature"],t,atol=1e-12)
    assert abs(got["h"])+abs(got["le"])+abs(got["ground_heat"]) < 1e-9


def test_failed_balance_is_explicit_and_input_is_not_mutated():
    initial = np.full(11,290.)
    before = initial.copy()
    with pytest.raises(RuntimeError,match="did not converge"):
        solve_canopy_energy([.3]*10,800.,350.,boundary(),initial_temperatures=initial,max_iterations=1)
    np.testing.assert_array_equal(initial,before)
    with pytest.raises(ValueError):
        solve_canopy_energy([.3]*10,800.,350.,boundary(),canopy_evaporation_limit=-1.)
    with pytest.raises(ValueError):
        boundary(leaf_velocity=0.)


@pytest.mark.parametrize('ground_temperature',[250.,265.,273.15])
@pytest.mark.parametrize('fraction',[0.,.7,1.])
def test_fixed_snow_surface_against_independent_leaf_air_solve(ground_temperature,fraction):
    from noahmp_clm.physics.snowpack import Snowpack
    pack=Snowpack.from_ice([4.,12.,30.],ground_temperature)
    e=boundary(air_temperature=ground_temperature+2.,specific_humidity=.001,soil_temperature=274.)
    kw=dict(vegetated_fraction=fraction,fixed_ground_temperature=ground_temperature,
        ground_is_ice=True,snowpack=pack,ground_shares_soil=False,ground_evaporation_limit=1e-6,
        soil_supply=SoilWaterSupply(np.array([0.,2e-6,2e-6,2e-6]),np.array([.6,.3,.1,0.]),0.))
    got=solve_canopy_energy([.3]*10,500.,220.,e,**kw)
    ref=independent_energy_reference([.3]*10,500.,220.,e,**kw)
    np.testing.assert_allclose(np.r_[got['leaf_temperature'],got['ground_temperature']],ref['temperature'],atol=5e-5,rtol=0.)
    for key in ('h','le','ground_heat','gpp'):
        np.testing.assert_allclose(got[key],ref[key],atol=2e-4,rtol=1e-5)
    assert got['ground_temperature']==ground_temperature
    assert abs(got['ground_heat']-got['ground_nonshortwave_heat']-got['snow_absorbed_shortwave'].sum()-got['soil_absorbed_shortwave'])<1e-10
    assert abs(720.-got['outgoing_shortwave']-got['outgoing_longwave']-got['h']-got['le']-got['ground_heat'])<6e-4
    assert got['ground_latent_heat']==2834000.
    assert got['ground_evaporation']<=1e-6+1e-15
    assert np.max(np.abs(got['layer_residual']))<5e-5


def test_ice_saturation_reference_and_derivative():
    from noahmp_clm.physics.canopy_energy import _ice_saturation
    t=np.linspace(208.15,273.15,80)
    q,deriv,es=_ice_saturation(t,90000.)
    # Independent Smithsonian/Goff--Gratch curve, NCAR EOL reference table.
    reference=100.*10.**(-9.09718*(273.16/t-1)-3.56654*np.log10(273.16/t)+.876793*(1-t/273.16)+np.log10(6.1071))
    assert np.max(np.abs(es/reference-1.))<.01
    finite=(_ice_saturation(t+1e-4,90000.)[0]-_ice_saturation(t-1e-4,90000.)[0])/2e-4
    np.testing.assert_allclose(deriv,finite,rtol=1e-7,atol=1e-12)


def test_gpu_packer_does_not_silently_drop_winter_boundary():
    from noahmp_clm.gpu.canopy_energy import pack_canopy_case
    with pytest.raises(NotImplementedError,match='Winter canopy'):
        pack_canopy_case([.3]*10,500.,220.,boundary(),fixed_ground_temperature=265.)
