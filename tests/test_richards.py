"""Independent transient ODE and hydrostatic tests of vertical water physics."""
from dataclasses import replace
import json
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from noahmp_clm.physics.richards import HydraulicProfile,redistribute_liquid


def profile(n=4):
    return HydraulicProfile(np.asarray([.1,.3,.6,1.]) if n==4 else np.geomspace(.03,1.,n),.45,1e-5,.355,5.5)


def independent_ode(initial,p,duration):
    """Integrate a separate theta-form finite-volume RHS and flux quadrature.

    The oracle never calls the production retention, flux, Newton or adaptive
    routines. These transient cases remain below saturated pore volume.
    """
    dz=p.dz_m;z=np.cumsum(dz)-dz/2.;n=len(dz)
    def rhs(_,value):
        theta=value[:n]
        # Bound coefficients only. The integrated storage has no water floor.
        wet=np.maximum(theta,0.)
        head=-np.minimum(p.saturated_suction_m*(p.porosity/np.maximum(wet,.01*p.conductivity_porosity))**p.retention_exponent,1e5)
        q=[]
        for j in range(n-1):
            relative=(wet[j]+wet[j+1])/(p.conductivity_porosity[j]+p.conductivity_porosity[j+1])
            conductivity=np.sqrt(p.saturated_conductivity_m_s[j]*p.saturated_conductivity_m_s[j+1])*relative**(p.retention_exponent[j]+p.retention_exponent[j+1]+3.)
            q.append(conductivity*(1.-(head[j+1]-head[j])/(z[j+1]-z[j])))
        divergence=np.diff(np.r_[0.,q,0.])
        return np.r_[-divergence/dz,1000.*np.asarray(q)]
    result=solve_ivp(rhs,(0.,duration),np.r_[initial/(1000.*dz),np.zeros(n-1)],method='Radau',rtol=2e-10,atol=1e-12)
    assert result.success and np.all(result.y[:n]>=-1e-13) and np.all(result.y[:n]<p.porosity[:,None])
    return result.y[:n,-1]*1000.*dz,result.y[n:,-1]


@pytest.mark.parametrize('theta,duration,n',[
    ([.35,.24,.28,.25],1800.,4),([.1,.15,.2,.25],21600.,4),([.4,.39,.25,.2],7200.,4),
    ([.28]*4,86400.,4),([.32,.25,.3,.2],86400.,4),([.28]*10,21600.,10)])
def test_transient_closed_flow_against_independent_radau(theta,duration,n):
    p=profile(n);old=np.asarray(theta)*1000.*p.dz_m
    got=redistribute_liquid(old,p,duration);expected,flux=independent_ode(old,p,duration)
    soil_error=float(np.max(abs(got['water_kg_m2']-expected)/(1000.*p.dz_m)))
    assert soil_error<=.002
    difference=abs(got['interface_transfer_kg_m2']-flux)
    assert np.all(difference<=np.where(abs(flux)<=.02,.002,.1*abs(flux)))
    assert abs(got['mass_residual_kg_m2'])<=got['mass_bound_kg_m2']
    assert np.all(got['water_kg_m2']>0.) and np.all(got['water_kg_m2']<=1000.*p.dz_m*p.porosity+1e-9)
    print('Richards independent Radau',json.dumps(dict(layers=n,seconds=duration,max_soil_water_error_m3m3=soil_error,
        max_face_transfer_error_kg_m2=float(difference.max()),mass_residual_kg_m2=got['mass_residual_kg_m2'],
        mass_bound_kg_m2=got['mass_bound_kg_m2'],substeps=got['accepted_substeps'],rejections=got['rejected_substeps'])),flush=True)


@pytest.mark.parametrize('n',[4,10])
def test_analytic_hydrostatic_profile_has_no_gravity_or_capillary_drift(n):
    p=profile(n);z=p.midpoint_m;water_table=float(z[-1]+2.)
    theta=p.porosity*(p.saturated_suction_m/(water_table-z))**(1./p.retention_exponent)
    initial=theta*1000.*p.dz_m;got=redistribute_liquid(initial,p,86400.)
    assert np.max(abs(got['water_kg_m2']-initial))<1e-9
    assert np.max(abs(got['interface_transfer_kg_m2']))<1e-9
    assert abs(got['mass_residual_kg_m2'])<=got['mass_bound_kg_m2']


def test_closed_saturated_column_selects_hydrostatic_pressure_without_water_creation():
    p=profile();initial=1000.*p.dz_m*p.porosity;got=redistribute_liquid(initial,p,86400.)
    np.testing.assert_array_equal(got['water_kg_m2'],initial)
    np.testing.assert_array_equal(got['interface_transfer_kg_m2'],0.)
    np.testing.assert_allclose(np.diff(got['pressure_head_m']),np.diff(p.midpoint_m),rtol=0.,atol=1e-14)


def test_wet_column_approaches_independent_partly_saturated_hydrostatic_equilibrium():
    p=profile();initial=np.full(4,.43)*1000.*p.dz_m;z=p.midpoint_m
    def equilibrium(table):
        return p.porosity*(p.saturated_suction_m/np.maximum(table-z,p.saturated_suction_m))**(1./p.retention_exponent)
    table=brentq(lambda v:float((equilibrium(v)*1000.*p.dz_m).sum()-initial.sum()),-2.,10.,xtol=1e-13)
    got=redistribute_liquid(initial,p,10.*86400.)
    assert np.max(abs(got['water_kg_m2']/(1000.*p.dz_m)-equilibrium(table)))<.002
    assert abs(got['mass_residual_kg_m2'])<=got['mass_bound_kg_m2']
    print('Richards partly saturated equilibrium',json.dumps(dict(table_m=table,
        soil_water_max_error_m3m3=float(np.max(abs(got['water_kg_m2']/(1000.*p.dz_m)-equilibrium(table)))),
        mass_residual_kg_m2=got['mass_residual_kg_m2'],substeps=got['accepted_substeps'])),flush=True)


def test_impermeable_layer_blocks_both_faces_without_disabling_other_layers():
    p=replace(profile(),saturated_conductivity_m_s=[1e-5,0.,1e-5,1e-5])
    initial=np.array([.32,.25,.35,.2])*1000.*p.dz_m;got=redistribute_liquid(initial,p,1800.)
    np.testing.assert_array_equal(got['water_kg_m2'][:2],initial[:2])
    np.testing.assert_array_equal(got['interface_transfer_kg_m2'][:2],0.)
    assert got['interface_transfer_kg_m2'][2]>0.
    assert abs(got['mass_residual_kg_m2'])<=got['mass_bound_kg_m2']


@pytest.mark.parametrize('bad',['negative_water','overfull','negative_conductivity','dimension'])
def test_reference_domain_cannot_silently_discard_water_or_reshape_layers(bad):
    p=profile();water=np.full(4,.3)*1000.*p.dz_m
    with pytest.raises(ValueError):
        if bad=='negative_water':water[0]=-1e-8
        if bad=='overfull':water[0]=1000.*p.porosity[0]*p.dz_m[0]+1.
        if bad=='negative_conductivity':p=replace(p,saturated_conductivity_m_s=-1.)
        if bad=='dimension':water=water[:3]
        redistribute_liquid(water,p,1800.)


def initial_soil(theta,temperature,dz=None,option=1):
    from noahmp_clm.column import SoilParams
    from noahmp_clm.physics.soil_enthalpy import SoilEnthalpyState,enthalpy_and_capacity
    dz=profile().dz_m if dz is None else dz;p=SoilParams();water=1000.*dz*np.asarray(theta)
    heat=enthalpy_and_capacity(water,np.asarray(temperature),dz,p,option)[0]
    return SoilEnthalpyState(water,heat,dz,p,option)


def independent_thermal_ode(state,duration):
    """Independent continuous water/heat RHS; reuse only verified phase inversion.

    Pressure, ice impedance and donor heat are calculated directly from the
    evolving states, not from any Richards function or split transport step.
    """
    from noahmp_clm.physics.soil_enthalpy import SoilEnthalpyState
    from noahmp_clm.physics.snowpack import TF,LF,CW
    n=len(state.water);p=state.params;dz=state.dz;z=np.cumsum(dz)-.5*dz
    def rhs(_,value):
        w,h=value[:n],value[n:];one=SoilEnthalpyState(w,h,dz,p,state.freezing_option)
        liquid=one.liquid;ice=w-liquid;t=one.temperature
        saturation=liquid/(1000.*dz*p.smcmax)
        psi=-np.minimum(p.saturated_suction_m*np.maximum(saturation,.01)**(-p.bexp),1e5)
        material=p.satdk*10.**(-6.*ice/(1000.*dz*p.smcmax))
        water_flux=[];energy_flux=[]
        for j in range(n-1):
            face=np.sqrt(material[j]*material[j+1])*((saturation[j]+saturation[j+1])*.5)**(2.*p.bexp+3.)
            mass=1000.*face*(1.-(psi[j+1]-psi[j])/(z[j+1]-z[j]))
            water_flux.append(mass);energy_flux.append(mass*(LF+CW*(t[j if mass>=0. else j+1]-TF)))
        return np.r_[-np.diff(np.r_[0.,water_flux,0.]),-np.diff(np.r_[0.,energy_flux,0.])]
    answer=solve_ivp(rhs,(0.,duration),np.r_[state.water,state.enthalpy],method='Radau',rtol=2e-9,
        atol=np.r_[np.full(n,1e-10),np.full(n,1e-3)])
    assert answer.success
    return SoilEnthalpyState(answer.y[:n,-1],answer.y[n:,-1],dz,p,state.freezing_option)


@pytest.mark.parametrize('kind,option',[('warm',1),('warm_day',1),('ten_layers',1),('frozen',1),('frozen',2)])
def test_water_enthalpy_transport_against_independent_continuous_ode(kind,option):
    from noahmp_clm.physics.richards import redistribute_soil_enthalpy
    if kind=='ten_layers':
        dz=profile(10).dz_m;theta=np.full(10,.28);temperature=285.+5.*np.sin(np.arange(10));duration=21600.
    else:
        dz=profile().dz_m;theta=np.array([.35,.24,.28,.25]);temperature=np.array([285.,294.,281.,300.]);duration=1800. if kind=='warm' else 86400.
        if kind=='frozen':theta=np.array([.3,.31,.29,.35]);temperature=np.array([271.,274.,272.,276.]);duration=21600.
    state=initial_soil(theta,temperature,dz,option);expected=independent_thermal_ode(state,duration)
    got,ledger=redistribute_soil_enthalpy(state,duration)
    water_error=float(np.max(abs(got.water-expected.water)/(1000.*dz)))
    temperature_error=float(np.max(abs(got.temperature-expected.temperature)))
    assert water_error<=.002 and temperature_error<=.05
    assert np.all(abs(ledger['residual_mass_heat'])<=ledger['bound_mass_heat'])
    assert ledger['maximum_substep_liquid_fraction']<=.05 and ledger['maximum_substep_temperature_change_k']<=.05
    print('Richards independent water/heat',json.dumps(dict(kind=kind,freezing_option=option,
        maximum_soil_water_error_m3m3=water_error,maximum_temperature_error_k=temperature_error,
        residual_mass_heat=ledger['residual_mass_heat'].tolist(),original_bounds=ledger['bound_mass_heat'].tolist(),
        accepted_substeps=ledger['accepted_substeps'],rejected_substeps=ledger['rejected_substeps'])),flush=True)


def test_frozen_pore_transform_retains_original_total_pore_pressure_and_conductivity():
    from noahmp_clm.physics.richards import soil_hydraulic_profile,hydraulic_properties
    state=initial_soil([.3]*4,[270.,271.,272.,274.]);p=state.params
    profile_=soil_hydraulic_profile(state);liquid=state.liquid;ice=state.water-liquid
    pressure=-profile_.saturated_suction_m*(profile_.porosity/(liquid/(1000.*state.dz)))**p.bexp
    theta,_,k,_=hydraulic_properties(pressure,profile_)
    direct_pressure=-p.saturated_suction_m*(p.smcmax/(liquid/(1000.*state.dz)))**p.bexp
    direct_k=p.satdk*(liquid/(1000.*state.dz*p.smcmax))**(2*p.bexp+3)*10.**(-6.*ice/(1000.*state.dz*p.smcmax))
    np.testing.assert_allclose(pressure,direct_pressure,rtol=1e-12)
    np.testing.assert_allclose(theta,liquid/(1000.*state.dz),rtol=1e-12)
    np.testing.assert_allclose(k,direct_k,rtol=1e-12,atol=1e-30)


@pytest.mark.parametrize('theta',[[0.,.3,.25,.28],[.3,0.,.25,.28],[.3,0.,0.,.28]])
def test_strict_dry_layers_wet_without_created_storage_against_radau(theta):
    p=profile();old=np.asarray(theta)*1000.*p.dz_m;duration=1800.
    expected,flux=independent_ode(old,p,duration);got=redistribute_liquid(old,p,duration)
    error=float(np.max(abs(got['water_kg_m2']-expected)/(1000.*p.dz_m)))
    face_error=abs(got['interface_transfer_kg_m2']-flux)
    assert error<=.002
    assert np.all(face_error<=np.where(abs(flux)<=.02,.002,.1*abs(flux)))
    assert np.all(got['water_kg_m2'][old==0.]>0.) and np.all(got['water_kg_m2']>=0.)
    assert abs(got['mass_residual_kg_m2'])<=got['mass_bound_kg_m2']
    print('Richards dry wetting',json.dumps(dict(initial_theta=theta,max_soil_water_error_m3m3=error,
        max_face_error_kg_m2=float(face_error.max()),final_water_kg_m2=got['water_kg_m2'].tolist(),
        mass_residual_kg_m2=got['mass_residual_kg_m2'],mass_bound_kg_m2=got['mass_bound_kg_m2'],
        substeps=got['accepted_substeps'],rejections=got['rejected_substeps'])),flush=True)


def test_fully_dry_column_retains_exact_zero_storage_and_finite_limited_pressure():
    p=profile();got=redistribute_liquid(np.zeros(4),p,86400.)
    np.testing.assert_array_equal(got['water_kg_m2'],0.)
    np.testing.assert_array_equal(got['interface_transfer_kg_m2'],0.)
    np.testing.assert_array_equal(got['pressure_head_m'],-1e5)
    assert np.all(got['pressure_defined']) and np.all(got['pressure_floor_active'])


@pytest.mark.parametrize('all_blocked',[False,True])
def test_zero_available_pore_volume_is_an_explicit_ice_barrier(all_blocked):
    pore=np.zeros(4) if all_blocked else np.array([.45,0.,.45,.45])
    k=np.where(pore>0.,1e-5,0.);p=replace(profile(),porosity=pore,saturated_conductivity_m_s=k)
    old=np.array([.3,0.,.35,.2])*1000.*p.dz_m if not all_blocked else np.zeros(4)
    got=redistribute_liquid(old,p,1800.)
    np.testing.assert_array_equal(got['pressure_defined'],pore>0.)
    np.testing.assert_array_equal(got['water_kg_m2'][pore==0.],0.)
    np.testing.assert_array_equal(got['interface_transfer_kg_m2'][:2],0.)
    assert np.all(np.isfinite(got['pressure_head_m']))
    assert abs(got['mass_residual_kg_m2'])<=got['mass_bound_kg_m2']
    if not all_blocked:assert got['interface_transfer_kg_m2'][2]>0.


def test_warm_dry_layer_transports_actual_water_and_heat_against_continuous_ode():
    from noahmp_clm.physics.richards import redistribute_soil_enthalpy
    state=initial_soil([.3,0.,.25,.28],[285.,294.,281.,300.]);duration=1800.
    expected=independent_thermal_ode(state,duration);got,ledger=redistribute_soil_enthalpy(state,duration)
    water_error=float(np.max(abs(got.water-expected.water)/(1000.*state.dz)))
    temperature_error=float(np.max(abs(got.temperature-expected.temperature)))
    assert got.water[1]>0. and water_error<=.002 and temperature_error<=.05
    assert np.all(abs(ledger['residual_mass_heat'])<=ledger['bound_mass_heat'])
    print('Richards dry water/heat',json.dumps(dict(max_soil_water_error_m3m3=water_error,
        max_temperature_error_k=temperature_error,residual_mass_heat=ledger['residual_mass_heat'].tolist(),
        original_bounds=ledger['bound_mass_heat'].tolist(),accepted_substeps=ledger['accepted_substeps'],
        rejected_substeps=ledger['rejected_substeps'])),flush=True)
