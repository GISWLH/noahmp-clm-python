import numpy as np
import json
from pathlib import Path
import pytest
from noahmp_clm.column import ColumnState
from noahmp_clm.physics.soil_enthalpy import SoilEnthalpyState,infiltrate,enthalpy_and_capacity
from noahmp_clm.physics.snowpack import Snowpack,TF,LF,drain_liquid
from noahmp_clm.physics.column_heat import coupled_heat_step
from clm_soil_oracle import reference_liquid as unfrozen_water_content


def soil(temperature):
    return SoilEnthalpyState.from_column(ColumnState(tsoil=np.full(4,temperature),smc=np.full(4,.25)))


def test_shared_interface_energy_and_total_conservation():
    snow=Snowpack.from_ice([3.,10.],[TF-5.,TF-10.],[100.,250.]);ground=soil(TF+2.)
    s,g,d=coupled_heat_step(snow,ground,1800.,30.,snow_absorbed=[1.,2.],soil_absorbed=[3.,0.,0.,0.])
    assert d['snow_enthalpy_change_jm2']==pytest.approx(1800*(33.-d['snow_soil_flux_wm2']),abs=2e-6)
    assert d['soil_enthalpy_change_jm2']==pytest.approx(1800*(3.+d['snow_soil_flux_wm2']),abs=2e-6)
    assert abs(d['residual_jm2'])<2e-6
    np.testing.assert_array_equal(s.mass,snow.mass);np.testing.assert_array_equal(g.water,ground.water)


def test_full_melt_energy_actually_enters_soil_or_runoff():
    snow=Snowpack.from_ice([.1],TF,100.);ground=soil(TF-1.)
    before=snow.enthalpy.sum()+ground.enthalpy.sum()
    s,g,d=coupled_heat_step(snow,ground,1800.,300.)
    s,drain=drain_liquid(s,return_ledger=True)
    assert len(s.mass)==0
    assert drain['drainage_enthalpy_jm2']>LF*drain['drainage_kgm2']
    g,overflow,overflow_h=infiltrate(g,drain['drainage_kgm2'],drain['drainage_enthalpy_jm2'])
    energy_after=s.enthalpy.sum()+g.enthalpy.sum()+overflow_h+drain['runoff_enthalpy_jm2']
    assert energy_after-before==pytest.approx(1800*300.,abs=3e-6)
    assert g.water.sum()+overflow+drain['runoff_kgm2']==pytest.approx(ground.water.sum()+.1,abs=1e-12)


def test_uniform_insulated_snow_and_soil_remain_in_equilibrium():
    snow=Snowpack.from_ice([2.,5.,11.,23.,60.],TF-5.,300.);ground=soil(TF-5.)
    for _ in range(20): snow,ground,d=coupled_heat_step(snow,ground,1800.,0.)
    np.testing.assert_allclose(snow.temperature,TF-5.,rtol=0,atol=1e-11)
    np.testing.assert_allclose(ground.temperature,TF-5.,rtol=0,atol=1e-11)
    assert abs(d['residual_jm2'])<2e-6


def test_no_snow_linear_warm_soil_against_independent_matrix_solution():
    g=soil(TF+5.);dt=1800.;k=np.array([.3,.5,.8,1.])
    capacity=(1-g.params.smcmax)*2e6*g.dz+4180*g.water
    matrix=np.diag(capacity/dt);rhs=capacity/dt*5.
    for j in range(3):
        conductance=1/(g.dz[j]/(2*k[j])+g.dz[j+1]/(2*k[j+1]))
        matrix[j,j]+=conductance;matrix[j+1,j+1]+=conductance
        matrix[j,j+1]-=conductance;matrix[j+1,j]-=conductance
    matrix[0,0]+=8.;rhs[0]+=10.+8.*10.
    expected=np.linalg.solve(matrix,rhs)
    _,new,d=coupled_heat_step(Snowpack.empty(),g,dt,10.,top_temperature=TF+10,
        top_conductance=8.,thermal_conductivity=k)
    np.testing.assert_allclose(new.temperature-TF,expected,rtol=0,atol=1e-11)
    assert abs(d['residual_jm2'])<2e-6


def independent_coupled_reference(snow,ground,dt,top,sw,soil_sw,k,air,exchange):
    """Independent simultaneous MINPACK solve in snow enthalpy/soil temperature.

    Uses the pre-existing soil phase function, not the new inverse, analytic
    derivative, assembled operator or enthalpy Newton iteration.
    """
    root=pytest.importorskip('scipy.optimize').root
    ns=len(snow.mass);n=ns+len(ground.water);z=np.r_[snow.thickness,ground.dz]
    original=np.r_[snow.enthalpy,ground.enthalpy]
    def evaluate(x,details=False):
        hs=x[:ns]*snow.mass*334000.
        ts=[]
        for mass,energy in zip(snow.mass,hs):
            ts.append(energy/(2100*mass) if energy<0 else
                (energy-334000*mass)/(4180*mass) if energy>334000*mass else 0.)
        tg=x[ns:]*10.
        liquid=unfrozen_water_content(ground.water/(1000*ground.dz),tg+273.15,
            ground.params,ground.freezing_option)*1000*ground.dz
        hg=((1-ground.params.smcmax)*2e6*ground.dz+2100*(ground.water-liquid)+4180*liquid)*tg+334000*liquid
        temperatures=np.r_[ts,tg];energy=np.r_[hs,hg]
        flux=[top+exchange*(air-273.15-temperatures[0])]
        for j in range(n-1): flux.append((temperatures[j]-temperatures[j+1])/(z[j]/(2*k[j])+z[j+1]/(2*k[j+1])))
        flux.append(0.)
        net=np.array(flux[:-1])-flux[1:]+np.r_[sw,soil_sw]
        residual=(energy-original)/dt-net
        return (energy,temperatures+273.15,residual) if details else residual
    initial=np.r_[snow.enthalpy/(snow.mass*334000.),(ground.temperature-273.15)/10.]
    sol=root(evaluate,initial,tol=1e-10)
    energy,temp,res=evaluate(sol.x,True)
    assert np.max(np.abs(res))<2e-6,(sol.message,res)
    return energy,temp


def test_random_coupled_freezing_against_independent_solver():
    from noahmp_clm.column import default_soil_params
    rng=np.random.default_rng(58439)
    for case in range(60):
        ns=case%6
        mass=rng.uniform(.5,50.,ns)
        snow=Snowpack(mass,mass*rng.uniform(-2100*20,LF*1.1,ns),mass/rng.uniform(80,550,ns))
        c=ColumnState(soil=default_soil_params(('sand','loam','clay')[case%3]),
            tsoil=rng.uniform(TF-12,TF+8,4),smc=rng.uniform(.1,.35,4))
        ground=SoilEnthalpyState.from_column(c)
        k=rng.uniform(.05,1.5,ns+4);sw=rng.uniform(0,50,ns);sg=rng.uniform(0,20,4)
        dt=rng.uniform(30,3600);top=rng.uniform(-100,150);air=rng.uniform(TF-20,TF+15);exchange=rng.uniform(0,15)
        s,g,d=coupled_heat_step(snow,ground,dt,top,snow_absorbed=sw,soil_absorbed=sg,
            top_temperature=air,top_conductance=exchange,thermal_conductivity=k)
        href,tref=independent_coupled_reference(snow,ground,dt,top,sw,sg,k,air,exchange)
        np.testing.assert_allclose(np.r_[s.temperature,g.temperature],tref,rtol=0,atol=1e-7)
        np.testing.assert_allclose(np.r_[s.enthalpy,g.enthalpy],href,rtol=1e-9,atol=.003)
        assert abs(d['residual_jm2'])<5e-6


def test_annual_thin_snow_crosses_complete_melt_without_line_search_stall():
    from noahmp_clm.column import SoilParams
    from noahmp_clm.physics.snow_heat import conductivity
    from noahmp_clm.physics.soil import thermal_conductivity
    data=json.loads((Path(__file__).parent/'fixtures/column_phase_boundary.json').read_text())
    snow=Snowpack(**data['snow']);params=data['soil'].pop('params')
    ground=SoilEnthalpyState(**data['soil'],params=SoilParams(**params))
    s,g,ledger=coupled_heat_step(snow,ground,data['dt'],data['flux'],**data['kwargs'])
    assert s.temperature[0]>TF and s.ice[0]==0.
    assert ledger['iterations']<10 and abs(ledger['residual_jm2'])<5e-6
    kw=data['kwargs'];k=np.r_[conductivity(snow),thermal_conductivity(ground.water/(1000*ground.dz),ground.params)]
    href,tref=independent_coupled_reference(snow,ground,data['dt'],data['flux'],
        kw['snow_absorbed'],kw['soil_absorbed'],k,kw['top_temperature'],kw['top_conductance'])
    np.testing.assert_allclose(np.r_[s.enthalpy,g.enthalpy],href,atol=.003,rtol=1e-9)
    np.testing.assert_allclose(np.r_[s.temperature,g.temperature],tref,atol=1e-7,rtol=0.)


@pytest.mark.parametrize('mass_scale',[1.,1e3,1e6,1e9])
def test_vanishing_snow_exact_latent_endpoint_advances_with_equal_rounded_norm(mass_scale):
    from noahmp_clm.column import SoilParams
    from noahmp_clm.physics.snow_heat import conductivity
    from noahmp_clm.physics.soil import thermal_conductivity
    packet=json.loads((Path(__file__).parent/'fixtures/vanishing_snow_event.json').read_text())
    snow=Snowpack(**packet['snow'])
    snow=Snowpack(snow.mass*mass_scale,snow.enthalpy*mass_scale,snow.thickness*mass_scale)
    p=dict(packet['soil']);props=p.pop('params');earth=SoilEnthalpyState(**p,params=SoilParams(**props))
    dt=packet['dt'];kw=packet['kwargs'];s,g,ledger=coupled_heat_step(snow,earth,dt,packet['top_flux'],**kw)
    # Independently solve the all-liquid endpoint in temperature coordinates.
    # Its tiny surface heat capacity is retained, rather than clipping mass.
    z=np.r_[snow.thickness,earth.dz];k=np.r_[conductivity(snow),thermal_conductivity(earth.water/(1000.*earth.dz),earth.params)]
    cap=np.r_[4180.*snow.mass,(1.-earth.params.smcmax)*2e6*earth.dz+4180.*earth.water]
    fusion=334000.*np.r_[snow.mass,earth.water]
    matrix=np.diag(cap/dt);rhs=(np.r_[snow.enthalpy,earth.enthalpy]-fusion)/dt+np.r_[kw['snow_absorbed'],kw['soil_absorbed']];rhs[0]+=packet['top_flux']
    for j in range(len(z)-1):
        conduct=1./(.5*z[j]/k[j]+.5*z[j+1]/k[j+1])
        matrix[j,j]+=conduct;matrix[j+1,j+1]+=conduct;matrix[j,j+1]-=conduct;matrix[j+1,j]-=conduct
    exchange=kw.get('top_conductance',0.);matrix[0,0]+=exchange
    if kw.get('top_temperature') is not None:rhs[0]+=exchange*(kw['top_temperature']-TF)
    expected=np.linalg.solve(matrix,rhs);assert np.all(expected>0.)
    actual=np.r_[s.temperature,g.temperature]-TF;h=np.r_[s.enthalpy,g.enthalpy]
    np.testing.assert_allclose(actual,expected,rtol=0.,atol=1e-8)
    np.testing.assert_allclose(h,fusion+cap*expected,rtol=0.,atol=ledger['convergence_tolerance_jm2'])
    assert ledger['iterations']<=6
    assert np.max(abs(ledger['layer_residual_jm2']))<=ledger['convergence_tolerance_jm2']
    assert abs(ledger['residual_jm2'])<1e-6
    np.testing.assert_array_equal(s.mass,snow.mass);np.testing.assert_array_equal(g.water,earth.water)
    print('Vanishing snow event',mass_scale,dict(temperature_error_k=float(abs(actual-expected).max()),
        energy_residual_jm2=ledger['residual_jm2'],iterations=ledger['iterations']),flush=True)
