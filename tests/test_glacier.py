"""Independent glacier Stefan, conduction and mass/enthalpy references."""
from dataclasses import replace
from decimal import Decimal,localcontext
import numpy as np
import pytest
from noahmp_clm.physics.glacier import GlacierIce,GlacierParameters,glacier_heat_step,add_glacier_surface_water
from noahmp_clm.physics.snowpack import Snowpack,TF,CI,LF,add_snowfall


@pytest.mark.parametrize('q',[-100.,0.,100.,500.,2000.])
def test_single_ice_cell_independent_decimal_stefan_energy(q):
    initial=GlacierIce(np.array([5.]),np.array([-105000.]),np.array([[1.,2.,3.,4.,5.]])*1e-8)
    _,ice,d=glacier_heat_step(Snowpack.empty(),initial,1800.,q,parameters=GlacierParameters(geothermal_wm2=0.))
    with localcontext() as ctx:
        ctx.prec=45;D=Decimal;h=D(-105000)+D(1800)*D(str(q));m=D(5)
        lost=min(m,max(D(0),h)/D(334000));out_h=max(D(0),h)
        expected_mass=float(m-lost);expected_h=float(min(D(0),h))
    assert ice.mass.sum()==pytest.approx(expected_mass,abs=1e-13)
    assert ice.enthalpy.sum()==pytest.approx(expected_h,abs=1e-8)
    assert d['runoff_kgm2']==pytest.approx(float(lost),abs=1e-13)
    assert d['runoff_enthalpy_jm2']==pytest.approx(float(out_h),abs=1e-8)
    np.testing.assert_allclose(ice.impurities.sum(0)+d['runoff_impurities_kgm2'],initial.impurities.sum(0),rtol=0,atol=1e-22)
    if q==2000.:
        assert len(ice.mass)==0 and d['runoff_enthalpy_jm2']>LF*5.
    print('glacier Stefan',q,'mass/heat',d['runoff_kgm2'],d['runoff_enthalpy_jm2'],'budget',d['water_residual_kgm2'],d['energy_residual_jm2'],flush=True)


def test_cold_snow_ice_interface_has_one_analytic_backward_euler_flux():
    snow=Snowpack.from_ice([10.],263.15,300.);ice=GlacierIce.from_thickness([.5],271.15)
    dt=3600.;ks=.023+(7.75e-5*300.+1.105e-6*300.**2)*(2.29-.023)
    g=1./(.5*(10./300.)/ks+.5*.5/2.29)
    cs=10.*2100.;ci=458.5*2100.;a=cs+dt*g;b=ci+dt*g
    # Closed 2x2 inverse, independent of the production Newton linear solver.
    t0=(-10.*cs*b+dt*g*(-2.*ci))/(a*b-(dt*g)**2)
    t1=(a*(-2.*ci)+dt*g*(-10.*cs))/(a*b-(dt*g)**2)
    ns,ni,d=glacier_heat_step(snow,ice,dt,0.,parameters=GlacierParameters(geothermal_wm2=0.))
    expected=np.array([t0,t1])+TF
    np.testing.assert_allclose(d['thermal_temperature_k'],expected,rtol=0,atol=1e-12)
    # Drainage conservatively regrids this 3.33 cm snow layer into two cells.
    np.testing.assert_allclose(ns.temperature,expected[0],rtol=0,atol=1e-12)
    np.testing.assert_allclose(ni.temperature,expected[1],rtol=0,atol=1e-12)
    assert d['thermal']['interface_flux_wm2'][0]==pytest.approx(g*(t0-t1),abs=1e-12)
    assert abs(d['energy_residual_jm2'])<1e-7


def test_independent_scipy_piecewise_enthalpy_system():
    root=pytest.importorskip('scipy.optimize').root;rng=np.random.default_rng(17301);max_t=max_h=0.;count=0
    for ns in (0,1,3,5):
        for ni in (1,4):
            for temp,top in ((250.,-40.),(270.,100.),(273.15,300.)):
                snow=Snowpack.from_ice(np.linspace(2.,10.,ns),temp,300.) if ns else Snowpack.empty()
                ice=GlacierIce.from_thickness(np.geomspace(.1,1.,ni),temp)
                mass=np.r_[snow.mass,ice.mass];old=np.r_[snow.enthalpy,ice.enthalpy];z=np.r_[snow.thickness,ice.thickness]
                k=np.r_[np.full(ns,.023+(7.75e-5*300.+1.105e-6*300.**2)*(2.29-.023)),np.full(ni,2.29)]
                dt=1800.;source=np.zeros(len(mass));source[0]=top;source[-1]+=.05
                def residual(specific):
                    temperature=np.where(specific<0.,specific/2100.,np.where(specific>334000.,(specific-334000.)/4180.,0.))
                    budget=mass*specific-old-dt*source
                    for i in range(len(mass)-1):
                        flux=(temperature[i]-temperature[i+1])/(z[i]/(2*k[i])+z[i+1]/(2*k[i+1]))
                        budget[i]+=dt*flux;budget[i+1]-=dt*flux
                    return budget/np.maximum(mass,1.)
                solved=root(residual,(old+dt*source)/mass,tol=1e-10)
                assert np.max(abs(residual(solved.x)))<1e-6,(solved.message,residual(solved.x))
                _,_,d=glacier_heat_step(snow,ice,dt,top)
                expected=TF+np.where(solved.x<0.,solved.x/2100.,np.where(solved.x>334000.,(solved.x-334000.)/4180.,0.))
                error=float(abs(d['thermal_enthalpy_jm2']-mass*solved.x).max());max_h=max(max_h,error)
                max_t=max(max_t,float(abs(d['thermal_temperature_k']-expected).max()));count+=1
                np.testing.assert_allclose(d['thermal_enthalpy_jm2'],mass*solved.x,rtol=0,atol=1e-5)
                np.testing.assert_allclose(d['thermal_temperature_k'],expected,rtol=0,atol=1e-8)
    print('glacier SciPy cases/max H/T',count,max_h,max_t,flush=True)


def test_surface_rain_freezes_with_all_heat_and_full_melt_exposes_deeper_ice():
    ice=GlacierIce.from_thickness([.1,.5],268.15);rain=2.;heat=rain*(334000.+4180.*(285.-273.15));tracer=np.arange(1,6)*1e-9
    new,d=add_glacier_surface_water(ice,rain,heat,tracer)
    assert d['runoff_kgm2']==0. and d['net_ice_formation_kgm2']==pytest.approx(2.)
    assert new.enthalpy.sum()-ice.enthalpy.sum()==pytest.approx(heat,abs=1e-8)
    np.testing.assert_array_equal(new.impurities.sum(0),tracer)
    small=GlacierIce(np.array([.1,10.]),np.array([0.,-210000.]))
    final,out=add_glacier_surface_water(small,1.,334000.+4180.*20.,tracer)
    assert len(final.mass)==1 and final.enthalpy[0]==0.
    assert final.mass.sum()+out['runoff_kgm2']==pytest.approx(11.1,abs=1e-13)
    assert final.enthalpy.sum()+out['runoff_enthalpy_jm2']-small.enthalpy.sum()==pytest.approx(417600.,abs=1e-8)
    np.testing.assert_allclose(final.impurities.sum(0)+out['runoff_impurities_kgm2'],tracer,rtol=0,atol=1e-22)


def test_full_year_snow_ice_heat_and_melt_ledger():
    ice=GlacierIce.from_thickness([.1,.3,.6,9.],263.15);snow=Snowpack.empty()
    initial_m=float(ice.mass.sum());initial_h=float(ice.enthalpy.sum());inputs=np.zeros(2);outputs=np.zeros(2);bounds=np.zeros(2);maxres=np.zeros(2)
    for day in range(365):
        angle=2.*np.pi*(day-90.)/365.;top=50.*np.sin(angle)
        if top<0.:
            snow=add_snowfall(snow,.5,263.15,100.);inputs+=[.5,.5*2100.*(-10.)]
        snow,ice,d=glacier_heat_step(snow,ice,86400.,top)
        inputs[1]+=d['thermal']['input_energy_jm2'];outputs+=[d['runoff_kgm2'],d['runoff_enthalpy_jm2']]
        bounds+=[d['water_bound_kgm2'],d['energy_bound_jm2']]
        residual=np.array([snow.mass.sum()+ice.mass.sum()-initial_m,snow.enthalpy.sum()+ice.enthalpy.sum()-initial_h])-inputs+outputs
        maxres=np.maximum(maxres,abs(residual));assert np.all(abs(residual)<=bounds+1e-6)
    print('glacier 365 prescribed flux days cumulative M/H',maxres.tolist(),'bounds',bounds.tolist(),'runoff',outputs.tolist(),'ice',ice.mass.sum(),flush=True)
    assert len(ice.mass)>0 and np.all(ice.enthalpy<=0.)


def test_glacier_contract_does_not_invent_replacement_ice():
    with pytest.raises(ValueError):GlacierIce([1.],[1.])
    with pytest.raises(ValueError):GlacierIce.from_thickness([1.],280.)
    with pytest.raises(ValueError):glacier_heat_step(Snowpack.empty(),GlacierIce.empty(),1800.,0.)
    with pytest.raises(ValueError):replace(GlacierParameters(),thermal_conductivity_wmk=-1.)
    with pytest.raises(ValueError):add_glacier_surface_water(GlacierIce.from_thickness([1.]),1.,0.)
