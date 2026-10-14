"""Independent capacity-curve integration and extensive VIC/ARNO ledgers."""
from dataclasses import replace
from decimal import Decimal,localcontext
import numpy as np
import pytest
from scipy.integrate import quad
from noahmp_clm.physics.vic import VICParameters,vic_infiltration,arno_baseflow,infiltrate_vic
from noahmp_clm.physics.soil_drainage import drain_soil
from noahmp_clm.physics.soil_enthalpy import SoilEnthalpyState
from noahmp_clm.column import ColumnState


def test_vic_pulse_against_independent_area_integral_and_decimal_reference():
    maximum=relative=0.;count=0
    for b in (0.,1e-5,.01,.2,.4,1.,2.):
        for capacity in (1.,180.,600.):
            for wet in (0.,.1,.5,.9,.99999,1.):
                for pulse in (0.,1e-12,1e-6,.1,5.,1000.):
                    water=capacity*wet
                    got,runoff,asat=vic_infiltration(pulse,water,capacity,b)
                    with localcontext() as ctx:
                        ctx.prec=65
                        P,W,C,B=map(Decimal,(pulse,water,capacity,b))
                        if W==C:expected=0.
                        elif b==0.:expected=float(min(P,C-W))
                        else:
                            im=(1+B)*C;x=((C-W)/C)**(1/(1+B))
                            expected=float(C-W-C*max(Decimal(0),x-P/im)**(1+B))
                    error=abs(got-expected);maximum=max(maximum,error)
                    if expected>1e-8:relative=max(relative,error/expected)
                    assert error<2e-10
                    assert 0<=got<=pulse and 0<=runoff<=pulse and 0<=asat<=1
                    assert got+runoff==pytest.approx(pulse,rel=1e-15,abs=1e-15)
                    # Independent quadrature of the unsaturated area as
                    # the infiltration-capacity threshold increases.
                    if b>0 and wet<1 and pulse>.001:
                        x=(1.-wet)**(1./(1.+b));im=(1.+b)*capacity
                        limit=min(pulse,im*x)
                        integral=quad(lambda s:max(0.,x-s/im)**b,0.,limit,epsabs=2e-9)[0]
                        assert got==pytest.approx(integral,rel=2e-8,abs=2e-8)
                    count+=1
    print('VIC independent',dict(cases=count,max_mass_difference_kgm2=maximum,max_relative_percent=100.*relative),flush=True)


def test_vic_wetter_and_more_heterogeneous_soils_produce_more_surface_runoff():
    for b in (.01,.2,.4):
        q=[vic_infiltration(10.,w,180.,b)[1] for w in (0.,50.,100.,179.,180.)]
        assert np.all(np.diff(q)>=0)
    q=[vic_infiltration(10.,100.,180.,b)[1] for b in (0.,.01,.2,.4)]
    assert np.all(np.diff(q)>0)
    assert vic_infiltration(20.,10.,20.,0.)[:2]==(10.,10.)
    assert vic_infiltration(10.,0.,0.)[:2]==(0.,10.)


def test_arno_curve_endpoints_and_independent_decimal_formula():
    maximum=0.
    for threshold in (.5,.9):
        for ds in (0.,.1,threshold):
            for exponent in (1.,2.,3.):
                p=VICParameters(baseflow_fraction=ds,baseflow_threshold=threshold,baseflow_exponent=exponent)
                for wet in (0.,.01,threshold-1e-8,threshold,threshold+1e-8,.99,1.):
                    got=arno_baseflow(500.*wet,500.,3600.,p)
                    with localcontext() as ctx:
                        ctx.prec=50;D,S,W,C=map(Decimal,(ds,threshold,wet,exponent))
                        tail=max(Decimal(0),(W-S)/(1-S))
                        expected=float(Decimal(10)/24*(D/S*W+(1-D/S)*tail**C))
                    maximum=max(maximum,abs(got-expected));assert got==pytest.approx(expected,rel=1e-12,abs=1e-14)
                assert arno_baseflow(500.,500.,86400.,p)==10.
    assert arno_baseflow(1.,500.,1e12)==1.
    print('ARNO independent max mass difference kgm2',maximum,flush=True)


@pytest.mark.parametrize('temperature',[260.,272.,280.,300.])
def test_vic_admission_and_arno_drainage_conserve_mass_heat_and_ice(temperature):
    old=SoilEnthalpyState.from_column(ColumnState(tsoil=np.full(4,temperature),smc=np.full(4,.3)))
    parameters=VICParameters(residual_pore_fraction=.1)
    incoming=20.;heat=incoming*(334000.+4180.*7.)
    admitted,runoff,runh=infiltrate_vic(old,incoming,heat,parameters)
    assert 0<runoff<incoming
    mass=admitted.water.sum()+runoff-old.water.sum()-incoming
    thermal=admitted.enthalpy.sum()+runh-old.enthalpy.sum()-heat
    assert abs(mass)<1e-10 and abs(thermal)<1e-6
    drained,ledger=drain_soil(admitted,1800.,bottom_parameters=parameters)
    assert abs(drained.water.sum()+ledger['bottom_mass_kgm2']-admitted.water.sum())<1e-10
    assert abs(drained.enthalpy.sum()+ledger['bottom_enthalpy_jm2']-admitted.enthalpy.sum())<1e-6
    assert ledger['bottom_mass_kgm2']<=admitted.liquid[-1]+ledger['interface_mass_kgm2'][-2]+1e-10
    print('VIC transfer',temperature,dict(surface_kgm2=runoff,bottom_kgm2=ledger['bottom_mass_kgm2'],
        admission_heat_residual_jm2=thermal,drainage_heat_residual_jm2=ledger['energy_residual_jm2']),flush=True)


def test_vic_invalid_parameters_and_water():
    for kw in (dict(infiltration_shape=-.1),dict(baseflow_fraction=.95),dict(baseflow_threshold=1.),
               dict(baseflow_exponent=.5),dict(residual_pore_fraction=1.)):
        with pytest.raises(ValueError):VICParameters(**kw)
    for values in ((-1.,0.,10.),(1.,11.,10.),(1.,0.,-1.),(np.nan,0.,10.)):
        with pytest.raises(ValueError):vic_infiltration(*values)
