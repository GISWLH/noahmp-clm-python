"""Independent degree-day sums and analytic seasonal timing/leaf-area curves."""
from dataclasses import replace
from decimal import Decimal,localcontext
import math
import numpy as np
import pytest
from noahmp_clm.physics.phenology import (DAY,TF,PhenologyState,PhenologyParameters,
    PhenologyPhase,critical_gdd,degree_days,step_phenology)


def test_gdd_decimal_reference():
    largest=0.
    with localcontext() as ctx:
        ctx.prec=60
        for temperature in np.linspace(250.,310.,101):
            for dt in (60.,1800.,3600.,86400.):
                ref=max(Decimal(0),Decimal(float(temperature))-Decimal(TF))*Decimal(dt)/Decimal(86400)
                largest=max(largest,abs(degree_days(temperature,dt)-float(ref)))
    assert largest<1e-13
    assert critical_gdd(TF)==pytest.approx(math.exp(4.8),rel=1e-15)
    print('Phenology degree-day max absolute K days',largest,flush=True)


def test_two_years_against_independent_piecewise_season():
    p=PhenologyParameters(gdd_threshold=100.,lai_maximum=5.)
    state=PhenologyState(armed=False);error=0.;onsets=[];offsets=[];gain=loss=0.
    for day in range(1,731):
        d=(day-1)%365+1
        state,out=step_phenology(state,TF+10.,43200. if d<300 else 36000.,d<183,d==1,DAY,p)
        # 100 degree-days is reached on day 10; the strict > trigger is day 11.
        expected=0. if d<=10 or d>=314 else min(1.,(d-10)/30.) if d<300 else (314-d)/15.
        error=max(error,abs(out['lai']-5.*expected))
        if out['onset_triggered']:onsets.append(day)
        if out['offset_triggered']:offsets.append(day)
        gain+=out['leaf_gain_fraction'];loss+=out['leaf_loss_fraction']
    assert onsets==[11,376] and offsets==[300,665]
    assert error<1e-14 and abs(gain-loss-state.leaf_fraction)<1e-14
    print('Phenology two-year LAI max absolute',error,'onsets',onsets,'offsets',offsets,flush=True)


def test_missed_spring_cannot_start_in_autumn_and_solstice_rearms():
    p=PhenologyParameters(gdd_threshold=100.)
    state=PhenologyState(gdd=99.)
    state,out=step_phenology(state,TF+30.,50000.,False,False,DAY,p)
    assert not state.armed and state.gdd==0. and state.phase==0
    for _ in range(100):state,out=step_phenology(state,TF+30.,40000.,False,False,DAY,p)
    assert state.leaf_fraction==0.
    state,out=step_phenology(state,TF+1.,30000.,True,True,DAY,p)
    assert state.armed and state.gdd==1.


def test_phase_counters_and_leaf_fraction_close_at_subdaily_resolution():
    p=PhenologyParameters(gdd_threshold=1.)
    state=PhenologyState();onset_time=None;active_time=None
    for step in range(31*48):
        state,out=step_phenology(state,TF+10.,50000.,True,False,1800.,p)
        if out['onset_triggered']:onset_time=step
        if state.phase==2 and active_time is None:active_time=step
    # Counters use seconds, retaining exact half-hour transitions.
    assert active_time-onset_time==30*48-1
    assert state.leaf_fraction==1. and state.remaining_seconds==0.
    for _ in range(16*48):state,out=step_phenology(state,TF+10.,30000.,False,False,1800.,p)
    assert state.phase==0 and state.leaf_fraction==0. and not state.armed


def test_phenology_configuration_rejects_inconsistent_state():
    with pytest.raises(ValueError):PhenologyParameters(gdd_threshold=0.)
    with pytest.raises(ValueError):PhenologyState(phase=2,leaf_fraction=.5,armed=False)
    with pytest.raises(ValueError):step_phenology(PhenologyState(),280.,40000.,True,False,2*DAY)
    with pytest.raises(ValueError):step_phenology(PhenologyState(),280.,-1.,True,False,DAY)
