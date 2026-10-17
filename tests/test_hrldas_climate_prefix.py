"""Integrity guards for incomplete-year diagnostic scoring, without model integration."""
from datetime import datetime,timedelta
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
from compare_hrldas_climate_prefix import compare_prefix

def inputs():
    start=datetime(2003,1,1);n=336;dt=timedelta(minutes=30)
    h=10.+np.sin(np.arange(n));le=20.+np.cos(np.arange(n))
    trace=dict(time_start=np.array([(start+i*dt).strftime('%Y%m%d%H%M') for i in range(n)]),
        time_end=np.array([(start+(i+1)*dt).strftime('%Y%m%d%H%M') for i in range(n)]),
        h_wm2=h,le_wm2=le,et_mm=np.full(n,.01),runoff_mm=np.zeros(n),swe_m=np.zeros(n),
        tsoil_k=np.full((n,4),273.15),smc_m3m3=np.full((n,4),.3))
    report=dict(steps_completed=n,steps_requested=17520,status='running',forcing_npz_sha256='forcing',configuration_sha256='config',utc_offset_hours=8)
    rows=[dict(time_start=(start+i*dt-timedelta(hours=8)).isoformat(),time_end=(start+(i+1)*dt-timedelta(hours=8)).isoformat(),
        h_wm2=float(h[i]),le_wm2=float(le[i]),et_mm=.01,runoff_mm=0.,swe_mm=0.,soil_temperature_k=[273.15]*4,soil_total_water_m3m3=[.3]*4) for i in range(n)]
    reference=dict(physical_intervals=17520,file_count=17521,timestep_seconds=1800,intervals=rows+[rows[-1]]*(17520-n))
    prepared=dict(inputs_sha256={'input.npz':'forcing','site-config.json':'config'},steps=17520,dt_seconds=1800,utc_offset_hours=8,initial_time_local=start.isoformat())
    return trace,report,reference,prepared

def test_exact_utc_conversion_keeps_original_year_incomplete():
    result=compare_prefix(*inputs())
    assert result['original_model_status']=='running' and result['original_model_steps_requested']==17520
    assert result['scored_days']==7 and result['scientific_acceptance_claim'] is False
    assert result['diagnostic_all_daily_bias_criteria_met'] and result['diagnostic_flux_r_criteria_met']
    assert result['daily']['2003-01-01']['et_mm_day']['model_mean']==pytest.approx(.48)

@pytest.mark.parametrize('fault',['wrong_timezone','gap','source','partial_model','partial_reference'])
def test_reject_time_identity_and_coverage_faults(fault):
    trace,report,reference,prepared=inputs()
    if fault=='wrong_timezone':prepared['utc_offset_hours']=0
    elif fault=='gap':trace['time_start'][40]='200301012100'
    elif fault=='source':report['configuration_sha256']='other'
    elif fault=='partial_model':report['steps_completed']=335
    else:reference['physical_intervals']=336
    with pytest.raises(ValueError):compare_prefix(trace,report,reference,prepared)

def test_near_zero_temperature_uses_celsius_absolute_tolerance():
    trace,report,reference,prepared=inputs();trace['tsoil_k']+=.6
    result=compare_prefix(trace,report,reference,prepared)
    metric=result['daily']['2003-01-01']['tsoil_c_layer_1']
    assert metric['near_zero'] and metric['relative_bias'] is None and not metric['bias_pass']
    assert metric['bias']==pytest.approx(.6)
