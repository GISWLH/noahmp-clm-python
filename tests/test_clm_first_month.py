"""A complete January remains an incomplete original annual trajectory."""
from copy import deepcopy
from datetime import datetime,timedelta
import hashlib,json
import numpy as np
import pytest
from test_hrldas_suction_binding import case
from compare_clm_first_month import compare_official,first_month_times
from compare_hrldas_native_year import compare_native_year
from compare_hrldas_column import compare_columns


def month_inputs():
    _,report,reference,prepared,old_raw,new=case();n=31*48;begin=datetime(2003,1,1);step=timedelta(minutes=30)
    new_raw=json.dumps(new).encode();report.update(configuration_sha256=hashlib.sha256(new_raw).hexdigest(),steps_completed=n,
        time_convention='local standard time',source_unchanged=True,initial_run=None)
    times=[begin+i*step for i in range(17521)];h=20.+np.sin(np.arange(17520));le=30.+np.cos(np.arange(17520))
    trace=dict(time_start=np.array([x.strftime('%Y%m%d%H%M') for x in times[:n]]),time_end=np.array([x.strftime('%Y%m%d%H%M') for x in times[1:n+1]]),
        h_wm2=h[:n].copy(),le_wm2=le[:n].copy(),et_mm=np.full(n,.01),runoff_mm=np.zeros(n),swe_m=np.zeros(n),tsoil_k=np.full((n,4),273.15),smc_m3m3=np.full((n,4),.3))
    rows=[dict(time_start=(times[i]-timedelta(hours=8)).isoformat(),time_end=(times[i+1]-timedelta(hours=8)).isoformat(),
        h_wm2=float(h[i]),le_wm2=float(le[i]),et_mm=.01,runoff_mm=0.,swe_mm=0.,soil_temperature_k=[273.15]*4,soil_total_water_m3m3=[.3]*4) for i in range(17520)]
    reference.update(intervals=rows,initial_time=rows[0]['time_start'],time_convention='original UTC')
    return trace,report,reference,prepared,old_raw,new_raw,'CN-Qia'


def test_first_month_retains_original_annual_status_identity_and_integrals():
    values=month_inputs();before=deepcopy(values[1:4]);result=compare_official(*values)
    assert values[1:4]==before
    assert result['original_model_status']=='running' and result['original_model_steps_requested']==17520
    assert result['original_model_steps_completed']==1488 and result['original_official_intervals']==17520
    assert len(result['daily'])==31 and list(result['monthly'])==['2003-01'] and not result['scientific_acceptance_claim']
    month=result['monthly']['2003-01']
    assert month['et_mm_day']['model_integral']==pytest.approx(14.88)
    assert month['h_wm2']['model_integral']==pytest.approx(values[0]['h_wm2'].sum()*1800.)
    assert result['monthly_bias_criteria_met'] and result['instantaneous_flux_criteria_met']
    with pytest.raises(ValueError):compare_columns(dict(status='running'),values[2])
    with pytest.raises(ValueError):compare_native_year(*values[:4],configuration_pair=values[4:])


@pytest.mark.parametrize('fault',['short','long','complete','request','source','carried','timezone','gap','end','soil_layers','nonfinite','partial_reference','forcing'])
def test_incomplete_month_identity_or_time_drift_cannot_be_scored(fault):
    t,r,ref,p,old,new,site=values=month_inputs()
    if fault=='short':r['steps_completed']-=1
    elif fault=='long':r['steps_completed']+=1
    elif fault=='complete':r['status']='complete'
    elif fault=='request':r['steps_requested']=1488
    elif fault=='source':r['source_unchanged']=False
    elif fault=='carried':r['initial_run']={'checkpoint_sha256':'other'}
    elif fault=='timezone':r['utc_offset_hours']=0
    elif fault=='gap':t['time_start'][40]='200301012100'
    elif fault=='end':t['time_end'][-1]='200302010030'
    elif fault=='soil_layers':t['tsoil_k']=t['tsoil_k'][:,:3]
    elif fault=='nonfinite':t['et_mm'][4]=np.nan
    elif fault=='partial_reference':ref['intervals'].pop()
    else:r['forcing_npz_sha256']='changed'
    with pytest.raises(ValueError):compare_official(*values)


def test_monthly_near_zero_is_absolute_and_constant_flux_is_unmeasured():
    values=month_inputs();t,r,ref,*_=values;t['tsoil_k']+=.6;t['et_mm']+=.001
    for key in ('h_wm2','le_wm2'):
        t[key][:]=0.
        for row in ref['intervals']:row[key]=0.
    result=compare_official(*values);month=result['monthly']['2003-01']
    assert month['tsoil_c_layer_1']['near_zero'] and month['tsoil_c_layer_1']['relative_bias'] is None and not month['tsoil_c_layer_1']['bias_pass']
    assert month['et_mm_day']['near_zero'] and month['et_mm_day']['bias_pass']
    assert month['et_mm_day']['bias']==pytest.approx(.048)
    assert result['instantaneous_flux_criteria_status']=='unmeasured' and not result['instantaneous_flux_criteria_met']


def test_first_month_cannot_select_a_different_initial_season():
    t,r,_,p,*_=month_inputs()
    with pytest.raises(ValueError):first_month_times(t,r,initial_time='2003-02-01',dt_seconds=1800,full_steps=17520,utc_offset_hours=8)
