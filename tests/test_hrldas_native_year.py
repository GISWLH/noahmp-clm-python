"""Synthetic calendar/integral/provenance tests, not physical annual evidence."""
from copy import deepcopy
from datetime import datetime,timedelta
import hashlib
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'benchmarks'))
from compare_hrldas_column import compare_columns
from compare_hrldas_native_year import compare_native_year,check_ledgers,check_final_checkpoint


def year_inputs(year=2004):
    begin=datetime(year,1,1);n=(datetime(year+1,1,1)-begin).days;dt=timedelta(days=1)
    starts=[begin+i*dt for i in range(n)];h=100.+np.sin(np.arange(n));le=50.+np.cos(np.arange(n))
    trace=dict(time_start=np.array([v.strftime('%Y%m%d%H%M') for v in starts]),
        time_end=np.array([(v+dt).strftime('%Y%m%d%H%M') for v in starts]),
        h_wm2=h,le_wm2=le,et_mm=np.ones(n),runoff_mm=np.full(n,.1),swe_m=np.zeros(n),
        tsoil_k=np.full((n,2),274.15),smc_m3m3=np.full((n,2),.3),
        step_residual_m_h_c_n=np.zeros((n,4)),step_bound_m_h_c_n=np.full((n,4),.001),
        cumulative_residual_m_h_c_n=np.zeros((n,4)))
    report=dict(status='complete',source_unchanged=True,steps_requested=n,steps_completed=n,
        configuration_sha256='config',forcing_npz_sha256='forcing',forcing_manifest_sha256='metadata',
        utc_offset_hours=8,time_convention='local standard time',ledger_steps=n,initial_run=None,
        cumulative_numerical_bounds_m_h_c_n=np.full(n,.001).cumsum()[-1]*np.ones(4))
    prepared=dict(steps=n,dt_seconds=86400,utc_offset_hours=8,initial_time_local=begin.isoformat(),
        inputs_sha256={'site.npz':'forcing','site.json':'metadata','site-config.json':'config'})
    rows=[dict(time_start=(v-timedelta(hours=8)).isoformat(),time_end=(v+dt-timedelta(hours=8)).isoformat(),
        h_wm2=float(h[i]),le_wm2=float(le[i]),et_mm=1.,runoff_mm=.1,swe_mm=0.,
        soil_temperature_k=[274.15]*2,soil_total_water_m3m3=[.3]*2) for i,v in enumerate(starts)]
    reference=dict(initial_time=rows[0]['time_start'],timestep_seconds=86400,physical_intervals=n,file_count=n+1,
        intervals=rows,time_convention='UTC interval ends')
    return trace,report,reference,prepared


@pytest.mark.parametrize('year,days,february',[(2003,365,28),(2004,366,29)])
def test_exact_calendar_and_integrals_preserve_originals(year,days,february):
    t,r,ref,p=year_inputs(year);old=deepcopy(ref);oldp=deepcopy(p)
    result=compare_native_year(t,r,ref,p)
    assert len(result['daily'])==days and len(result['monthly'])==12
    assert ref==old and p==oldp and not result['scientific_acceptance_claim']
    assert 'local standard time UTC+8' in result['time_convention']
    assert result['daily_bias_criteria_met'] and result['monthly_bias_criteria_met'] and result['instantaneous_flux_criteria_met']
    month=result['monthly'][f'{year}-02']
    assert month['et_mm_day']['model_integral']==february
    assert month['runoff_mm_day']['model_integral']==pytest.approx(february*.1)
    assert result['full_span']['et_mm_day']['model_integral']==days
    assert result['full_span']['h_wm2']['model_integral']==pytest.approx(t['h_wm2'].sum()*86400.)
    assert result['full_span']['h_wm2']['integral_units']=='J/m2'
    assert month['tsoil_c_layer_1']['state_units']=='deg C'


@pytest.mark.parametrize('fault',['running','source','short_request','incomplete_year','forcing','metadata','config','timezone','gap','offset','soil_layers','nonfinite','original_date'])
def test_invalid_annual_input_is_rejected(fault):
    t,r,ref,p=year_inputs()
    if fault=='running':r['status']='running'
    elif fault=='source':r['source_unchanged']=False
    elif fault=='short_request':r['steps_requested']+=1
    elif fault=='incomplete_year':
        for k in t:t[k]=t[k][:-1]
        r['steps_completed']-=1;r['steps_requested']-=1;p['steps']-=1
        ref['intervals'].pop();ref['physical_intervals']-=1;ref['file_count']-=1
    elif fault in ('forcing','metadata','config'):r[{'forcing':'forcing_npz_sha256','metadata':'forcing_manifest_sha256','config':'configuration_sha256'}[fault]]='changed'
    elif fault=='timezone':p['utc_offset_hours']=0
    elif fault=='gap':t['time_start'][40]='200402100100'
    elif fault=='offset':ref['intervals'][0]['time_end']='2004-01-01T17:00:00'
    elif fault=='soil_layers':t['tsoil_k']=t['tsoil_k'][:,:1]
    elif fault=='nonfinite':t['le_wm2'][3]=np.nan
    else:p['initial_time_local']='2004-01-02T00:00:00'
    with pytest.raises(ValueError):compare_native_year(t,r,ref,p)


def test_partial_month_and_constant_flux_are_unmeasured():
    t,r,ref,p=year_inputs();rows=deepcopy(ref['intervals'][:7])
    for i,row in enumerate(rows):
        row['time_start']=(datetime(2004,1,1)+timedelta(days=i)).isoformat()
        row['time_end']=(datetime(2004,1,2)+timedelta(days=i)).isoformat()
        row['h_wm2']=2.;row['le_wm2']=0.
    ref.update(initial_time=rows[0]['time_start'],intervals=rows)
    result=compare_columns(dict(status='complete',steps_completed=7,intervals=deepcopy(rows)),ref)
    assert result['monthly_bias_criteria_met'] is None
    assert result['monthly']['2004-01']['et_mm_day']['status']=='unmeasured'
    assert result['instantaneous_flux_criteria_status']=='unmeasured' and not result['instantaneous_flux_criteria_met']


def test_half_hour_amounts_and_watts_have_distinct_monthly_integrals():
    begin=datetime(2003,1,1);n=31*48;rows=[]
    for i in range(n):
        rows.append(dict(time_start=(begin+timedelta(minutes=30*i)).isoformat(),
            time_end=(begin+timedelta(minutes=30*(i+1))).isoformat(),et_mm=.025,runoff_mm=.001,
            h_wm2=100.,le_wm2=50.,swe_mm=0.,soil_temperature_k=[274.15],soil_total_water_m3m3=[.3]))
    result=compare_columns(dict(status='complete',steps_completed=n,intervals=deepcopy(rows)),
        dict(initial_time=begin.isoformat(),timestep_seconds=1800,intervals=rows,time_convention='test local'))
    month=result['monthly']['2003-01']
    assert month['et_mm_day']['model_integral']==pytest.approx(.025*n)
    assert month['h_wm2']['model_integral']==100.*31*86400
    assert month['et_mm_day']['model_mean']==pytest.approx(1.2)


@pytest.mark.parametrize('fault',['step','cumulative','negative_bound','reported_bound','fresh_steps','missing'])
def test_conservation_cannot_use_science_tolerance(fault):
    t,r,_,_=year_inputs()
    if fault=='step':t['step_residual_m_h_c_n'][0,1]=.001001
    elif fault=='cumulative':t['cumulative_residual_m_h_c_n'][0,0]=.001001
    elif fault=='negative_bound':t['step_bound_m_h_c_n'][0,0]=-.001
    elif fault=='reported_bound':r['cumulative_numerical_bounds_m_h_c_n'][0]+=.001
    elif fault=='fresh_steps':r['ledger_steps']+=1
    else:del t['step_bound_m_h_c_n']
    with pytest.raises(ValueError):check_ledgers(t,r)


def test_carried_ledger_requires_original_checkpoint_and_cumulative_bounds(tmp_path):
    t,r,_,_=year_inputs();path=tmp_path/'initial.npz';start=np.array([1.,2.,3.,4.])
    np.savez(path,bounds=start[None,:]);r['batch_column']=0
    r['initial_run']=dict(checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),previous_ledger_steps=365)
    r['ledger_steps']+=365;r['cumulative_numerical_bounds_m_h_c_n']+=start
    with pytest.raises(ValueError):check_ledgers(t,r)
    result=check_ledgers(t,r,path)
    assert result['starting_bounds_m_h_c_n']==start.tolist()
    t['cumulative_residual_m_h_c_n'][0,0]=1.002
    with pytest.raises(ValueError):check_ledgers(t,r,path)


def test_final_storage_is_summed_from_checkpoint_not_reported_alias(tmp_path):
    t,r,_,_=year_inputs();r['batch_column']=0
    arrays={f'a{i}':np.zeros((1,1)) for i in range(16)}
    for i in (0,1,7):arrays[f'a{i}']=np.zeros((1,2,9))
    arrays['a11']=np.zeros((1,8));arrays['a13']=np.zeros((1,14))
    arrays['a1'][0,0,0]=2.;arrays['a11'][0,6]=3.;arrays['a13'][0,7]=4.
    path=tmp_path/'final.npz'
    def save():
        np.savez(path,**arrays,index=np.array(r['steps_completed']),initial=np.array([[2.,0.,3.,4.]]),net=np.zeros((1,4)),bounds=r['cumulative_numerical_bounds_m_h_c_n'][None,:])
        r['batch_checkpoint_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    save();result=check_final_checkpoint(path,t,r)
    assert result['extensive_storage_m_h_c_n']==[2.,0.,3.,4.]
    arrays['a11'][0,6]+=.00001;save()
    with pytest.raises(ValueError,match='storage/ledger'):check_final_checkpoint(path,t,r)
