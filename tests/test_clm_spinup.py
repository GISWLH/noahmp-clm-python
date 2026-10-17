"""Synthetic provenance/calendar/endpoint audits; no actual spinup evidence."""
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime,timedelta
import json,sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'benchmarks'))
import assess_clm_spinup as audit
import run_fluxnet_native as native
from noahmp_clm.column import ColumnState
from noahmp_clm.options import PhysicsOptions
from noahmp_clm.physics.biogeochemistry import BGCParameters


def endpoints():
    return dict(soil_temperature_k=np.array([280.,281.]),soil_water_m3m3=np.array([.25,.3]),swe_m=.01,
                soil_carbon_gc_m2=np.array([0.,10.,20.,0.,100.,1000.,3000.]))


def test_endpoint_layer_maxima_and_carbon_cancellation_remain_visible():
    a=endpoints();b=deepcopy(a);b['soil_temperature_k'][1]+=.11;b['soil_water_m3m3'][0]+=.0009
    b['soil_carbon_gc_m2'][4]+=10.;b['soil_carbon_gc_m2'][5]-=10.
    r=audit.compare_endpoints(a,b)
    assert not r['water_heat_criteria_met'] and r['soil_temperature_k']==pytest.approx(.11)
    assert r['total_soil_carbon_change_gc_m2']==0. and r['soil_carbon_change_gc_m2'][4:6]==[10.,-10.]
    assert r['per_pool_rate_percent_per_100_climate_years'][0] is None


@pytest.mark.parametrize('key,change',[('soil_temperature_k',.101),('soil_water_m3m3',.00101),('swe_m',.000501)])
def test_each_fixed_water_heat_limit_is_independent(key,change):
    a=endpoints();b=deepcopy(a);b[key]+=change
    assert not audit.compare_endpoints(a,b)['water_heat_criteria_met']


def test_near_zero_carbon_never_uses_relative_error_or_pass():
    a=endpoints();a['soil_carbon_gc_m2'][:]=0.;b=deepcopy(a);b['soil_carbon_gc_m2'][4]=.2
    r=audit.compare_endpoints(a,b)
    assert r['signed_total_soil_carbon_rate_percent_per_100_climate_years'] is None
    assert r['total_soil_carbon_rate_criterion_met'] is None and r['total_soil_carbon_change_gc_m2']==.2


def test_a_scaled_annual_rate_does_not_become_an_observed_century():
    a=endpoints();b=deepcopy(a);b['soil_carbon_gc_m2']*=1.0001
    r=audit.compare_endpoints(a,b);assert r['signed_total_soil_carbon_rate_percent_per_100_climate_years']==pytest.approx(1.)
    history=lambda amounts:[[[0.,0.,0.,0.,x,0.,0.]] for x in amounts]
    assert audit.observed_carbon_century(history([1000.,1000.1]),['site'])[0]['actual_100_cycle_change_percent'] is None
    history=history([500.]+[1000.]*100+[1040.])
    r=audit.observed_carbon_century(history,['site'])[0]
    assert r['actual_100_cycle_change_percent']==4. and r['actual_100_cycle_criterion_met']
    assert r['actual_100_cycle_soc_change_percent']==4. and r['actual_100_cycle_soc_criterion_met']


def test_litter_gain_cannot_hide_three_pool_soil_carbon_drift():
    a=endpoints();b=deepcopy(a);b['soil_carbon_gc_m2'][0]+=10.;b['soil_carbon_gc_m2'][4]-=10.
    r=audit.compare_endpoints(a,b)
    assert r['total_soil_carbon_change_gc_m2']==0. and r['total_soil_carbon_rate_criterion_met']
    assert r['soil_organic_carbon_change_gc_m2']==-10. and not r['soc_rate_criterion_met']
    old=[0.,0.,0.,0.,100.,0.,0.];new=[10.,0.,0.,0.,90.,0.,0.]
    history=[[old]]*100+[[new]];r=audit.observed_carbon_century(history,['site'])[0]
    assert r['actual_100_cycle_change_percent']==0. and r['actual_100_cycle_criterion_met']
    assert r['actual_100_cycle_soc_change_percent']==-10. and not r['actual_100_cycle_soc_criterion_met']


def fixture(tmp_path,cycles=2,year=2003):
    inputs=tmp_path/'inputs';inputs.mkdir();begin=datetime(year,1,1);n=int((datetime(year+1,1,1)-begin).total_seconds()/3600)
    starts=np.array([(begin+timedelta(hours=i)).strftime('%Y%m%d%H%M') for i in range(n)])
    ends=np.array([(begin+timedelta(hours=i+1)).strftime('%Y%m%d%H%M') for i in range(n)])
    f=np.tile([3600.,280.,.005,2.,101325.,100.,250.,0.],(n,1));env=np.tile([380.,.1,.5],(n,1))
    np.savez_compressed(inputs/'forcing.npz',forcing=f,environment=env,time_start=starts,time_end=ends)
    geometry=dict(latitude=26.7414,longitude=115.0581,utc_offset_hours=8)
    p=BGCParameters();plant=np.array([120.,600.,150.]);soil=np.array([50.,100.,100.,0.,300.,3000.,3000.])
    column=ColumnState(tsoil=np.full(4,280.),smc=np.full(4,.28),options=PhysicsOptions(OptSnowAlbedo=3,OptSoilTemperatureBottom=1))
    config=dict(site='CN-Qia',scope='Synthetic audited annual transactions, no physical integration',**geometry,column=asdict(column),phenology=None,leaf_dimension_m=.04,leaf_parameters={},
        biogeochemistry=dict(plant_parameters=asdict(p.plant),decomposition_parameters=asdict(p.decomposition),maximum_lai=p.maximum_lai,saturated_potential_mpa=p.saturated_potential_mpa,
            plant_state=dict(carbon=plant,nitrogen=plant/np.array(p.plant.target_cn),reserve_carbon=30.,reserve_nitrogen=.05),
            soil_state=dict(carbon=soil,nitrogen=soil/25.,mineral_nitrogen=1.)))
    native.atomic_json(inputs/'config.json',config);config=json.loads((inputs/'config.json').read_text())
    meta=dict(**geometry,prepared_npz_sha256=native.sha(inputs/'forcing.npz'),archive_name='FLX_CN-Qia_FLUXNET2015_FULLSET.zip')
    native.atomic_json(inputs/'meta.json',meta)
    batch=tmp_path/'batch.json';native.atomic_json(batch,dict(cases=[dict(case='site',input=str(inputs/'forcing.npz'),input_manifest=str(inputs/'meta.json'),configuration=str(inputs/'config.json'))]))
    identity=dict(case='site',forcing_npz_sha256=native.sha(inputs/'forcing.npz'),forcing_manifest_sha256=native.sha(inputs/'meta.json'),configuration_sha256=native.sha(inputs/'config.json'))
    sources=native.source_hashes();manifest=tmp_path/'source.json';native.atomic_json(manifest,sources)
    a=native.pack_clm_bgc_columns([native.build_state(config)],dtype=np.float64,co2_ppm=env[:1,0],cosine_zenith=env[:1,1],diffuse_fraction=env[:1,2],leaf_dimension=np.array([.04]))
    saved={f'a{i}':v for i,v in enumerate(a)};saved.update(index=np.array(0),initial=native.totals(a),net=np.zeros((1,4)),bounds=np.zeros((1,4)),maximum=np.zeros((1,4)),stepmax=np.zeros((1,4)),tracermax=np.zeros((1,5)))
    folders=[];previous=None
    for index in range(cycles):
        folder=tmp_path/('cycle'+str(index));sub=folder/'site';sub.mkdir(parents=True);folders.append(folder)
        link=None if previous is None else dict(mode='cycle',batch_report_sha256=native.sha(previous/'batch.json'),checkpoint_sha256=native.sha(previous/'state-final.npz'),previous_ledger_steps=index*n,previous_segments=index)
        gid=dict(batch_sha256=native.sha(batch),source_sha256=sources,backend='jax64',compare_jax32=False,cases=[identity])
        if link:gid['initial_run']=link
        initial={k:v.copy() for k,v in saved.items()};initial['index']=np.array(0);native.write_npz(folder/'state-0000000-seed.npz',**initial)
        native.write_npz(sub/'empty.npz',time_start=np.array([],dtype='U12'),time_end=np.array([],dtype='U12'))
        base=dict(**identity,configuration=config,source_sha256=sources,source_unchanged=True,batch_column=0,time_convention='local standard time',utc_offset_hours=8,initial_run=link,site='CN-Qia')
        r0=dict(base,status='running',steps_completed=0,model_npz='empty.npz',model_npz_sha256=native.sha(sub/'empty.npz'),batch_checkpoint_sha256=native.sha(folder/'state-0000000-seed.npz'))
        native.atomic_json(sub/'report-0000000-seed.json',r0)
        bounds=np.cumsum(np.vstack((saved['bounds'],np.full((n,4),.001))),axis=0)[1:][-1:]
        saved={k:v.copy() for k,v in saved.items()};saved['index']=np.array(n);saved['bounds']=bounds
        native.write_npz(folder/'state-final.npz',**saved);frame=audit._frame(saved,0)
        native.write_npz(sub/'trace.npz',time_start=starts,time_end=ends,tsoil_k=np.tile(frame['soil_temperature_k'],(n,1)),smc_m3m3=np.tile(frame['soil_water_m3m3'],(n,1)),
            swe_m=np.full(n,frame['swe_m']),step_residual_m_h_c_n=np.zeros((n,4)),step_bound_m_h_c_n=np.full((n,4),.001),cumulative_residual_m_h_c_n=np.zeros((n,4)))
        r=dict(base,status='complete',steps_requested=n,steps_completed=n,ledger_steps=(index+1)*n,ledger_segments=index+1,model_npz='trace.npz',model_npz_sha256=native.sha(sub/'trace.npz'),
            batch_checkpoint_sha256=native.sha(folder/'state-final.npz'),cumulative_numerical_bounds_m_h_c_n=bounds[0].tolist())
        native.atomic_json(sub/'report-final.json',r)
        native.atomic_json(folder/'batch.json',dict(status='complete',source_unchanged=True,steps_requested=n,steps_completed=n,ledger_steps=(index+1)*n,ledger_segments=index+1,identity=gid,
            checkpoint_file='state-final.npz',checkpoint_sha256=native.sha(folder/'state-final.npz'),case_reports={'site':dict(file='report-final.json',sha256=native.sha(sub/'report-final.json'))}))
        previous=folder
    return batch,inputs,folders,manifest


@pytest.mark.parametrize('year',[2003,2004])
def test_two_complete_linked_calendar_cycles_can_meet_water_heat_only(tmp_path,year):
    args=fixture(tmp_path,year=year);r=audit.assess_cycles(*args)
    assert not r['cycles'][0]['cases'][0]['water_heat_converged'] and r['all_columns_water_heat_converged']
    assert r['cycles'][-1]['cases'][0]['consecutive_water_heat_checks']==2
    assert r['soil_carbon_observed_century'][0]['actual_100_cycle_criterion_met'] is None
    assert not r['scientific_acceptance_claim'] and not r['soil_c_long_term_equilibrium_claim'] and not r['nitrogen_equilibrium_claim']


def test_one_full_cycle_is_not_spinup(tmp_path):
    assert not audit.assess_cycles(*fixture(tmp_path,cycles=1))['all_columns_water_heat_converged']


@pytest.mark.parametrize('fault',['running','failed','source','forcing','order','descriptor','continue','link','ledger_reset','carried_alias','cold_seed','trace_time','temperature','zero_duplicate'])
def test_incomplete_or_unbound_cycles_are_rejected(tmp_path,fault):
    batch,inputs,folders,manifest=fixture(tmp_path)
    folder=folders[1];global_path=folder/'batch.json';g=json.loads(global_path.read_text());rpath=folder/'site/report-final.json';r=json.loads(rpath.read_text())
    if fault in ('running','failed'):g['status']=fault
    elif fault=='source':g['identity']['source_sha256']['changed']='foreign'
    elif fault=='forcing':
        with (inputs/'forcing.npz').open('ab') as stream:stream.write(b'corrupt')
    elif fault=='order':g['identity']['cases'][0]['case']='other'
    elif fault=='descriptor':g['case_reports']={}
    elif fault in ('continue','link','ledger_reset'):
        if fault=='continue':g['identity']['initial_run']['mode']='continue'
        elif fault=='link':g['identity']['initial_run']['checkpoint_sha256']='foreign'
        else:g['ledger_steps']=g['steps_completed']
    elif fault in ('carried_alias','cold_seed'):
        target=folder if fault=='carried_alias' else folders[0];p=target/'state-0000000-seed.npz'
        with np.load(p,allow_pickle=False) as z:saved=dict(z)
        saved['a4'][0,0]+=.01;native.write_npz(p,**saved)
        p0=target/'site/report-0000000-seed.json';r0=json.loads(p0.read_text());r0['batch_checkpoint_sha256']=native.sha(p);native.atomic_json(p0,r0)
    elif fault in ('trace_time','temperature'):
        p=folder/'site/trace.npz'
        with np.load(p,allow_pickle=False) as z:trace=dict(z)
        if fault=='trace_time':trace['time_start'][2]='200301010130'
        else:trace['tsoil_k'][-1,0]+=1.
        native.write_npz(p,**trace);r['model_npz_sha256']=native.sha(p);native.atomic_json(rpath,r);g['case_reports']['site']['sha256']=native.sha(rpath)
    else:
        p=folder/'state-0000000-seed.npz';(folder/'state-0000000-other.npz').write_bytes(p.read_bytes())
    native.atomic_json(global_path,g)
    with pytest.raises(ValueError):audit.assess_cycles(batch,inputs,folders,manifest)
