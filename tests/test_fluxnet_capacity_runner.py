"""Actual station continuity, causal acclimation and complete native history."""
from dataclasses import asdict
from datetime import datetime
import copy,json,os,pickle
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
import run_fluxnet_column as numpy_runner
import run_fluxnet_native as native_runner
from test_fluxnet_stem_runner import configuration as stem_configuration
from test_fluxnet_native_runner import batch_fixture


def configuration(structured=False,seasonal=False):
    config,forcing=stem_configuration()
    config['case']='capacity-station-test'
    config['scope']='Synthetic actual station physics/restart test; not observation acceptance'
    if not structured:del config['canopy_structure']
    if seasonal:
        from test_clm_bgc_column import case
        state,_,_=case()
        config['column']['options']['OptDynamicVeg']=5
        config['phenology']=dict(parameters=asdict(state.phenology_parameters),initial=asdict(state.phenology_state))
    config['canopy_capacity']=dict(parameters=dict(specific_leaf_area=.01,leaf_cn=58.,rubisco_nitrogen_fraction=.0509),
        time_step_seconds=forcing.dt,initial_temperature_k=293.15)
    return config,forcing


def station_inputs(tmp_path,structured=False,seasonal=False):
    config,forcing=configuration(structured,seasonal)
    config_path=tmp_path/'config.json';numpy_runner.atomic_json(config_path,config)
    data=tmp_path/'forcing.npz';f=np.asarray([list(vars(forcing).values())]*4)
    f[:,1]=[282.,284.,286.,288.]
    np.savez_compressed(data,time_start=np.asarray(['200312312300','200312312330','200401010000','200401010030']),
        time_end=np.asarray(['200312312330','200401010000','200401010030','200401010100']),
        forcing=f,environment=np.asarray([[380.,.5,.5]]*4))
    meta=tmp_path/'meta.json';numpy_runner.atomic_json(meta,dict(site=config['site'],prepared_npz_sha256=numpy_runner.sha(data),
        latitude=config['latitude'],longitude=config['longitude'],utc_offset_hours=config['utc_offset_hours']))
    return data,meta,config_path


@pytest.mark.parametrize('structured,seasonal',[(False,False),(True,True)])
def test_actual_numpy_capacity_station_carries_history_across_restart_and_year(tmp_path,structured,seasonal):
    paths=station_inputs(tmp_path,structured,seasonal)
    whole=numpy_runner.run(*paths,tmp_path/'whole')
    part=numpy_runner.run(*paths,tmp_path/'split',stop_after=2)
    assert part['status']=='running' and part['steps_completed']==2
    final=numpy_runner.run(*paths,tmp_path/'split',resume=True)
    assert whole['status']==final['status']=='complete' and final['steps_completed']==4
    with np.load(tmp_path/'whole'/whole['model_npz']) as a,np.load(tmp_path/'split'/final['model_npz']) as b:
        assert set(a.files)==set(b.files) and set(numpy_runner.CAPACITY_TRACE)<=set(a.files)
        for key in a.files:np.testing.assert_array_equal(a[key],b[key],err_msg=key)
        np.testing.assert_array_equal(a['capacity_mean_temperature_used_k'],[293.15,282.,283.,284.])
        np.testing.assert_array_equal(a['capacity_next_mean_temperature_k'],[282.,283.,284.,285.])
        np.testing.assert_array_equal(a['capacity_temperature_count'],[1,2,3,4])
        np.testing.assert_array_equal(a['capacity_temperature_cursor'],[1,2,3,4])
        assert a['capacity_daylength_factor'][2]!=a['capacity_daylength_factor'][1]
    a=pickle.loads((tmp_path/'whole'/whole['checkpoint_file']).read_bytes())
    b=pickle.loads((tmp_path/'split'/final['checkpoint_file']).read_bytes())
    for x,y in zip(a[1:],b[1:]):np.testing.assert_array_equal(x,y)
    for key in ('water','enthalpy'):np.testing.assert_array_equal(getattr(a[0].ground,key),getattr(b[0].ground,key))
    memory=a[0].canopy_capacity_state
    np.testing.assert_array_equal(memory.values,b[0].canopy_capacity_state.values)
    np.testing.assert_array_equal(memory.values[:4],[282.,284.,286.,288.])
    assert memory.count==memory.cursor==4 and not memory.values.flags.writeable
    assert a[0].canopy_capacity_parameters==b[0].canopy_capacity_parameters
    if seasonal:assert a[0].options.OptDynamicVeg==b[0].options.OptDynamicVeg==5 and a[0].phenology_state==b[0].phenology_state
    if os.environ.get('NOAHMP_CAPACITY_STATION_REFERENCE'):
        folder=Path(os.environ['NOAHMP_CAPACITY_STATION_REFERENCE']);folder.mkdir(exist_ok=True)
        with np.load(paths[0]) as inputs,np.load(tmp_path/'whole'/whole['model_npz']) as trace:
            artifact={**{'input/'+k:inputs[k] for k in inputs.files},**{'trace/'+k:trace[k] for k in trace.files},
                'config_json':np.asarray(paths[2].read_text()),'metadata_json':np.asarray(paths[1].read_text()),
                'report_json':np.asarray(json.dumps(whole,default=numpy_runner.encode))}
            np.savez_compressed(folder/('case'+str(int(structured))+'.npz'),**artifact)
    print('Actual NumPy capacity station',json.dumps(dict(structured=structured,seasonal=seasonal,
        split_trace_max_absolute=0.,mean_temperature_used_k=[293.15,282.,283.,284.],
        maximum_cumulative_m_h_c_n=final['maximum_cumulative_residual_m_h_c_n'],
        maximum_step_m_h_c_n=final['maximum_step_residual_m_h_c_n']),default=numpy_runner.encode),flush=True)


def test_capacity_forcing_interval_disagreement_fails_before_cold_run(tmp_path):
    paths=station_inputs(tmp_path);config=json.loads(paths[2].read_text())
    config['canopy_capacity']['time_step_seconds']=3600.;numpy_runner.atomic_json(paths[2],config)
    with pytest.raises(ValueError,match='interval'):numpy_runner.run(*paths,tmp_path/'bad')
    assert not (tmp_path/'bad').exists()


@pytest.mark.parametrize('change',['missing_initial','unknown_outer','missing_trait','unknown_trait','bool_interval','bad_interval','bool_trait'])
def test_capacity_station_requires_explicit_valid_configuration(change):
    config,_=configuration();value=config['canopy_capacity']
    if change=='missing_initial':del value['initial_temperature_k']
    if change=='unknown_outer':value['fit_to_observations']=True
    if change=='missing_trait':del value['parameters']['leaf_cn']
    if change=='unknown_trait':value['parameters']['vcmax_fit']=30.
    if change=='bool_interval':value['time_step_seconds']=True
    if change=='bad_interval':value['time_step_seconds']=1801.
    if change=='bool_trait':value['parameters']['leaf_cn']=True
    with pytest.raises(ValueError):numpy_runner.capacity_configuration(config)


def test_capacity_photoperiod_uses_full_year_geometry_without_future_weather():
    meta=dict(latitude=45.,longitude=0.,utc_offset_hours=0.)
    winter=numpy_runner.capacity_environment(datetime(2004,1,1),meta,True)
    summer=numpy_runner.capacity_environment(datetime(2004,6,21),meta,True)
    assert winter.maximum_daylength_seconds==summer.maximum_daylength_seconds
    assert 30000.<winter.daylength_seconds<33000.<summer.daylength_seconds<=summer.maximum_daylength_seconds<57000.
    assert numpy_runner.capacity_environment(datetime(2004,1,1),meta,False) is None


@pytest.mark.parametrize('count',[20,23])
@pytest.mark.parametrize('offset',range(4))
@pytest.mark.parametrize('shadow',[False,True])
def test_native_checkpoint_cannot_drop_any_capacity_history_array(count,offset,shadow):
    saved={prefix+str(k):np.ones((1,1)) for prefix in ('a','shadow') for k in range(count)}
    identity=dict(native_state_arrays=count)
    assert len(native_runner.checkpoint_arrays(saved,identity,shadow=shadow))==count
    del saved[('shadow' if shadow else 'a')+str(count-4+offset)]
    with pytest.raises(ValueError,match='every declared'):native_runner.checkpoint_arrays(saved,identity,shadow=shadow)


def test_native_capacity_batch_retains_distinct_column_gas_templates(tmp_path):
    path,row,config=batch_fixture(tmp_path);capacity,_=configuration()
    config['canopy_capacity']=capacity['canopy_capacity'];numpy_runner.atomic_json(tmp_path/'config.json',config)
    second=copy.deepcopy(config);second['leaf_parameters']={'ball_berry_slope':4.4713363873}
    numpy_runner.atomic_json(tmp_path/'second.json',second)
    other=dict(row,case='second',configuration='second.json')
    numpy_runner.atomic_json(path,dict(cases=[row,other]))
    loaded=native_runner.load_batch(path)
    assert native_runner.native_state_count(loaded)==20
    assert loaded[0]['config']['leaf_parameters']!=loaded[1]['config']['leaf_parameters']
    del second['canopy_capacity'];numpy_runner.atomic_json(tmp_path/'second.json',second)
    with pytest.raises(ValueError,match='Split capacity'):native_runner.load_batch(path)
