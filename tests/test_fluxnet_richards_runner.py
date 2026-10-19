"""Explicit hydrology survives continuous station runs and restart identity."""
import copy,json,os,pickle
from pathlib import Path
import numpy as np
import pytest
from test_fluxnet_capacity_runner import station_inputs
import run_fluxnet_column as numpy_runner
import run_fluxnet_native as native_runner
from test_fluxnet_native_runner import batch_fixture


def closed_configuration(config):
    config['soil_hydrology']=dict(method='richards_zero_flux',parameters={})
    config['column']['options']['OptRunoffSubsurface']=3
    return config


@pytest.mark.parametrize('structured',[False,True])
def test_actual_closed_station_continuity_and_configuration_bound_restart(tmp_path,monkeypatch,structured):
    paths=station_inputs(tmp_path,structured,structured)
    config=closed_configuration(json.loads(paths[2].read_text()))
    if not structured:del config['canopy_capacity']
    config['column']['smc']=[.38,.18,.28,.12]
    config['scope']='Synthetic actual zero-flux Richards station/checkpoint; not observed acceptance'
    numpy_runner.atomic_json(paths[2],config)
    run=numpy_runner.step_clm_enthalpy_column;transfers=[]
    def actual(*a,**kw):
        assert kw['soil_hydrology']==numpy_runner.hydrology_configuration(config)
        result,d=run(*a,**kw)
        hydro=d['soil_hydrology']
        assert hydro['method']=='richards_zero_flux' and hydro['lower_boundary_condition']==2
        transfers.append(float(abs(hydro['interface_transfer_kg_m2']).sum()))
        assert d['runoff_deep_mm']==0.
        return result,d
    monkeypatch.setattr(numpy_runner,'step_clm_enthalpy_column',actual)
    whole=numpy_runner.run(*paths,tmp_path/'whole')
    partial=numpy_runner.run(*paths,tmp_path/'split',stop_after=2)
    assert partial['status']=='running' and partial['steps_completed']==2
    final=numpy_runner.run(*paths,tmp_path/'split',resume=True)
    assert whole['status']==final['status']=='complete' and final['steps_completed']==4
    assert len(transfers)==8 and min(transfers)>0.
    with np.load(tmp_path/'whole'/whole['model_npz']) as a,np.load(tmp_path/'split'/final['model_npz']) as b:
        assert set(a.files)==set(b.files)
        for key in a.files:np.testing.assert_array_equal(a[key],b[key],err_msg=key)
        if os.environ.get('NOAHMP_RICHARDS_STATION_REFERENCE'):
            folder=Path(os.environ['NOAHMP_RICHARDS_STATION_REFERENCE']);folder.mkdir(exist_ok=True)
            with np.load(paths[0]) as inputs:
                artifact={**{'input/'+k:inputs[k] for k in inputs.files},**{'trace/'+k:a[k] for k in a.files},
                    'config_json':np.asarray(paths[2].read_text()),'metadata_json':np.asarray(paths[1].read_text()),
                    'report_json':np.asarray(json.dumps(whole,default=numpy_runner.encode))}
                np.savez_compressed(folder/('case'+str(int(structured))+'.npz'),**artifact)
    a=pickle.loads((tmp_path/'whole'/whole['checkpoint_file']).read_bytes())
    b=pickle.loads((tmp_path/'split'/final['checkpoint_file']).read_bytes())
    for key in ('water','enthalpy'):np.testing.assert_array_equal(getattr(a[0].ground,key),getattr(b[0].ground,key))
    for x,y in zip(a[1:],b[1:]):np.testing.assert_array_equal(x,y)
    former=paths[2].read_bytes()
    config['soil_hydrology']['parameters']['maximum_liquid_fraction']=.04
    numpy_runner.atomic_json(paths[2],config)
    with pytest.raises(ValueError,match='Resume identity mismatch'):
        numpy_runner.run(*paths,tmp_path/'split',resume=True)
    paths[2].write_bytes(former)
    print('Actual Richards station',json.dumps(dict(structured=structured,split_trace_max_absolute=0.,
        minimum_actual_face_transfer_kg_m2=min(transfers),maximum_cumulative_m_h_c_n=final['maximum_cumulative_residual_m_h_c_n'],
        maximum_step_m_h_c_n=final['maximum_step_residual_m_h_c_n']),default=numpy_runner.encode),flush=True)


@pytest.mark.parametrize('change',['type','method','missing_parameters','unknown_outer','unknown_control','bool','nonfinite','fraction','integer','outlet'])
def test_explicit_hydrology_configuration_rejects_ambiguous_controls(change):
    config=closed_configuration(dict(column=dict(options={})));v=config['soil_hydrology']
    if change=='type':config['soil_hydrology']=[]
    if change=='method':v['method']='richards_aquifer'
    if change=='missing_parameters':del v['parameters']
    if change=='unknown_outer':v['fit_runoff']=True
    if change=='unknown_control':v['parameters']['bottom_drainage']=0.
    if change=='bool':v['parameters']['maximum_liquid_fraction']=True
    if change=='nonfinite':v['parameters']['ice_impedance_exponent']=float('nan')
    if change=='fraction':v['parameters']['maximum_liquid_fraction']=0.
    if change=='integer':v['parameters']['max_substeps']=100.
    if change=='outlet':config['column']['options']['OptRunoffSubsurface']=6
    with pytest.raises(ValueError):numpy_runner.hydrology_configuration(config)


def test_legacy_absent_and_null_keep_original_hydrology():
    assert numpy_runner.hydrology_configuration({}) is None
    assert numpy_runner.hydrology_configuration(dict(soil_hydrology=None)) is None


@pytest.mark.parametrize('change',['method','controls'])
def test_native_batches_do_not_silently_share_different_hydrology(tmp_path,change):
    path,row,config=batch_fixture(tmp_path)
    config=closed_configuration(config|dict(column=dict(options={})))
    other=copy.deepcopy(config)
    if change=='method':other['soil_hydrology']=None
    else:other['soil_hydrology']['parameters']['maximum_liquid_fraction']=.04
    numpy_runner.atomic_json(tmp_path/'config.json',config)
    numpy_runner.atomic_json(tmp_path/'second.json',other)
    numpy_runner.atomic_json(path,dict(cases=[row,dict(row,case='second',configuration='second.json')]))
    with pytest.raises(ValueError,match='differing soil hydrology'):native_runner.load_batch(path)
    numpy_runner.atomic_json(tmp_path/'second.json',config)
    cases=native_runner.load_batch(path)
    assert native_runner.batch_hydrology(cases)==numpy_runner.hydrology_configuration(config)
