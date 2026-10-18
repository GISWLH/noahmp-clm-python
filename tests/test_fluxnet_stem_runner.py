"""Actual NumPy trajectory continuity and strict native geometry persistence."""
from dataclasses import asdict
import copy,json,pickle
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
import run_fluxnet_column as numpy_runner
import run_fluxnet_native as native_runner
from test_fluxnet_native_runner import committed_fixture


def configuration(scheme='CN_NONCROP'):
    from test_clm_bgc_column import case
    from noahmp_clm.column import ColumnState
    state,f,_=case(None,285.)
    c=ColumnState(tsoil=state.tsoil,smc=state.smc,dz=state.ground.dz,soil=state.ground.params,
        lai=state.lai,fveg=state.fveg,ts=285.,canw=0.,root_frac=state.root_frac,options=state.options)
    p=state.bgc_parameters
    config=dict(site='CN-Qia',case='structured-runner-test',scope='Synthetic forcing, actual NumPy physics/checkpoint test only',
        latitude=26.7414,longitude=115.0581,utc_offset_hours=8,column=asdict(c),phenology=None,
        leaf_parameters={},leaf_dimension_m=.04,most_geometry={},aerosol_deposition_kg_m2_s=[0.]*5,
        nitrogen_deposition_gn_m2_s=1e-7,biogeochemistry=dict(plant_state=asdict(state.bgc_state.plant),
            soil_state=asdict(state.bgc_state.soil),plant_parameters=asdict(p.plant),decomposition_parameters=asdict(p.decomposition),
            maximum_lai=p.maximum_lai,saturated_potential_mpa=p.saturated_potential_mpa),
        canopy_structure=dict(parameters=dict(scheme=scheme,canopy_top_m=3.,canopy_bottom_m=.5),initial_sai=.8))
    return config,f


@pytest.mark.parametrize('scheme',['PRESCRIBED','CN_NONCROP'])
def test_actual_numpy_station_split_retains_all_structure_state_and_diagnostics(tmp_path,scheme):
    config,f=configuration(scheme);config_path=tmp_path/'config.json';numpy_runner.atomic_json(config_path,config)
    data=tmp_path/'forcing.npz';np.savez_compressed(data,
        time_start=np.asarray(['200301010000','200301010030','200301010100','200301010130']),
        time_end=np.asarray(['200301010030','200301010100','200301010130','200301010200']),
        forcing=np.asarray([list(vars(f).values())]*4),environment=np.asarray([[380.,.5,.5]]*4))
    meta=tmp_path/'meta.json';numpy_runner.atomic_json(meta,dict(site=config['site'],prepared_npz_sha256=numpy_runner.sha(data),
        latitude=config['latitude'],longitude=config['longitude'],utc_offset_hours=config['utc_offset_hours']))
    continuous=numpy_runner.run(data,meta,config_path,tmp_path/'continuous')
    partial=numpy_runner.run(data,meta,config_path,tmp_path/'split',stop_after=2)
    assert partial['status']=='running' and partial['steps_completed']==2
    final=numpy_runner.run(data,meta,config_path,tmp_path/'split',resume=True)
    assert continuous['status']==final['status']=='complete' and final['steps_completed']==4
    with np.load(tmp_path/'continuous'/continuous['model_npz']) as a,np.load(tmp_path/'split'/final['model_npz']) as b:
        assert set(a.files)==set(b.files) and set(numpy_runner.STRUCTURE_TRACE)<=set(a.files)
        for key in a.files:np.testing.assert_array_equal(a[key],b[key],err_msg=key)
    a=pickle.loads((tmp_path/'continuous'/continuous['checkpoint_file']).read_bytes())
    b=pickle.loads((tmp_path/'split'/final['checkpoint_file']).read_bytes())
    assert a[0].sai==b[0].sai and a[0].canopy_structure_parameters==b[0].canopy_structure_parameters
    for key in ['water','enthalpy']:np.testing.assert_array_equal(getattr(a[0].ground,key),getattr(b[0].ground,key))
    for x,y in zip(a[1:],b[1:]):np.testing.assert_array_equal(x,y)
    print('NumPy actual structured station',json.dumps(dict(scheme=scheme,split_trace_maximum_absolute=0.,
        sai_after=a[0].sai,maximum_cumulative_m_h_c_n=final['maximum_cumulative_residual_m_h_c_n'],
        maximum_step_m_h_c_n=final['maximum_step_residual_m_h_c_n']),default=numpy_runner.encode),flush=True)


@pytest.mark.parametrize('change',['missing_sai','negative_sai','nan_sai','scheme','unknown_key'])
def test_station_structure_requires_explicit_valid_initial_geometry(change):
    config,_=configuration()
    value=config['canopy_structure']
    if change=='missing_sai':del value['initial_sai']
    if change=='negative_sai':value['initial_sai']=-1.
    if change=='nan_sai':value['initial_sai']=float('nan')
    if change=='scheme':value['parameters']['scheme']='unknown'
    if change=='unknown_key':value['fitted_sai']=1.
    with pytest.raises(ValueError):numpy_runner.structure_configuration(config)


@pytest.mark.parametrize('missing',[16,17,18])
def test_native_checkpoint_cannot_drop_any_structure_array(tmp_path,missing):
    folder,_,_,_=committed_fixture(tmp_path);path=folder/'batch.json';report=json.loads(path.read_text())
    report['identity']['native_state_arrays']=19
    with np.load(folder/'state.npz') as archive:saved=dict(archive)
    for k,shape in [(16,(1,1)),(17,(1,2)),(18,(1,2))]:saved['a'+str(k)]=np.ones(shape,np.int32 if k==18 else float)
    del saved['a'+str(missing)]
    native_runner.write_npz(folder/'state.npz',**saved);report['checkpoint_sha256']=native_runner.sha(folder/'state.npz')
    native_runner.atomic_json(path,report)
    with pytest.raises(ValueError,match='every declared'):native_runner.read_generation(folder)


def test_native_shadow_structure_is_independent_and_requires_all_arrays():
    identity=dict(native_state_arrays=19)
    saved={prefix+str(k):np.full((1,1),k) for prefix in ('a','shadow') for k in range(19)}
    saved['shadow16']=np.array([[.4]]);saved['a16']=np.array([[.8]])
    assert native_runner.checkpoint_arrays(saved,identity)[16][0,0]==.8
    assert native_runner.checkpoint_arrays(saved,identity,shadow=True)[16][0,0]==.4
    del saved['shadow18']
    with pytest.raises(ValueError,match='every declared'):native_runner.checkpoint_arrays(saved,identity,shadow=True)


def test_native_leaf_only_metadata_cannot_hide_extra_geometry():
    saved={f'a{k}':np.zeros((1,1)) for k in range(19)}
    with pytest.raises(ValueError,match='every declared'):native_runner.checkpoint_arrays(saved,{})


def test_native_batch_cannot_mix_undeclared_and_retained_structure():
    config,_=configuration();plain=copy.deepcopy(config);del plain['canopy_structure']
    assert native_runner.native_state_count([dict(config=config)])==19
    assert native_runner.native_state_count([dict(config=plain)])==16
    with pytest.raises(ValueError,match='Split structured'):native_runner.native_state_count([dict(config=config),dict(config=plain)])
