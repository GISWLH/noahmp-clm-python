from dataclasses import asdict
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'benchmarks'))
from fluxnet_soil_operator import depth_weights, soil_products, initial_soil_from_manifest
from evaluate_fluxnet_series import evaluate
from test_fluxnet_evaluation import example


def soil_example():
    model,obs=example();n=len(model['time_start'])
    dz=np.array([.1,.3,.6]);z=np.cumsum(dz)-.5*dz
    initial={'dz':dz,'tsoil_k':283.15+10*z,'smc_m3m3':.2+.1*z,'liquid_m3m3':.15+.1*z}
    for key,slope in [('tsoil_k',.1),('smc_m3m3',.0001),('liquid_m3m3',.0001)]:
        model[key]=initial[key][None,:]+np.arange(1,n+1)[:,None]*slope
    operator={'source':'Analytic linear profiles for independent test',
        'temperature_depth_m':.15,'water_depth_m':.05,
        'spatial_method':'linear_centres_constant_outer_half_cell','temporal_method':'trapezoid_endpoints'}
    obs['obs_TS_F_MDS_1']=10+10*.15+(np.arange(n)+.5)*.1
    obs['obs_SWC_F_MDS_1']=100*(.15+.1*.05+(np.arange(n)+.5)*.0001)
    obs['qc_TS_F_MDS_1']=np.zeros(n);obs['qc_SWC_F_MDS_1']=np.zeros(n)
    return model,obs,initial,operator


def test_linear_depth_and_interval_mean_use_the_initial_state_and_correct_units():
    m,o,i,p=soil_example();o['qc_TS_F_MDS_1'][:2]=1;o['qc_SWC_F_MDS_1'][2]=np.nan
    r=evaluate(m,o,utc_offset_hours=8,soil_operator=p,initial_soil=i)
    d=r['daily']['2004-02-01'];meta=r['soil_observation_operator']
    assert r['instantaneous']['TS_F_MDS_1']['metrics']['n']==46
    assert r['instantaneous']['SWC_F_MDS_1_liquid']['metrics']['n']==47
    assert d['TS_F_MDS_1']['metrics']['absolute_bias']<1e-12
    assert d['TS_F_MDS_1']['metrics']['model_mean']==pytest.approx(13.9)
    assert d['SWC_F_MDS_1_liquid']['metrics']['absolute_bias']<1e-14
    assert d['SWC_F_MDS_1_total']['metrics']['bias']==pytest.approx(.05)
    assert meta['maximum_liquid_total_difference_m3m3']==pytest.approx(.05)
    np.testing.assert_allclose(meta['temperature_weights'],[.5,.5,0.])
    assert 'model_integral' not in d['TS_F_MDS_1']
    assert r['monthly']['2004-02']['TS_F_MDS_1']['metrics']['status']=='unmeasured'


def test_soil_missing_records_do_not_turn_into_complete_daily_means():
    m,o,i,p=soil_example();o['obs_TS_F_MDS_1'][12]=np.nan
    r=evaluate(m,o,utc_offset_hours=8,soil_operator=p,initial_soil=i)
    assert r['daily']['2004-02-01']['TS_F_MDS_1']['metrics']['status']=='unmeasured'
    assert r['daily']['2004-02-01']['SWC_F_MDS_1_liquid']['metrics']['status']=='measured'


@pytest.mark.parametrize('fault',['depth','spatial','temporal','initial','shape','nan'])
def test_invalid_soil_operators_are_rejected(fault):
    m,o,i,p=soil_example()
    if fault=='depth':p['water_depth_m']=2.
    if fault=='spatial':p['spatial_method']='pick_best_layer'
    if fault=='temporal':p['temporal_method']='endpoints_as_observed_means'
    if fault=='initial':i=None
    if fault=='shape':m['tsoil_k']=m['tsoil_k'][:,:2]
    if fault=='nan':m['liquid_m3m3'][1,0]=np.nan
    with pytest.raises(ValueError):evaluate(m,o,utc_offset_hours=8,soil_operator=p,initial_soil=i)


def test_outer_half_cells_are_constant_without_extrapolating_past_soil_domain():
    np.testing.assert_array_equal(depth_weights([.1,.3,.6],0.),[1.,0.,0.])
    np.testing.assert_array_equal(depth_weights([.1,.3,.6],1.),[0.,0.,1.])
    with pytest.raises(ValueError):depth_weights([.1,.3,.6],1.001)
    with pytest.raises(ValueError):depth_weights([.1,0.,.6],.1)


def test_near_freezing_temperature_bias_is_in_celsius():
    m,o,i,p=soil_example();i['tsoil_k'][:]=273.55;m['tsoil_k'][:]=273.55;o['obs_TS_F_MDS_1'][:]=0.
    d=evaluate(m,o,utc_offset_hours=8,soil_operator=p,initial_soil=i)['daily']['2004-02-01']['TS_F_MDS_1']['metrics']
    assert d['near_zero'] and d['bias_pass'] and d['relative_bias'] is None
    assert d['absolute_bias']==pytest.approx(.4)


def test_initial_soil_uses_extensive_state_including_frozen_liquid_water():
    from noahmp_clm.column import ColumnState
    from noahmp_clm.physics.soil_enthalpy import SoilEnthalpyState
    original=SoilEnthalpyState.from_column(ColumnState(tsoil=np.full(4,270.)))
    actual=initial_soil_from_manifest({'initial_state':{'ground':asdict(original)}})
    np.testing.assert_allclose(actual['tsoil_k'],270.)
    np.testing.assert_allclose(actual['liquid_m3m3'],original.liquid/(1000*original.dz))
    assert np.all(actual['liquid_m3m3']<actual['smc_m3m3'])


def test_historical_enthalpy_without_suction_cannot_be_silently_reinterpreted():
    from noahmp_clm.column import ColumnState
    from noahmp_clm.physics.soil_enthalpy import SoilEnthalpyState
    ground=asdict(SoilEnthalpyState.from_column(ColumnState(tsoil=np.full(4,270.))))
    del ground['params']['saturated_suction_m']
    with pytest.raises(ValueError,match='original frozen source'):
        initial_soil_from_manifest({'initial_state':{'ground':ground}})

@pytest.mark.parametrize('fault',[None,'site','length','source_changed'])
def test_soil_cli_binds_complete_trace_operator_and_instrument_hashes(tmp_path,fault):
    import hashlib,json,subprocess
    from noahmp_clm.column import SoilParams
    from noahmp_clm.physics.soil_enthalpy import SoilEnthalpyState,enthalpy_and_capacity
    m,o,i,p=soil_example();p['site']='TEST'
    params=SoilParams();w=i['smc_m3m3']*1000*i['dz']
    h=enthalpy_and_capacity(w,i['tsoil_k'],i['dz'],params)[0]
    ground=asdict(SoilEnthalpyState(w,h,i['dz'],params))
    np.savez(tmp_path/'m.npz',**m);np.savez(tmp_path/'o.npz',**o)
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    mm={'status':'complete','site':'TEST','time_convention':'local standard time','utc_offset_hours':8,
        'model_npz_sha256':sha(tmp_path/'m.npz'),'forcing_npz_sha256':sha(tmp_path/'o.npz'),
        'steps_completed':48,'steps_requested':48,'source_unchanged':True,'initial_state':{'ground':ground}}
    om={'prepared_npz_sha256':sha(tmp_path/'o.npz'),'utc_offset_hours':8}
    if fault=='site':p['site']='OTHER'
    if fault=='length':mm['steps_completed']=47
    if fault=='source_changed':mm['source_unchanged']=False
    for name,obj in [('m.json',mm),('o.json',om),('operator.json',p)]:
        (tmp_path/name).write_text(json.dumps(obj,default=lambda x:x.tolist()),encoding='utf-8')
    root=Path(__file__).parents[1]
    command=[sys.executable,str(root/'benchmarks/evaluate_fluxnet_series.py')]
    for flag,name in [('model','m.npz'),('model-manifest','m.json'),('observations','o.npz'),
                      ('observation-manifest','o.json'),('soil-operator','operator.json'),('output','result.json')]:
        command.extend(['--'+flag,str(tmp_path/name)])
    completed=subprocess.run(command,text=True,capture_output=True,cwd=root)
    if fault:
        assert completed.returncode!=0 and 'ValueError' in completed.stderr
        assert not (tmp_path/'result.json').exists()
    else:
        assert completed.returncode==0,completed.stderr
        result=json.loads((tmp_path/'result.json').read_text())
        assert result['source_sha256']['soil_instrument']==sha(root/'benchmarks/fluxnet_soil_operator.py')
        assert result['source_sha256']['soil_operator']==sha(tmp_path/'operator.json')
        assert result['daily']['2004-02-01']['TS_F_MDS_1']['metrics']['absolute_bias']<1e-12
