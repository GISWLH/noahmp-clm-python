"""Native station histories against retained EHPC NumPy traces, never a CPU forward fallback."""
import json,os
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
import run_fluxnet_column as numpy_runner
import run_fluxnet_native as native_runner
from test_gpu_snicar import assert_science_close


def inputs(tmp_path,structured):
    fixture=Path(__file__).parent/'fixtures/clm_capacity_station_reference.npz'
    with np.load(fixture,allow_pickle=False) as z:
        prefix=str(int(structured))+'/'
        reference={n[len(prefix):]:z[n] for n in z.files if n.startswith(prefix)}
    assert reference
    config=json.loads(str(reference['config_json']));meta=json.loads(str(reference['metadata_json']))
    forcing=tmp_path/'forcing.npz'
    with forcing.open('wb') as stream:np.savez_compressed(stream,**{n[6:]:v for n,v in reference.items() if n.startswith('input/')})
    meta['prepared_npz_sha256']=native_runner.sha(forcing)
    native_runner.atomic_json(tmp_path/'config.json',config);native_runner.atomic_json(tmp_path/'meta.json',meta)
    batch=tmp_path/'batch.json';native_runner.atomic_json(batch,dict(cases=[dict(case='one',input='forcing.npz',input_manifest='meta.json',configuration='config.json')]))
    return batch,reference


def no_cpu_forward(*args,**kwargs):
    raise AssertionError('Native station test must not execute the NumPy physical tendency')


@pytest.mark.parametrize('structured,backend',[(False,'cuda32'),(True,'cuda32'),(True,'jax64')])
def test_capacity_station_all_native_history_and_cross_year_restart_against_ehpc(tmp_path,monkeypatch,structured,backend):
    pytest.importorskip('jax')
    if backend=='cuda32':pytest.importorskip('cupy')
    batch,reference=inputs(tmp_path,structured)
    assert np.all(reference['input/forcing'][:,0]==1800.)  # The explicit per-day factors below are for this HH fixture.
    monkeypatch.setattr(numpy_runner,'step_clm_enthalpy_column',no_cpu_forward)
    options=dict(backend=backend,compare_jax32=backend=='cuda32')
    whole=native_runner.run(batch,tmp_path/'whole',**options)
    part=native_runner.run(batch,tmp_path/'split',stop_after=2,**options)
    assert part['status']=='running' and part['steps_completed']==2
    final=native_runner.run(batch,tmp_path/'split',resume=True,**options)
    assert whole['status']==final['status']=='complete' and final['steps_completed']==4
    count=23 if structured else 20
    assert final['identity']['native_state_arrays']==count
    _,a,rows=native_runner.read_generation(tmp_path/'whole')
    _,b,other=native_runner.read_generation(tmp_path/'split')
    for key in a:
        assert a[key].dtype==b[key].dtype
        np.testing.assert_array_equal(a[key],b[key],err_msg=key)
    trace=rows[0][1]
    for key in trace:np.testing.assert_array_equal(trace[key],other[0][1][key],err_msg=key)
    arrays=native_runner.checkpoint_arrays(a,whole['identity'])
    np.testing.assert_array_equal(arrays[-3],[[4,4]])
    np.testing.assert_array_equal(arrays[-4][0,:4],[282.,284.,286.,288.])
    if backend=='cuda32':
        shadow=native_runner.checkpoint_arrays(a,whole['identity'],shadow=True)
        for index,(x,y) in enumerate(zip(arrays,shadow)):
            native_runner.compare_same_precision({str(index):x},{str(index):y},'final_history')
    maximum={}
    for name,scale,absolute,floor,offset in [
        ('h_wm2',1.,1.,10.,0.),('le_wm2',1.,1.,10.,0.),
        ('gpp_gc_m2',48.,.1,1.,0.),('nee_gc_m2',48.,.1,1.,0.),
        ('et_mm',48.,.05,.5,0.),('runoff_mm',48.,.05,.5,0.),
        ('tsoil_k',1.,.5,5.,273.15),('smc_m3m3',1.,.002,.02,0.),('swe_m',1.,.0001,.001,0.)]:
        expected=(reference['trace/'+name].astype(float)-offset)*scale
        actual=(trace[name].astype(float)-offset)*scale
        assert_science_close(actual,expected,relative=.1,absolute=absolute,zero_floor=floor)
        maximum[name]=float(abs(actual-expected).max(initial=0.))
    np.testing.assert_allclose(trace['capacity_mean_temperature_used_k'],reference['trace/capacity_mean_temperature_used_k'],rtol=0.,atol=.001)
    np.testing.assert_allclose(trace['capacity_next_mean_temperature_k'],reference['trace/capacity_next_mean_temperature_k'],rtol=0.,atol=.001)
    saved=dict(scope='Actual native station4 intervals spanning calendar year; all original step/cumulative MHCN and optional independently carried JAX32 shadow enforced by runner. Full trace/checkpoint split/restart exact. Not observed multi-year/spinup/FLUXNET acceptance.',
        structured=structured,backend=backend,maximum_reference_absolute_error=maximum,split_restart_max_absolute=0.,
        same_precision_max_absolute=final['same_precision_max_absolute'],
        maximum_cumulative_m_h_c_n=final['maximum_cumulative_residual_m_h_c_n'].tolist(),
        all_state_array_count=count)
    if os.environ.get('NOAHMP_CAPACITY_STATION_DIAGNOSTICS'):
        folder=Path(os.environ['NOAHMP_CAPACITY_STATION_DIAGNOSTICS']);folder.mkdir(exist_ok=True)
        prefix=backend+'-'+str(int(structured))
        native_runner.atomic_json(folder/(prefix+'.json'),saved)
        native_runner.write_npz(folder/(prefix+'.npz'),**{**{'trace/'+k:v for k,v in trace.items()},**a})
    print('Native capacity station',json.dumps(saved),flush=True)
