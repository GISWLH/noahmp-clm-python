from copy import deepcopy
from datetime import datetime,timedelta
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'benchmarks'))
from compare_hrldas_column import compare_columns


def example():
    begin=datetime(2001,1,1);rows=[]
    for i in range(24):
        rows.append(dict(time_start=(begin+timedelta(hours=i)).isoformat(),time_end=(begin+timedelta(hours=i+1)).isoformat(),
            et_mm=.001,h_wm2=20.+i,le_wm2=1.+i,runoff_mm=0.,swe_mm=0.,
            soil_temperature_k=[274.15,280.15],soil_total_water_m3m3=[.25,.3]))
    model=dict(status='complete',steps_completed=24,intervals=deepcopy(rows))
    ref=dict(initial_time=begin.isoformat(),timestep_seconds=3600,intervals=rows,time_convention='Test UTC interval ends')
    return model,ref


def test_near_zero_et_and_celsius_state_are_not_diluted_by_kelvin():
    m,r=example()
    for row in m['intervals']:
        row['et_mm']+=.002
        row['soil_temperature_k'][0]+=.6
    report=compare_columns(m,r);d=report['daily']['2001-01-01']
    assert d['et_mm_day']['bias']==pytest.approx(.048)
    assert d['et_mm_day']['near_zero'] and d['et_mm_day']['bias_pass']
    assert d['tsoil_c_layer_1']['reference_mean']==pytest.approx(1.)
    assert d['tsoil_c_layer_1']['absolute_bias']==pytest.approx(.6)
    assert not d['tsoil_c_layer_1']['bias_pass']
    assert report['instantaneous_flux_criteria_met']


@pytest.mark.parametrize('fault',['shift','missing','partial','not_complete'])
def test_misaligned_or_partial_runs_cannot_be_scored_as_complete(fault):
    m,r=example()
    if fault=='shift':m['intervals'][0]['time_end']='2001-01-01T01:30:00'
    if fault=='missing':m['intervals'].pop()
    if fault=='partial':
        m['intervals'].pop();m['steps_completed']-=1;r['intervals'].pop()
    if fault=='not_complete':m['status']='running'
    with pytest.raises(ValueError):compare_columns(m,r)
