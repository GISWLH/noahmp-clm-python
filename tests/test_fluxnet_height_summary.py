from copy import deepcopy
from datetime import datetime,timedelta
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'benchmarks'))
from summarize_fluxnet_height_sensitivity import compare,same_nonheight_configuration


def data():
    t=datetime(2003,7,1);a={
        'time_start':np.array([(t+timedelta(minutes=30*k)).strftime('%Y%m%d%H%M') for k in range(48)]),
        'time_end':np.array([(t+timedelta(minutes=30*(k+1))).strftime('%Y%m%d%H%M') for k in range(48)])}
    for k,v in {'h_wm2':20.,'le_wm2':0.,'et_mm':.01,'runoff_mm':0.,'gpp_gc_m2':.1,'nee_gc_m2':0.,'swe_m':0.,'lai':2.}.items():a[k]=np.full(48,v)
    a['tsoil_k']=np.full((48,4),280.15);a['smc_m3m3']=np.full((48,4),.3)
    return a


def test_complete_daily_height_response_and_near_zero_no_division():
    b=data();a=deepcopy(b);a['h_wm2']+=2.;a['nee_gc_m2']+=.001;a['tsoil_k']+=.2
    r=compare(a,b)['2003-07-01']
    assert r['h_wm2']['relative_bias']==pytest.approx(.1)
    assert r['h_wm2']['model_integral']==pytest.approx(22.*86400)
    assert r['et_mm']['model_integral']==pytest.approx(.48)
    assert r['gpp_gc_m2']['model_integral']==pytest.approx(4.8)
    assert r['nee_gc_m2']['near_zero'] and r['nee_gc_m2']['relative_bias'] is None
    assert r['nee_gc_m2']['bias']==pytest.approx(.048)
    assert r['tsoil_k'][0]['bias']==pytest.approx(.2)
    assert 'bias_pass' not in r['h_wm2']


@pytest.mark.parametrize('fault',['partial','gap','shift','nonfinite'])
def test_incomplete_or_incompatible_height_traces_are_rejected(fault):
    b=data();a=deepcopy(b)
    if fault=='partial':
        for x in (a,b):
            for k in x:x[k]=x[k][:-1]
    if fault=='gap':
        for x in (a,b):
            for k in x:x[k]=np.delete(x[k],5,axis=0)
    if fault=='shift':a['time_start'][0]='200307010030'
    if fault=='nonfinite':a['h_wm2'][5]=np.nan
    with pytest.raises(ValueError):compare(a,b)


def test_only_measurement_heights_may_change():
    a={'case':'a','baseline':True,'soil':{'porosity':.45},'most_geometry':dict(wind_height=2.,temperature_height=6.,humidity_height=6.,momentum_roughness=.043)}
    b=deepcopy(a);b['case']='b';b['baseline']=False;b['most_geometry']['wind_height']=4.
    same_nonheight_configuration(a,b)
    b['most_geometry']['momentum_roughness']=.075
    with pytest.raises(ValueError):same_nonheight_configuration(a,b)
