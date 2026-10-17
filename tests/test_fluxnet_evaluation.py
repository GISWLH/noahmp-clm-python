from datetime import datetime,timedelta
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'benchmarks'))
from evaluate_fluxnet_series import evaluate, CARBON_G_PER_UMOL, PRODUCTS


def example(days=1,start=datetime(2004,2,1)):
    n=days*48;dt=1800
    obs={'time_start':np.array([(start+timedelta(seconds=i*dt)).strftime('%Y%m%d%H%M') for i in range(n)]),
         'time_end':np.array([(start+timedelta(seconds=(i+1)*dt)).strftime('%Y%m%d%H%M') for i in range(n)])}
    model={k:v.copy() for k,v in obs.items()}
    for field,(key,_,carbon) in PRODUCTS.items():
        values=np.linspace(1.,3.,n) if carbon else np.linspace(20.,40.,n)
        obs['obs_'+field]=values.copy();obs['qc_'+field]=np.full(n,np.nan if field.startswith('GPP') else 0.)
        model[key]=values*CARBON_G_PER_UMOL*dt if carbon else values.copy()
    return model,obs


def test_daily_carbon_and_heat_are_integrated_with_own_qc():
    m,o=example();o['qc_H_F_MDS'][0]=1;o['qc_LE_F_MDS'][1]=np.nan
    r=evaluate(m,o,utc_offset_hours=8);d=r['daily']['2004-02-01']
    assert d['H_F_MDS']['model_integral']==pytest.approx(30*86400)
    assert d['GPP_NT_VUT_REF']['model_integral']==pytest.approx(2*12.0107e-6*86400)
    assert d['H_F_MDS']['own_qc_counts']=={'zero':47,'nonzero':1,'missing':0}
    assert r['instantaneous']['H_F_MDS']['metrics']['n']==47
    assert r['instantaneous']['LE_F_MDS']['metrics']['n']==47
    assert r['instantaneous']['NEE_VUT_REF']['metrics']['n']==48
    assert r['instantaneous']['GPP_NT_VUT_REF']['metrics']['n']==48
    assert d['GPP_NT_VUT_REF']['own_qc_counts']['missing']==48
    assert r['monthly']['2004-02']['H_F_MDS']['metrics']['status']=='unmeasured'


def test_full_leap_month_and_missing_partition_product_are_distinct():
    m,o=example(29);o['obs_GPP_NT_VUT_REF'][10]=np.nan
    r=evaluate(m,o,utc_offset_hours=8);month=r['monthly']['2004-02']
    assert month['H_F_MDS']['expected_intervals']==1392
    assert month['H_F_MDS']['model_integral']==pytest.approx(30*86400*29)
    assert month['GPP_NT_VUT_REF']['metrics']['bias_pass'] is None
    assert month['GPP_DT_VUT_REF']['metrics']['bias_pass']
    assert r['daily']['2004-02-02']['GPP_NT_VUT_REF']['metrics']['bias_pass']


@pytest.mark.parametrize('fault',['shift','gap','nan_model','shape'])
def test_malformed_traces_are_rejected(fault):
    m,o=example()
    if fault=='shift':m['time_start'][0]='200402010030'
    if fault=='gap':
        for data in (m,o):
            for k in data:data[k]=np.delete(data[k],5)
    if fault=='nan_model':m['h_wm2'][5]=np.nan
    if fault=='shape':m['h_wm2']=m['h_wm2'][:-1]
    with pytest.raises(ValueError):evaluate(m,o,utc_offset_hours=8)


def test_near_zero_nee_uses_absolute_carbon_tolerance():
    m,o=example();o['obs_NEE_VUT_REF'][:]=0.;m['nee_gc_m2'][:]=.09/48
    r=evaluate(m,o,utc_offset_hours=8)['daily']['2004-02-01']['NEE_VUT_REF']['metrics']
    assert r['bias']==pytest.approx(.09) and r['near_zero'] and r['bias_pass']


def test_corrected_and_uncorrected_heat_products_remain_separate():
    m,o=example();o['obs_H_CORR']*=1.25;o['obs_LE_CORR']*=1.5
    o['qc_H_F_MDS'][0]=1;o['qc_LE_F_MDS'][1:3]=2
    # Corrected-QC aliases are not an independent quality source. Use the
    # official underlying H/LE QC even if a caller supplies different aliases.
    o['qc_H_CORR'][:]=0;o['qc_LE_CORR'][:]=0
    r=evaluate(m,o,utc_offset_hours=8);d=r['daily']['2004-02-01']
    assert d['H_F_MDS']['metrics']['bias_pass']
    assert d['LE_F_MDS']['metrics']['bias_pass']
    assert not d['H_CORR']['metrics']['bias_pass']
    assert not d['LE_CORR']['metrics']['bias_pass']
    assert d['H_CORR']['reference_integral']==pytest.approx(37.5*86400)
    assert d['LE_CORR']['reference_integral']==pytest.approx(45.*86400)
    assert r['instantaneous']['H_CORR']['metrics']['n']==47
    assert r['instantaneous']['LE_CORR']['metrics']['n']==46
    assert r['instantaneous']['H_CORR']['qc_source']=='H_F_MDS_QC'
    assert d['LE_CORR']['qc_source']=='LE_F_MDS_QC'
    assert 'corrected' in d['H_CORR']['reference_kind']


def test_missing_corrected_heat_does_not_use_uncorrected_as_replacement():
    m,o=example();o['obs_H_CORR'][3]=np.nan;o['obs_LE_CORR'][:]=np.nan
    r=evaluate(m,o,utc_offset_hours=8);d=r['daily']['2004-02-01']
    assert d['H_F_MDS']['metrics']['bias_pass'] and d['LE_F_MDS']['metrics']['bias_pass']
    assert d['H_CORR']['metrics']['status']=='unmeasured'
    assert d['LE_CORR']['metrics']['status']=='unmeasured'
    assert r['instantaneous']['H_CORR']['metrics']['n']==47
    assert r['instantaneous']['LE_CORR']['metrics']['status']=='unmeasured'
