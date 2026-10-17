"""Report every predeclared height case relative to its fixed baseline.

This is a sensitivity comparison, not observation skill or height selection.
Only complete trajectories with identical forcing and non-height parameters
can be compared. Missing/running/failed cases remain visible in the report.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime,timedelta
import hashlib,json
from pathlib import Path
import numpy as np
from noahmp_clm.validation import compare_series

FIELDS={'h_wm2':('h_wm2','W/m2'),'le_wm2':('le_wm2','W/m2'),
        'et_mm':('et_mm_day','mm/day'),'runoff_mm':('runoff_mm_day','mm/day'),
        'gpp_gc_m2':('gpp_gc_day','g C/m2/day'),'nee_gc_m2':('nee_gc_day','g C/m2/day'),
        'swe_m':('swe_m','m'),'lai':(None,'m2/m2'),
        'tsoil_k':('tsoil_c','K difference; C for relative mean'),'smc_m3m3':('smc','m3/m3')}
INTEGRATED={'et_mm','runoff_mm','gpp_gc_m2','nee_gc_m2'}


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def same_nonheight_configuration(a,b):
    def strip(c):
        c=deepcopy(c)
        for key in ('case','baseline'):c.pop(key,None)
        for key in ('wind_height','temperature_height','humidity_height'):c['most_geometry'].pop(key)
        return c
    if strip(a)!=strip(b):raise ValueError('Non-height parameters differ between scenarios')


def compare(model,baseline):
    for key in ('time_start','time_end'):
        if not np.array_equal(model[key],baseline[key]):raise ValueError('Height traces have different times')
    start=[datetime.strptime(str(x),'%Y%m%d%H%M') for x in baseline['time_start']]
    end=[datetime.strptime(str(x),'%Y%m%d%H%M') for x in baseline['time_end']]
    if not start or len(start)!=len(end):raise ValueError('Empty or malformed trace')
    dt=(end[0]-start[0]).total_seconds()
    if dt not in (1800,3600) or any((b-a).total_seconds()!=dt for a,b in zip(start,end)) or any(a!=b for a,b in zip(end[:-1],start[1:])):
        raise ValueError('Noncontiguous or variable time intervals')
    periods={'whole_window':np.ones(len(start),bool)}
    labels=np.array([t.strftime('%Y-%m-%d') for t in start])
    for label in sorted(set(labels)):
        mask=labels==label;idx=np.flatnonzero(mask);begin=datetime.strptime(label,'%Y-%m-%d')
        if start[idx[0]]!=begin or end[idx[-1]]!=begin+timedelta(days=1):raise ValueError('Partial day cannot stand in for daily sensitivity')
        periods[label]=mask
    results={}
    for name,mask in periods.items():
        values={}
        for key,(variable,units) in FIELDS.items():
            a=np.asarray(model[key],float);b=np.asarray(baseline[key],float)
            if a.shape!=b.shape or a.shape[0]!=len(start) or a.ndim not in (1,2):raise ValueError('Shape mismatch')
            if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):raise ValueError('Nonfinite model trajectory')
            if key in INTEGRATED:a=a*86400/dt;b=b*86400/dt
            if key=='tsoil_k':a=a-273.15;b=b-273.15
            if a.ndim==1:a=a[:,None];b=b[:,None]
            columns=[]
            for k in range(a.shape[1]):
                x,y=a[mask,k],b[mask,k]
                if variable:
                    full=compare_series(x,y,variable)
                    metric={k:v for k,v in full.items() if k not in ('bias_pass','correlation_pass','relative_tolerance','absolute_tolerance','status')}
                else:
                    delta=x-y;metric=dict(n=len(x),model_mean=float(x.mean()),reference_mean=float(y.mean()),bias=float(delta.mean()),
                        max_absolute_difference=float(abs(delta).max()),rmse=float(np.sqrt(np.mean(delta**2))))
                metric.update(units=units,baseline_role='fixed literature assumption; not observation')
                if key in INTEGRATED:
                    metric.update(model_integral=float(x.sum()*dt/86400),baseline_integral=float(y.sum()*dt/86400),
                                  integral_units='g C/m2' if key.endswith('gc_m2') else 'mm')
                if key in ('h_wm2','le_wm2'):
                    metric.update(model_integral=float(x.sum()*dt),baseline_integral=float(y.sum()*dt),integral_units='J/m2')
                columns.append(metric)
            values[key]=columns[0] if len(columns)==1 else columns
        results[name]=values
    return results


def summarize(root,cases_path,protocol_path):
    root,cases_path,protocol_path=map(Path,(root,cases_path,protocol_path))
    cases=json.loads(cases_path.read_text(encoding='utf-8'));protocol=json.loads(protocol_path.read_text(encoding='utf-8'))
    if cases['protocol_sha256']!=sha(protocol_path):raise ValueError('Height protocol hash mismatch')
    expected={(site,tq,w) for site,g in protocol['sites'].items() for tq in g['temperature_humidity_candidates_m'] for w in g['wind_candidates_m']}
    reports={};traces={};observed=set();results={}
    for entry in cases['cases']:
        name=entry['case'];path=root/name/'report.json'
        config_path=cases_path.parent/entry['configuration']
        if sha(config_path)!=entry['configuration_sha256']:raise ValueError('Frozen configuration hash mismatch')
        c=json.loads(config_path.read_text(encoding='utf-8'));g=c['most_geometry']
        declared=protocol['sites'][entry['site']]['baseline']
        is_baseline=all(g[k]==v for k,v in declared.items())
        if entry['baseline']!=is_baseline or c['baseline']!=is_baseline:raise ValueError('Baseline differs from locked protocol')
        point=(entry['site'],g['temperature_height'],g['wind_height'])
        if point not in expected or point in observed or g['humidity_height']!=g['temperature_height']:raise ValueError('Unexpected or duplicate height scenario')
        observed.add(point)
        if not path.exists():results[name]={'status':'not_started'};continue
        r=json.loads(path.read_text(encoding='utf-8'));reports[name]=r
        if r['configuration']!=c:raise ValueError('Report configuration differs from frozen case')
        if r['configuration_sha256']!=entry['configuration_sha256'] or c['protocol_sha256']!=sha(protocol_path):raise ValueError('Configuration identity mismatch')
        results[name]={'status':r['status'],'steps_completed':r['steps_completed'],'baseline':entry['baseline'],
                       'measurement_heights_m':{k:g[k] for k in ('temperature_height','humidity_height','wind_height')},
                       'maximum_cumulative_residual_m_h_c_n':r['maximum_cumulative_residual_m_h_c_n']}
        if r['status']!='complete':
            if 'failure' in r:results[name]['failure']=r['failure']
            continue
        trace=root/name/r['model_npz']
        if sha(trace)!=r['model_npz_sha256'] or not r.get('source_unchanged'):raise ValueError('Unverified model trajectory')
        with np.load(trace,allow_pickle=False) as a:traces[name]=dict(a)
        if len(traces[name]['time_start'])!=r['steps_requested'] or r['steps_completed']!=r['steps_requested']:
            raise ValueError('Complete status has an incomplete trace')
        results[name]['model_npz_sha256']=r['model_npz_sha256']
    if observed!=expected:raise ValueError('Case list omits a predeclared height combination')
    for entry in cases['cases']:
        name=entry['case'];baseline=[x for x in cases['cases'] if x['site']==entry['site'] and x['baseline']]
        if len(baseline)!=1:raise ValueError('Require exactly one predeclared baseline per site')
        base=baseline[0]['case']
        if name not in traces or base not in traces:continue
        a,b=reports[name],reports[base]
        if a['forcing_npz_sha256']!=b['forcing_npz_sha256'] or a['source_sha256']!=b['source_sha256']:
            raise ValueError('Forcing or model differs between height cases')
        same_nonheight_configuration(a['configuration'],b['configuration'])
        results[name]['comparison_to_baseline']=compare(traces[name],traces[base]);results[name]['baseline_case']=base
    return dict(scope='Height sensitivity conditional on fixed forcing, initial state and parameter assumptions; no height optimization or observation acceptance',
                protocol_sha256=sha(protocol_path),cases_sha256=sha(cases_path),expected_cases=len(expected),
                complete_cases=sum(v['status']=='complete' for v in results.values()),cases=results)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','cases','protocol','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();r=summarize(a.root,a.cases,a.protocol);a.output.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'complete':r['complete_cases'],'expected':r['expected_cases']}))
