"""Compare complete CLM interval traces with independently audited HRLDAS output.

Time matching is exact, without truncation, gap filling, or a timezone shift.
Daily state means use interval ends in both series. Scientific criteria are
those in V1_ACCEPTANCE, separate from configuration/spinup comparability.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timedelta
import hashlib,json
from pathlib import Path
import numpy as np
from noahmp_clm.validation import compare_series


def compare_columns(model,reference):
    if model.get('status')!='complete':raise ValueError('Model trajectory must be complete')
    if len(model['intervals'])!=model['steps_completed']:raise ValueError('Complete model length disagrees with report')
    return compare_interval_rows(model['intervals'],reference)


def compare_interval_rows(a,reference):
    """Score already scoped data rows; makes no claim about a run's status.

    Callers must separately validate original reports and declare whether the
    rows cover a full run or an incomplete-year diagnostic. This avoids ever
    fabricating a complete report for an incomplete original trajectory.
    """
    b=reference['intervals']
    dt=reference['timestep_seconds']
    if dt<=0 or 86400%dt:raise ValueError('Require a timestep dividing a day')
    if len(a)!=len(b) or not a:
        raise ValueError('Both trajectories must have exactly matching lengths')
    initial=datetime.fromisoformat(reference['initial_time'])
    for i,(x,y) in enumerate(zip(a,b)):
        for key,j in [('time_start',i),('time_end',i+1)]:
            expected=(initial+timedelta(seconds=j*dt)).isoformat()
            if x[key]!=expected or y[key]!=expected:raise ValueError('Exact time alignment required')
    def series(rows):
        result={'et_mm_day':np.array([r['et_mm'] for r in rows])*86400/dt,
                'runoff_mm_day':np.array([r['runoff_mm'] for r in rows])*86400/dt,
                'h_wm2':np.array([r['h_wm2'] for r in rows]),
                'le_wm2':np.array([r['le_wm2'] for r in rows]),
                'swe_m':np.array([r['swe_mm'] for r in rows])/1000.}
        for field,var,offset in [('soil_temperature_k','tsoil_c',273.15),('soil_total_water_m3m3','smc',0.)]:
            array=np.array([r[field] for r in rows],dtype=float)-offset
            if array.ndim!=2 or array.shape[1]==0:raise ValueError('Require explicit soil layer arrays')
            for layer in range(array.shape[1]):result[f'{var}_layer_{layer+1}']=array[:,layer]
        if any(not np.all(np.isfinite(v)) for v in result.values()):raise ValueError('Nonfinite trajectory')
        return result
    aa,bb=series(a),series(b)
    if aa.keys()!=bb.keys():raise ValueError('Soil layer counts differ; explicit remapping required')
    def period(selected):
        group={}
        for key in aa:
            variable=key.split('_layer_')[0]
            metric=compare_series(aa[key][selected],bb[key][selected],variable)
            if key in ('et_mm_day','runoff_mm_day','h_wm2','le_wm2'):
                factor=dt/86400. if key.endswith('_mm_day') else dt
                metric.update(model_integral=float(np.sum(aa[key][selected])*factor),
                    reference_integral=float(np.sum(bb[key][selected])*factor),
                    integral_units='mm' if key.endswith('_mm_day') else 'J/m2',integration_seconds=int(selected.sum()*dt))
            else:metric['state_units']='deg C' if key.startswith('tsoil_c') else 'm3/m3' if key.startswith('smc') else 'm water equivalent'
            group[key]=metric
        return group
    dates=np.array([r['time_start'][:10] for r in a]);daily={}
    for date in sorted(set(dates)):
        selected=dates==date
        if int(selected.sum())*dt!=86400:raise ValueError('Partial days cannot masquerade as daily acceptance')
        daily[date]=period(selected)
    months=np.array([r['time_start'][:7] for r in a]);monthly={}
    for month in sorted(set(months)):
        selected=months==month;indices=np.flatnonzero(selected);begin=datetime.strptime(month,'%Y-%m')
        end=begin.replace(year=begin.year+1,month=1) if begin.month==12 else begin.replace(month=begin.month+1)
        complete=a[indices[0]]['time_start']==begin.isoformat() and a[indices[-1]]['time_end']==end.isoformat()
        if complete:monthly[month]=period(selected)
        else:monthly[month]={key:dict(status='unmeasured',n=len(indices),bias_pass=None,correlation_pass=None,reason='partial calendar month') for key in aa}
    instant={key:compare_series(aa[key],bb[key],key) for key in ('h_wm2','le_wm2')}
    return {'scope':'Numerical scientific comparisons only; configuration/spinup representativeness requires separate review',
            'physical_intervals':len(a),'daily':daily,'monthly':monthly,'full_span':period(np.ones(len(a),bool)),'instantaneous_fluxes':instant,
            'time_convention':reference['time_convention'],
            'state_convention':'Arithmetic means of equally spaced interval-end states; soil temperature in Celsius',
            'daily_bias_criteria_met':all(x['bias_pass'] for d in daily.values() for x in d.values()),
            'monthly_bias_criteria_met':None if any(x['bias_pass'] is None for d in monthly.values() for x in d.values()) else all(x['bias_pass'] for d in monthly.values() for x in d.values()),
            'instantaneous_flux_criteria_met':all(x['correlation_pass'] is True for x in instant.values()),
            'instantaneous_flux_criteria_status':'unmeasured' if any(x['correlation_pass'] is None for x in instant.values()) else 'pass' if all(x['correlation_pass'] for x in instant.values()) else 'fail'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('model','reference','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    result=compare_columns(json.loads(args.model.read_text(encoding='utf-8')),json.loads(args.reference.read_text(encoding='utf-8')))
    result['source_sha256']={key:hashlib.sha256(path.read_bytes()).hexdigest() for key,path in
                             [('model',args.model),('reference',args.reference),('instrument',Path(__file__))]}
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('physical_intervals','daily_bias_criteria_met','instantaneous_flux_criteria_met')}))


if __name__=='__main__':main()
