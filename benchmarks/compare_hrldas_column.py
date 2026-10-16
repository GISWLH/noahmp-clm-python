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
    a,b=model['intervals'],reference['intervals']
    dt=reference['timestep_seconds']
    if dt<=0 or 86400%dt:raise ValueError('Require a timestep dividing a day')
    if len(a)!=len(b) or len(a)!=model['steps_completed'] or not a:
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
    dates=np.array([r['time_start'][:10] for r in a]);daily={}
    for date in sorted(set(dates)):
        selected=dates==date
        if int(selected.sum())*dt!=86400:raise ValueError('Partial days cannot masquerade as daily acceptance')
        group={}
        for key in aa:
            variable=key.split('_layer_')[0]
            group[key]=compare_series(aa[key][selected],bb[key][selected],variable)
        daily[date]=group
    instant={key:compare_series(aa[key],bb[key],key) for key in ('h_wm2','le_wm2')}
    return {'scope':'Numerical scientific comparisons only; configuration/spinup representativeness requires separate review',
            'physical_intervals':len(a),'daily':daily,'instantaneous_fluxes':instant,
            'time_convention':reference['time_convention'],
            'state_convention':'Arithmetic means of equally spaced interval-end states; soil temperature in Celsius',
            'daily_bias_criteria_met':all(x['bias_pass'] for d in daily.values() for x in d.values()),
            'instantaneous_flux_criteria_met':all(x['correlation_pass'] is True for x in instant.values())}


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
