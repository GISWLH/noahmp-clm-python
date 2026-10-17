"""Score a fixed first-week CLM prefix against an independently complete official year.

An incomplete model year stays incomplete. Both original report and trace hashes
are retained; no replacement complete model report or shortened annual request
is constructed. UTC/local conversion is explicit and checked at every interval.
"""
from __future__ import annotations
import argparse,gzip,hashlib,json
from datetime import datetime,timedelta
from pathlib import Path
import numpy as np
from noahmp_clm.validation import compare_series

SITES=('CN-Qia','CN-Cha','CN-HaM')
STEPS=336

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_json(path):
    data=path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix=='.gz' else data)

def compare_prefix(trace,model_report,reference,prepared):
    """Only the predeclared first336 intervals; require exact source/time identity."""
    identity=prepared['inputs_sha256']
    for suffix,key in [('.npz','forcing_npz_sha256'),('-config.json','configuration_sha256')]:
        hashes=[v for k,v in identity.items() if k.endswith(suffix)]
        if hashes!=[model_report[key]]:raise ValueError('Original forcing/configuration identities disagree')
    return _compare_aligned_prefix(trace,model_report,reference,prepared)


def _compare_aligned_prefix(trace,model_report,reference,prepared):
    """Shared scoring after a caller has proved its explicit input binding.

    The original compare_prefix guard remains unchanged. A separate derived
    binding can reuse the identical official reference without rewriting its
    original prepared/configuration identities.
    """
    if model_report['steps_completed']!=STEPS or model_report['steps_requested']!=17520:
        raise ValueError('Require original first336 prefix of the unchanged17520 request')
    if model_report['status']!='running':raise ValueError('This diagnostic expects an explicitly incomplete original model year')
    if reference['physical_intervals']!=17520 or len(reference['intervals'])!=17520 or reference['file_count']!=17521:
        raise ValueError('Official full first-climate-year integrity required')
    dt=reference['timestep_seconds']
    if dt!=1800 or prepared['steps']!=17520 or prepared['dt_seconds']!=dt:raise ValueError('Unexpected protocol duration')
    offset=timedelta(hours=prepared['utc_offset_hours'])
    if model_report['utc_offset_hours']!=prepared['utc_offset_hours']:raise ValueError('UTC offsets disagree')
    times=[datetime.strptime(str(x),'%Y%m%d%H%M') for x in trace['time_start']]
    ends=[datetime.strptime(str(x),'%Y%m%d%H%M') for x in trace['time_end']]
    if len(times)!=STEPS or len(ends)!=STEPS:raise ValueError('Exact first-week trace required')
    if times[0]!=datetime.fromisoformat(prepared['initial_time_local']):raise ValueError('Prefix must start at declared initial time')
    if times[0].hour or times[0].minute or times[0].second:raise ValueError('Complete local days required')
    rows=reference['intervals'][:STEPS]
    for i,(start,end,row) in enumerate(zip(times,ends,rows)):
        if start!=times[0]+timedelta(seconds=i*dt) or end-start!=timedelta(seconds=dt):raise ValueError('Model time gap or overlap')
        if any(datetime.fromisoformat(row[k])+offset!=t for k,t in [('time_start',start),('time_end',end)]):
            raise ValueError('Exact explicit UTC to local-standard alignment required')
    model={};official={}
    for name,factor in [('et_mm',86400./dt),('runoff_mm',86400./dt),('h_wm2',1.),('le_wm2',1.)]:
        key=name+'_day' if name.endswith('_mm') else name
        model[key]=np.asarray(trace[name],float)*factor;official[key]=np.asarray([x[name] for x in rows],float)*factor
    model['swe_m']=np.asarray(trace['swe_m'],float);official['swe_m']=np.asarray([r['swe_mm']/1000. for r in rows])
    for old,new,var,shift in [('tsoil_k','soil_temperature_k','tsoil_c',273.15),('smc_m3m3','soil_total_water_m3m3','smc',0.)]:
        a=np.asarray(trace[old],float)-shift;b=np.asarray([r[new] for r in rows],float)-shift
        if a.shape!=(STEPS,4) or b.shape!=a.shape:raise ValueError('Same explicit four-layer support required')
        for k in range(4):model[f'{var}_layer_{k+1}']=a[:,k];official[f'{var}_layer_{k+1}']=b[:,k]
    if any(a.shape!=(STEPS,) or not np.all(np.isfinite(a)) for a in [*model.values(),*official.values()]):raise ValueError('Nonfinite or malformed series')
    daily={}
    for day in range(7):
        selected=slice(day*48,(day+1)*48);date=times[day*48].date().isoformat()
        daily[date]={key:compare_series(model[key][selected],official[key][selected],key.split('_layer_')[0]) for key in model}
    instant={key:compare_series(model[key],official[key],key) for key in ['h_wm2','le_wm2']}
    whole={key:compare_series(model[key],official[key],key.split('_layer_')[0]) for key in model}
    return dict(scope='Predeclared first-week cold-start structural diagnostic only; no annual, spinup or scientific-acceptance claim.',
        original_model_status=model_report['status'],original_model_steps_requested=model_report['steps_requested'],
        original_model_steps_completed=model_report['steps_completed'],official_physical_intervals=reference['physical_intervals'],
        scored_steps=STEPS,scored_days=7,scientific_acceptance_claim=False,
        time_convention='Every official UTC interval shifted by the protocol fixed offset, then compared exactly with original CLM local-standard times; no rephasing or missing intervals.',
        state_convention='Equally spaced interval-end means; soil temperature Celsius; same four layer supports declared in input configurations.',
        daily=daily,instantaneous_fluxes=instant,whole_prefix_means=whole,
        diagnostic_all_daily_bias_criteria_met=all(x['bias_pass'] for d in daily.values() for x in d.values()),
        diagnostic_flux_r_criteria_met=all(x['correlation_pass'] is True for x in instant.values()))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model-root',type=Path,required=True)
    p.add_argument('--reference-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise ValueError('Preserve previous diagnostic')
    protocol=read_json(args.reference_root/'protocol.json');results={}
    for site in SITES:
        directory=args.model_root/(site+'-initial-climate-cycle')
        candidates=list(directory.glob('report-0000336-*.json'))+list(directory.glob('report-0000336-*.json.gz'))
        if len(candidates)!=1:raise ValueError('One immutable original336 per-case report required')
        report_path=candidates[0];report=read_json(report_path)
        trace_path=directory/Path(report['model_npz']).name
        if sha(trace_path)!=report['model_npz_sha256']:raise ValueError('Original trace SHA256 mismatch')
        with np.load(trace_path,allow_pickle=False) as z:trace=dict(z)
        reference_path=args.reference_root/site/'audit.json';prepared_path=args.reference_root/site/'prepared-manifest.json'
        prepared=read_json(prepared_path);reference=read_json(reference_path)
        if reference['source_protocol_sha256']!=sha(args.reference_root/'protocol.json'):raise ValueError('Official source protocol mismatch')
        result=compare_prefix(trace,report,reference,prepared)
        result['source_sha256']={str(x):sha(x) for x in [report_path,trace_path,reference_path,prepared_path,args.reference_root/'protocol.json',Path(__file__)]}
        results[site]=result
    args.output.write_text(json.dumps(dict(protocol_scope=protocol['scope'],results=results),indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({site:dict(h_r=r['instantaneous_fluxes']['h_wm2']['r'],le_r=r['instantaneous_fluxes']['le_wm2']['r'],
        week_h_bias=r['whole_prefix_means']['h_wm2']['relative_bias'],week_et_bias=r['whole_prefix_means']['et_mm_day']['relative_bias'],
        daily_bias_criteria=r['diagnostic_all_daily_bias_criteria_met'],flux_r_criteria=r['diagnostic_flux_r_criteria_met']) for site,r in results.items()}),flush=True)

if __name__=='__main__':main()
