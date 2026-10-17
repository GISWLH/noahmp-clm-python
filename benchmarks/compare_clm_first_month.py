"""Fixed January diagnostic from immutable incomplete-year native output.

Original running reports and full annual requests are retained. All official
and observational products are reported, without choosing by model skill.
Run only on an EHPC PBS compute node; no physical integration is performed.
"""
from pathlib import Path
from datetime import datetime,timedelta
import argparse,hashlib,json,os,tarfile
import numpy as np
from compare_hrldas_column import compare_interval_rows
from compare_hrldas_suction_prefix import bind_suction_inputs,SITES,PROTOCOL_SHA,ARCHIVE_SHA,GENERATOR_SHA
from compare_hrldas_native_year import check_ledgers,check_final_checkpoint
from evaluate_fluxnet_series import evaluate
from fluxnet_soil_operator import initial_soil_from_manifest

sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def first_month_times(trace,report,*,initial_time,dt_seconds,full_steps,utc_offset_hours):
    begin=datetime.fromisoformat(initial_time);finish=datetime(begin.year,2,1);annual_end=datetime(begin.year+1,1,1)
    if begin!=datetime(begin.year,1,1) or dt_seconds!=1800 or full_steps*dt_seconds!=(annual_end-begin).total_seconds():
        raise ValueError('Original complete January-start annual protocol required')
    n=int((finish-begin).total_seconds()/dt_seconds)
    if (report.get('status')!='running' or report.get('source_unchanged') is not True or report.get('initial_run') is not None
            or report['steps_requested']!=full_steps or report['steps_completed']!=n):
        raise ValueError('Exact immutable first-month cold-start prefix of original annual request required')
    if report['time_convention']!='local standard time' or report['utc_offset_hours']!=utc_offset_hours:
        raise ValueError('Explicit matching local-standard convention required')
    starts=np.asarray(trace['time_start']);ends=np.asarray(trace['time_end'])
    if starts.shape!=(n,) or ends.shape!=(n,):raise ValueError('Exact first-month trace length required')
    times=[begin+timedelta(seconds=i*dt_seconds) for i in range(n+1)]
    if not np.array_equal(starts,[t.strftime('%Y%m%d%H%M') for t in times[:-1]]) or not np.array_equal(ends,[t.strftime('%Y%m%d%H%M') for t in times[1:]]):
        raise ValueError('Exact contiguous original January local intervals required')
    return times


def compare_official(trace,report,reference,prepared,old_raw,new_raw,site):
    binding=bind_suction_inputs(report,prepared,old_raw,new_raw,site)
    times=first_month_times(trace,report,initial_time=prepared['initial_time_local'],dt_seconds=prepared['dt_seconds'],
        full_steps=prepared['steps'],utc_offset_hours=prepared['utc_offset_hours'])
    n=len(times)-1;full=prepared['steps'];dt=prepared['dt_seconds'];offset=timedelta(hours=prepared['utc_offset_hours'])
    if (reference['physical_intervals']!=full or reference['file_count']!=full+1 or len(reference['intervals'])!=full
            or reference['timestep_seconds']!=dt or datetime.fromisoformat(reference['initial_time'])+offset!=times[0]):
        raise ValueError('Unchanged complete official annual reference required')
    values={k:np.asarray(trace[k],float) for k in ['h_wm2','le_wm2','et_mm','runoff_mm','swe_m','tsoil_k','smc_m3m3']}
    if any(v.shape!=((n,4) if k in ('tsoil_k','smc_m3m3') else (n,)) or not np.isfinite(v).all() for k,v in values.items()):
        raise ValueError('Finite same-four-layer first-month data required')
    model=[];official=[]
    for i,row in enumerate(reference['intervals'][:n]):
        converted=dict(row)
        for key,t in [('time_start',times[i]),('time_end',times[i+1])]:
            converted[key]=(datetime.fromisoformat(row[key])+offset).isoformat()
            if converted[key]!=t.isoformat():raise ValueError('Exact official UTC/local alignment required')
        official.append(converted)
        model.append(dict(time_start=times[i].isoformat(),time_end=times[i+1].isoformat(),
            h_wm2=float(values['h_wm2'][i]),le_wm2=float(values['le_wm2'][i]),et_mm=float(values['et_mm'][i]),
            runoff_mm=float(values['runoff_mm'][i]),swe_mm=float(values['swe_m'][i]*1000.),
            soil_temperature_k=values['tsoil_k'][i].tolist(),soil_total_water_m3m3=values['smc_m3m3'][i].tolist()))
    view=dict(reference,initial_time=times[0].isoformat(),intervals=official,time_convention='Fixed local standard time; original official UTC explicitly converted')
    result=compare_interval_rows(model,view)
    if len(result['daily'])!=31 or list(result['monthly'])!=[times[0].strftime('%Y-%m')]:raise ValueError('Exactly one complete January required')
    result.update(scope='Fixed first January cold-start diagnostic, not annual/spinup/observational acceptance.',configuration_binding=binding,
        original_model_status=report['status'],original_model_steps_requested=report['steps_requested'],original_model_steps_completed=report['steps_completed'],
        original_official_intervals=full,scientific_acceptance_claim=False)
    return result


def compare_observed(trace,report,observed,metadata,operator):
    times=first_month_times(trace,report,initial_time=metadata['start'],dt_seconds=metadata['dt_seconds'],full_steps=metadata['steps'],utc_offset_hours=metadata['utc_offset_hours'])
    if operator['site']!=report['site']:raise ValueError('Original soil operator/site mismatch')
    n=len(times)-1
    for key in ('time_start','time_end'):
        if np.shape(observed[key])!=(metadata['steps'],) or not np.array_equal(trace[key],observed[key][:n]):raise ValueError('Exact original full observation source/prefix alignment required')
    result=evaluate(trace,{key:np.asarray(value)[:n] for key,value in observed.items()},utc_offset_hours=metadata['utc_offset_hours'],
        soil_operator=operator,initial_soil=initial_soil_from_manifest(report))
    if len(result['daily'])!=31 or len(result['monthly'])!=1:raise ValueError('One complete January required')
    result.update(scope='Fixed first January observational diagnostic from original incomplete native year; no spinup/calibration/holdout or scientific-acceptance claim.',
        original_model_status=report['status'],original_model_steps_requested=report['steps_requested'],original_model_steps_completed=report['steps_completed'],scientific_acceptance_claim=False)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['model-root','configuration-root','reference-root','reference-archive','soil-operators','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if not os.environ.get('PBS_JOBID'):raise RuntimeError('Run on an EHPC PBS compute node')
    if a.output.exists():raise ValueError('Preserve previous diagnostic')
    protocol=a.reference_root/'protocol.json'
    if sha(a.reference_archive)!=ARCHIVE_SHA or sha(protocol)!=PROTOCOL_SHA or json.loads(protocol.read_text())['instrument_sha256']!=GENERATOR_SHA:
        raise ValueError('Original immutable official reference/protocol/generator required')
    official={};observed_results={};ledgers={};identities={}
    with tarfile.open(a.reference_archive) as archive:
        for site in ['CN-Cng',*SITES]:
            folder=a.model_root/(site+'-suction-climate-cycle');paths=list(folder.glob('report-0001488-*.json'))
            if len(paths)!=1:raise ValueError('One immutable original1488 report required: '+site)
            report_path=paths[0];report=json.loads(report_path.read_text());trace_path=folder/Path(report['model_npz']).name
            checkpoint=folder/report['batch_checkpoint_file'];weather=a.configuration_root/(site+'.npz');metadata=a.configuration_root/(site+'.json');config=a.configuration_root/(site+'-config.json')
            for path,key in [(trace_path,'model_npz_sha256'),(weather,'forcing_npz_sha256'),(metadata,'forcing_manifest_sha256'),(config,'configuration_sha256')]:
                if sha(path)!=report[key]:raise ValueError('Original model/input identity mismatch: '+str(path))
            meta=json.loads(metadata.read_text());op=a.soil_operators/(site+'.json')
            if meta['prepared_npz_sha256']!=sha(weather):raise ValueError('Original prepared observation hash mismatch')
            with np.load(trace_path,allow_pickle=False) as z:trace=dict(z)
            ledgers[site]=dict(ledger=check_ledgers(trace,report),checkpoint=check_final_checkpoint(checkpoint,trace,report))
            with np.load(weather,allow_pickle=False) as z:
                observed_results[site]=compare_observed(trace,report,z,meta,json.loads(op.read_text()))
            bound_paths=[report_path,trace_path,checkpoint,weather,metadata,config,op]
            if site in SITES:
                prepared_path=a.reference_root/site/'prepared-manifest.json';reference_path=a.reference_root/site/'audit.json'
                for name,path in [('protocol.json',protocol),(site+'/prepared-manifest.json',prepared_path),(site+'/audit.json',reference_path),(site+'/run-receipt.json',a.reference_root/site/'run-receipt.json')]:
                    if archive.extractfile(name).read()!=path.read_bytes():raise ValueError('Original official metadata differs from archive')
                prepared=json.loads(prepared_path.read_text());reference=json.loads(reference_path.read_text())
                if reference['source_protocol_sha256']!=PROTOCOL_SHA or not all(sha(k)==v for k,v in prepared['inputs_sha256'].items()):raise ValueError('Original official source input mismatch')
                old=next(Path(k) for k in prepared['inputs_sha256'] if k.endswith('-config.json'))
                official[site]=compare_official(trace,report,reference,prepared,old.read_bytes(),config.read_bytes(),site)
                bound_paths += [old,prepared_path,reference_path]
            identities[site]={str(path):sha(path) for path in bound_paths}
    result=dict(scope='Fixed first January I1488 diagnostics; original running/annual-request identities retained, no complete report fabricated.',
        node=os.uname().nodename,pbs_job=os.environ['PBS_JOBID'],official=official,observational=observed_results,stored_independent_ledgers=ledgers,
        source_sha256=identities,official_archive_sha256=ARCHIVE_SHA,official_protocol_sha256=PROTOCOL_SHA,
        instrument_sha256={str(Path(__file__)):sha(__file__)},scientific_acceptance_claim=False)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(official_sites=list(official),observed_sites=list(observed_results),scored_steps=1488,original_status='running',scientific_acceptance_claim=False)),flush=True)

if __name__=='__main__':main()
