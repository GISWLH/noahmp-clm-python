"""Complete native calendar-year metrics plus stored independent ledger checks.

This evaluates audited data. It does not certify spinup, calibration/holdout,
reference representativeness, or a release, and never promotes a prefix.
"""
from pathlib import Path
from datetime import datetime,timedelta
import argparse,hashlib,json,os,tarfile
import numpy as np
import run_fluxnet_native as native
from compare_hrldas_column import compare_columns
from compare_hrldas_suction_prefix import bind_suction_inputs,ARCHIVE_SHA,PROTOCOL_SHA,GENERATOR_SHA,SITES

sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def check_ledgers(trace,report,initial_checkpoint=None):
    n=report['steps_completed'];names=('step_residual_m_h_c_n','step_bound_m_h_c_n','cumulative_residual_m_h_c_n')
    if n<=0:raise ValueError('Nonempty ledger required')
    if any(k not in trace for k in names):raise ValueError('Stored independent M/H/C/N ledger traces required')
    residual,bound,cumulative=(np.asarray(trace[k],float) for k in names)
    if any(v.shape!=(n,4) or not np.isfinite(v).all() for v in (residual,bound,cumulative)) or np.any(bound<0):
        raise ValueError('Malformed native ledger trace')
    link=report.get('initial_run');start=np.zeros(4)
    if link:
        if initial_checkpoint is None or sha(initial_checkpoint)!=link['checkpoint_sha256']:raise ValueError('Carried ledger requires its exact linked initial checkpoint')
        with np.load(initial_checkpoint,allow_pickle=False) as z:start=np.asarray(z['bounds'][report['batch_column']],float)
        if report['ledger_steps']!=link['previous_ledger_steps']+n:raise ValueError('Carried ledger length mismatch')
    elif initial_checkpoint is not None or report['ledger_steps']!=n:raise ValueError('Fresh ledger provenance mismatch')
    if start.shape!=(4,) or np.any(start<0) or not np.isfinite(start).all():raise ValueError('Invalid starting numerical bounds')
    accumulated=np.cumsum(np.vstack((start,bound)),axis=0)[1:]
    if np.any(abs(residual)>bound) or np.any(abs(cumulative)>accumulated):raise ValueError('Independent native conservation trace exceeds original numerical bounds')
    reported=np.asarray(report['cumulative_numerical_bounds_m_h_c_n'],float)
    allowance=64*np.finfo(float).eps*max(n,1)*np.maximum(abs(accumulated[-1]),1.)
    if reported.shape!=(4,) or not np.isfinite(reported).all() or np.any(abs(reported-accumulated[-1])>allowance):
        raise ValueError('Reported cumulative bounds disagree with original step bounds')
    return dict(scope='Checks original stored independent ledger residuals against original step and carried cumulative numerical bounds; conservation has no 10-percent waiver.',
        starting_bounds_m_h_c_n=start.tolist(),maximum_step_residual_m_h_c_n=abs(residual).max(0).tolist(),
        maximum_cumulative_residual_m_h_c_n=abs(cumulative).max(0).tolist(),final_bounds_m_h_c_n=accumulated[-1].tolist())

def check_final_checkpoint(path,trace,report):
    """Check all declared arrays, then independently sum conserved storage."""
    if sha(path)!=report['batch_checkpoint_sha256']:raise ValueError('Final checkpoint identity mismatch')
    k=report['batch_column'];n=report['steps_completed']
    with np.load(path,allow_pickle=False) as z:
        count=native.native_state_count([dict(config=report.get('configuration',{}))])
        if z['index'].item()!=n:raise ValueError('Final checkpoint generation/arrays mismatch')
        original=native.checkpoint_arrays(z,dict(native_state_arrays=count));a=[np.asarray(v,float) for v in original]
        if any(not np.isfinite(v).all() for v in a):raise ValueError('Nonfinite checkpoint state')
        if count in (19,23):
            p,_=native.structure_configuration(report['configuration']);ss,pp,oo=original[16:19];columns=len(original[0])
            if ss.shape!=(columns,1) or pp.shape!=(columns,2) or oo.shape!=(columns,2) or oo.dtype.kind not in 'iu' or np.any(ss<0):
                raise ValueError('Malformed structure checkpoint arrays')
            if ss.dtype!=original[0].dtype or pp.dtype!=original[0].dtype or not np.array_equal(pp[k],np.asarray([p.canopy_top_m,p.canopy_bottom_m],pp.dtype)) or not np.array_equal(oo[k],[int(p.scheme),p.short_vegetation]):
                raise ValueError('Checkpoint structure parameters disagree with configuration')
            if 'sai_m2m2' not in trace or np.asarray(trace['sai_m2m2']).shape!=(n,) or trace['sai_m2m2'][-1]!=ss[k,0]:
                raise ValueError('Final SAI state/trace alias mismatch')
        if count in (20,23):
            from dataclasses import astuple
            from noahmp_clm.physics.canopy_capacity import TemperatureMemory
            p,cold=native.capacity_configuration(report['configuration']);samples,control,configuration,traits=original[-4:]
            columns=len(original[0]);dtype=original[0].dtype
            if (samples.shape!=(columns,len(cold.values)) or control.shape!=(columns,2) or control.dtype.kind not in 'iu'
                    or configuration.shape!=(columns,2) or traits.shape!=(columns,4)
                    or any(v.dtype!=dtype for v in (samples,configuration,traits))):
                raise ValueError('Malformed capacity checkpoint arrays')
            if not np.array_equal(configuration[k],np.asarray([cold.time_step_seconds,cold.initial_temperature_k],dtype)) or not np.array_equal(traits[k],np.asarray(astuple(p),dtype)):
                raise ValueError('Checkpoint capacity parameters disagree with configuration')
            memory=TemperatureMemory(samples[k],float(configuration[k,0]),int(control[k,0]),int(control[k,1]),float(configuration[k,1]))
            for key,value in [('capacity_temperature_count',memory.count),('capacity_temperature_cursor',memory.cursor)]:
                if key not in trace or np.asarray(trace[key]).shape!=(n,) or trace[key][-1]!=value:
                    raise ValueError('Final capacity control/trace alias mismatch')
            if 'capacity_next_mean_temperature_k' not in trace:raise ValueError('Missing capacity mean trace')
            np.testing.assert_allclose(trace['capacity_next_mean_temperature_k'][-1],memory.mean_temperature_k,rtol=1e-6,atol=1e-6)
        snow,soil,canopy,plant,earth=(a[i][k] for i in (0,1,7,11,13))
        storage=np.array([sum(v[:,0].sum() for v in (snow,soil,canopy)),sum(v[:,1].sum() for v in (snow,soil,canopy)),
            plant[:3].sum()+plant[6]+earth[:7].sum(),plant[3:6].sum()+plant[7]+earth[7:].sum()])
        residual=storage-np.asarray(z['initial'][k],float)-np.asarray(z['net'][k],float)
        bounds=np.asarray(z['bounds'][k],float)
    original=np.asarray(trace['cumulative_residual_m_h_c_n'][-1],float)
    # Same extensive expression and float64 order as producer: exact equality.
    if not np.array_equal(residual,original) or not np.array_equal(bounds,np.asarray(report['cumulative_numerical_bounds_m_h_c_n'])):
        raise ValueError('Final extensive storage/ledger disagrees with original trace/report')
    if not np.isfinite(residual).all() or np.any(abs(residual)>bounds):raise ValueError('Final checkpoint conservation failure')
    return dict(checkpoint_sha256=sha(path),extensive_storage_m_h_c_n=storage.tolist(),residual_m_h_c_n=residual.tolist())

def compare_native_year(trace,report,reference,prepared,*,configuration_pair=None,initial_checkpoint=None):
    if report.get('status')!='complete' or report.get('source_unchanged') is not True:
        raise ValueError('Unchanged complete native year required; prefixes remain incomplete')
    n=report['steps_completed']
    if report['steps_requested']!=n or n<=0:raise ValueError('Full original requested trajectory required')
    if configuration_pair is None:
        for predicate,key in [(lambda k:k.endswith('.npz'),'forcing_npz_sha256'),
                (lambda k:k.endswith('-config.json'),'configuration_sha256'),
                (lambda k:k.endswith('.json') and not k.endswith('-config.json'),'forcing_manifest_sha256')]:
            if [v for k,v in prepared['inputs_sha256'].items() if predicate(k)]!=[report[key]]:raise ValueError('Original input identity mismatch')
        binding=dict(scope='Direct exact original forcing/configuration identity')
    else:binding=bind_suction_inputs(report,prepared,*configuration_pair)
    offset=timedelta(hours=prepared['utc_offset_hours'])
    if report['utc_offset_hours']!=prepared['utc_offset_hours'] or report['time_convention']!='local standard time':
        raise ValueError('Explicit matching fixed local-standard time convention required')
    if reference['physical_intervals']!=n or reference['file_count']!=n+1 or len(reference['intervals'])!=n:
        raise ValueError('Complete matching official reference year required')
    dt=float(reference['timestep_seconds'])
    if dt<=0 or 86400%dt or prepared['dt_seconds']!=dt or prepared['steps']!=n:raise ValueError('Protocol timestep/length mismatch')
    starts=np.asarray(trace['time_start']);ends=np.asarray(trace['time_end'])
    if starts.shape!=(n,) or ends.shape!=(n,):raise ValueError('Native time length mismatch')
    times=[datetime.strptime(str(v),'%Y%m%d%H%M') for v in starts];stops=[datetime.strptime(str(v),'%Y%m%d%H%M') for v in ends]
    begin=times[0];finish=datetime(begin.year+1,1,1)
    if begin!=datetime(begin.year,1,1) or stops[-1]!=finish or n*dt!=(finish-begin).total_seconds():
        raise ValueError('Exactly one complete local calendar year required')
    if begin!=datetime.fromisoformat(prepared['initial_time_local']):raise ValueError('Declared original initial date mismatch')
    required=('h_wm2','le_wm2','et_mm','runoff_mm','swe_m','tsoil_k','smc_m3m3')
    layers=len(reference['intervals'][0]['soil_temperature_k'])
    values={k:np.asarray(trace[k],float) for k in required}
    for key,v in values.items():
        expected=(n,layers) if key in ('tsoil_k','smc_m3m3') else (n,)
        if v.shape!=expected or not np.isfinite(v).all():raise ValueError('Malformed native physical series: '+key)
    model_rows=[];reference_rows=[]
    for i,(start,end,row) in enumerate(zip(times,stops,reference['intervals'])):
        if start!=begin+timedelta(seconds=i*dt) or end-start!=timedelta(seconds=dt):raise ValueError('Native time gap or overlap')
        translated=dict(row)
        for key,t in [('time_start',start),('time_end',end)]:
            translated[key]=(datetime.fromisoformat(row[key])+offset).isoformat()
            if translated[key]!=t.isoformat():raise ValueError('Exact original UTC/local alignment required')
        reference_rows.append(translated)
        model_rows.append(dict(time_start=start.isoformat(),time_end=end.isoformat(),
            h_wm2=float(values['h_wm2'][i]),le_wm2=float(values['le_wm2'][i]),et_mm=float(values['et_mm'][i]),
            runoff_mm=float(values['runoff_mm'][i]),swe_mm=float(values['swe_m'][i]*1000.),
            soil_temperature_k=values['tsoil_k'][i].tolist(),soil_total_water_m3m3=values['smc_m3m3'][i].tolist()))
    if datetime.fromisoformat(reference['initial_time'])+offset!=begin:raise ValueError('Original reference initial time mismatch')
    ledger=check_ledgers(trace,report,initial_checkpoint)
    view=dict(reference,initial_time=begin.isoformat(),intervals=reference_rows,
        time_convention=f'Fixed local standard time UTC{prepared["utc_offset_hours"]:+g}; original official UTC timestamps converted explicitly')
    result=compare_columns(dict(status='complete',steps_completed=n,intervals=model_rows),view)
    if len(result['daily'])!=(finish-begin).days or len(result['monthly'])!=12 or result['monthly_bias_criteria_met'] is None:
        raise ValueError('Complete daily/monthly coverage required')
    result.update(scope='Complete native calendar-year numerical diagnostic, explicit original input binding, time conversion, integrals and stored independent conservation traces. Spinup/calibration/reference representativeness require separate evidence.',
        calendar_year=begin.year,configuration_binding=binding,stored_independent_ledgers=ledger,
        scientific_acceptance_claim=False,not_certified=['climate spinup','soil-C long-term equilibrium','calibration/holdout','observation representativeness','independent impurity re-audit'])
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['model','model-manifest','checkpoint','reference','prepared','reference-protocol','reference-archive','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--original-configuration',type=Path);p.add_argument('--configuration',type=Path);p.add_argument('--site',choices=sorted(SITES),required=True);p.add_argument('--initial-checkpoint',type=Path)
    a=p.parse_args()
    if not os.environ.get('PBS_JOBID'):raise RuntimeError('Run evaluation on an EHPC PBS compute node')
    if a.output.exists():raise ValueError('Preserve previous evaluation')
    report=json.loads(a.model_manifest.read_text());reference=json.loads(a.reference.read_text());prepared=json.loads(a.prepared.read_text())
    if report.get('status')!='complete':raise ValueError('Incomplete real native run cannot be evaluated as an annual result')
    if sha(a.model)!=report['model_npz_sha256']:raise ValueError('Original native trace hash mismatch')
    if sha(a.reference_archive)!=ARCHIVE_SHA or sha(a.reference_protocol)!=PROTOCOL_SHA:raise ValueError('Immutable official reference identity required')
    if json.loads(a.reference_protocol.read_text())['instrument_sha256']!=GENERATOR_SHA or reference['source_protocol_sha256']!=PROTOCOL_SHA:
        raise ValueError('Original reviewed official generator/audit protocol required')
    with tarfile.open(a.reference_archive) as archive:
        for name,path in [('protocol.json',a.reference_protocol),(a.site+'/prepared-manifest.json',a.prepared),(a.site+'/audit.json',a.reference)]:
            if archive.extractfile(name).read()!=path.read_bytes():raise ValueError('Reference metadata differs from immutable archive: '+name)
    if not all(sha(k)==v for k,v in prepared['inputs_sha256'].items()):raise ValueError('Original prepared inputs changed')
    pair=None
    supplied=[a.original_configuration is not None,a.configuration is not None]
    if any(supplied) and not all(supplied):raise ValueError('Supply both original/new configuration files and station')
    if all(supplied):pair=(a.original_configuration.read_bytes(),a.configuration.read_bytes(),a.site)
    with np.load(a.model,allow_pickle=False) as z:
        result=compare_native_year(z,report,reference,prepared,configuration_pair=pair,initial_checkpoint=a.initial_checkpoint)
        result['final_checkpoint']=check_final_checkpoint(a.checkpoint,z,report)
    result.update(native_source_sha256=report['source_sha256'],native_backend=report['backend'],native_initial_run=report.get('initial_run'))
    paths=[a.model,a.model_manifest,a.checkpoint,a.reference,a.prepared,a.reference_protocol,a.reference_archive,Path(__file__)]
    paths += [v for v in [a.original_configuration,a.configuration,a.initial_checkpoint] if v is not None]
    result['source_sha256']={str(path):sha(path) for path in paths}
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(calendar_year=result['calendar_year'],days=len(result['daily']),months=len(result['monthly']),daily_bias=result['daily_bias_criteria_met'],monthly_bias=result['monthly_bias_criteria_met'],instantaneous_flux=result['instantaneous_flux_criteria_met'],scientific_acceptance_claim=False)),flush=True)

if __name__=='__main__':main()
