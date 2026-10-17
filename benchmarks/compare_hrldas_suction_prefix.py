"""Bind a new explicit-suction CLM configuration to unchanged official inputs.

The official original prepared manifest is never rewritten. This is limited
to the declared suction revision: every other configuration value must match,
including all fields not used by the official generator. Scoring stays a
cold first-week diagnostic; complete annual/spinup acceptance is separate.
"""
from pathlib import Path
import argparse,copy,hashlib,json,os,tarfile
import numpy as np
from compare_hrldas_climate_prefix import _compare_aligned_prefix,read_json

PROTOCOL_SHA='a13336a71c3aad8ea612a0e85489959df8b7f568f834f866c6f5c3d8f71d3eff'
ARCHIVE_SHA='168131a520bcfa9f6cd5a2a274778beb0e7483d34f80337d22708a22a275c82c'
GENERATOR_SHA='7c4d7963b11526942ca326d105088bfd4d74f23c2b0ecf4ebfa2520fc4fafca6'
SITES={'CN-Qia':.355,'CN-Cha':.355,'CN-HaM':.263}
digest=lambda raw:hashlib.sha256(raw).hexdigest()
sha=lambda p:digest(Path(p).read_bytes())

def bind_suction_configurations(old_raw,new_raw,prepared,site):
    old=json.loads(old_raw);new=json.loads(new_raw);before=digest(old_raw);after=digest(new_raw)
    identities=[v for k,v in prepared['inputs_sha256'].items() if k.endswith('-config.json')]
    if identities!=[before]:raise ValueError('Original official configuration bytes disagree')
    if site not in SITES or old.get('case')!=site+'-initial-climate-cycle' or new.get('case')!=site+'-suction-climate-cycle':
        raise ValueError('Declared station/case revision required')
    if new.get('parent_configuration_sha256')!=before:raise ValueError('Exact original parent configuration required')
    revision=new.get('physics_revision',{})
    if (revision.get('law')!='CLM5Eq2.6.52 with explicit positive saturated suction head'
            or revision.get('saturated_suction_m')!=SITES[site] or revision.get('old_checkpoints_reused') is not False
            or revision.get('forcing_water_temperature_carbon_nitrogen_seed_policy_unchanged') is not True):
        raise ValueError('Unexpected suction experiment declaration')
    a,b=copy.deepcopy(old),copy.deepcopy(new)
    for v in [a,b]:
        for key in ['case','scope','parent_configuration_sha256','physics_revision']:v.pop(key,None)
    if 'saturated_suction_m' in a['column']['soil'] or b['column']['soil'].pop('saturated_suction_m',None)!=SITES[site]:
        raise ValueError('Only the declared new explicit suction is permitted')
    if a!=b:raise ValueError('Another physical/configuration field changed; this reference-reuse binding is invalid')
    return dict(scope='Derived comparison binding only; original official prepared manifest and input identities remain unchanged.',
        site=site,original_configuration_sha256=before,new_configuration_sha256=after,
        saturated_suction_m=SITES[site],all_other_configuration_values_equal=True,
        permitted_changes=['case','scope','parent_configuration_sha256','physics_revision','column.soil.saturated_suction_m'])

def bind_suction_inputs(report,prepared,old_raw,new_raw,site):
    binding=bind_suction_configurations(old_raw,new_raw,prepared,site)
    if report['configuration_sha256']!=binding['new_configuration_sha256']:raise ValueError('New model report/configuration bytes disagree')
    if report.get('case')!=site+'-suction-climate-cycle':raise ValueError('New model report case mismatch')
    for predicate,key in [(lambda k:k.endswith('.npz'),'forcing_npz_sha256'),
            (lambda k:k.endswith('.json') and not k.endswith('-config.json'),'forcing_manifest_sha256')]:
        values=[v for k,v in prepared['inputs_sha256'].items() if predicate(k)]
        if values!=[report[key]]:raise ValueError('Original forcing/metadata byte identity required')
    return binding


def compare_suction_prefix(trace,report,reference,prepared,old_raw,new_raw,site):
    binding=bind_suction_inputs(report,prepared,old_raw,new_raw,site)
    result=_compare_aligned_prefix(trace,report,reference,prepared)
    result['configuration_binding']=binding
    result['scope']='Declared explicit-suction first-week cold-start diagnostic against an unchanged official full year. Derived configuration binding; not regenerated reference, annual/spinup or scientific acceptance.'
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model-root',type=Path,required=True)
    p.add_argument('--configuration-root',type=Path,required=True);p.add_argument('--reference-root',type=Path,required=True)
    p.add_argument('--reference-archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if not os.environ.get('PBS_JOBID'):raise RuntimeError('Execute comparison on an EHPC PBS compute node')
    if args.output.exists():raise ValueError('Preserve previous comparison')
    if sha(args.reference_archive)!=ARCHIVE_SHA or sha(args.reference_root/'protocol.json')!=PROTOCOL_SHA:
        raise ValueError('Original immutable official reference identity required')
    protocol=read_json(args.reference_root/'protocol.json')
    if protocol['instrument_sha256']!=GENERATOR_SHA:raise ValueError('Original reviewed official generator required')
    results={}
    with tarfile.open(args.reference_archive) as archive:
        names=['protocol.json']+[site+'/'+name for site in SITES for name in ['prepared-manifest.json','audit.json','run-receipt.json']]
        for name in names:
            if archive.extractfile(name).read()!=(args.reference_root/name).read_bytes():raise ValueError('Official metadata differs from original archive: '+name)
        for site in SITES:
            folder=args.model_root/(site+'-suction-climate-cycle');candidates=list(folder.glob('report-0000336-*.json'))+list(folder.glob('report-0000336-*.json.gz'))
            if len(candidates)!=1:raise ValueError('Require one immutable original336 model report')
            report_path=candidates[0];report=read_json(report_path);trace_path=folder/Path(report['model_npz']).name
            if sha(trace_path)!=report['model_npz_sha256']:raise ValueError('Original model trace hash mismatch')
            with np.load(trace_path,allow_pickle=False) as z:trace=dict(z)
            prepared_path=args.reference_root/site/'prepared-manifest.json';reference_path=args.reference_root/site/'audit.json'
            prepared=read_json(prepared_path);reference=read_json(reference_path)
            if reference['source_protocol_sha256']!=PROTOCOL_SHA:raise ValueError('Reference audit protocol mismatch')
            paths=prepared['inputs_sha256'];old_config=next(Path(k) for k in paths if k.endswith('-config.json'))
            if not all(sha(k)==v for k,v in paths.items()):raise ValueError('Original source input hash mismatch')
            new_config=args.configuration_root/(site+'-config.json')
            # These hashes bind actual original weather and metadata files,
            # not only a model-reported string or a cached equality boolean.
            for suffix in ['.npz','.json']:
                old=next(Path(k) for k in paths if k.endswith(suffix) and not k.endswith('-config.json'))
                if sha(old)!=sha(args.configuration_root/(site+suffix)):raise ValueError('New weather/metadata differs from official source')
            result=compare_suction_prefix(trace,report,reference,prepared,old_config.read_bytes(),new_config.read_bytes(),site)
            result['source_sha256']={str(x):sha(x) for x in [report_path,trace_path,prepared_path,reference_path,old_config,new_config,Path(__file__)]}
            results[site]=result
    output=dict(scope='New explicit-suction I336 versus unchanged original HRLDAS reference. Every old/new configuration and weather identity verified; no manifest or report rewriting.',
        node=os.uname().nodename,pbs_job=os.environ['PBS_JOBID'],original_official_protocol_sha256=PROTOCOL_SHA,original_official_archive_sha256=ARCHIVE_SHA,results=results)
    args.output.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps({site:dict(h_r=r['instantaneous_fluxes']['h_wm2']['r'],le_r=r['instantaneous_fluxes']['le_wm2']['r'],week_h_bias=r['whole_prefix_means']['h_wm2']['relative_bias'],daily_bias_criteria=r['diagnostic_all_daily_bias_criteria_met'],flux_r_criteria=r['diagnostic_flux_r_criteria_met']) for site,r in results.items()}),flush=True)

if __name__=='__main__':main()
