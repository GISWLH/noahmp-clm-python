"""Audit complete repeated climate cycles; report endpoints, not a release.

Original source/input/transaction hashes and carried numerical ledgers are
required. Annualized soil-C rates and an actually observed 100-cycle change
are different outputs. No physical time stepping or observational scoring.
"""
from pathlib import Path
from datetime import datetime
import argparse,json,os,re
import numpy as np
import run_fluxnet_native as native
from run_fluxnet_column import sha,validate_input
from compare_hrldas_native_year import check_ledgers,check_final_checkpoint
from noahmp_clm.column import SoilParams
from noahmp_clm.physics.soil_enthalpy import SoilEnthalpyState
from noahmp_clm.physics.decomposition import POOLS

LIMITS=dict(soil_temperature_k=.1,soil_water_m3m3=.001,swe_m=.0005,
            minimum_cycles=2,consecutive_endpoint_checks=2,
            canopy_sai_relative=.1,canopy_sai_near_zero_m2m2=.0001,canopy_sai_absolute_m2m2=1e-6,
            soil_carbon_percent_per_century=5.,soil_carbon_near_zero_gc_m2=1.)


def require(condition,message):
    if not condition:raise ValueError(message)


def compare_endpoints(before,after):
    result={}
    require(('canopy_sai_m2m2' in before)==('canopy_sai_m2m2' in after),'Canopy structure endpoint schema mismatch')
    if 'canopy_sai_m2m2' in before:
        a,b=(np.asarray(v['canopy_sai_m2m2'],float) for v in (before,after))
        require(a.shape==b.shape==() and np.isfinite(a) and np.isfinite(b) and a>=0 and b>=0,'Invalid canopy SAI endpoint')
        change=float(abs(b-a));near=bool(a<=LIMITS['canopy_sai_near_zero_m2m2'])
        result.update(canopy_sai_change_m2m2=change,canopy_sai_before_m2m2=float(a),canopy_sai_after_m2m2=float(b),
            canopy_sai_relative_change=None if near else change/float(a),
            canopy_structure_criterion_met=change<=(LIMITS['canopy_sai_absolute_m2m2'] if near else LIMITS['canopy_sai_relative']*float(a)))
    for key in ('soil_temperature_k','soil_water_m3m3','swe_m'):
        a,b=(np.asarray(v[key],float) for v in (before,after))
        require(a.shape==b.shape and a.size>0 and np.isfinite(a).all() and np.isfinite(b).all(),'Invalid endpoint '+key)
        result[key]=float(abs(b-a).max())
    a,b=(np.asarray(v['soil_carbon_gc_m2'],float) for v in (before,after))
    require(a.shape==b.shape==(7,) and np.isfinite(a).all() and np.isfinite(b).all() and np.all(a>=0) and np.all(b>=0),'Invalid seven-pool soil carbon endpoint')
    delta=b-a;total=float(a.sum());change=float(delta.sum());soc_before=float(a[4:].sum());soc_after=float(b[4:].sum());soc_change=float(delta[4:].sum())
    rate=None if total<=LIMITS['soil_carbon_near_zero_gc_m2'] else 10000.*change/total
    soc_rate=None if soc_before<=LIMITS['soil_carbon_near_zero_gc_m2'] else 10000.*soc_change/soc_before
    result.update(water_heat_criteria_met=all(result[k]<=LIMITS[k] for k in ('soil_temperature_k','soil_water_m3m3','swe_m')),
        soil_carbon_change_gc_m2=delta.tolist(),total_soil_carbon_change_gc_m2=change,
        total_soil_carbon_before_gc_m2=total,total_soil_carbon_after_gc_m2=float(b.sum()),
        signed_total_soil_carbon_rate_percent_per_100_climate_years=rate,
        total_soil_carbon_rate_criterion_met=None if rate is None else abs(rate)<=LIMITS['soil_carbon_percent_per_century'],
        carbon_pool_names=POOLS,total_carbon_scope='All seven litter/wood/SOM pools; SOC is separately restricted to fast/slow/passive.',
        soil_organic_carbon_before_gc_m2=soc_before,soil_organic_carbon_after_gc_m2=soc_after,soil_organic_carbon_change_gc_m2=soc_change,
        signed_soc_rate_percent_per_100_climate_years=soc_rate,
        soc_rate_criterion_met=None if soc_rate is None else abs(soc_rate)<=LIMITS['soil_carbon_percent_per_century'],
        per_pool_rate_percent_per_100_climate_years=[None if x<=LIMITS['soil_carbon_near_zero_gc_m2'] else float(10000.*y/x) for x,y in zip(a,delta)],
        rate_scope='One complete repeated climate year, scaled to 100 years; not an observed century. Near-zero carbon has only absolute changes, no relative pass.')
    return result


def _frame(saved,k,identity=None):
    arrays=native.checkpoint_arrays(saved,identity or {})
    ground=native.initial_ground(arrays,k);ground['params']=SoilParams(**ground['params'])
    temperature=SoilEnthalpyState(**ground).temperature
    soil=np.asarray(arrays[1][k],float);carbon=np.asarray(arrays[13][k,:7],float)
    frame=dict(soil_temperature_k=temperature,soil_water_m3m3=soil[:,0]/(1000.*soil[:,2]),
        swe_m=float(np.asarray(arrays[0][k,:,0],float).sum()/1000.),soil_carbon_gc_m2=carbon)
    if len(arrays) in (19,23):frame['canopy_sai_m2m2']=float(arrays[16][k,0])
    return frame


def _inputs(batch,input_root):
    rows=json.loads(Path(batch).read_text())['cases'];require(bool(rows),'Nonempty original batch required')
    loaded=[];names=set()
    for row in rows:
        name=row['case'];require(name not in names and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',name),'Unique safe original cases required');names.add(name)
        paths={k:Path(input_root)/Path(row[k]).name for k in ('input','input_manifest','configuration')}
        meta=json.loads(paths['input_manifest'].read_text());config=json.loads(paths['configuration'].read_text())
        with np.load(paths['input'],allow_pickle=False) as z:data=dict(z)
        stamps=validate_input(data,meta,paths['input']);n=len(stamps);dt=float(data['forcing'][0,0])
        require(stamps[0]==datetime(stamps[0].year,1,1) and str(data['time_end'][-1])==datetime(stamps[0].year+1,1,1).strftime('%Y%m%d%H%M')
                and n*dt==(datetime(stamps[0].year+1,1,1)-stamps[0]).total_seconds(),'Exactly one complete original forcing calendar year required')
        require(all(config[key]==meta[key] for key in ('latitude','longitude','utc_offset_hours')),'Original configuration/forcing geometry mismatch')
        site=meta.get('site');require(site==config['site'] if site is not None else '_'+config['site']+'_' in meta['archive_name'],'Original station identity mismatch')
        identity=dict(case=name,forcing_npz_sha256=sha(paths['input']),forcing_manifest_sha256=sha(paths['input_manifest']),configuration_sha256=sha(paths['configuration']))
        loaded.append(dict(identity=identity,paths=paths,config=config,metadata=meta,data=data,steps=n))
    require(len({c['steps'] for c in loaded})==1,'Original batch length mismatch')
    return loaded


def _initial_generation(folder,report,rows,source_hashes):
    candidates=list(folder.glob('state-0000000-*.npz'));require(len(candidates)==1,'One original zero-step generation required')
    path=candidates[0];generation=path.stem.removeprefix('state-');state_sha=sha(path);evidence={str(path):state_sha}
    with np.load(path,allow_pickle=False) as z:saved=dict(z)
    require(saved['index'].item()==0,'Initial zero-step generation required')
    native.checkpoint_arrays(saved,report['identity'])
    if report['identity']['compare_jax32']:native.checkpoint_arrays(saved,report['identity'],shadow=True)
    for k,(case,(_,trace)) in enumerate(zip(report['identity']['cases'],rows)):
        sub=folder/case['case'];p=sub/('report-'+generation+'.json');r=json.loads(p.read_text());trace0=sub/r['model_npz']
        require(trace0.parent==sub and sha(trace0)==r['model_npz_sha256'],'Original initial trace hash mismatch')
        require(r['steps_completed']==0 and r['status']=='running' and r['source_unchanged'] and r['batch_checkpoint_sha256']==state_sha
                and r['batch_column']==k and r['source_sha256']==source_hashes and all(r[key]==v for key,v in case.items() if key!='case'),'Original initial generation identity mismatch')
        with np.load(trace0,allow_pickle=False) as z:require(all(len(z[n])==0 for n in z.files),'Initial trace must be empty')
        evidence.update({str(p):sha(p),str(trace0):sha(trace0)})
    return path,saved,evidence


def assess_cycles(batch,input_root,folders,model_source_manifest):
    loaded=_inputs(batch,input_root);require(bool(folders),'At least one complete cycle required')
    model_manifest=json.loads(Path(model_source_manifest).read_text());sources=native.source_hashes()
    require(all(model_manifest.get(k)==v for k,v in sources.items()),'Evaluator and original model physical source mismatch')
    identities=[c['identity'] for c in loaded];evidence={str(Path(batch)):sha(batch),str(Path(model_source_manifest)):sha(model_source_manifest)}
    for c in loaded:
        evidence.update({str(p):sha(p) for p in c['paths'].values()})
    state_count=native.native_state_count(loaded)
    previous=None;history=[];consecutive=[0]*len(loaded);structure_checks=[0]*len(loaded);carbon_history=[];last_frames=None
    for cycle,folder in enumerate(map(Path,folders),1):
        declared=json.loads((folder/'batch.json').read_text());n=loaded[0]['steps']
        # Authenticate the committed case list before following its paths.
        require(declared['status']=='complete' and declared['source_unchanged'] and declared['steps_completed']==declared['steps_requested']==n,'Unchanged complete original year required; running/failed prefixes are not cycles')
        require(declared['identity']['cases']==identities and declared['identity']['batch_sha256']==sha(batch) and declared['identity']['source_sha256']==sources,'Cycle source/input/order identity mismatch')
        require(declared['identity'].get('native_state_arrays',16)==state_count,'Cycle structure state/configuration mismatch')
        require(set(declared.get('case_reports',{}))=={v['case'] for v in identities},'Immutable case generation descriptors required')
        report,saved,rows=native.read_generation(folder);require(report==declared,'Committed generation changed while reading')
        link=report['identity'].get('initial_run')
        if previous is None:require(link is None and report['ledger_steps']==n and report['ledger_segments']==1,'Full cold-root cycle history required')
        else:
            old,old_state,old_folder=previous
            require(report['identity']['backend']==old['identity']['backend'] and report['identity']['compare_jax32']==old['identity']['compare_jax32'],'Cycle backend/shadow identity mismatch')
            require(link is not None and link['mode']=='cycle' and link['batch_report_sha256']==sha(old_folder/'batch.json')
                    and link['checkpoint_sha256']==old['checkpoint_sha256'] and link['previous_ledger_steps']==old['ledger_steps']
                    and link['previous_segments']==old['ledger_segments'] and report['ledger_steps']==old['ledger_steps']+n
                    and report['ledger_segments']==old['ledger_segments']+1,'Original linked cycle/ledger chain required')
        initial_path,initial,initial_evidence=_initial_generation(folder,report,rows,sources);evidence.update(initial_evidence)
        if previous is None:
            dtype=initial['a0'].dtype;environment=np.asarray([c['data']['environment'][0] for c in loaded],dtype)
            expected=native.pack_station_arrays([native.build_state(c['config']) for c in loaded],state_count,dtype=dtype,
                co2_ppm=environment[:,0],cosine_zenith=environment[:,1],diffuse_fraction=environment[:,2],
                leaf_dimension=np.asarray([c['config']['leaf_dimension_m'] for c in loaded],dtype))
            for i,value in enumerate(expected):
                actual=initial['a'+str(i)]
                require(actual.dtype==value.dtype and actual.shape==value.shape and actual.tobytes()==value.tobytes(),'Cold state differs from original configuration/forcing seed: a'+str(i))
                if report['identity']['compare_jax32']:
                    shadow=initial['shadow'+str(i)]
                    require(shadow.dtype==value.dtype and shadow.shape==value.shape and shadow.tobytes()==value.tobytes(),'Cold shadow differs from original configuration/forcing seed: shadow'+str(i))
            require(np.array_equal(initial['initial'],native.totals(native.checkpoint_arrays(initial,report['identity']))),'Cold initial extensive storage mismatch')
            require(all(np.all(initial[k]==0) for k in ('net','bounds','maximum','stepmax','tracermax')),'Cold numerical ledgers must start at zero')
            if report['identity']['compare_jax32']:
                require(all(np.all(initial[k]==0) for k in ('shadow_net','shadow_bounds','shadow_maximum')),'Cold shadow numerical ledgers must start at zero')
            last_frames=[_frame(initial,k,report['identity']) for k in range(len(loaded))];carbon_history.append([v['soil_carbon_gc_m2'].copy() for v in last_frames])
        else:
            old_state=previous[1];require(initial.keys()==old_state.keys(),'Carried checkpoint field schema mismatch')
            for key in initial.keys()-{'index'}:
                a,b=initial[key],old_state[key]
                require(a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes(),'Carried state/ledger bytes differ: '+key)
        frame=[];case_results=[];checkpoint=folder/report['checkpoint_file']
        for k,((r,trace),c) in enumerate(zip(rows,loaded)):
            require(r['status']=='complete' and r['source_unchanged'] and r['source_sha256']==sources and r['batch_column']==k
                    and r['steps_completed']==r['steps_requested']==n and r.get('initial_run')==link and r['configuration']==c['config']
                    and r['time_convention']=='local standard time' and r['utc_offset_hours']==c['metadata']['utc_offset_hours'],'Original case completion/configuration/time identity mismatch')
            require(all(np.array_equal(trace[key],c['data'][key]) for key in ('time_start','time_end')),'Complete original calendar alignment required')
            ledger=check_ledgers(trace,r,None if previous is None else previous[2]/previous[0]['checkpoint_file'])
            storage=check_final_checkpoint(checkpoint,trace,r);endpoint=_frame(saved,k,report['identity'])
            require(np.array_equal(endpoint['soil_water_m3m3'],trace['smc_m3m3'][-1]) and endpoint['swe_m']==trace['swe_m'][-1],'Final water/snow state aliases mismatch')
            native_temperature=np.asarray(trace['tsoil_k'][-1],float)
            require(native_temperature.shape==endpoint['soil_temperature_k'].shape and np.isfinite(native_temperature).all(),'Finite matching soil temperature layers required')
            decode=float(abs(native_temperature-endpoint['soil_temperature_k']).max())
            require(decode<=LIMITS['soil_temperature_k'],'Native/independent endpoint temperature decoding differs by more than the convergence scale')
            result=compare_endpoints(last_frames[k],endpoint);consecutive[k]=consecutive[k]+1 if result['water_heat_criteria_met'] else 0
            structure=result.get('canopy_structure_criterion_met')
            structure_checks[k]=structure_checks[k]+1 if structure else 0
            result.update(case=identities[k]['case'],site=c['config']['site'],consecutive_water_heat_checks=consecutive[k],
                water_heat_converged=cycle>=LIMITS['minimum_cycles'] and consecutive[k]>=LIMITS['consecutive_endpoint_checks'],
                consecutive_structure_checks=None if structure is None else structure_checks[k],
                canopy_structure_converged=None if structure is None else cycle>=LIMITS['minimum_cycles'] and structure_checks[k]>=LIMITS['consecutive_endpoint_checks'],
                maximum_temperature_decode_difference_k=decode,stored_independent_ledgers=ledger,final_checkpoint=storage)
            case_results.append(result);frame.append(endpoint)
            p=folder/identities[k]['case']/report['case_reports'][identities[k]['case']]['file'];t=p.parent/r['model_npz']
            evidence.update({str(p):sha(p),str(t):sha(t)})
        evidence.update({str(folder/'batch.json'):sha(folder/'batch.json'),str(checkpoint):sha(checkpoint)})
        history.append(dict(cycle=cycle,cases=case_results));carbon_history.append([v['soil_carbon_gc_m2'].copy() for v in frame]);last_frames=frame;previous=(report,saved,folder)
    observed_century=observed_carbon_century(carbon_history,[c['identity']['case'] for c in loaded])
    return dict(scope=__doc__,limits=LIMITS,cycles=history,all_columns_water_heat_converged=all(v['water_heat_converged'] for v in history[-1]['cases']),
        all_columns_water_heat_sai_converged=all(v['water_heat_converged'] and v['canopy_structure_converged'] is not False for v in history[-1]['cases']),
        soil_carbon_observed_century=observed_century,soil_c_long_term_equilibrium_claim=False,nitrogen_equilibrium_claim=False,
        scientific_acceptance_claim=False,temperature_method='Same NumPy enthalpy decoder applied to original initial/final stores; native final temperature discrepancy reported and limited to the predeclared 0.1K convergence scale.',
        not_certified=['Long-term soil-C equilibrium by extrapolating short-window rates','Total N steady state in a positive-deposition/no-loss simplification','Observation/calibration/holdout acceptance','Independent re-derivation of every boundary flux from stored traces'],
        evidence_sha256=evidence)


def observed_carbon_century(carbon_history,cases):
    values=np.asarray(carbon_history,float)
    require(values.ndim==3 and values.shape[1:]==(len(cases),7) and len(values)>=2 and np.isfinite(values).all() and np.all(values>=0),'Finite seven-pool carbon endpoint history required')
    result=[];cycles=len(values)-1
    for k,case in enumerate(cases):
        old=values[-101,k] if cycles>=100 else None;now=values[-1,k]
        change=None if old is None or old.sum()<=LIMITS['soil_carbon_near_zero_gc_m2'] else float(100.*(now.sum()-old.sum())/old.sum())
        soc=None if old is None or old[4:].sum()<=LIMITS['soil_carbon_near_zero_gc_m2'] else float(100.*(now[4:].sum()-old[4:].sum())/old[4:].sum())
        per_pool=[None]*7 if old is None else [None if a<=LIMITS['soil_carbon_near_zero_gc_m2'] else float(100.*(b-a)/a) for a,b in zip(old,now)]
        result.append(dict(case=case,observed_complete_climate_years=cycles,actual_100_cycle_change_percent=change,
            actual_100_cycle_criterion_met=None if change is None else abs(change)<=LIMITS['soil_carbon_percent_per_century'],
            actual_100_cycle_soc_change_percent=soc,actual_100_cycle_soc_criterion_met=None if soc is None else abs(soc)<=LIMITS['soil_carbon_percent_per_century'],
            carbon_pool_names=POOLS,actual_100_cycle_per_pool_change_percent=per_pool))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('batch','input-root','model-source-manifest','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--segments',type=Path,nargs='+',required=True);a=p.parse_args()
    if not os.environ.get('PBS_JOBID'):raise RuntimeError('Run evaluation on an EHPC PBS compute node')
    if a.output.exists():raise ValueError('Preserve previous assessment')
    result=assess_cycles(a.batch,a.input_root,a.segments,a.model_source_manifest)
    result['instrument_sha256']=sha(__file__);result['pbs_job']=os.environ['PBS_JOBID']
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(cycles=len(result['cycles']),water_heat_converged=result['all_columns_water_heat_converged'],soil_c_long_term_equilibrium_claim=False,scientific_acceptance_claim=False)))


if __name__=='__main__':main()
