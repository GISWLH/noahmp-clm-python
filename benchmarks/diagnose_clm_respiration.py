"""Replay a fixed original January day, exposing existing respiration components.

No physical code or parameter changes. Original trace fields and every final
state/ledger array must be bitwise identical. Atkin comparisons are conditional
diagnostics under declared trait/temperature assumptions, not a replacement
trajectory, calibration, or a causal attribution of observational errors.
"""
from pathlib import Path
from dataclasses import astuple
from datetime import datetime,timezone
import argparse,hashlib,json,time,traceback
import numpy as np
import run_fluxnet_native as native
from noahmp_clm.gpu.clm_bgc_column import next_bgc_arrays,pack_bgc_environment
from noahmp_clm.physics.canopy_most import MOSTGeometry
from noahmp_clm.physics.photosynthesis import LeafParameters
from fluxnet_deposition import DepositionSchedule

sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
INTERCEPTS={'CN-Cng':2.196,'CN-Qia':1.499,'CN-Cha':1.756,'CN-HaM':2.196}


def single(paths):
    paths=list(paths)
    if len(paths)!=1:raise ValueError('One original immutable generation required')
    return paths[0]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['batch','model-root','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();begin,end=672,720
    if a.output.exists():raise ValueError('Use a fresh diagnostic directory')
    source_root=native.ROOT;manifest=json.loads((source_root/'SOURCE_SHA256.json').read_text())
    verify=lambda:all(sha(source_root/k)==v for k,v in manifest.items())
    if not verify():raise ValueError('Original frozen model source changed')
    cases=native.load_batch(a.batch);count=len(cases)
    if count!=4 or {c['config']['site'] for c in cases}!=set(INTERCEPTS):raise ValueError('Original four-site I batch required')
    for c in cases:
        if c['stamps'][begin].strftime('%m%d%H%M')!='01150000' or c['stamps'][end].strftime('%m%d%H%M')!='01160000':
            raise ValueError('Predeclared January15 diagnostic window required')
    start_path=single(a.model_root.glob(f'state-{begin:07d}-*.npz'));end_path=single(a.model_root.glob(f'state-{end:07d}-*.npz'))
    with np.load(start_path,allow_pickle=False) as z:start=dict(z)
    with np.load(end_path,allow_pickle=False) as z:expected=dict(z)
    if int(start['index'])!=begin or int(expected['index'])!=end:raise ValueError('Original checkpoint index mismatch')
    sources=native.source_hashes();references=[];bindings=[]
    for k,c in enumerate(cases):
        folder=a.model_root/c['case'];report_path=single(folder.glob(f'report-{end:07d}-*.json'));r=json.loads(report_path.read_text())
        trace=folder/Path(r['model_npz']).name
        if (r['status']!='running' or r['steps_requested']!=17520 or r['steps_completed']!=end or r['batch_column']!=k
                or not r['source_unchanged'] or r['source_sha256']!=sources or r['batch_checkpoint_sha256']!=sha(end_path)
                or r['configuration_sha256']!=sha(c['paths']['configuration']) or r['forcing_npz_sha256']!=sha(c['paths']['input'])
                or r['forcing_manifest_sha256']!=sha(c['paths']['input_manifest']) or r['model_npz_sha256']!=sha(trace)):
            raise ValueError('Original report/state/input/trace identity mismatch')
        with np.load(trace,allow_pickle=False) as z:references.append(dict(z))
        bindings.append({str(path):sha(path) for path in [report_path,trace,*c['paths'].values()]})
    a.output.mkdir();started=time.monotonic();engine=native.Backend('cuda32');dtype=np.float32
    leaf=LeafParameters(**cases[0]['config']['leaf_parameters'])
    geometry=np.asarray([astuple(MOSTGeometry(**c['config']['most_geometry'])) for c in cases],dtype)
    schedules=[DepositionSchedule(c['config']) for c in cases];nrates=np.asarray([v.nitrogen_rate for v in schedules],dtype)
    cells=tuple(start[f'a{k}'] for k in range(16));device=tuple(engine.xp.asarray(v) for v in cells)
    initial,net,bounds,maximum,stepmax,tracermax=(start[k].copy() for k in ['initial','net','bounds','maximum','stepmax','tracermax'])
    records={key:[] for key in ['maintenance','growth','unmet','gpp','ra','rh','nee','leaf_potential','leaf_n_area','leaf_area_valid','nitrogen_factor']}
    result=dict(scope=__doc__,backend='cuda32',fixed_begin_index=begin,fixed_end_index=end,source_sha256=sources,
        input_sha256=bindings,checkpoint_sha256={str(start_path):sha(start_path),str(end_path):sha(end_path)},instrument_sha256=sha(__file__),
        started_utc=datetime.now(timezone.utc).isoformat(),steps_completed=0,scientific_acceptance_claim=False)
    try:
        for index in range(begin,end):
            f=np.asarray([c['data']['forcing'][index] for c in cases],dtype)
            env=np.asarray([c['data']['environment'][index] for c in cases],dtype)
            e,cal=pack_bgc_environment([native.calendar(c['stamps'][index],c['metadata'],c['config']['phenology'] is not None) for c in cases],dtype=dtype)
            nitrogen=nrates*f[:,0]
            deposition=np.asarray([d.aerosol(c['stamps'][index],f[k,0]) for k,(c,d) in enumerate(zip(cases,schedules))],dtype)
            before_device=engine.update_forcing(device,env)
            out_device=engine.step(*before_device,f,environment=e,calendar=cal,nitrogen_deposition_gn_m2=nitrogen,
                deposition=deposition,most_geometry=geometry,leaf_parameters=leaf,return_surface=True)
            out=engine.host(out_device);next_device=next_bgc_arrays(before_device,out_device);after=tuple(engine.host(v) for v in next_device)
            incoming,residual,bound,tracer=native.audit_step(cells,after,f,deposition,nitrogen,out)
            next_net=net+incoming;next_bounds=bounds+bound;cumulative=native.totals(after)-initial-next_net
            if not np.all(abs(cumulative)<=next_bounds):raise RuntimeError('Original cumulative numerical conservation bounds exceeded')
            temperature,liquid=engine.soil(next_device);d=out['diagnostics'];g=after[1].astype(float)
            values={key:d[field] for key,field in native.TRACE.items()}
            values.update(swe_m=after[0][:,:,0].astype(float).sum(1)/1000.,lai=after[4][:,0],tsoil_k=temperature,
                smc_m3m3=g[:,:,0]/(1000.*g[:,:,2]),liquid_m3m3=liquid/(1000.*g[:,:,2]),step_residual_m_h_c_n=residual,
                step_bound_m_h_c_n=bound,cumulative_residual_m_h_c_n=cumulative)
            for k,reference in enumerate(references):
                for key,v in values.items():
                    left,right=np.asarray(v[k]),reference[key][index]
                    if left.dtype!=right.dtype or left.tobytes()!=right.tobytes():raise ValueError('Original trace bytes differ: '+cases[k]['case']+'/'+key+'/'+str(index))
                    np.testing.assert_array_equal(left,right)
            bgc=out['biogeochemistry'];allocation=bgc['allocation'];plant=bgc['turnover']['state'].astype(float)
            leaf_area=plant[:,0]*cells[12][:,19].astype(float);valid=leaf_area>0.
            n_area=np.divide(plant[:,3],leaf_area,out=np.zeros(count),where=valid)
            extra=dict(maintenance=allocation['maintenance_respiration_gc_m2'],growth=allocation['growth_respiration_gc_m2'],
                unmet=allocation['unmet_maintenance_gc_m2'],gpp=d['gpp_gc_m2'],ra=d['autotrophic_respiration_gc_m2'],rh=d['heterotrophic_respiration_gc_m2'],nee=d['nee_gc_m2'],
                leaf_potential=engine.host(out_device['diagnostics']['leaf_respiration_umol_m2s']*12.0107e-6*engine.xp.asarray(f[:,0])),leaf_n_area=n_area,leaf_area_valid=valid,nitrogen_factor=bgc['nitrogen_factor_used'])
            for key,v in extra.items():records[key].append(np.array(v,copy=True))
            cells,device=after,next_device;net,bounds=next_net,next_bounds
            maximum=np.maximum(maximum,abs(cumulative));stepmax=np.maximum(stepmax,abs(residual));tracermax=np.maximum(tracermax,tracer)
            result['steps_completed']+=1
            if result['steps_completed']%8==0:print(json.dumps(dict(steps=result['steps_completed'],seconds=time.monotonic()-started)),flush=True)
        final={f'a{k}':v for k,v in enumerate(cells)}
        final.update(index=np.array(end),initial=initial,net=net,bounds=bounds,maximum=maximum,stepmax=stepmax,tracermax=tracermax)
        if set(final)!=set(expected):raise ValueError('Original final sixteen-state/ledger schema mismatch')
        for key,v in final.items():
            if v.dtype!=expected[key].dtype or v.tobytes()!=expected[key].tobytes():raise ValueError('Original checkpoint bytes differ: '+key)
            np.testing.assert_array_equal(v,expected[key],err_msg='Final original checkpoint/'+key)
        arrays={key:np.asarray(v) for key,v in records.items()};np.savez_compressed(a.output/'components.npz',**arrays)
        np.savez_compressed(a.output/'final-checkpoint.npz',**final);site_results={}
        for k,c in enumerate(cases):
            valid=arrays['leaf_area_valid'][:,k];n=arrays['leaf_n_area'][:,k];intercept=INTERCEPTS[c['config']['site']]
            # Conditional published Eq2.17.1 with the existing declared static
            # growth-temperature assumption, not a new inferred ten-day state.
            reference=intercept+.2061*n-.0402*(leaf.growth_temperature-273.15)
            site_results[c['config']['site']]=dict(local_day=c['stamps'][begin].date().isoformat(),
                maintenance_gc_m2=arrays['maintenance'][:,k].astype(float).sum(0).tolist(),growth_gc_m2=arrays['growth'][:,k].astype(float).sum(0).tolist(),
                unmet_maintenance_gc_m2=arrays['unmet'][:,k].astype(float).sum(0).tolist(),
                carbon_integrals_gc_m2={key:float(arrays[key][:,k].astype(float).sum()) for key in ['gpp','ra','rh','nee','leaf_potential']},
                leaf_n_area_range_gn_m2_leaf=None if not valid.any() else [float(n[valid].min()),float(n[valid].max())],
                existing_respiration25_umol_m2s=leaf.respiration25,existing_growth_temperature_k=leaf.growth_temperature,
                conditional_atkin_intercept=intercept,conditional_atkin_rd25_range=None if not valid.any() else [float(reference[valid].min()),float(reference[valid].max())],
                conditional_rd25_max_absolute_difference=None if not valid.any() else float(abs(reference[valid]-leaf.respiration25).max()),
                scope='Model components and conditional reference only. No PFT selection by skill, dynamic ten-day acclimation, parameter change, observational attribution or acceptance.')
        result.update(status='complete',all_original_trace_fields_bitwise_equal=True,final_all_23_arrays_bitwise_equal=True,sites=site_results,
            maximum_step_residual_m_h_c_n=stepmax.max(0).tolist(),maximum_cumulative_residual_m_h_c_n=maximum.max(0).tolist())
    except BaseException:result.update(status='failed',failure=traceback.format_exc())
    result.update(seconds=time.monotonic()-started,source_unchanged=verify(),finished_utc=datetime.now(timezone.utc).isoformat())
    if not result['source_unchanged']:result['status']='failed'
    result['artifacts_sha256']={p.name:sha(p) for p in a.output.glob('*.npz')}
    (a.output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_sha256','input_sha256')}),flush=True)
    if result['status']!='complete':raise RuntimeError('Original-day diagnostic incomplete; retain its failure record')

if __name__=='__main__':main()
