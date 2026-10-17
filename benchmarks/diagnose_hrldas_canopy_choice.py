"""Controlled first-week official canopy-option diagnostic on a PBS CPU node.

Compare unchanged monthly-table option4 with prescribed initial-LAI option9.
Option9 may also alter SAI/fveg; report all three rather than call this a
leaf-only intervention. No production parameters, original reference or model
year request changes. Every chosen station/mode is retained regardless of score.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime,timedelta
import hashlib,json,os,re,shutil,socket,subprocess,tarfile,time
from pathlib import Path,PurePosixPath
import numpy as np
from scipy.io import netcdf_file
from audit_hrldas_point_output import audit
from compare_hrldas_climate_prefix import read_json,SITES
from noahmp_clm.column import SoilParams
from noahmp_clm.physics.soil_enthalpy import SoilEnthalpyState
from noahmp_clm.validation import compare_series

sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def save(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def extract_selected(archive,destination,selected,expected):
    destination.mkdir(parents=True);found=set()
    with tarfile.open(archive,'r|gz') as t:
        for m in t:
            if m.name not in selected:continue
            name=PurePosixPath(m.name);assert m.isfile() and len(name.parts)==1 and not name.is_absolute()
            data=t.extractfile(m).read();assert hashlib.sha256(data).hexdigest()==expected[m.name]
            (destination/m.name).write_bytes(data);found.add(m.name)
            if found==selected:break
    assert found==selected,(archive,selected-found)

def condition_arrays(directory,files):
    keys=['LAI','SAI','FVEG','SWFORC','LWFORC','COSZ','FSA','FIRA','SAV','SAG','GRDFLX',
          'TG','TV','HFX','LH','ALBEDO','SOIL_T','SOIL_W','SOIL_M','SNEQV','CANLIQ','CANICE']
    result={k:[] for k in keys};units={}
    for filename in files:
        with netcdf_file(directory/filename,mmap=False) as nc:
            for key in keys:
                v=nc.variables[key];unit=v.units.decode() if isinstance(v.units,bytes) else str(v.units)
                if key in units:assert units[key]==unit
                units[key]=unit;result[key].append(np.asarray(v.data,float).ravel().copy())
    return {k:np.asarray(v) for k,v in result.items()},units

def score(model,reference):
    """Descriptive counterfactual scores only, never a final acceptance decision."""
    pairs={}
    for key in ['h_wm2','le_wm2','et_mm']:
        factor=48. if key=='et_mm' else 1.;variable='et_mm_day' if key=='et_mm' else key
        pairs[variable]=compare_series(np.asarray(model[key],float)*factor,np.asarray([r[key] for r in reference],float)*factor,variable)
    pairs['tsoil_c_layer_1']=compare_series(np.asarray(model['tsoil_k'])[:,0]-273.15,
        np.asarray([r['soil_temperature_k'][0] for r in reference])-273.15,'tsoil_c')
    return pairs

def one(task):
    reference,model_root,runtime,output,site=map(Path,task);directory=output/site;directory.mkdir()
    prepared=read_json(reference/site/'prepared-manifest.json');original=read_json(reference/site/'audit.json')
    start=datetime.fromisoformat(prepared['initial_time_utc']);files=[(start+timedelta(seconds=1800*i)).strftime('%Y%m%d%H%M')+'.LDASOUT_DOMAIN1' for i in range(337)]
    forcing_files={f.replace('LDASOUT','LDASIN') for f in files}
    expected=dict(prepared['forcing_sha256']);expected['hrldas_setup_single_point.nc']=prepared['setup_sha256']
    extract_selected(reference/(str(site)+'-forcing.tar.gz'),directory/'forcing',forcing_files|{'hrldas_setup_single_point.nc'},expected)
    original_hashes={Path(k).name:v for k,v in original['source_sha256'].items()}
    extract_selected(reference/(str(site)+'-output.tar.gz'),directory/'original-output',set(files),original_hashes)
    candidates=list((model_root/(str(site)+'-initial-climate-cycle')).glob('report-0000336-*.json'))
    assert len(candidates)==1;model_report=read_json(candidates[0]);trace_path=candidates[0].parent/Path(model_report['model_npz']).name
    assert sha(trace_path)==model_report['model_npz_sha256'] and model_report['status']=='running' and model_report['steps_requested']==17520
    assert model_report['steps_completed']==336 and model_report['utc_offset_hours']==prepared['utc_offset_hours']
    assert model_report['configuration_sha256']==next(v for k,v in prepared['inputs_sha256'].items() if k.endswith('-config.json'))
    assert model_report['forcing_npz_sha256']==next(v for k,v in prepared['inputs_sha256'].items() if k.endswith('.npz'))
    with np.load(trace_path,allow_pickle=False) as z:model=dict(z)
    for i,row in enumerate(original['intervals'][:336]):
        for name in ['time_start','time_end']:
            assert datetime.strptime(str(model[name][i]),'%Y%m%d%H%M')==datetime.fromisoformat(row[name])+timedelta(hours=prepared['utc_offset_hours'])
    ground=model_report['initial_state']['ground'].copy()
    if 'saturated_suction_m' not in ground['params']:
        raise ValueError('Historical soil enthalpy must be diagnosed using its original frozen source')
    ground['params']=SoilParams(**ground['params'])
    for key in ['water','enthalpy','dz']:ground[key]=np.asarray(ground[key],float)
    state=SoilEnthalpyState(**ground)
    result=dict(site=str(site),original_model_status=model_report['status'],original_model_steps_requested=17520,
        scientific_acceptance_claim=False,model_lai_min_max=[float(np.min(model['lai'])),float(np.max(model['lai']))],
        initial_model_tsoil_k=state.temperature.tolist(),initial_model_liquid_m3m3=(state.liquid/(1000*state.dz)).tolist(),
        original_official_initial_state=original['initial_state'],modes={})
    nml_template=(reference/site/'namelist.hrldas').read_text()
    for mode,option in [('monthly4-control',4),('prescribed-initial9',9)]:
        run=directory/mode;run.mkdir();(run/'output').mkdir()
        for p in (runtime/'bondville').iterdir():
            if p.suffix=='.TBL' or p.name.startswith('snicar_'):(run/p.name).symlink_to(p.resolve())
        nml=nml_template
        replacements={'HRLDAS_SETUP_FILE':'"'+str(directory/'forcing/hrldas_setup_single_point.nc')+'"',
            'INDIR':'"'+str(directory/'forcing')+'/"','OUTDIR':'"'+str(run/'output')+'/"','KDAY':7,'DYNAMIC_VEG_OPTION':option}
        for key,value in replacements.items():
            nml,count=re.subn(r'(?m)^(\s*'+re.escape(key)+r'\s*=)[^\n]*',lambda m:m[1]+' '+str(value),nml);assert count==1
        (run/'namelist.hrldas').write_text(nml)
        environment=os.environ.copy();environment.pop('LD_LIBRARY_PATH',None)
        command=[str(runtime/'lib/ld-linux-x86-64.so.2'),'--library-path',str(runtime/'lib'),str(runtime/'bin/hrldas.exe')]
        begun=time.monotonic()
        with (run/'official.log').open('wb') as log:code=subprocess.run(command,cwd=run,env=environment,stdout=log,stderr=subprocess.STDOUT).returncode
        save(run/'run-receipt.json',dict(returncode=code,seconds=time.monotonic()-begun,command=command,node=socket.gethostname(),pbs_job=os.environ['PBS_JOBID'],protocol_sha256=sha(output/'protocol.json')))
        assert code==0
        audited=audit(run/'output',initial_time=start,timestep_seconds=1800);assert audited['physical_intervals']==336
        save(run/'audit.json',audited)
        arrays,units=condition_arrays(run/'output',files);np.savez_compressed(run/'raw-conditions.npz',**arrays)
        if option==4:
            same=sum(sha(run/'output'/f)==original_hashes[f] for f in files)
            result['control_ldasout_byte_equal']=same;assert same==337,'Original full-year prefix replay must be unchanged before interpreting the intervention'
        active=arrays['LAI'][1:,0]>0.
        for key in ['LAI','SAI','FVEG','FSA','FIRA','SAV','SAG','GRDFLX']:
            assert np.all(np.isfinite(arrays[key][1:])) and np.all(abs(arrays[key][1:])<1e30),key
        conditions={key:dict(units=units[key],initial=arrays[key][0].tolist() if key in ['LAI','SAI','FVEG'] else None,
            first_physical=arrays[key][1].tolist(),minimum=float(np.min(arrays[key][1:])),maximum=float(np.max(arrays[key][1:]))) for key in ['LAI','SAI','FVEG','FSA','FIRA','SAV','SAG','GRDFLX']}
        if np.any(active):assert np.all(np.isfinite(arrays['TV'][1:][active]))
        result['modes'][mode]=dict(option=option,initial_fluxes_undefined_and_not_scored=True,conditions=conditions,comparisons_to_fixed_clm_prefix=score(model,audited['intervals']))
        archive=run/'official-output.tar.gz'
        with tarfile.open(archive,'w:gz') as t:
            for f in sorted((run/'output').iterdir()):t.add(f,arcname=f.name,recursive=False)
    save(directory/'summary.json',result)
    print(json.dumps(dict(site=str(site),control_byte_equal=result['control_ldasout_byte_equal'],model_lai=result['model_lai_min_max'],official_lai={k:v['conditions']['LAI'] for k,v in result['modes'].items()})),flush=True)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['reference','model-root','runtime','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();assert os.environ.get('PBS_JOBID'),'Physical official runs require an EHPC PBS allocation'
    assert not a.output.exists();a.output.mkdir()
    manifest=read_json(a.runtime/'MANIFEST.json')['files'];assert all(sha(a.runtime/k)==v for k,v in manifest.items())
    protocol=dict(scope=__doc__,site_modes={s:[4,9] for s in SITES},scored_steps=336,days=7,dt_seconds=1800,
        reference_protocol_sha256=sha(a.reference/'protocol.json'),source_sha256=sha(__file__),
        changes='Only dynamic vegetation4->9 within separately matched seven-day runs; paths and duration relocated for both. SAI/fveg coeffects are reported. Existing setup initial LAI unchanged, not fitted.',
        node=socket.gethostname(),pbs_job=os.environ['PBS_JOBID'])
    save(a.output/'protocol.json',protocol)
    tasks=[tuple(map(str,[a.reference,a.model_root,a.runtime,a.output]))+(s,) for s in SITES]
    with ProcessPoolExecutor(max_workers=3) as pool:results=list(pool.map(one,tasks))
    assert all(sha(a.runtime/k)==v for k,v in manifest.items())
    save(a.output/'completed.json',dict(results=results,original_runtime_unchanged=True,protocol_sha256=sha(a.output/'protocol.json')))

if __name__=='__main__':main()
