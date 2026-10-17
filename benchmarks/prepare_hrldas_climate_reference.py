"""Prepare and execute independent official three-site first-climate-year references.

Run physical integration in an EHPC PBS job. This freezes standard Noah-MP
table/phenology choices before outputs; it is not a matched-parameter or spun-up
CLM acceptance experiment. Input timestamps are explicitly local standard -> UTC.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime,timedelta
import hashlib,json,os,re,socket,subprocess,time
from pathlib import Path
import numpy as np
from scipy.io import netcdf_file

SITES={
    'CN-Qia':dict(vegetation=1,soil=6,elevation_m=102.,elevation_source='https://asiaflux.net/?action=multidatabase_action_main_filedownload&download_flag=1&metadata_id=41&upload_id=53'),
    'CN-Cha':dict(vegetation=5,soil=6,elevation_m=738.,elevation_source='https://asiaflux.net/?action=multidatabase_action_main_filedownload&download_flag=1&metadata_id=41&upload_id=53'),
    'CN-HaM':dict(vegetation=10,soil=9,elevation_m=3250.,elevation_source='Existing selected JPF/ICOS product metadata, docs/v1-results/fluxnet-site-metadata-audit.json'),
}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def clone_nc(template,destination,updates,title):
    """Keep the official generator's schema/units; replace explicit site values."""
    with netcdf_file(template,mmap=False) as src,netcdf_file(destination,'w',version=1) as dst:
        for name,size in sorted(src.dimensions.items(),key=lambda x:x[1] is not None):dst.createDimension(name,size)
        for name,value in src._attributes.items():setattr(dst,name,value)
        dst.TITLE=title
        if not set(updates)<=set(src.variables):raise ValueError('Unknown official NetCDF variable')
        for name,old in src.variables.items():
            var=dst.createVariable(name,old.typecode(),old.dimensions)
            for key,value in old._attributes.items():setattr(var,key,value)
            data=np.array(old.data,copy=True)
            if name in updates:data[...] = np.asarray(updates[name]).reshape(data.shape) if np.size(updates[name])>1 else updates[name]
            var[:]=data
    with netcdf_file(destination,mmap=False) as check:
        for name,value in updates.items():
            expected=np.broadcast_to(np.asarray(value).reshape(check.variables[name].data.shape) if np.size(value)>1 else value,check.variables[name].data.shape)
            np.testing.assert_array_equal(check.variables[name].data,np.asarray(expected,dtype=check.variables[name].data.dtype))

def prepare(inputs,runtime,output):
    if output.exists():raise ValueError('Preserve previous experiment directory')
    runtime=runtime.resolve();output=output.resolve();output.mkdir(parents=True)
    template=runtime/'bondville/forcing';nml_template=(runtime/'bondville/namelist.hrldas').read_text()
    forcing_template=next(iter(sorted(template.glob('*.LDASIN_DOMAIN1'))))
    runtime_manifest=json.loads((runtime/'MANIFEST.json').read_text())['files']
    assert all(sha(runtime/k)==v for k,v in runtime_manifest.items())
    specification=dict(scope='Official three-site full first-climate-year cold references; no spinup, calibration or CLM acceptance claim.',
        node=socket.gethostname(),pbs_job=os.environ.get('PBS_JOBID'),instrument_sha256=sha(__file__),
        official_binary_sha256=sha(runtime/'bin/hrldas.exe'),runtime=str(runtime),runtime_manifest_sha256=sha(runtime/'MANIFEST.json'),
        source_upstream=dict(hrldas='cd96df470220f7d7133cdbccd5f9c5355cf173e2',noahmp='17751dcd7a2442a3c57138f530f523daf08dc1c5'),
        fixed_choices=dict(dynamic_vegetation=4,precipitation_partition=5,surface_runoff=6,subsurface_runoff=6,
            snow_albedo=1,bottom_temperature=1,co2_ppm=395.,direct_shortwave_fraction=.7),
        limitations=[
            'Three sites chosen a priori because literature-baseline temperature/humidity/wind heights agree within each site; CN-Cng needs a separately justified unequal-height conversion.',
            'Noah standard MODIS monthly-table phenology (option4), roots, optics and leaf parameters differ from coupled CLM C/N/GDD phenology; this is a structural reference, not an isolated solver comparison.',
            'Noah STAS loam6 for Qia/Cha differs from the existing nominal CLM loam. HaM uses STAS9 in both. No table is changed to fit outputs.',
            'BATS snow albedo versus CLM SNICAR, fixed395ppm versus station CO2, and fixed0.7/0.3 shortwave split versus station estimates remain explicit differences.',
            'Same unobserved initial uniform temperature/water and zero snow seeds as current CLM climate cycle. No model is spun up; cold-start scores cannot establish final acceptance.',
            'HaM precipitation is the original gap-filled JPF product with unresolved credibility. No scaling or score-based exclusion.',
            'Qia/Cha elevation102/738m uses site-researcher2007 newsletter; later broader AsiaFlux site pages list100/731m. Values are fixed before outputs.',
        ],cases=[])
    for site,site_parameters in SITES.items():
        npz=inputs/(site+'.npz');metadata=inputs/(site+'.json');configuration=inputs/(site+'-config.json')
        meta=json.loads(metadata.read_text());config=json.loads(configuration.read_text())
        if sha(npz)!=meta['prepared_npz_sha256']:raise ValueError('Original forcing mismatch')
        with np.load(npz,allow_pickle=False) as z:data={k:z[k] for k in ['forcing','time_start','time_end']}
        f=data['forcing'];dt=float(f[0,0]);assert dt==1800. and np.all(f[:,0]==dt) and np.all(np.isfinite(f))
        start=[datetime.strptime(str(x),'%Y%m%d%H%M') for x in data['time_start']]
        end=[datetime.strptime(str(x),'%Y%m%d%H%M') for x in data['time_end']]
        assert all((b-a).total_seconds()==dt for a,b in zip(start,end)) and start[1:]==end[:-1]
        assert start[0].hour==start[0].minute==0 and len(f)*dt%86400==0
        offset=timedelta(hours=config['utc_offset_hours']);utc_start=start[0]-offset
        geometry=config['most_geometry'];height=float(geometry['temperature_height'])
        assert height==geometry['humidity_height']==geometry['wind_height']
        column=config['column'];dz=np.asarray(column['dz']);assert len(dz)==4
        directory=output/site;forcing=directory/'forcing';forcing.mkdir(parents=True);(directory/'output').mkdir()
        for p in (runtime/'bondville').iterdir():
            if p.suffix=='.TBL' or p.name.startswith('snicar_'):(directory/p.name).symlink_to(p.resolve())
        values=dict(XLAT=config['latitude'],XLONG=config['longitude'],TMN=column['tsoil'][-1],
            HGT=site_parameters['elevation_m'],SEAICE=0.,SHDMAX=100.*column['fveg'],SHDMIN=100.*column['fveg'],
            LAI=column['lai'],XLAND=1,IVGTYP=site_parameters['vegetation'],ISLTYP=site_parameters['soil'],
            SNOW=1000.*column['swe'],SNODEP=0.,CANWAT=1000.*column['canw'],TSK=column['ts'],
            DZS=dz,ZS=np.cumsum(dz)-dz/2.,TSLB=column['tsoil'],SMOIS=column['smc'])
        assert values['SNOW']==0.,'Nonzero initial snow requires an observed or declared density/depth'
        clone_nc(template/'hrldas_setup_single_point.nc',forcing/'hrldas_setup_single_point.nc',values,'Prepared official-schema FLUXNET first-climate-year reference')
        forcing_hashes={}
        # HRLDAS reads the right-end record at equal forcing/model dt. Initial
        # record duplicates the first interval only for IO initialization.
        records=[(utc_start,f[0])]+[(stamp-offset,row) for stamp,row in zip(end,f)]
        for stamp,row in records:
            target=forcing/(stamp.strftime('%Y%m%d%H%M')+'.LDASIN_DOMAIN1')
            updates=dict(T2D=row[1],Q2D=row[2],U2D=row[3],V2D=0.,PSFC=row[4],SWDOWN=row[5],LWDOWN=row[6],RAINRATE=row[7]*1000.)
            clone_nc(forcing_template,target,updates,'Original interval meteorology, local-standard timestamps converted to UTC')
            forcing_hashes[target.name]=sha(target)
        replacements=dict(HRLDAS_SETUP_FILE='"'+str(forcing/'hrldas_setup_single_point.nc')+'"',INDIR='"'+str(forcing)+'/"',
            OUTDIR='"'+str(directory/'output')+'/"',START_YEAR=utc_start.year,START_MONTH=utc_start.month,START_DAY=utc_start.day,
            START_HOUR=utc_start.hour,START_MIN=utc_start.minute,KDAY=int(len(f)*dt/86400),ZLVL=height,
            DYNAMIC_VEG_OPTION=4,PCP_PARTITION_OPTION=5,SURFACE_RUNOFF_OPTION=6,SUBSURFACE_RUNOFF_OPTION=6)
        replacements.update({f'soil_thick_input({i+1})':float(x) for i,x in enumerate(dz)})
        nml=nml_template
        for key,value in replacements.items():
            nml,count=re.subn(r'(?m)^(\s*'+re.escape(key)+r'\s*=)[^\n]*',lambda m:m[1]+' '+str(value),nml)
            assert count==1,(key,count)
        (directory/'namelist.hrldas').write_text(nml)
        case=dict(site=site,**site_parameters,height_m=height,initial_time_utc=utc_start.isoformat(),
            initial_time_local=start[0].isoformat(),utc_offset_hours=config['utc_offset_hours'],
            steps=len(f),dt_seconds=dt,days=int(len(f)*dt/86400),
            inputs_sha256={str(p):sha(p) for p in [npz,metadata,configuration]},
            setup_sha256=sha(forcing/'hrldas_setup_single_point.nc'),namelist_sha256=sha(directory/'namelist.hrldas'),
            forcing_records=len(records),forcing_sha256=forcing_hashes,
            time_convention='FLUXNET interval-end stamp minus explicit UTC offset; original interval value at that UTC right endpoint. First interval duplicated only for initial IO.',
            original_clm_soil=column['soil'],original_clm_geometry=geometry)
        save(directory/'prepared-manifest.json',case)
        specification['cases'].append({k:v for k,v in case.items() if k!='forcing_sha256'})
        print(json.dumps(dict(prepared=site,records=len(records),utc_start=utc_start.isoformat())),flush=True)
    save(output/'protocol.json',specification)

def run_one(task):
    output,site,runtime=map(Path,task);directory=output/site;spec=json.loads((directory/'prepared-manifest.json').read_text())
    assert not list((directory/'output').iterdir())
    assert sha(directory/'namelist.hrldas')==spec['namelist_sha256']
    assert sha(directory/'forcing/hrldas_setup_single_point.nc')==spec['setup_sha256']
    assert all(sha(directory/'forcing'/k)==v for k,v in spec['forcing_sha256'].items())
    environment=os.environ.copy();environment.pop('LD_LIBRARY_PATH',None)
    command=[str(runtime/'lib/ld-linux-x86-64.so.2'),'--library-path',str(runtime/'lib'),str(runtime/'bin/hrldas.exe')]
    started=time.monotonic()
    with (directory/'official.log').open('wb') as log:code=subprocess.run(command,cwd=directory,env=environment,stdout=log,stderr=subprocess.STDOUT).returncode
    result=dict(site=str(site),returncode=code,seconds=time.monotonic()-started,node=socket.gethostname(),pbs_job=os.environ.get('PBS_JOBID'),
        command=command,protocol_sha256=sha(output/'protocol.json'),manifest_sha256=sha(directory/'prepared-manifest.json'),
        ldasout_files=len(list((directory/'output').glob('*.LDASOUT_DOMAIN1'))))
    save(directory/'run-receipt.json',result)
    if code!=0:raise RuntimeError(f'Official {site} exited {code}; retain original log')
    from audit_hrldas_point_output import audit
    audited=audit(directory/'output',initial_time=datetime.fromisoformat(spec['initial_time_utc']),timestep_seconds=spec['dt_seconds'])
    assert audited['physical_intervals']==spec['steps']
    audited['source_protocol_sha256']=result['protocol_sha256'];save(directory/'audit.json',audited)
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['prepare','run'])
    parser.add_argument('--inputs',type=Path);parser.add_argument('--runtime',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.action=='prepare':prepare(args.inputs,args.runtime,args.output)
    else:
        if not os.environ.get('PBS_JOBID'):raise RuntimeError('Official physical simulation requires PBS compute allocation')
        protocol=json.loads((args.output/'protocol.json').read_text());tasks=[(str(args.output),s,str(args.runtime)) for s in SITES]
        assert [x['site'] for x in protocol['cases']]==list(SITES)
        with ProcessPoolExecutor(max_workers=3) as pool:results=list(pool.map(run_one,tasks))
        runtime_manifest=json.loads((args.runtime/'MANIFEST.json').read_text())['files']
        assert all(sha(args.runtime/k)==v for k,v in runtime_manifest.items())
        save(args.output/'completed.json',dict(results=results,runtime_unchanged=True,protocol_sha256=sha(args.output/'protocol.json')))
        print(json.dumps(results),flush=True)

if __name__=='__main__':main()
