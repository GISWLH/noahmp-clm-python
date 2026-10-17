"""Audit measured curve conditions and ordering without fitting or model calls.

The source C3-grass archive is CC-BY-4.0, Han et al. (2022),
https://doi.org/10.25581/ornlsfa.027/1887896 . This reports evidence about
experimental comparability; it neither excludes curves nor changes gates.
"""
from pathlib import Path
import argparse,csv,hashlib,io,json,math,statistics,zipfile
from datetime import datetime

SOURCE_SHA256='936cff54c647e84bb4fc59e380c6d234888f8e47722635b00d24e55068c83cd5'
FIELDS=('PARi','Tleaf','CO2S','CO2R','CO2i','AnetCO2','FmorFm.','FoorFs','StomCond','Trmmol','VpdL','Tair','RH_R','AirPress','OxygenPress')

def audit(path):
    payload=Path(path).read_bytes();actual=hashlib.sha256(payload).hexdigest()
    if actual!=SOURCE_SHA256:raise ValueError('Archive differs from the recorded source')
    files=[];dark_groups={}
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for name in sorted(archive.namelist()):
            raw=list(csv.DictReader(io.StringIO(archive.read(name).decode('cp1252'))))
            if not raw:raise ValueError('Empty measurement file: '+name)
            units=raw[0];measurements=[]
            for line,row in enumerate(raw[1:],start=3):
                stamp=datetime.strptime(row['ObsDate']+' '+row['HHMMSS'],'%Y-%m-%d %H:%M:%S')
                values={key:float(row[key]) for key in FIELDS}
                if not all(math.isfinite(v) and v!=-9999 for v in values.values()):
                    raise ValueError(f'Missing or nonfinite required measurement in {name}:{line}')
                code=float(row['DataType'])
                if not code.is_integer():raise ValueError('Nonintegral DataType')
                measurements.append(dict(stamp=stamp,values=values,code=int(code),raw=row))
            groups={label:[r for r in measurements if predicate(r['code'])] for label,predicate in
                [('dark',lambda c:c==0),('aci',lambda c:11<=c<=25),('light',lambda c:31<=c<=45),
                 ('other',lambda c:not(c==0 or 11<=c<=25 or 31<=c<=45))]}
            summaries={}
            for label,rows in groups.items():
                summary=dict(count=len(rows),codes=sorted({r['code'] for r in rows}))
                if rows:
                    stamps=[r['stamp'] for r in rows]
                    summary.update(first=min(stamps).isoformat(),last=max(stamps).isoformat(),
                        duration_minutes=(max(stamps)-min(stamps)).total_seconds()/60.,
                        chronological_in_file=stamps==sorted(stamps),ranges={})
                    for key in FIELDS:
                        values=[r['values'][key] for r in rows]
                        summary['ranges'][key]=dict(min=min(values),max=max(values),mean=statistics.fmean(values),unit=units[key])
                summaries[label]=summary
            dark=groups['dark'];comparability={}
            if len(dark)==1:
                fingerprint=hashlib.sha256(json.dumps(dark[0]['raw'],sort_keys=True).encode()).hexdigest()
                dark_groups.setdefault(fingerprint,[]).append(name)
                comparability['dark_record_sha256']=fingerprint
                for label in ('aci','light'):
                    if groups[label]:
                        stamps=[r['stamp'] for r in groups[label]]
                        comparability['dark_minus_'+label+'_end_minutes']=(dark[0]['stamp']-max(stamps)).total_seconds()/60.
            if groups['aci'] and groups['light']:
                for key in ('Tleaf','CO2i','CO2S','VpdL','StomCond'):
                    comparability['light_minus_aci_mean_'+key]=summaries['light']['ranges'][key]['mean']-summaries['aci']['ranges'][key]['mean']
                comparability['light_end_minus_aci_start_minutes']=(max(r['stamp'] for r in groups['light'])-min(r['stamp'] for r in groups['aci'])).total_seconds()/60.
            files.append(dict(file=name,groups=summaries,comparability=comparability))
    return dict(scope='Read-only audit of original measured conditions/timestamps. No fitting, production physics, exclusions or acceptance changes. Differences do not by themselves prove physiological state changes.',
        source_doi='https://doi.org/10.25581/ornlsfa.027/1887896',license='CC-BY-4.0',archive_sha256=actual,
        instrument_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),files=files,
        identical_dark_record_groups=[dict(record_sha256=k,files=v) for k,v in sorted(dark_groups.items())])

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('archive',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=audit(a.archive);a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(files=len(result['files']),identical_dark_record_groups=result['identical_dark_record_groups'])),flush=True)
