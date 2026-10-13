"""Additional state metrics from archived trajectories; never rerun physics."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
from noahmp_clm.validation import compare_series


def summarize(report_path,output):
    report=json.loads(report_path.read_text())
    path=report_path.with_suffix('.npz')
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest!=report['series_sha256']: raise ValueError('Trajectory hash mismatch')
    arrays=np.load(path)
    result={'scope':'extra state metrics of the same synthetic archived trajectories',
        'input_report_sha256':hashlib.sha256(report_path.read_bytes()).hexdigest(),'series_sha256':digest,
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'backends':{}}
    ref=arrays['numpy64_states']
    for backend in ('jax64','jax32','cuda32'):
        a=arrays[f'{backend}_states']
        result['backends'][backend]={
            'max_soil_temperature_difference_k':float(np.max(np.abs(a[:,:,:4]-ref[:,:,:4]))),
            'max_soil_water_difference':float(np.max(np.abs(a[:,:,4:8]-ref[:,:,4:8]))),
            'max_canopy_water_difference_m':float(np.max(np.abs(a[:,:,30]-ref[:,:,30]))),
            'canw_daily':[[compare_series(a[c,j:j+48,30],ref[c,j:j+48,30],'canw_m')
                for j in range(0,a.shape[1],48)] for c in range(3)],
            'swe_daily':[[compare_series(a[c,j:j+48,26],ref[c,j:j+48,26],'swe_m')
                for j in range(0,a.shape[1],48)] for c in range(3)]}
    output.write_text(json.dumps(result,indent=2,allow_nan=False))
    failed=sum(not m['bias_pass'] for v in result['backends'].values() for key in ('canw_daily','swe_daily') for c in v[key] for m in c)
    print(json.dumps({'failed':failed,'state_maxima':{k:{f:x for f,x in v.items() if f.startswith('max_')} for k,v in result['backends'].items()}}))
    return bool(failed)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('report',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();raise SystemExit(summarize(a.report,a.output))
