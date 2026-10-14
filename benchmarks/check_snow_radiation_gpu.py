"""Large three-backend SNICAR/canopy optical audit, not a snow evolution run."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,time
import numpy as np
import cupy as cp
import jax
from noahmp_clm.gpu.snicar import run_snow_shortwave_jax
from noahmp_clm.gpu.cuda.snicar import run_snow_shortwave_cuda
from noahmp_clm.gpu.canopy_snow_radiation import run_canopy_snow_jax
from noahmp_clm.gpu.cuda.canopy_snow_radiation import run_canopy_snow_cuda

ROOT=Path(__file__).resolve().parents[1]


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def host(result):
    return {k:host(v) if isinstance(v,dict) else (cp.asnumpy(v) if isinstance(v,cp.ndarray) else np.asarray(v)) for k,v in result.items()}


def flatten(result,prefix=''):
    for k,v in result.items():
        if isinstance(v,dict):yield from flatten(v,prefix+k+'.')
        else:yield prefix+k,v


def same_precision(a,b):
    aa,bb=dict(flatten(a)),dict(flatten(b));errors={}
    for key in aa:
        np.testing.assert_allclose(aa[key],bb[key],rtol=1e-6,atol=1e-6)
        errors[key]=float(np.max(aa[key]!=bb[key]) if aa[key].dtype==bool else np.max(np.abs(aa[key]-bb[key])))
    return errors


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path);args=ap.parse_args()
    start=time.perf_counter();assert jax.config.x64_enabled and any(d.platform=='gpu' for d in jax.devices())
    snicar=load('snow_reference','tests/test_gpu_snicar.py');coupled=load('coupled_reference','tests/test_gpu_canopy_snow_radiation.py')
    report={'scope':__doc__,'backends':str(jax.devices())}
    for name,ref,functions,count in [('snow',snicar,(run_snow_shortwave_jax,run_snow_shortwave_cuda),360),
                                    ('canopy_snow',coupled,(run_canopy_snow_jax,run_canopy_snow_cuda),320)]:
        data=ref.cases(np.float64,count)
        quant=tuple(v.astype(np.float32) if v.dtype.kind=='f' else v for v in data)
        runs={'jax64':host(functions[0](*data)),'jax32':host(functions[0](*quant)),'cuda32':host(functions[1](*quant))}
        entry={'cases':len(data[0]),'same_precision':same_precision(runs['cuda32'],runs['jax32'])}
        for backend,results in runs.items():
            values=data if backend=='jax64' else quant;metrics={};max_budget=0.;minimum_absorbed=0.
            names=('absorbed','spectral_absorbed','spectral_ground_absorbed','spectral_reflected','ground_absorbed','reflected','albedo') if name=='snow' else (
                'canopy_absorbed','absorbed_par','snow_absorbed','soil_absorbed','reflected','snow_direct_albedo','snow_diffuse_albedo')
            for row in range(len(data[0])):
                expected=ref.reference(*(v[row] for v in values))
                for key in names:
                    actual=np.asarray(results[key][row]);target=np.asarray(expected[key]);err=np.abs(actual-target)
                    floor=1e-3 if 'albedo' in key else 1.;mask=np.abs(target)>floor
                    metric=metrics.setdefault(key,dict(max_absolute=0.,max_relative_percent=0.,relative_reference_floor=floor))
                    metric['max_absolute']=max(metric['max_absolute'],float(np.max(err)))
                    metric['max_relative_percent']=max(metric['max_relative_percent'],float(np.max(err[mask]/np.abs(target[mask]),initial=0.)*100.))
                    snicar.assert_science_close(actual,target,relative=.1 if backend!='jax64' else 1e-7,
                        absolute=.1 if backend!='jax64' and 'albedo' not in key else 1e-6,zero_floor=floor)
                if name=='snow':
                    absorbed=results['absorbed'][row];incoming=values[2][row,:4].astype(float).sum();ground=float(results['ground_absorbed'][row])
                else:
                    absorbed=np.r_[results['canopy_absorbed'][row],results['snow_absorbed'][row]]
                    incoming=float(values[3][row,0]);ground=float(results['soil_absorbed'][row])
                budget=incoming-float(results['reflected'][row])-absorbed.astype(float).sum()-ground
                max_budget=max(max_budget,abs(budget));minimum_absorbed=min(minimum_absorbed,float(np.min(absorbed)))
                assert abs(budget)<(.005 if backend!='jax64' else 1e-8)
            assert minimum_absorbed>=(-.001 if backend!='jax64' else -1e-8)
            entry[backend]=dict(numpy64=metrics,max_independent_radiation_budget_wm2=max_budget,minimum_absorbed_wm2=minimum_absorbed)
        report[name]=entry;print(name,json.dumps(entry,indent=2),flush=True)
    files=[Path(__file__),ROOT/'tests/test_gpu_snicar.py',ROOT/'tests/test_gpu_canopy_snow_radiation.py']+list((ROOT/'src/noahmp_clm').rglob('*.py'))+list((ROOT/'src/noahmp_clm/gpu/cuda').glob('*.cu'))+[ROOT/'src/noahmp_clm/data/snow_tables.npz']
    report['source_sha256']={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    report['wall_seconds']=time.perf_counter()-start
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2,allow_nan=False))
    print('Saved',args.output,'seconds',report['wall_seconds'],flush=True)


if __name__=='__main__':main()
