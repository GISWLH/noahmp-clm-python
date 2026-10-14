"""Expanded native fixed surface/thermal water boundary audit; synthetic only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tests'))
from test_gpu_canopy_boundary import cases,numpy_reference,check_science,check_ledgers,reference_arguments
from test_canopy_energy import independent_energy_reference
from test_gpu_snicar import assert_science_close


def host(result,cp):
    return {k:host(v,cp) if isinstance(v,dict) else cp.asnumpy(v) if isinstance(v,cp.ndarray) else np.asarray(v)
        for k,v in result.items()}


def metrics(result,refs):
    data={}
    for key,value in result.items():
        if key in ('iterations','converged','gas_exchange'):continue
        expected=np.asarray([r[key] for r in refs]);difference=abs(value-expected)
        floor=1e-6 if key in ('canopy_humidity','wet_evaporation','transpiration','ground_evaporation','root_uptake') else .001 if key=='canopy_water_after_kgm2' else 1.
        valid=abs(expected)>floor
        data[key]=dict(max_absolute=float(difference.max()),zero_floor=floor,
            max_relative_percent=None if 'temperature' in key or not np.any(valid) else float(np.max(difference[valid]/abs(expected[valid]))*100.))
    return data


def main():
    import jax
    import cupy as cp
    from noahmp_clm.gpu.canopy_boundary import run_canopy_boundary_jax
    from noahmp_clm.gpu.cuda.canopy_boundary import run_canopy_boundary_cuda
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Use a fresh report path')
    start=time.perf_counter();report={'kind':'synthetic fixed ground canopy and stored water heat, not winter column/observations',
        'cases':425,'versions':dict(numpy=np.__version__,jax=jax.__version__,cupy=cp.__version__),
        'devices':str(jax.devices()),'backends':{},'same_precision':{},'independent':[]}
    for dtype in (np.float64,np.float32):
        arrays=cases(dtype,400);refs=numpy_reference(arrays)
        runs=[('jax64' if dtype==np.float64 else 'jax32',run_canopy_boundary_jax)]
        if dtype==np.float32:runs.append(('cuda32',run_canopy_boundary_cuda))
        for name,run in runs:
            got=host(run(*arrays),cp)
            check_science(got,refs,dtype)
            for key,values in got['gas_exchange'].items():
                expected=np.asarray([r['gas_exchange'][key] for r in refs])
                assert_science_close(values,expected,absolute=.1 if dtype==np.float32 else .0005,
                    relative=.1 if dtype==np.float32 else 1e-5,zero_floor=1.)
            report['backends'][name]=dict(errors=metrics(got,refs),ledgers=check_ledgers(got,arrays),
                max_layer_residual_wm2=float(np.max(abs(got['layer_residual']))),max_iterations=int(got['iterations'].max()))
            if name=='jax32':j32=got
            if name=='cuda32':
                for key,values in got.items():
                    fields=values if isinstance(values,dict) else [None]
                    for field in fields:
                        a,b=(values[field],j32[key][field]) if field else (values,j32[key])
                        label=key+'.'+field if field else key
                        if a.dtype.kind in 'ib':
                            np.testing.assert_array_equal(a,b);report['same_precision'][label]=int(np.max(a!=b))
                        else:
                            np.testing.assert_allclose(a,b,rtol=1e-6,atol=1e-6)
                            report['same_precision'][label]=float(np.max(abs(a-b)))
            for i in (0,3,5,11,17,21,23):
                a,s,l,e,kw=reference_arguments(tuple(v[i] for v in arrays))
                independent=independent_energy_reference(a,s,l,e,**kw)
                delta={'backend':name,'case':i,'temperature_k':float(np.max(abs(got['leaf_temperature'][i]-independent['temperature'][:10])))}
                assert delta['temperature_k']<.05
                for key in ('h','le','ground_heat','gpp'):
                    delta[key]=float(abs(got[key][i]-independent[key]))
                    assert_science_close(got[key][i],independent[key],absolute=.1,zero_floor=1.)
                report['independent'].append(delta)
    report['elapsed_s']=time.perf_counter()-start
    report['source_sha256']={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in ('src','tests','benchmarks') for p in (ROOT/folder).rglob('*')
        if p.is_file() and '__pycache__' not in p.parts and not any(v.endswith('.egg-info') for v in p.parts)}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('backends','source_sha256','independent')}) ,flush=True)


if __name__=='__main__':main()
