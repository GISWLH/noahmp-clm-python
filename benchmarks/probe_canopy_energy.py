"""Locate the first canopy backend divergence, including pre-Newton fluxes."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tests"))
from test_gpu_canopy_energy import cases
from noahmp_clm.gpu.canopy_energy import run_canopy_energy_jax
from noahmp_clm.gpu.cuda.canopy_energy import run_canopy_energy_cuda


def compare(j,c,cp):
    result={}
    for key in j:
        if key=="gas_exchange":
            result[key]=compare(j[key],c[key],cp)
            continue
        a,e=cp.asnumpy(c[key]),np.asarray(j[key])
        if a.dtype==bool:
            result[key]={"mismatches":int(np.count_nonzero(a!=e))}
            continue
        delta=abs(a-e)
        bad=~np.isclose(a,e,rtol=1e-6,atol=1e-6)
        index=np.unravel_index(np.argmax(delta),delta.shape)
        result[key]={"max_abs":float(delta[index]),"failed":int(bad.sum()),
            "max_index":list(map(int,index)),"cuda":float(a[index]),"jax":float(e[index])}
    return result


if __name__=="__main__":
    import cupy as cp
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    inputs=cases(np.float32)
    report={}
    for iterations in (1,2,100):
        j=run_canopy_energy_jax(*inputs,max_iterations=iterations,raise_on_failure=False)
        c=run_canopy_energy_cuda(*inputs,max_iterations=iterations,raise_on_failure=False)
        report[str(iterations)]=compare(j,c,cp)
        print(iterations,json.dumps(report[str(iterations)]),flush=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
