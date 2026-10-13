"""Detailed GPU canopy report: matched inputs, all diagnostics, both gates."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tests"))
from test_gpu_canopy_energy import cases,numpy_reference
from check_canopy_energy import random_cases
from noahmp_clm.gpu.canopy_energy import pack_canopy_case,run_canopy_energy_jax
from noahmp_clm.gpu.cuda.canopy_energy import run_canopy_energy_cuda


def flatten(result):
    for key,value in result.items():
        if isinstance(value,dict):
            for field,array in value.items():
                yield key+"."+field,array
        else:
            yield key,value


def main(output,count):
    import jax,cupy as cp
    inputs=cases(np.float32)
    extra=[pack_canopy_case(area,sw,lw,e,dtype=np.float32,**kw)
           for _,area,sw,lw,e,kw in random_cases(count)]
    if extra:
        inputs=tuple(np.concatenate((a,np.stack([x[i] for x in extra])),axis=0) for i,a in enumerate(inputs))
    j=run_canopy_energy_jax(*inputs,raise_on_failure=False)
    c=run_canopy_energy_cuda(*inputs,raise_on_failure=False)
    report={"scope":"coupled canopy component, not complete column GPU stepping or observational validation",
        "cases":len(inputs[0]),"jax_devices":[str(d) for d in jax.devices()],
        "cuda_pci_bus_id":cp.cuda.runtime.deviceGetPCIBusId(0),
        "numpy_reference":"same float32 inputs and leaf parameters promoted to float64",
        "jax_unconverged":np.flatnonzero(~np.asarray(j["converged"])).tolist(),
        "cuda_unconverged":np.flatnonzero(~cp.asnumpy(c["converged"])).tolist(),
        "cuda_vs_jax32":{},"jax32_vs_numpy64":{}}
    jc,cc=dict(flatten(j)),dict(flatten(c))
    for key in jc:
        actual,expected=cp.asnumpy(cc[key]),np.asarray(jc[key])
        exact=key in ("converged","iterations")
        bad=(actual!=expected) if exact else ~np.isclose(actual,expected,atol=1e-6,rtol=1e-6)
        report["cuda_vs_jax32"][key]={"failed":int(bad.sum()),
            "max_abs":float(np.max(np.abs(actual.astype(float)-expected.astype(float)))),
            "nonfinite":int((~np.isfinite(actual)|~np.isfinite(expected)).sum())}
    references=numpy_reference(inputs)
    thresholds={"gpp":(.1,.1),"leaf_respiration":(.1,.1),"h":(10.,1.),"le":(10.,1.),
        "ground_heat":(10.,1.),"rnet":(10.,1.),"outgoing_shortwave":(10.,1.),"outgoing_longwave":(10.,1.),
        "canopy_humidity":(1e-5,1e-6)}
    for key,(zero,tol) in thresholds.items():
        actual=np.asarray(j[key]);expected=np.asarray([r[key] for r in references])
        delta=abs(actual-expected);near=abs(expected)<=zero
        allowed=np.where(near,tol,.1*abs(expected))
        report["jax32_vs_numpy64"][key]={"max_abs":float(delta.max()),"failed":int((delta>allowed).sum()),
            "nearzero_count":int(near.sum()),"nearzero_max_abs":float(delta[near].max()) if near.any() else None,
            "max_relative_nonzero":float((delta[~near]/abs(expected[~near])).max()) if (~near).any() else None}
    for key,tolerance in (("leaf_temperature",.05),("ground_temperature",.05),("canopy_temperature",.05)):
        delta=abs(np.asarray(j[key])-np.asarray([r[key] for r in references]))
        report["jax32_vs_numpy64"][key]={"max_abs":float(delta.max()),"failed":int((delta>tolerance).sum())}
    report["budgets"]={key:float(np.max(abs(np.asarray(j[key])))) for key in
        ("layer_residual","residual","air_heat_residual","air_water_residual","canopy_water_residual")}
    report["maximum_iterations"]=int(np.asarray(j["iterations"]).max())
    paths=["src/noahmp_clm/gpu/canopy_energy.py","src/noahmp_clm/gpu/cuda/canopy_energy.py",
        "src/noahmp_clm/gpu/cuda/canopy_energy_kernel.cu","src/noahmp_clm/gpu/cuda/leaf_kernel.cu",
        "src/noahmp_clm/gpu/cuda/radiation_kernel.cu","src/noahmp_clm/physics/canopy_energy.py",
        "src/noahmp_clm/physics/soil_uptake.py","src/noahmp_clm/gpu/photosynthesis.py",
        "src/noahmp_clm/gpu/canopy_radiation.py","benchmarks/check_canopy_energy_gpu.py",
        "tests/test_gpu_canopy_energy.py"]
    report["source_sha256"]={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2),flush=True)
    assert not report["jax_unconverged"] and not report["cuda_unconverged"]
    assert all(not row["failed"] for group in ("cuda_vs_jax32","jax32_vs_numpy64") for row in report[group].values())
    assert report["budgets"]["residual"]<.06 and report["budgets"]["air_water_residual"]<1e-8


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--random-cases",type=int,default=256)
    args=parser.parse_args()
    main(args.output,args.random_cases)
