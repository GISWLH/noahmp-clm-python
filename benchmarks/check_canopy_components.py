"""Quantified CPU continuous-equation and optional real GPU component audit."""
from __future__ import annotations
import argparse
import hashlib
from itertools import product
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tests"))
from test_canopy_radiation import continuous_reference
from test_gpu_canopy_radiation import cases,random_cases,longwave_cases
from test_gpu_photosynthesis import leaf_cases,ordinary,random_leaves
from noahmp_clm.physics.canopy_radiation import two_stream_shortwave,canopy_longwave
from noahmp_clm.physics.photosynthesis import LeafParameters
from noahmp_clm.gpu.photosynthesis import LEAF_OUTPUTS


def metrics(actual,expected,near=1.,same_precision=False):
    a,e=np.asarray(actual,dtype=float),np.asarray(expected,dtype=float)
    err=abs(a-e)
    mask=abs(e)>near
    result = {"max_abs":float(np.max(err,initial=0.)),
        "max_relative_above_near_zero":float(np.max(err[mask]/abs(e[mask]),initial=0.)),
        "near_zero_max_abs":float(np.max(err[~mask],initial=0.)),
        "near_zero_reference_threshold":near,"n_values":int(a.size),
        "all_finite":bool(np.all(np.isfinite(a)))}
    if same_precision:
        result["same_precision_1e_6_failures"]=int(np.sum(err>1e-6+1e-6*abs(e)))
    return result


def audit(gpu=False):
    actual={k:[] for k in ("absorbed","reflected","ground_absorbed")}
    expected={k:[] for k in actual}
    residual=[]
    for lai,opt,mu in product([0.,.1,3.,8.],[(.1,.05),(.45,.25),(.6,.4)],[.05,.5,1.]):
        area=np.full(10,lai/10.)
        a=two_stream_shortwave(area,400.,200.,leaf_reflectance=opt[0],
            leaf_transmittance=opt[1],ground_albedo=.3,cosine_zenith=mu)
        e=continuous_reference(area,*opt,mu)
        for k in actual:
            actual[k].extend(np.atleast_1d(a[k]));expected[k].extend(np.atleast_1d(e[k]))
        residual.append(abs(a["residual"]))
    result={"radiation_continuous_reference":{
        k:metrics(actual[k],expected[k]) for k in actual},
        "radiation_cpu_budget_max_wm2":max(residual)}
    if gpu:
        import jax
        import cupy as cp
        from noahmp_clm.gpu.photosynthesis import run_leaves_jax
        from noahmp_clm.gpu.cuda.photosynthesis import run_leaves_cuda
        from noahmp_clm.gpu.canopy_radiation import run_shortwave_jax,run_longwave_jax
        from noahmp_clm.gpu.cuda.canopy_radiation import run_shortwave_cuda,run_longwave_cuda
        result["devices"]=[str(d) for d in jax.devices()]
        result["versions"]={"numpy":np.__version__,"jax":jax.__version__,"cupy":cp.__version__}
        result["leaf"]={}
        for name,u,p in [("grid",leaf_cases(np.float32),LeafParameters(tpu25=8.)),
                ("random",random_leaves(),LeafParameters(vcmax25=85.,jmax25=155.,tpu25=6.5,
                    respiration25=.7,ball_berry_slope=7.,growth_temperature=293.15))]:
            j=np.asarray(run_leaves_jax(u,p));c=cp.asnumpy(run_leaves_cuda(u,p));e=ordinary(u,p)
            result["leaf"][name]={"n":len(u),"cuda_jax32":{k:metrics(c[:,i],j[:,i],same_precision=True) for i,k in enumerate(LEAF_OUTPUTS)},
                "jax32_numpy64":{k:metrics(j[:,i],e[:,i]) for i,k in enumerate(LEAF_OUTPUTS)},
                "diffusion_residual_max":float(np.max(abs(j[:,11])))}
        result["shortwave"]={}
        for name,(leaf,u) in [("grid",cases(np.float32)),("random",random_cases())]:
            j=run_shortwave_jax(leaf,u);c=run_shortwave_cuda(leaf,u)
            e=[two_stream_shortwave(l.astype(float),float(v[0]),float(v[1]),ground_albedo=float(v[2]),
                 cosine_zenith=float(v[3]),leaf_reflectance=float(v[4]),leaf_transmittance=float(v[5]))
                 for l,v in zip(leaf,u)]
            result["shortwave"][name]={"n":len(u),"cuda_jax32":{k:metrics(cp.asnumpy(c[k]),np.asarray(j[k]),same_precision=True) for k in j},
                "jax32_numpy64":{k:metrics(np.asarray(j[k]),[r[k] for r in e]) for k in j},
                "budget_max_wm2":float(np.max(abs(np.asarray(j["residual"]))))}
        lai,temp,u=longwave_cases(np.float32)
        j=run_longwave_jax(lai,temp,u);c=run_longwave_cuda(lai,temp,u)
        e=[canopy_longwave(l.astype(float),t.astype(float),*map(float,v[:2]),ground_emissivity=float(v[2]))
           for l,t,v in zip(lai,temp,u)]
        result["longwave"]={"n":len(u),
            "cuda_jax32":{k:metrics(cp.asnumpy(c[k]),np.asarray(j[k]),same_precision=True) for k in j},
            "jax32_numpy64":{k:metrics(np.asarray(j[k]),[r[k] for r in e]) for k in j},
            "budget_max_wm2":float(np.max(abs(np.asarray(j["residual"]))))}
    paths=["src/noahmp_clm/physics/photosynthesis.py","src/noahmp_clm/physics/canopy_radiation.py",
        "src/noahmp_clm/gpu/photosynthesis.py","src/noahmp_clm/gpu/canopy_radiation.py",
        "src/noahmp_clm/gpu/cuda/photosynthesis.py","src/noahmp_clm/gpu/cuda/canopy_radiation.py",
        "src/noahmp_clm/gpu/cuda/leaf_kernel.cu","src/noahmp_clm/gpu/cuda/radiation_kernel.cu"]
    result["source_sha256"]={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}
    result["scope"]="Standalone components; not connected to column stepping; no canopy GPP observation acceptance"
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu",action="store_true")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    result=audit(args.gpu)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({"output":str(args.output),"gpu":args.gpu,"cpu_radiation":result["radiation_continuous_reference"]}))
