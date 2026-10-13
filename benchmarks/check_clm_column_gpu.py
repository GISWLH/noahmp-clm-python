"""Synthetic CLM column trajectory/roundoff audit; no observational claim."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import cupy as cp
import jax
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tests"))
from test_gpu_clm_column import column_cases,reference
from noahmp_clm.column import ColumnState,Forcing
from noahmp_clm.physics.energy import qsat
from noahmp_clm.gpu.pack import forcing_to_vec
from noahmp_clm.gpu.clm_column import pack_clm_columns,site_array,step_clm_columns_jax,CLM_DIAGNOSTICS
from noahmp_clm.gpu.cuda.clm_column import step_clm_columns_cuda
from noahmp_clm.validation import compare_series


def same_precision(a,b):
    a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    return {"max_absolute_difference":float(np.max(np.abs(a-b),initial=0)),
        "failed_values":int((~np.isclose(a,b,rtol=1e-6,atol=1e-6)).sum())}


def audit(days,checkpoint):
    start=time.perf_counter()
    inputs=column_cases(random_count=64)
    j=step_clm_columns_jax(*inputs,return_canopy=True);c=step_clm_columns_cuda(*inputs,return_canopy=True)
    ns,nd=reference(*inputs)
    columns={"jax_cuda_state":same_precision(j[0],cp.asnumpy(c[0])),
        "jax_cuda_diagnostics":{name:same_precision(j[2][:,k],cp.asnumpy(c[2][:,k])) for k,name in enumerate(CLM_DIAGNOSTICS)},
        "numpy_max_soil_temperature_difference_k":float(np.max(np.abs(np.asarray(j[0])[:,:4]-ns[:,:4]))),
        "numpy_max_soil_water_difference":float(np.max(np.abs(np.asarray(j[0])[:,4:8]-ns[:,4:8]))),
        "numpy_max_flux_difference_wm2":{CLM_DIAGNOSTICS[k]:float(np.max(np.abs(np.asarray(j[2])[:,k]-nd[:,k]))) for k in (2,3,4)}}
    columns["jax_cuda_canopy"]={name:same_precision(a,cp.asnumpy(c[3][name])) for name,a in j[3].items() if name!="gas_exchange"}
    columns["jax_cuda_leaf_gas"]={name:same_precision(a,cp.asnumpy(c[3]["gas_exchange"][name])) for name,a in j[3]["gas_exchange"].items()}
    states=[ColumnState(ts=294.,tsoil=np.full(4,292.),smc=np.full(4,.28)),
        ColumnState(ts=294.,tsoil=np.full(4,292.),smc=np.array([.121,.125,.14,.14])),
        ColumnState(ts=294.,tsoil=np.full(4,292.),smc=np.full(4,.28),fveg=0.)]
    initial,i=pack_clm_columns(states,np.float32)
    current={"numpy64":initial.copy(),"jax64":initial.astype(np.float64),"jax32":initial.copy(),"cuda32":initial.copy()}
    rows={k:[] for k in current};trace={k:[] for k in current}
    for step in range(days*48):
        hour=step*.5%24.;sun=max(0.,np.sin(np.pi*(hour-6.)/12.))
        tair=294.+5.*np.sin(2.*np.pi*(hour-9.)/24.)
        rain=2e-6 if 64<=step<68 else 0.
        forcing=Forcing(1800.,tair,.75*qsat(tair,101325.),3.,101325.,800.*sun,340.,rain)
        f=np.tile(forcing_to_vec(forcing).astype(np.float32),(3,1))
        site=site_array(3,np.float32,cosine_zenith=max(.1,sun),diffuse_fraction=.35)
        # Initial state, forcing, site and leaf parameters are identically
        # quantized; the NumPy trajectory itself remains float64 thereafter.
        from noahmp_clm.clm_driver import step_clm_column
        from noahmp_clm.gpu.pack import vec_to_column_state,forcing_vec_to_forcing
        from noahmp_clm.physics.photosynthesis import LeafParameters
        from noahmp_clm.gpu.photosynthesis import pack_leaf_parameters
        params=LeafParameters(*pack_leaf_parameters(dtype=np.float32).astype(float))
        np_states=[];np_diag=[]
        for s,iv,fv,v in zip(current["numpy64"],i,f,site):
            sn,d=step_clm_column(vec_to_column_state(s,iv,[.1,.3,.6,1.]),forcing_vec_to_forcing(fv),
                co2_ppm=float(v[0]),cosine_zenith=float(v[1]),diffuse_fraction=float(v[2]),
                leaf_dimension=float(v[3]),nitrogen_factor=float(v[4]),leaf_parameters=params)
            np_states.append(pack_clm_columns([sn])[0][0]);np_diag.append([d.get(k,1.) for k in CLM_DIAGNOSTICS])
        current["numpy64"]=np.asarray(np_states);rows["numpy64"].append(np.asarray(np_diag))
        for backend in ("jax64","jax32","cuda32"):
            fun=step_clm_columns_cuda if backend=="cuda32" else step_clm_columns_jax
            typ=np.float64 if backend=="jax64" else np.float32
            s,_,d=fun(current[backend],i,f.astype(typ),site.astype(typ),leaf_parameters=params)
            current[backend]=s
            rows[backend].append(cp.asnumpy(d) if backend=="cuda32" else np.asarray(d))
        for backend in current:
            trace[backend].append(cp.asnumpy(current[backend]) if backend=="cuda32" else np.asarray(current[backend]))
        if step%48==47: print(f"day {step//48+1}/{days}",flush=True)
    rows={k:np.stack(v,axis=1) for k,v in rows.items()}
    trace={k:np.stack(v,axis=1) for k,v in trace.items()}
    arrays={**{f"{k}_diagnostics":v for k,v in rows.items()},**{f"{k}_states":v for k,v in trace.items()}}
    checkpoint.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(checkpoint,**arrays)
    report={"scope":"synthetic snow-free trajectories, not CLM/HRLDAS/FLUXNET observation validation",
        "days":days,"columns":3,"steps_per_column":days*48,"jax_devices":list(map(str,jax.devices())),
        "single_step_88_columns":columns,"cuda_vs_jax32":{
            "all_states":same_precision(trace["jax32"],trace["cuda32"]),
            **{name:same_precision(rows["jax32"][:,:,k],rows["cuda32"][:,:,k]) for k,name in enumerate(CLM_DIAGNOSTICS)}},
        "backends":{}}
    names=("vegetated","drying","bare")
    for backend in rows:
        values=rows[backend];summary={}
        for col,name in enumerate(names):
            d=values[col];r=rows["numpy64"][col]
            variables={"h_wm2":(d[:,2],r[:,2]),"le_wm2":(d[:,3],r[:,3]),"g_wm2":(d[:,4],r[:,4]),
                "et_mm_day":(d[:,5]*48,r[:,5]*48),"runoff_mm_day":(d[:,7]*48,r[:,7]*48),
                "gpp_gc_day":(d[:,13]*48,r[:,13]*48)}
            for layer in range(4):
                variables[f"tsoil_c_{layer}"]=(trace[backend][col,:,layer]-273.15,trace["numpy64"][col,:,layer]-273.15)
                variables[f"smc_{layer}"]=(trace[backend][col,:,4+layer],trace["numpy64"][col,:,4+layer])
            scores={key:{"period":compare_series(a,b,key.rsplit('_',1)[0] if key.rsplit('_',1)[-1].isdigit() else key),
                "daily":[compare_series(a[j*48:(j+1)*48],b[j*48:(j+1)*48],key.rsplit('_',1)[0] if key.rsplit('_',1)[-1].isdigit() else key) for j in range(days)]}
                for key,(a,b) in variables.items()}
            storage0=np.dot(initial[col,4:8].astype(float),[.1,.3,.6,1.])+float(initial[col,30])
            storage1=np.dot(trace[backend][col,-1,4:8].astype(float),[.1,.3,.6,1.])+float(trace[backend][col,-1,30])
            summary[name]={"metrics":scores,"water_residual_m":storage1-storage0+float(np.sum(d[:,5].astype(float)+d[:,7]-d[:,6]))/1000.,
                "max_step_water_residual_m":float(np.max(np.abs(d[:,17]))),
                "max_latent_water_residual_wm2":float(np.max(np.abs(d[:,18]))),
                "max_surface_energy_residual_wm2":float(np.max(np.abs(d[:,11]))),
                "cumulative_surface_energy_residual_jm2":float(np.sum(d[:,11].astype(float))*1800.),
                "maximum_iterations":float(np.max(d[:,19])),"et_mm":float(np.sum(d[:,5].astype(float))),
                "runoff_mm":float(np.sum(d[:,7].astype(float))),"gpp_gcm2":float(np.sum(d[:,13].astype(float)))}
        report["backends"][backend]=summary
    report["seconds"]=time.perf_counter()-start
    return report,arrays


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,required=True);parser.add_argument("--days",type=int,default=7)
    args=parser.parse_args();report,arrays=audit(args.days,args.output.with_suffix(".npz"))
    sources=list(Path("src/noahmp_clm").rglob("*.py"))+list(Path("src/noahmp_clm").rglob("*.cu"))
    sources += [Path("benchmarks/check_clm_column_gpu.py"),Path("tests/test_gpu_clm_column.py")]
    report["source_sha256"]={p.as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output.with_suffix(".npz"),**arrays)
    report["series_sha256"]=hashlib.sha256(args.output.with_suffix(".npz").read_bytes()).hexdigest()
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False))
    failed=sum(v["failed_values"] for v in report["cuda_vs_jax32"].values())
    single=report["single_step_88_columns"]
    failed+=single["jax_cuda_state"]["failed_values"]+sum(v["failed_values"] for v in single["jax_cuda_diagnostics"].values())
    failed+=sum(v["failed_values"] for group in ("jax_cuda_canopy","jax_cuda_leaf_gas") for v in single[group].values())
    for backend in report["backends"].values():
        failed+=sum(not d["bias_pass"] for c in backend.values() for m in c["metrics"].values() for d in m["daily"])
    print(json.dumps({"seconds":report["seconds"],"failed":failed,"output":str(args.output)}))
    raise SystemExit(bool(failed))
