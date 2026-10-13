"""J2 locate the first distinct state/diagnostic in repeated one-step runs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from probe_gpu_numerics import inputs

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",required=True);args=parser.parse_args()
    import cupy as cp
    import jax
    from noahmp_clm.column import ColumnState
    from noahmp_clm.gpu.pack import column_state_to_vec
    from noahmp_clm.gpu.cuda.loader import run_columns_cuda
    from noahmp_clm.gpu.jax_core import run_columns_jax
    jax.config.update("jax_enable_x64",True)
    if jax.default_backend()!="gpu":raise RuntimeError("GPU required")
    f,u,_=inputs(156);i=np.broadcast_to(column_state_to_vec(ColumnState())[1],(len(f),33)).copy()
    us=np.broadcast_to(u[:,None,:],(len(f),48,8)).copy();us[:,12:,7]=0.
    full=np.asarray(run_columns_jax(f,i,us)[2]);jf=f.copy();cf=f.copy();ji=i.copy();ci=i.copy();rows=[];saved=False
    jdiags=[];cdiags=[]
    for step in range(48):
        forcing=us[:,step:step+1,:];old_jf=jf.copy()
        jf,ji,jd=map(np.asarray,run_columns_jax(jf,ji,forcing));cf,ci,cd=map(cp.asnumpy,run_columns_cuda(cf,ci,forcing))
        jdiags.append(jd[:,0]);cdiags.append(cd[:,0])
        row={"step":step+1,"state":[],"diag":[]}
        for key,a,b in (("state",cf,jf),("diag",cd[:,0],jd[:,0])):
            for k in range(a.shape[-1]):
                delta=abs(a[:,k]-b[:,k]);w=int(np.argmax(delta))
                if delta[w]:row[key].append({"field":k,"max_abs":float(delta[w]),"column":w,
                                            "cuda":float(a[w,k]),"jax":float(b[w,k])})
        rows.append(row);print(json.dumps(row),flush=True)
        if not saved and row["state"]:
            np.savez_compressed(args.output+"-first-input.npz",f=old_jf,u=forcing[:,0],t=jd[:,0,0],
                                cuda_state=cf,jax_state=jf,cuda_diag=cd,jax_diag=jd)
            saved=True
    jdiags=np.stack(jdiags,axis=1);cdiags=np.stack(cdiags,axis=1)
    summary={"one_step_vs_scan_max_abs":float(np.max(abs(jdiags-full))),"one_step_vs_scan_not_equal":int(np.sum(jdiags!=full)),
             "cuda_jax_max_abs":float(np.max(abs(cdiags-jdiags)))}
    print(json.dumps(summary),flush=True)
    Path(args.output+".json").write_text(json.dumps({"summary":summary,"steps":rows},indent=2)+"\n",encoding="utf-8")

if __name__=="__main__":main()
