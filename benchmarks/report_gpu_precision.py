"""Report unchanged H4 tolerances without assertion short-circuiting.

This report does not replace either acceptance test. The residual 2e-3 columns
are measurements for discussion, not an accepted tolerance or passing gate.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

from probe_gpu_numerics import inputs
from noahmp_clm.column import ColumnState
from noahmp_clm.driver import run_column
from noahmp_clm.gpu.pack import column_state_to_vec, vec_to_column_state, forcing_vec_to_forcing

NAMES=("ts","rnet","h","le","g","et_mm","prcp_mm","runoff_mm","melt_mm","swe","canw","residual")

def report(actual, expected, *, backend, kind, steps, rtol):
    rows=[]
    names=NAMES if kind=="diagnostics" else [f"fvec_{k}" for k in range(70)]
    for k,name in enumerate(names):
        a,b=actual[...,k],expected[...,k];delta=abs(a-b);allowed=1e-6+rtol*abs(b)
        w=np.unravel_index(np.argmax(delta/allowed),a.shape)
        row={"steps":steps,"comparison":backend,"kind":kind,"field":name,"rtol":rtol,"atol":1e-6,
             "failed":int(np.sum(~np.isclose(a,b,rtol=rtol,atol=1e-6))),"max_abs":float(delta.max()),
             "max_ratio":float((delta/allowed)[w]),"worst_index":[int(v) for v in w],
             "cuda":float(a[w]),"reference":float(b[w])}
        if name=="residual":
            row.update(max_abs_cuda_residual=float(abs(a).max()),max_abs_reference_residual=float(abs(b).max()),
                       abs_difference_exceeds_2e3=int(np.sum(delta>2e-3)),abs_cuda_residual_exceeds_2e3=int(np.sum(abs(a)>2e-3)))
        rows.append(row);print(json.dumps(row),flush=True)
    return rows

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",required=True);args=parser.parse_args()
    import cupy as cp
    import jax
    from noahmp_clm.gpu.cuda.loader import run_columns_cuda
    from noahmp_clm.gpu.jax_core import run_columns_jax
    jax.config.update("jax_enable_x64",True)
    if jax.default_backend()!="gpu":raise RuntimeError("GPU required")
    f,u,_=inputs(1024);initial=ColumnState();i=np.broadcast_to(column_state_to_vec(initial)[1],(len(f),33)).copy();rows=[]
    for steps in (1,48):
        forcing=np.broadcast_to(u[:,None],(len(f),steps,8)).copy();forcing[:,12:,7]=0.
        cf,ci,cd=map(cp.asnumpy,run_columns_cuda(f,i,forcing));jf,ji,jd=map(np.asarray,run_columns_jax(f,i,forcing))
        cache={};ds=[];fs=[]
        for index in range(len(f)):
            key=f[index].tobytes()+i[index].tobytes()+forcing[index].tobytes()
            if key not in cache:
                state=vec_to_column_state(f[index],i[index],initial.dz)
                diag,final=run_column(state,[forcing_vec_to_forcing(row) for row in forcing[index]])
                cache[key]=(np.array([[d[name] for name in NAMES] for d in diag]),column_state_to_vec(final)[0])
            d,s=cache[key];ds.append(d);fs.append(s)
        rows+=report(cd,np.asarray(ds),backend="numpy64",kind="diagnostics",steps=steps,rtol=1e-4)
        rows+=report(cf,np.asarray(fs),backend="numpy64_exploratory_state",kind="state",steps=steps,rtol=1e-4)
        rows+=report(cd,jd,backend="jax32_original_H4",kind="diagnostics",steps=steps,rtol=1e-5)
        rows+=report(cf,jf,backend="jax32_original_H4",kind="state",steps=steps,rtol=1e-5)
        rows+=report(cd,jd,backend="jax32_J2",kind="diagnostics",steps=steps,rtol=1e-6)
        rows+=report(cf,jf,backend="jax32_J2",kind="state",steps=steps,rtol=1e-6)
        print(json.dumps({"steps":steps,"integer_cuda_jax_equal":bool(np.array_equal(ci,ji))}),flush=True)
    Path(args.output).write_text(json.dumps(rows,indent=2)+"\n",encoding="utf-8")

if __name__=="__main__":main()
