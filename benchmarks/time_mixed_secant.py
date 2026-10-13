"""Matched, alternating device-resident K1 prototype timings, no profiling."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from probe_mixed_secant import mixed_source
from probe_gpu_numerics import inputs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--n",type=int,default=100000)
    args=p.parse_args()
    import cupy as cp
    from noahmp_clm.gpu.cuda import loader
    from noahmp_clm.gpu.pack import column_state_to_vec
    from noahmp_clm.column import ColumnState
    f,one,_=inputs(args.n)
    i=np.broadcast_to(column_state_to_vec(ColumnState())[1],(len(f),33)).copy()
    u=np.broadcast_to(one[:,None],(len(f),48,8)).copy();u[:,12:,7]=0
    initial=cp.asarray(f.T.copy());opts=cp.asarray(i.T.copy());forcing=cp.asarray(u.transpose(1,2,0).copy())
    source=loader.KERNEL_SOURCE
    kernels={}
    for name,code in (("baseline",source),("mixed",mixed_source(source))):
        module=cp.RawModule(code=code,options=("--std=c++14","--fmad=false","--prec-div=false"),name_expressions=("step_columns",))
        kernels[name]=module.get_function("step_columns")
    expected={};samples={k:[] for k in kernels};hashes={}
    for repeat in range(6):
        for name in (("baseline","mixed") if repeat%2==0 else ("mixed","baseline")):
            state=initial.copy();options=opts.copy();out=cp.empty_like(state);oi=cp.empty_like(options)
            diag=cp.empty((48,12,args.n),cp.float32)
            start,end=cp.cuda.Event(),cp.cuda.Event()
            start.record()
            for t in range(48):
                kernels[name]((min((args.n+255)//256,4096),),(256,),
                              (state,options,forcing[t],out,oi,diag[t],np.int32(args.n)))
                state,out=out,state;options,oi=oi,options
            end.record();end.synchronize();seconds=cp.cuda.get_elapsed_time(start,end)/1000
            host=tuple(map(cp.asnumpy,(state,options,diag)))
            if not all(np.all(np.isfinite(x)) for x in host):raise AssertionError("nonfinite")
            fingerprint=hashlib.sha256(b"".join(x.tobytes() for x in host)).hexdigest()
            if name in expected:assert fingerprint==expected[name]
            else:expected[name]=fingerprint
            samples[name].append(seconds)
            print(json.dumps({"variant":name,"phase":"warmup" if repeat<3 else "measured","repeat":repeat%3,"seconds":seconds,"full_output_sha256":fingerprint}),flush=True)
    result={"n":args.n,"steps":48,"source_sha256":hashlib.sha256(source.encode()).hexdigest(),
            "scope":"CUDA events; all 48 kernels on resident inputs, excludes H2D/D2H; full outputs read/hashed after each run",
            "alternating_order":True,"timings":{k:{"warmup":v[:3],"measured":v[3:],"median":float(np.median(v[3:])),"output_sha256":expected[k]} for k,v in samples.items()}}
    result["overhead_fraction"]=result["timings"]["mixed"]["median"]/result["timings"]["baseline"]["median"]-1
    args.output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result),flush=True)


if __name__=="__main__":main()
