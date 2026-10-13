"""Experimental K1 kernel: double F/root; unchanged float state/bulk physics.

Build in memory from an identified frozen source. No production dispatch or ABI
changes. Report every original H4 failure and measured overhead, never a new gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np


def mixed_source(source, round_root=False):
    start = source.index("namespace {") + len("namespace {")
    end = source.index("struct Runoff")
    double = source[start:end]
    double = re.sub(r"\bfloat\b", "double", double)
    double = re.sub(r"\b(fminf|fmaxf|fabsf|copysignf|expf|powf|logf|atanf|sqrtf|log10f)\b", lambda m:m[0][:-1], double)
    double = re.sub(r"(\d(?:[\d.eE+\-]*\d)?)f\b", r"\1", double)
    # Explicit scalar promotion before any F evaluation; no already-rounded Ts
    # is fed back into the final double energy evaluation in the mixed variant.
    prefix = "namespace mixed64 {\n" + double + "\n}\n"
    anchor = "// 对应普通版: driver.step_column。Default specialization"
    assert source.count(anchor) == 1
    source = source.replace(anchor, prefix + anchor)
    old = "    float ts=surface_temperature(s,u,is_snow,tr,drag,ground,water,conductivity);\n    Energy e=energy(ts,s,u,is_snow,tr,drag,ground,water,conductivity); float melt=0.0f;"
    assert source.count(old) == 1, "Frozen source no longer matches the experiment"
    new = """    double ds[70], du[8];
    for(int k=0;k<70;++k) ds[k]=s[k];
    for(int k=0;k<8;++k) du[k]=u[k];
    double root=mixed64::surface_temperature(ds,du,is_snow,(double)tr,drag,ground,water,conductivity);
    float ts=(float)root;
"""
    if round_root:
        new += "    Energy e=energy(ts,s,u,is_snow,tr,drag,ground,water,conductivity); float melt=0.0f;"
    else:
        new += """    mixed64::Energy de=mixed64::energy(root,ds,du,is_snow,(double)tr,drag,ground,water,conductivity);
    Energy e={(float)de.rn,(float)de.h,(float)de.le,(float)de.g,(float)de.ecan,(float)de.esoil,
              (float)(de.rn-de.h-de.le-de.g)};
    float melt=0.0f;"""
    return source.replace(old,new)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--forensics",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--timing-n",type=int,default=100000)
    args=p.parse_args()
    import cupy as cp
    from noahmp_clm.gpu.cuda import loader
    from probe_gpu_numerics import inputs
    from noahmp_clm.column import ColumnState
    from noahmp_clm.gpu.pack import column_state_to_vec
    data=np.load(args.forensics)
    f,i,u=data["initial"],data["options"],data["forcing"]
    reference=data["numpy_diagnostics"]
    source=loader.KERNEL_SOURCE
    result={"source_sha256":hashlib.sha256(source.encode()).hexdigest(),"device":cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
            "precision":"float64 F and secant; float32 bulk and unchanged ABI", "variants":{},"timing":{}}
    kernels={}
    for name in ("baseline","double_root_rounded_before_F","mixed"):
        code=source if name=="baseline" else mixed_source(source,name=="double_root_rounded_before_F")
        module=cp.RawModule(code=code,options=("--std=c++14","--fmad=false","--prec-div=false"),name_expressions=("step_columns",))
        kernels[name]=module.get_function("step_columns")
        loader.load_column_kernel=lambda: kernels[name]
        cf,ci,cd=map(cp.asnumpy,loader.run_columns_cuda(f,i,u))
        assert np.all(np.isfinite(cf)) and np.all(np.isfinite(cd))
        if name=="baseline":
            np.testing.assert_array_equal(cd,data["cuda_diagnostics"])
            np.testing.assert_array_equal(cf,data["cuda_states"][:,-1])
        delta=abs(cd.astype(float)-reference)
        row={"generated_source_sha256":hashlib.sha256(code.encode()).hexdigest(),
             "residual_002_count":int(np.sum(delta[:,:,11]>.002)),
             "residual_abs_max":float(abs(cd[:,:,11]).max()),
             "state_failed":int(np.sum(~np.isclose(cf,data["numpy_states"][:,-1],rtol=1e-4,atol=1e-6))),
             "integer_equal":bool(np.array_equal(ci,data["numpy_options"][:,-1])),
             "diagnostics":{str(key):{"max_abs":float(delta[:,:,k].max()),"strict_failed":int(np.sum(delta[:,:,k]>1e-6+1e-4*abs(reference[:,:,k])))} for k,key in enumerate(data["names"])}}
        result["variants"][name]=row
        np.savez_compressed(args.output.with_name(args.output.stem+"-"+name+".npz"),state=cf,options=ci,diagnostics=cd)
        print(json.dumps({"variant":name,**row}),flush=True)
    # Fixed heterogeneous workload, 3 warmups + 3 timed, matched input/flags.
    f,one,_=inputs(args.timing_n)
    i=np.broadcast_to(column_state_to_vec(ColumnState())[1],(len(f),33)).copy()
    u=np.broadcast_to(one[:,None],(len(f),48,8)).copy();u[:,12:,7]=0
    for name in ("baseline","mixed"):
        loader.load_column_kernel=lambda:kernels[name]
        expected=None; samples=[]
        for repeat in range(6):
            cp.cuda.get_current_stream().synchronize();started=time.perf_counter()
            out=loader.run_columns_cuda(f,i,u)
            # Full output host consumption inside timing, bitwise against this
            # variant's first run. Strict NumPy accuracy remains reported above.
            host=tuple(map(cp.asnumpy,out))
            elapsed=time.perf_counter()-started
            if not all(np.all(np.isfinite(x)) for x in host):raise AssertionError("nonfinite output")
            if expected is None:expected=host
            else:
                for a,b in zip(host,expected):np.testing.assert_array_equal(a,b)
            samples.append(elapsed)
            print(json.dumps({"variant":name,"phase":"warmup" if repeat<3 else "measured","repeat":repeat%3,"seconds":elapsed,"full_output_equal":True}),flush=True)
        result["timing"][name]={"n":len(f),"steps":48,"warmups":samples[:3],"samples":samples[3:],"median":float(np.median(samples[3:])),"includes_host_input_output":True,"full_output_bitwise_equal":True}
    result["overhead_fraction"]=result["timing"]["mixed"]["median"]/result["timing"]["baseline"]["median"]-1
    args.output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"overhead_fraction":result["overhead_fraction"]}),flush=True)


if __name__=="__main__":main()
