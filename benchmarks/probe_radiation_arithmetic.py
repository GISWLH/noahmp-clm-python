"""Inspect coefficient and optimized-HLO differences in the new GPU component.

Probe kernels are cut from local trusted production source before interface
solving. This tool does not alter production math or tolerance.
"""
from pathlib import Path
import inspect
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tests"))
import numpy as np
import jax
import cupy as cp
from test_gpu_canopy_radiation import random_cases
from noahmp_clm.gpu.canopy_radiation import shortwave_kernel_jax

leaf,u=random_cases()
body=inspect.getsource(shortwave_kernel_jax).split("    beam = ")[0]
body=body.replace("shortwave_kernel_jax", "probe_jax")
body+="    return jnp.stack((r,t,up,down,beam_t,jnp.full_like(lai,beta)))\n"
scope={}
exec(body,scope)
compiled=jax.jit(jax.vmap(scope["probe_jax"])).lower(leaf,u).compile()
Path("radiation-coeff-hlo.txt").write_text(compiled.as_text())
j=np.asarray(compiled(leaf,u))
path=Path(__file__).resolve().parents[1]/"src/noahmp_clm/gpu/cuda/radiation_kernel.cu"
source=path.read_text().split("        beam[0]=beam0;")[0].replace("run_shortwave", "probe_cuda")
source+='''
        float* out=output+60*row;
        for(int i=0;i<10;++i) {
            out[i]=r[i];out[10+i]=t[i];out[20+i]=su[i];out[30+i]=sd[i];out[40+i]=bt[i];out[50+i]=beta;
        }
    }
}
'''
module=cp.RawModule(code=source,options=("--std=c++14","--fmad=false","--prec-div=false"),name_expressions=("probe_cuda",))
out=cp.empty((len(u),6,10),dtype=cp.float32)
module.get_function("probe_cuda")((1,),(256,),(cp.asarray(leaf),cp.asarray(u),out,np.int32(len(u))))
c=cp.asnumpy(out)
for i,key in enumerate(("r","t","up","down","beam_t","beta")):
    err=abs(c[:,i]-j[:,i]);bad=np.argwhere(err>0)
    print(key,"max_abs",float(err.max()),"different",len(bad))
    for row,layer in bad[:3]:
        print(int(row),int(layer),"u",u[row].tolist(),"lai",float(leaf[row,layer]),
              "cuda",float(c[row,i,layer]),"jax",float(j[row,i,layer]))
