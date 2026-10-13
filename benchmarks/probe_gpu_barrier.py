"""J2 compiler experiment: test FMA contraction and JAX optimization barriers."""
import json
from pathlib import Path
import sys

import cupy as cp
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
if jax.default_backend() != "gpu":
    raise RuntimeError("GPU required")

a = np.array([1 + 2**-23, 1.234567, .99, .96], np.float32)
b = np.array([1 - 2**-23, 234.5678, -123.45678, -12.345678], np.float32)
c = np.array([-1, -289.589, 122.22222, 11.851852], np.float32)

def fused(x,y,z):
    return x*y+z

def barrier(x,y,z):
    product, addend = jax.lax.optimization_barrier((x*y,z))
    return product+addend

rows={}
for name,fn in (("plain",fused),("barrier",barrier)):
    compiled=jax.jit(jax.vmap(fn)).lower(a,b,c).compile()
    value=np.asarray(compiled(a,b,c))
    rows[name]={"values":value.tolist(),"bits":value.view(np.uint32).tolist(),
                "hlo_has_barrier":"optimization-barrier" in compiled.as_text()}
    Path(sys.argv[1]+"-"+name+".hlo.txt").write_text(compiled.as_text())
source=r'''
extern "C" __global__ void test(const float *a,const float *b,const float*c,float*out) {
  int j=threadIdx.x;
  out[j]=a[j]*b[j]+c[j];
  out[j+4]=__fadd_rn(__fmul_rn(a[j],b[j]),c[j]);
  out[j+8]=fmaf(a[j],b[j],c[j]);
}
'''
for mode,options in (("cuda_default",()),("cuda_fmad_false",("--fmad=false",))):
    k=cp.RawKernel(source,"test",options=options)
    out=cp.empty(12,cp.float32)
    k((1,),(4,),(cp.asarray(a),cp.asarray(b),cp.asarray(c),out))
    val=cp.asnumpy(out).reshape(3,4)
    rows[mode]={"values":val.tolist(),"bits":val.view(np.uint32).tolist()}
print(json.dumps(rows,indent=2),flush=True)
Path(sys.argv[1]+".json").write_text(json.dumps(rows,indent=2)+"\n")
