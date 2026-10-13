"""J2 read-only scalar math/constant-folding probe; optional XLA dump via env."""
from pathlib import Path
import json
import sys

import cupy as cp
import jax
import jax.numpy as jnp
import numpy as np

from probe_gpu_numerics import inputs
from noahmp_clm.gpu import jax_core as jc

jax.config.update("jax_enable_x64", True)
if jax.default_backend() != "gpu":
    raise RuntimeError("GPU required")

NAMES=("tp","tm","clausius_p","clausius_m","ratio_p","ratio_m","exponent",
       "pow_p","pow_m","liq_p","liq_m","derivative","base","phase","capacity",
       "exp_log_p","exp2_log2_p","root_weight","soil_k","log10_sr","pow_quarter","root_pair","root_cross")

def scalar(s):
    t,smc,smcmax,b=s[0],s[4],s[39],s[43]
    tp,tm=t+.02,t-.02
    pc=jc.LF/461.5*jnp.maximum(0.,jc.FREEZING_POINT-tp)/jc.FREEZING_POINT**2
    mc=jc.LF/461.5*jnp.maximum(0.,jc.FREEZING_POINT-tm)/jc.FREEZING_POINT**2
    pr,mr=jnp.maximum(pc,1e-8)/5e-5,jnp.maximum(mc,1e-8)/5e-5
    exponent=-1./jnp.maximum(b,1e-6)
    pp,mp=pr**exponent,mr**exponent
    lp=jc._unfrozen_water(smc,tp,smcmax,b,jnp.int32(1))
    lm=jc._unfrozen_water(smc,tm,smcmax,b,jnp.int32(1))
    deriv=(lp-lm)/.04
    base=smc*4.18e6+(1-smcmax)*2e6
    phase=jc.RHO_W*jc.LF*jnp.maximum(0.,deriv)
    _,capacity=jc._phase_and_capacity(smc,t,smcmax,b,jnp.int32(1))
    root=jnp.sum(s[12:16]*jnp.clip((s[4:8]-s[40])/(s[41]-s[40]),0.,1.))
    sr=jnp.clip(smc/smcmax,0.,1.)
    return jnp.stack((tp,tm,pc,mc,pr,mr,exponent,pp,mp,lp,lm,deriv,base,phase,capacity,
                      jnp.exp(jnp.log(pr)*exponent),jnp.exp2(jnp.log2(pr)*exponent),
                      root,jc._soil_conductivity(s[4:5],smcmax,s[44])[0],
                      jnp.log10(sr),jnp.power(jnp.float32(1.0017),.25),root,root))

SOURCE=r'''
extern "C" __global__ void probe_math(const float*states,float*output,int n) {
  int col=blockIdx.x*blockDim.x+threadIdx.x;if(col>=n)return;
  const float*s=states+col*70;float*out=output+col*23;
  float t=s[0],smc=s[4],sat=s[39],b=s[43],tp=t+.02f,tm=t-.02f;
  float pc=LF/461.5f*fmaxf(0,TF-tp)/(TF*TF),mc=LF/461.5f*fmaxf(0,TF-tm)/(TF*TF);
  float pr=fmaxf(pc,1e-8f)/5e-5f,mr=fmaxf(mc,1e-8f)/5e-5f;
  float exponent=-1.f/fmaxf(b,1e-6f),pp=powf(pr,exponent),mp=powf(mr,exponent);
  float lp=liquid(smc,tp,s),lm=liquid(smc,tm,s),deriv=(lp-lm)/.04f;
  float base=smc*4.18e6f+(1-sat)*2e6f,phase=RW*LF*fmaxf(0,deriv);
  out[0]=tp;out[1]=tm;out[2]=pc;out[3]=mc;out[4]=pr;out[5]=mr;out[6]=exponent;
  out[7]=pp;out[8]=mp;out[9]=lp;out[10]=lm;out[11]=deriv;
  out[12]=base;out[13]=phase;out[14]=capacity(smc,t,s);
  out[15]=expf(logf(pr)*exponent);out[16]=exp2f(log2f(pr)*exponent);
  float root=0;for(int j=0;j<4;++j)root+=s[12+j]*beta(s[4+j],s);
  out[17]=root;out[18]=soil_k(smc,s);out[19]=log10f(clip(smc/sat,0,1));
  out[20]=powf(1.0017f,.25f);
  float w[4];for(int j=0;j<4;++j)w[j]=s[12+j]*beta(s[4+j],s);
  out[21]=(w[0]+w[1])+(w[2]+w[3]);out[22]=(w[0]+w[2])+(w[1]+w[3]);
}
'''

f,_,_=inputs(156)
if len(sys.argv)>2:f=np.load(sys.argv[2])["f"]
compiled=jax.jit(jax.vmap(scalar)).lower(f).compile()
expected=np.asarray(compiled(f))
Path(sys.argv[1]+".hlo.txt").write_text(compiled.as_text())
module=cp.RawModule(code=Path("src/noahmp_clm/gpu/cuda/column_kernel.cu").read_text()+SOURCE,
                    options=("--std=c++14","--fmad=false","--prec-div=false"),name_expressions=("probe_math",))
out=cp.empty((len(f),len(NAMES)),cp.float32)
module.get_function("probe_math")((2,),(128,),(cp.asarray(f),out,np.int32(len(f))))
actual=cp.asnumpy(out)
rows=[]
for k,name in enumerate(NAMES):
    delta=abs(actual[:,k]-expected[:,k]);idx=int(np.argmax(delta))
    row={"component":name,"max_abs":float(delta[idx]),"not_equal":int(np.sum(delta!=0)),
         "worst_index":idx,"cuda":float(actual[idx,k]),"jax":float(expected[idx,k]),
         "cuda_bits":int(actual[idx,k].view(np.uint32)),"jax_bits":int(expected[idx,k].view(np.uint32))}
    print(json.dumps(row),flush=True);rows.append(row)
Path(sys.argv[1]+".json").write_text(json.dumps(rows,indent=2)+"\n")
