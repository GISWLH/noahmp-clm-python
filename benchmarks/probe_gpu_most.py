"""J2 trace the twenty MOST updates with identical state/forcing/temperature."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from probe_gpu_numerics import inputs

NAMES=("zeta","psi_m","psi_h","den_m","den_h","ustar","ce","h","obukhov")
SOURCE=r'''
extern "C" __global__ void probe_most(const float* states,const float* forcings,const float* temps,float* output,int n){
 int col=blockIdx.x*blockDim.x+threadIdx.x;if(col>=n)return;float s[70],u[8];
 for(int j=0;j<70;++j)s[j]=states[col*70+j];for(int j=0;j<8;++j)u[j]=forcings[col*8+j];
 float ts=temps[col],wind=fmaxf(u[3],.3f),rho=air_density(u),z0=clip(s[33],1e-5f,5.f);
 float lm=logf(10.f/z0),lh=logf(10.f/fmaxf(z0/10.f,1e-6f)),ob=1e30f;
 bool done=false;
 for(int it=0;it<20;++it){bool finite=fabsf(ob)<1e29f;float zeta=finite?10.f/ob:0.f,pm,ph;
 stability(zeta,&pm,&ph);float dm=fmaxf(1.f,lm-pm),dh=fmaxf(1.f,lh-ph);
 float us=.4f*wind/dm,ce=fmaxf(1e-6f,.16f*wind/(dm*dh)),h=rho*CP*ce*(ts-u[1]);
 float on=fabsf(h)<1e-8f?1e30f:clip((-rho*CP*u[1])*(us*(us*us))/(3.924f*h),-1e6f,1e6f);
 float row[9]={zeta,pm,ph,dm,dh,us,ce,h,on};for(int j=0;j<9;++j)output[(col*20+it)*9+j]=row[j];
 bool nf=fabsf(on)<1e29f;bool next=(finite&&nf&&fabsf(on-ob)/fmaxf(1.f,fabsf(ob))<1e-4f)||(!finite&&!nf);
 if(!done)ob=on;done=done||next;
 }
}
'''

def reference(s,u,t):
    import jax
    import jax.numpy as jnp
    from noahmp_clm.gpu import jax_core as jc
    wind=jnp.maximum(u[3],.3);rho=jc._air_density(u[1],u[2],u[4]);z0=jnp.clip(s[33],1e-5,5.)
    lm=jnp.log(10./z0);lh=jnp.log(10./jnp.maximum(z0/10.,1e-6))
    def body(carry,_):
        ob,done=carry;finite=jnp.abs(ob)<1e29;zeta=jnp.where(finite,10./ob,0.)
        pm,ph=jc._psi_m(zeta),jc._psi_h(zeta);dm=jnp.maximum(1.,lm-pm);dh=jnp.maximum(1.,lh-ph)
        us=.4*wind/dm;ce=jnp.maximum(1e-6,.4**2*wind/(dm*dh));h=rho*jc.CP*ce*(t-u[1])
        on=-rho*jc.CP*u[1]*us**3/(.4*9.81*jnp.where(jnp.abs(h)<1e-8,1.,h))
        on=jnp.where(jnp.abs(h)<1e-8,1e30,jnp.clip(on,-1e6,1e6));nf=jnp.abs(on)<1e29
        new=(finite&nf&(jnp.abs(on-ob)/jnp.maximum(1.,jnp.abs(ob))<1e-4))|(~finite&~nf)
        return (jnp.where(done,ob,on),done|new),jnp.array([zeta,pm,ph,dm,dh,us,ce,h,on])
    return jax.lax.scan(body,(jnp.float32(1e30),jnp.asarray(False)),None,length=20)[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",required=True);args=parser.parse_args()
    import cupy as cp
    import jax
    jax.config.update("jax_enable_x64",True)
    if jax.default_backend()!="gpu":raise RuntimeError("GPU required")
    f,u,t=inputs(156)
    compiled=jax.jit(jax.vmap(reference)).lower(f,u,t).compile()
    Path(args.output+".hlo.txt").write_text(compiled.as_text(),encoding="utf-8")
    ref=np.asarray(compiled(f,u,t))
    module=cp.RawModule(code=Path("src/noahmp_clm/gpu/cuda/column_kernel.cu").read_text()+SOURCE,
                        options=("--std=c++14","--fmad=false","--prec-div=false"),name_expressions=("probe_most",))
    actual=cp.empty(ref.shape,cp.float32)
    module.get_function("probe_most")((2,),(128,),(cp.asarray(f),cp.asarray(u),cp.asarray(t),actual,np.int32(len(f))))
    actual=cp.asnumpy(actual);rows=[]
    for iteration in range(20):
        for k,name in enumerate(NAMES):
            delta=abs(actual[:,iteration,k]-ref[:,iteration,k]);w=int(np.argmax(delta))
            row={"iteration":iteration,"component":name,"max_abs":float(delta[w]),
                 "not_equal":int(np.sum(actual[:,iteration,k]!=ref[:,iteration,k])),"worst_index":w,
                 "cuda":float(actual[w,iteration,k]),"jax":float(ref[w,iteration,k])}
            rows.append(row);print(json.dumps(row),flush=True)
    Path(args.output+".json").write_text(json.dumps(rows,indent=2)+"\n",encoding="utf-8")
    np.savez_compressed(args.output+".npz",cuda=actual,jax=ref)

if __name__=="__main__":main()
