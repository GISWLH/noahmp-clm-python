"""J2 same-input snowfall preparation and real soil matrix arithmetic probe."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from probe_gpu_numerics import inputs

NAMES = ([f"snow_{k}" for k in range(9)] + ["snow_total"]
         + [f"{name}_{k}" for name in ("capacity", "conductivity") for k in range(4)]
         + [f"conduct_{k}" for k in range(3)]
         + [f"{name}_{k}" for name in ("lower", "diagonal", "upper", "rhs", "solution",
                                      "capdz", "div_rn", "mul_recip", "div_fast") for k in range(4)]
         + ["snow_numerator", "snow_denominator", "snow_div_rn", "snow_recip", "snow_div_fast"])

SOURCE = r'''
extern "C" __global__ void probe_stages(const float* states,const float* forcings,float* output,int n) {
 int col=blockIdx.x*blockDim.x+threadIdx.x; if(col>=n)return;
 float s[70],u[8];for(int j=0;j<70;++j)s[j]=states[col*70+j];for(int j=0;j<8;++j)u[j]=forcings[col*8+j];
 float* out=output+col*62;int count=0;for(int j=0;j<3;++j)count+=s[16+j]>0;count=count>0?count:1;
 float amount=u[1]<TF?u[7]*u[0]:0.f,wt=0.f,tot=0.f;
 for(int j=0;j<count;++j){wt+=s[16+j]*s[19+j];tot+=s[16+j];}wt+=amount*u[1];tot+=amount;
 float den=fmaxf(tot,1e-12f);out[57]=wt;out[58]=tot;out[59]=__fdiv_rn(wt,den);
 out[60]=wt*(1.f/den);out[61]=__fdividef(wt,den);
 snowfall(s,&count,amount,u);
 for(int j=0;j<9;++j)out[j]=s[16+j];out[9]=s[26];
 float cap[4],k[4],g[3],a[4],b[4],c[4],rhs[4],sol[4];
 for(int j=0;j<4;++j){cap[j]=capacity(s[4+j],s[j],s);k[j]=soil_k(s[4+j],s);out[10+j]=cap[j];out[14+j]=k[j];}
 for(int j=0;j<3;++j){g[j]=(2.f/(1.f/k[j]+1.f/k[j+1]))/(.5f*(DZ[j]+DZ[j+1]));out[18+j]=g[j];}
 for(int j=0;j<4;++j){float up=j>0?g[j-1]:0.f,down=j<3?g[j]:0.f;
 a[j]=-up;c[j]=-down;b[j]=cap[j]*DZ[j]/u[0]+up+down;rhs[j]=cap[j]*DZ[j]/u[0]*s[j]+(j==0?12.345f:0.f);
 out[21+j]=a[j];out[25+j]=b[j];out[29+j]=c[j];out[33+j]=rhs[j];}
 thomas4(a,b,c,rhs,sol);for(int j=0;j<4;++j)out[37+j]=sol[j];
 for(int j=0;j<4;++j){float v=cap[j]*DZ[j];out[41+j]=v;out[45+j]=__fdiv_rn(v,u[0]);
 out[49+j]=v*(1.f/u[0]);out[53+j]=__fdividef(v,u[0]);}
}
'''


def reference(s, u):
    import jax.numpy as jnp
    from noahmp_clm.gpu import jax_core as jc
    one=jnp.int32(1)
    count=jnp.maximum(one,jnp.sum((s[16:19]>0).astype(jnp.int32))).astype(jnp.int32)
    amount=jnp.where(u[1]<jc.FREEZING_POINT,u[7]*u[0],0.)
    swe,t,d,count,total=jc._add_snowfall(s[16:19],s[19:22],s[22:25],count,amount,u[1])
    d=jc._compact_snow(swe,t,d,count,u[0],jnp.int32(2))
    _,cap=jc._phase_and_capacity(s[4:8],s[:4],s[39],s[43],one)
    k=jc._soil_conductivity(s[4:8],s[39],s[44])
    dz=jnp.asarray(jc.DZ,jnp.float32)
    g=jnp.array([(2./(1./k[j]+1./k[j+1]))/(.5*(dz[j]+dz[j+1])) for j in range(3)])
    a=jnp.array([0.,-g[0],-g[1],-g[2]])
    c=jnp.array([-g[0],-g[1],-g[2],0.])
    b=jnp.array([cap[0]*dz[0]/u[0]+g[0],cap[1]*dz[1]/u[0]+g[0]+g[1],
                 cap[2]*dz[2]/u[0]+g[1]+g[2],cap[3]*dz[3]/u[0]+g[2]])
    rhs=cap*dz/u[0]*s[:4]
    rhs=rhs.at[0].add(jnp.float32(12.345))
    sol=jc._thomas4(a,b,c,rhs)
    capdz=cap*dz
    q=capdz/u[0]
    num=jnp.sum(jnp.where(jnp.arange(3)<count,s[16:19]*s[19:22],0.))+amount*u[1]
    den=jnp.sum(s[16:19])+amount
    mean=num/jnp.maximum(den,1e-12)
    return jnp.concatenate((swe,t,d,jnp.array([total]),cap,k,g,a,b,c,rhs,sol,capdz,q,q,q,
                            jnp.array([num,den,mean,mean,mean])))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    parser.add_argument("--prec-div-false",action="store_true")
    parser.add_argument("--input-npz")
    args=parser.parse_args()
    import cupy as cp
    import jax
    jax.config.update("jax_enable_x64",True)
    if jax.default_backend()!="gpu":raise RuntimeError("GPU required")
    f,u,_=inputs(156)
    if args.input_npz:
        data=np.load(args.input_npz);f,u=data["f"],data["u"]
    compiled=jax.jit(jax.vmap(reference)).lower(f,u).compile()
    Path(args.output+".hlo.txt").write_text(compiled.as_text(),encoding="utf-8")
    ref=np.asarray(compiled(f,u))
    module=cp.RawModule(code=Path("src/noahmp_clm/gpu/cuda/column_kernel.cu").read_text()+SOURCE,
                        options=("--std=c++14","--fmad=false",*(("--prec-div=false",) if args.prec_div_false else ())),name_expressions=("probe_stages",))
    actual=cp.empty(ref.shape,cp.float32)
    module.get_function("probe_stages")((2,),(128,),(cp.asarray(f),cp.asarray(u),actual,np.int32(len(f))))
    actual=cp.asnumpy(actual)
    rows=[]
    for k,name in enumerate(NAMES):
        delta=abs(actual[:,k]-ref[:,k]);w=int(np.argmax(delta))
        row={"component":name,"max_abs":float(delta[w]),"not_equal":int(np.sum(actual[:,k]!=ref[:,k])),
             "failed_1e6":int(np.sum(~np.isclose(actual[:,k],ref[:,k],rtol=1e-6,atol=1e-6))),
             "worst_index":w,"cuda":float(actual[w,k]),"jax":float(ref[w,k])}
        rows.append(row);print(json.dumps(row),flush=True)
    Path(args.output+".json").write_text(json.dumps(rows,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":main()
