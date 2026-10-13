"""J2 expression-level float32 vapor/PM comparison without a root solve."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from probe_gpu_numerics import inputs

NAMES=("es", "qs_num", "qs_den", "qsat", "ea", "slope_num", "slope_den", "slope", "gamma",
       "rho", "pm_radiation", "pm_aerodynamic", "pm_numerator", "pm_denominator", "pm",
       "ground_dry", "ground_diff", "ground_resistance", "root_water", "ftemp", "assim", "rh", "gs_mol", "gs_ms", "rs")
SOURCE=r'''
extern "C" __global__ void probe_flux(const float* states,const float* forcings,const float* temps,float* output,int n){
 int col=blockIdx.x*blockDim.x+threadIdx.x;if(col>=n)return;float s[70],u[8];
 for(int j=0;j<70;++j)s[j]=states[col*70+j];for(int j=0;j<8;++j)u[j]=forcings[col*8+j];
 float t=temps[col],es=esat(t),qn=.622f*es,qd=fmaxf(1.f,u[4]-.378f*es),tc=u[1]-TF,e=esat(u[1]);
 float ea=u[2]*u[4]/(.622f+.378f*u[2]),sn=e*17.67f*243.5f,sd=(tc+243.5f)*(tc+243.5f),slope=sn/sd;
 float gamma=CP*u[4]/(.622f*LV),rho=air_density(u),rad=slope*123.f,aero=rho*CP*fmaxf(0.f,e-ea)/100.f;
 float num=rad+aero,den=slope+gamma*(1.f+150.f/100.f);
 float sat=fmaxf(s[39],1e-6f),frac=fminf(1.f,clip(s[4],0.f,sat)/sat);
 float dry=.1f*(expf((1.f-frac)*(1.f-frac))-1.f)/1.718281828459045f;
 float diff=2.2e-5f*sat*sat*powf(fmaxf(1e-6f,1.f-s[40]/sat),2.f+3.f/fmaxf(s[43],1e-6f));
 float w[4];for(int j=0;j<4;++j)w[j]=s[12+j]*beta(s[4+j],s);
 float water=fmaxf(.02f,clip((w[0]+w[2])+(w[1]+w[3]),0.f,1.f));
 float ft=clip(1.f-.0016f*(298.f-u[1])*(298.f-u[1]),.05f,1.f);
 float a=25.f*(1.f-expf(-fmaxf(0.f,u[5])*2.f/450.f))*ft*water;
 float rh=clip(ea/fmaxf(esat(u[1]),1.f),.05f,1.f),gm=.01f+9.f*a*rh/400.f;
 float ms=gm*8.314462618f*u[1]/fmaxf(u[4],1.f),rs=clip(1.f/fmaxf(ms,1e-6f),s[36],5000.f);
 float row[25]={es,qn,qd,qn/qd,ea,sn,sd,slope,gamma,rho,rad,aero,num,den,fmaxf(0.f,num/den),dry,diff,dry/diff,water,ft,a,rh,gm,ms,rs};
 for(int j=0;j<25;++j)output[col*25+j]=row[j];
}
'''

def reference(s,u,t):
    import jax.numpy as jnp
    from noahmp_clm.gpu import jax_core as jc
    es=jc._sat_vapor_pressure(t);qn=.622*es;qd=jnp.maximum(1.,u[4]-.378*es)
    tc=u[1]-273.15;e=jc._sat_vapor_pressure(u[1]);ea=u[2]*u[4]/(.622+.378*u[2])
    sn=e*17.67*243.5;sd=(tc+243.5)**2;slope=sn/sd;gamma=jc.CP*u[4]/(.622*jc.LV)
    rho=jc._air_density(u[1],u[2],u[4]);rad=slope*123.;aero=rho*jc.CP*jnp.maximum(0.,e-ea)/100.
    num=rad+aero;den=slope+gamma*(1.+150./100.)
    sat=jnp.maximum(s[39],1e-6);frac=jnp.minimum(1.,jnp.clip(s[4],0.,sat)/sat)
    dry=.1*(jnp.exp((1.-frac)**2)-1.)/(jnp.e-1.)
    diff=2.2e-5*sat**2*jnp.maximum(1e-6,1.-s[40]/sat)**(2.+3./jnp.maximum(s[43],1e-6))
    water=jnp.maximum(.02,jc._soil_stress(s[4:8],s[12:16],s[39],s[40],s[41],s[43],jnp.int32(1))[0])
    ft=jnp.clip(1.-.0016*(298.-u[1])**2,.05,1.)
    a=25.*(1.-jnp.exp(-jnp.maximum(0.,u[5])*2./450.))*ft*water
    rh=jnp.clip(ea/jnp.maximum(jc._sat_vapor_pressure(u[1]),1.),.05,1.)
    gm=.01+9.*a*rh/400.;ms=gm*8.314462618*u[1]/jnp.maximum(u[4],1.);rs=jnp.clip(1./jnp.maximum(ms,1e-6),s[36],5000.)
    return jnp.array([es,qn,qd,qn/qd,ea,sn,sd,slope,gamma,rho,rad,aero,num,den,jnp.maximum(0.,num/den),dry,diff,dry/diff,water,ft,a,rh,gm,ms,rs])

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",required=True);parser.add_argument("--input-npz");args=parser.parse_args()
    import cupy as cp
    import jax
    jax.config.update("jax_enable_x64",True)
    if jax.default_backend()!="gpu":raise RuntimeError("GPU required")
    f,u,t=inputs(156)
    if args.input_npz:
        data=np.load(args.input_npz);f,u,t=data["f"],data["u"],data["t"]
    compiled=jax.jit(jax.vmap(reference)).lower(f,u,t).compile()
    Path(args.output+".hlo.txt").write_text(compiled.as_text(),encoding="utf-8")
    ref=np.asarray(compiled(f,u,t))
    module=cp.RawModule(code=Path("src/noahmp_clm/gpu/cuda/column_kernel.cu").read_text()+SOURCE,
                        options=("--std=c++14","--fmad=false","--prec-div=false"),name_expressions=("probe_flux",))
    actual=cp.empty(ref.shape,cp.float32)
    module.get_function("probe_flux")((2,),(128,),(cp.asarray(f),cp.asarray(u),cp.asarray(t),actual,np.int32(len(f))))
    actual=cp.asnumpy(actual);rows=[]
    for k,name in enumerate(NAMES):
        delta=abs(actual[:,k]-ref[:,k]);w=int(np.argmax(delta))
        row={"component":name,"max_abs":float(delta[w]),"not_equal":int(np.sum(actual[:,k]!=ref[:,k])),
             "failed_1e6":int(np.sum(~np.isclose(actual[:,k],ref[:,k],rtol=1e-6,atol=1e-6))),
             "worst_index":w,"cuda":float(actual[w,k]),"jax":float(ref[w,k])}
        rows.append(row);print(json.dumps(row),flush=True)
    Path(args.output+".json").write_text(json.dumps(rows,indent=2)+"\n",encoding="utf-8")

if __name__=="__main__":main()
