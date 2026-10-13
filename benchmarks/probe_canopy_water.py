"""Bitwise probes of water-network arithmetic at identical trial inputs."""
from pathlib import Path
import json,sys
import numpy as np
import jax,jax.numpy as jnp
import cupy as cp
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tests"))
from test_gpu_canopy_energy import cases
from noahmp_clm.gpu.canopy_energy import _water

folder=ROOT/"src/noahmp_clm/gpu/cuda"
source="\n".join((folder/name).read_text() for name in ("leaf_kernel.cu","radiation_kernel.cu","canopy_energy_kernel.cu"))
source+=r'''
extern "C" __global__ void probe_water(const float* area,const float* u,const float* soil,const float* qs,
 const float* gs,const float* rho,float* out,int n) {
 int r=blockIdx.x*blockDim.x+threadIdx.x;if(r>=n)return;
 CanopyWater w;ca_water(u[r*23+1],qs+r*11,gs+r*10,area+r*10,u[r*23+13],rho[r],u+r*23,soil+r*10,w);
 for(int j=0;j<10;++j){out[r*39+j]=w.wet[j];out[r*39+10+j]=w.tr[j];out[r*39+20+j]=w.kleaf[j];}
 out[r*39+30]=w.eg;out[r*39+31]=w.eb;out[r*39+32]=w.kunder;out[r*39+33]=w.kbare;out[r*39+34]=w.residual;
 for(int j=0;j<4;++j)out[r*39+35+j]=w.uptake[j];
}
extern "C" __global__ void probe_environment(const float* u,const float* temp,float* out,int n) {
 int r=blockIdx.x*blockDim.x+threadIdx.x;if(r>=n)return;
 const float* v=u+23*r;
 for(int j=0;j<11;++j){float tc=temp[11*r+j]-273.15f;
 float es=611.2f*expf(17.67f*tc/(tc+243.5f)),den=v[2]-.378f*es;
 out[r*35+j]=es;out[r*35+11+j]=.622f*es/den;
 out[r*35+22+j]=.622f*v[2]*es*17.67f*243.5f/(den*den*(tc+243.5f)*(tc+243.5f));}
 out[r*35+33]=v[2]/(287.f*v[0]*(1.f+.61f*v[1]));
 out[r*35+34]=v[2]/(8.314462618f*v[0]);
}
'''
mod=cp.RawModule(code=source,options=("--std=c++14","--fmad=false","--prec-div=false","--prec-sqrt=false"),
                 name_expressions=("probe_water","probe_environment"))
area,u,factors,soil,temp=cases(np.float32)
n=len(u)
def environment(u,temp):
 tc=temp-273.15
 es=jax.lax.optimization_barrier(611.2*jnp.exp(17.67*tc/(tc+243.5)));den=u[2]-.378*es
 qs=.622*es/den
 numerator=jax.lax.optimization_barrier(.622*u[2]*es)
 numerator=jax.lax.optimization_barrier(numerator*17.67)*243.5
 dq=numerator/(den*den*(tc+243.5)*(tc+243.5))
 rho=u[2]/(287.*u[0]*(1.+.61*u[1]));mol=u[2]/(8.314462618*u[0])
 return jnp.concatenate((es,qs,dq,jnp.stack((rho,mol))))
env_j=np.asarray(jax.jit(jax.vmap(environment))(u,temp))
env_c=cp.empty((n,35),cp.float32)
mod.get_function("probe_environment")(((n+63)//64,),(64,),(cp.asarray(u),cp.asarray(temp),env_c,np.int32(n)))
env_c=cp.asnumpy(env_c)
def stats(a,b):
 bad=a!=b;index=np.argwhere(bad)
 return {"max_abs":float(np.max(abs(a-b))),"different":int(bad.sum()),"examples":
 [{"index":x.tolist(),"cuda":float(a[tuple(x)]),"jax":float(b[tuple(x)])} for x in index[:8]]}
report={"environment":stats(env_c,env_j)}
report["environment_fields"]={k:stats(env_c[:,s],env_j[:,s]) for k,s in
    (("es",slice(0,11)),("qs",slice(11,22)),("dq",slice(22,33)),("rho",slice(33,34)),("molar",slice(34,35)))}
(ROOT/"outputs/environment-probe-hlo.txt").write_text(jax.jit(jax.vmap(environment)).lower(u,temp).compile().as_text())
variant_source=r'''
extern "C" __global__ void variants(const float* env,const float* u,float* o,int n){
 int r=blockIdx.x*blockDim.x+threadIdx.x;if(r>=n)return;
 for(int j=0;j<11;++j){float es=env[r*35+j],den=u[r*23+2]-.378f*es;
 o[(r*11+j)*7]=.622f*es/den;
 o[(r*11+j)*7+1]=.622f*(es/den);
 o[(r*11+j)*7+2]=(.622f/den)*es;
 o[(r*11+j)*7+3]=(.622f*es)*__frcp_rn(den);
 o[(r*11+j)*7+4]=__fdividef(.622f*es,den);
 o[(r*11+j)*7+5]=(.622f*es)*__fdividef(1.f,den);
 o[(r*11+j)*7+6]=__fdiv_rn(.622f*es,den);
 }}'''
variant_kernel=cp.RawKernel(variant_source,"variants",options=("--std=c++14","--fmad=false","--prec-div=false"))
variants=cp.empty((n,11,7),cp.float32)
variant_kernel(((n+63)//64,),(64,),(cp.asarray(env_j),cp.asarray(u),variants,np.int32(n)))
report["qs_variants"]={str(i):stats(cp.asnumpy(variants[:,:,i]),env_j[:,11:22]) for i in range(7)}
qs=env_j[:,11:22].copy();rho=env_j[:,33].copy()
gs=np.random.default_rng(440).uniform(.001,.02,(n,10)).astype(np.float32)
def water(area,u,soil,qs,gs,rho):
 wet,tr,eg,eb,k,ku,kb,res,uptake=_water(u[1],qs,gs,area,u[13],rho,u,soil)
 return jnp.concatenate((wet,tr,k,jnp.stack((eg,eb,ku,kb,res)),uptake))
compiled=jax.jit(jax.vmap(water))
a=np.asarray(compiled(area,u,soil,qs,gs,rho))
b=cp.empty((n,39),cp.float32)
mod.get_function("probe_water")(((n+63)//64,),(64,),(*[cp.asarray(x) for x in (area,u,soil,qs,gs,rho)],b,np.int32(n)))
report["water"]=stats(cp.asnumpy(b),a)
(ROOT/"outputs/water-probe.json").write_text(json.dumps(report,indent=2)+"\n")
(ROOT/"outputs/water-probe-hlo.txt").write_text(compiled.lower(area,u,soil,qs,gs,rho).compile().as_text())
print(json.dumps(report,indent=2),flush=True)
