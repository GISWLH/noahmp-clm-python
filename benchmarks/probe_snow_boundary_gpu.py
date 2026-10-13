"""K2 same-input snow boundary arithmetic probe, on actual JAX/CUDA GPU."""
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import cupy as cp
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from check_gpu_options import scenarios
from noahmp_clm.gpu import jax_core as jc
from noahmp_clm.gpu.cuda import loader
from noahmp_clm.gpu.pack import column_state_to_vec,forcing_to_vec
from noahmp_clm.options import PhysicsOptions

SOURCE = r'''
extern "C" __global__ void probe(const float* input,const float* forcing,float* output){
 float s[70],u[8];for(int j=0;j<70;++j)s[j]=input[j];for(int j=0;j<8;++j)u[j]=forcing[j];
 int count=3; snowfall(s,&count,u[7]*u[0],u,2); SnowBoundary b=snow_response(s,u[0],1);
 for(int j=0;j<70;++j)output[j]=s[j];
 output[70]=b.reference;output[71]=b.effective_k;output[72]=b.surface_k;
 float ts=surface_temperature(s,u,true,0.,2,4,1,1);Energy e=energy(ts,s,u,true,0.,2,4,1,1);
 output[73]=ts;output[74]=e.g;output[75]=e.residual;
}
'''

def probe(s,i,u):
    sw,t,d,count,_=jc._add_snowfall(s[16:19],s[19:22],s[22:25],i[32],u[7]*u[0],u[1])
    d=jc._compact_snow(sw,t,d,count,u[0],jnp.int32(2))
    s=s.at[16:19].set(sw).at[19:22].set(t).at[22:25].set(d)
    response=jc._snow_surface_response(sw,t,d,count,u[0],jnp.int32(1))
    def func(ts):
        rn=(1.-s[28])*u[5]+.99*(u[6]-jc.SIGMA*ts**4)
        h=jc._air_density(u[1],u[2],u[4])*jc.CP*jc._exchange_velocity(u[3],u[1],u[2],u[4],s[33],ts,jnp.int32(2))*(ts-u[1])
        return rn-h-jc._coupled_snow_flux(ts,response)
    ts=jc._secant_solve(func,u[1],u[1]+1.)
    return jnp.concatenate((s,jnp.array([*response,ts,jc._coupled_snow_flux(ts,response),func(ts)])))

def main():
    cases=scenarios(24,replace(PhysicsOptions(),OptSurfaceDrag=2))
    packed=[column_state_to_vec(s) for _,s,_ in cases]
    f=np.array([x[0] for x in packed],np.float32);i=np.array([x[1] for x in packed],np.int32)
    u=np.array([[forcing_to_vec(x) for x in fs] for _,_,fs in cases],np.float32)
    cf,ci,cd=map(cp.asnumpy,loader.run_columns_cuda(f,i,u[:,:3]))
    jf,ji,jd=map(np.asarray,jc.run_columns_jax(f,i,u[:,:3]))
    np.savez('outputs/k2-snow-probe-input.npz',cuda_f=cf,jax_f=jf,i=i,u=u)
    module=cp.RawModule(code=loader.KERNEL_SOURCE+SOURCE,options=('--std=c++14','--fmad=false','--prec-div=false'),name_expressions=('probe',))
    compiled=jax.jit(probe).lower(cf[4],i[4],u[4,3]).compile()
    Path('outputs/k2-snow-probe.hlo.txt').write_text(compiled.as_text())
    for name,s in [('cuda_input',cf[4]),('jax_input',jf[4])]:
        out=cp.empty(76,cp.float32)
        module.get_function('probe')((1,),(1,),(cp.asarray(s),cp.asarray(u[4,3]),out))
        a=cp.asnumpy(out);b=np.asarray(compiled(s,i[4],u[4,3]))
        print(json.dumps(dict(name=name,input_snow=s[16:25].tolist(),
            cuda=a[70:].tolist(),jax=b[70:].tolist(),
            differences=[dict(index=int(k),cuda=float(a[k]),jax=float(b[k])) for k in np.flatnonzero(a!=b)])),flush=True)

if __name__=='__main__':main()
