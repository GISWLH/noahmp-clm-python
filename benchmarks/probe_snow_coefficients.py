"""Trace native device slab arithmetic; diagnostic code never changes physics."""
import inspect,json,importlib.util
from pathlib import Path
import numpy as np
import jax,jax.numpy as j
import cupy as cp
from noahmp_clm.gpu import snicar
root=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('reference',root/'tests/test_gpu_snicar.py')
ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
c,n,b=ref.cases(np.float32);optics=np.asarray(snicar.run_snow_shortwave_jax(c,n,b)['optics'])
rows=[];keys=[]
for row in range(len(c)):
    for mode in range(2):
        for band in range(5):
            for layer in range(5):
                rows.append([*optics[row,mode,band,:,layer],b[row,6],int(band==0)])
                keys.append((row,mode,band,layer))
v=np.array(rows,np.float32)
source=inspect.getsource(snicar.slab_coefficients_jax)
source=source.replace('_,transfer,_,integral=lax.fori_loop','_,transfer,last_abs_term,integral=lax.fori_loop')
source=source.replace('    def double(_,values):',
    '    initial=j.stack((r,t,up,down,direct,ad,ab))\n    def double(_,values):')
source=source.replace('    return j.stack(result[:5])',
    '    return j.concatenate((j.stack(result[:5]),j.stack((tau,w,g,loss,a,b,beta,count.astype(tau.dtype),width,direct)),transfer.ravel(),integral,initial,j.stack(result[5:]),source,last_abs_term))')
scope=dict(vars(snicar));exec(compile(source,__file__,'exec'),scope)
jresult=np.asarray(jax.jit(jax.vmap(lambda x:scope['slab_coefficients_jax'](*x[:4],x[4]>0)))(j.asarray(v)))
cuda=(root/'src/noahmp_clm/gpu/cuda/snicar_kernel.cu').read_text()
cuda=cuda.replace('    float r=-transfer[3]', '    for(int k=0;k<3;++k){out[36+k]=source[k];out[39+k]=abs_term[k];}\n    float r=-transfer[3]',1)
cuda=cuda.replace('    for(int i=0;i<count;++i) {',
    '''    out[5]=tau;out[6]=w;out[7]=g;out[8]=loss;out[9]=a;out[10]=b;out[11]=beta;
    out[12]=(float)count;out[13]=width;out[14]=beam;
    for(int k=0;k<9;++k)out[15+k]=transfer[k];
    for(int k=0;k<3;++k)out[24+k]=integral[k];
    out[27]=r;out[28]=t;out[29]=u;out[30]=d;out[31]=beam;out[32]=ad;out[33]=ab;
    for(int i=0;i<count;++i) {''',1)
cuda=cuda.replace('    out[0]=r;out[1]=t;out[2]=u;out[3]=d;out[4]=beam;',
    '    out[0]=r;out[1]=t;out[2]=u;out[3]=d;out[4]=beam;out[34]=ad;out[35]=ab;',1)
cuda+='''\nextern "C" __global__ void probe(const float* v,float* out,int n) {
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x)
        snow_slab(v[5*i],v[5*i+1],v[5*i+2],v[5*i+3],v[5*i+4]>0.f,out+42*i);
}\n'''
module=cp.RawModule(code=cuda,options=('--std=c++14','--fmad=false','--prec-div=false'),name_expressions=('probe',))
out=cp.empty((len(v),42),cp.float32)
module.get_function('probe')(((len(v)+63)//64,),(64,),(cp.asarray(v),out,np.int32(len(v))))
cresult=cp.asnumpy(out)
for field in range(42):
    delta=np.abs(cresult[:,field]-jresult[:,field]);ix=int(delta.argmax())
    if delta[ix]>1e-30:
        print(json.dumps(dict(field=field,maximum=float(delta[ix]),case=keys[ix],input=v[ix].tolist(),jax=jresult[ix].tolist(),cuda=cresult[ix].tolist())),flush=True)
