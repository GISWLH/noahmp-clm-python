"""Saved G614 phase context, actual extension, native equations and hard gates."""
from functools import lru_cache
from pathlib import Path
import numpy as np
import pytest
from test_phase_jacobian_boundary import operands
from test_canopy_phase_energy import independent_reference

def arrays(dtype):
    v,_,_=operands();snow=v['snow'][:,[0,3,4,5,6,7,8]]
    return tuple(np.asarray(x,np.int32 if k==8 else dtype) for k,x in enumerate(
        (v['leaf'],v['u'],v['factors'],v['soil'],np.zeros(10),v['condensate'],v['boundary'],snow,v['snow_count'])))

def jax_result(dtype):
    import jax
    import jax.numpy as j
    from noahmp_clm.gpu.canopy_phase_energy import phase_energy_kernel_jax
    from noahmp_clm.gpu.snicar import _jax_tables
    v,_,_=operands();p=j.asarray(v['leaf_parameters'],dtype);extension=j.asarray(v['extension'],dtype)
    ice,impurity=_jax_tables(str(np.dtype(dtype)))
    tolerance=.005 if dtype==np.float32 else 5e-5
    fn=jax.jit(lambda *x:phase_energy_kernel_jax(*x,p,ice,impurity,exchange_extension=extension,tolerance=tolerance))
    return np.asarray(fn(*map(j.asarray,arrays(dtype))))

@lru_cache(maxsize=1)
def cuda_kernel():
    import cupy as cp
    from noahmp_clm.gpu.cuda import canopy_phase_energy as module
    code='\n'.join(Path(module.__file__).with_name(name).read_text() for name in
        ('leaf_kernel.cu','radiation_kernel.cu','canopy_energy_kernel.cu','canopy_phase_energy_kernel.cu'))
    # Test adapter only: production context, evaluator and Newton solve run on
    # device. The saved MOST extension is passed exactly as in the full column.
    code+='''
extern "C" __global__ void saved_phase(const float* a,const float* u,const float* f,const float* soil,
 const float* stem,const float* c,const float* b,const float* radiation,const float* p,const float* extension,float* out) {
 PhaseContext v;phase_prepare(a,u,f,soil,stem,c,b,p,radiation,v);
 v.water_velocity=extension[0];v.heat_temperature=extension[1];phase_solve(v,out,100,.005f);
}
'''
    return cp.RawModule(code=code,options=('--std=c++14','--fmad=false','--prec-div=false','--prec-sqrt=false'),name_expressions=('saved_phase',)).get_function('saved_phase')

def cuda_result():
    import cupy as cp
    from noahmp_clm.gpu.cuda.canopy_snow_radiation import _kernel
    from noahmp_clm.gpu.cuda.snicar import _tables
    from noahmp_clm.gpu.canopy_boundary import radiation_inputs
    v,_,_=operands();a,u,f,soil,stem,c,b,snow,n=tuple(cp.ascontiguousarray(cp.asarray(x)) for x in arrays(np.float32))
    rad=cp.empty((1,584),cp.float32)
    _kernel(True)((1,),(32,),(a[None,:],snow[None,:,:],n.reshape(1),cp.ascontiguousarray(radiation_inputs(a[None,:],u[None,:],cp)),*_tables(),rad,np.int32(1)))
    out=cp.empty(300,cp.float32)
    cuda_kernel()((1,),(1,),(a,u,f,soil,stem,c,b,rad,cp.asarray(v['leaf_parameters'],cp.float32),cp.asarray(v['extension'],cp.float32),out))
    return cp.asnumpy(out)

def verify(out,dtype):
    v,args,kw=operands();ref=independent_reference(*args,**kw)
    tol=.005 if dtype==np.float32 else 5e-5
    assert out[68]==1. and out[292]<=tol and abs(out[55:66]).max()<=tol*1.05
    difference=float(abs(out[:11].astype(float)-ref['temperature']).max());assert difference<(.05 if dtype==np.float32 else 1e-4)
    for actual,key in [(out[26],'h'),(out[27],'le')]:np.testing.assert_allclose(actual,ref[key],rtol=.1,atol=.05 if dtype==np.float32 else .001)
    mass=out[195:205].astype(float);heat=out[205:215].astype(float);old=v['condensate'];vapor=(out[225:235].astype(float)+out[235:245].astype(float))
    residual=mass-old[:,0]+vapor
    np.testing.assert_array_less(abs(residual),64*np.finfo(dtype).eps*np.maximum(abs(mass)+abs(old[:,0])+abs(vapor),1e-12)+1e-14)
    np.testing.assert_allclose(mass,ref['mass'],rtol=.1,atol=1e-4 if dtype==np.float32 else 1e-7)
    np.testing.assert_allclose(heat,ref['enthalpy'],rtol=.1,atol=10. if dtype==np.float32 else .1)
    dt=float(v['boundary'][4]);thermal=out[215:225].astype(float);adv=out[245:255].astype(float)
    parcel=(thermal-heat)-adv;bound=64*np.finfo(dtype).eps*np.maximum(abs(thermal)+abs(heat)+abs(adv),1e-8)
    assert np.all(abs(parcel)<=bound+1e-10)
    atmospheric=(float(out[29])-float(out[26]))-float(out[27])-float(out[28])
    closure=atmospheric*dt-float((heat-old[:,1]).sum())-float(adv.sum())-float(out[299])*dt
    closure_bound=11*tol*1.05*dt+64*np.finfo(dtype).eps*max(1.,abs(atmospheric*dt)+abs(heat).sum()+abs(old[:,1]).sum()+abs(adv).sum()+abs(float(out[299])*dt))
    assert abs(closure)<=closure_bound
    print('G614 native phase',str(dtype),dict(maximum_temperature_k=difference,h_difference=float(out[26])-ref['h'],le_difference=float(out[27])-ref['le'],
        max_solver_residual_wm2=float(out[292]),water_residual_kgm2=float(abs(residual).max()),energy_residual_jm2=closure,energy_bound_jm2=closure_bound,
        iterations=float(out[67]),restarts=float(out[291])),flush=True)

@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_actual_phase_jax_independent_reference(dtype):
    jax=pytest.importorskip('jax')
    if dtype==np.float64 and not jax.config.x64_enabled:pytest.skip('JAX x64 required')
    verify(jax_result(dtype),dtype)

def test_actual_phase_cuda_jax_same_precision_and_independent_budget():
    pytest.importorskip('cupy');pytest.importorskip('jax')
    cu=cuda_result();ja=jax_result(np.float32)
    np.testing.assert_allclose(cu,ja,rtol=1e-6,atol=1e-6)
    verify(cu,np.float32)
    print('G614 phase all300 fields CUDA/JAX32',float(abs(cu.astype(float)-ja.astype(float)).max()),flush=True)
