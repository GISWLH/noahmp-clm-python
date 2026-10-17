import numpy as np
import pytest
from noahmp_clm.column import SoilParams
from noahmp_clm.physics.soil_enthalpy import enthalpy_and_capacity,temperature_from_enthalpy,liquid_and_derivative
from noahmp_clm.gpu.enthalpy import validate_soil_thermodynamics


def cases(dtype=np.float32,inverse=False):
    rng=np.random.default_rng(10352);n=2048
    z=rng.uniform(.01,2.,n);por=np.array([.4,.45,.5])[np.arange(n)%3]
    b=np.array([4.,5.5,9.])[np.arange(n)%3]
    water=rng.uniform(.001,.99,n)*por*1000*z
    offset=np.r_[rng.uniform(-120.,120.,n//2),-np.logspace(-7,1.5,n//2)]
    offset[:7]=[-123.15,-60.,-1.,0.,.001,80.,126.85];water[7:10]=0.
    options=(1+np.arange(n)%2).astype(np.int32)
    psi=np.array([.069,.355,.468])[np.arange(n)%3]
    u=np.column_stack((water,offset,z,por,b,psi)).astype(dtype)
    if inverse:
        # Interior energy values remain inside the physical bracket after
        # independent host float64-to-device float32 quantization.
        u[0,1]=-123.;u[6,1]=126.
        u[:,1]=reference(u,options,False)['enthalpy_jm2']
    return u,options


def reference(u,options,inverse):
    u=u.astype(float);out={k:np.empty(len(u)) for k in ('temperature_offset_k','temperature_k','liquid_kgm2','ice_kgm2','enthalpy_jm2','capacity_jm2k')}
    for por,b,psi in np.unique(u[:,3:6],axis=0):
        p=SoilParams(smcmax=por,bexp=b,saturated_suction_m=psi)
        for option in (1,2):
            mask=(u[:,3]==por)&(u[:,4]==b)&(u[:,5]==psi)&(options==option)
            w,q,z=u[mask,:3].T
            t=temperature_from_enthalpy(w,q,z,p,option) if inverse else q+273.15
            h,c=enthalpy_and_capacity(w,t,z,p,option)
            liq=liquid_and_derivative(w/(1000*z),t,p,option)[0]*1000*z
            for key,value in zip(out,(t-273.15,t,liq,w-liq,h,c)):out[key][mask]=value
    return out


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('inverse',[False,True])
def test_jax_soil_thermodynamics_against_numpy(dtype,inverse):
    jax=pytest.importorskip('jax')
    if dtype is np.float64 and not jax.config.x64_enabled: pytest.skip('JAX x64 required')
    from noahmp_clm.gpu.enthalpy import run_soil_thermodynamics_jax
    u,o=cases(dtype,inverse);got=run_soil_thermodynamics_jax(u,o,inverse=inverse);expected=reference(u,o,inverse)
    for key,value in expected.items():
        error=np.max(np.abs(np.asarray(got[key])-value));print(dtype,inverse,key,error,flush=True)
        if key in ('temperature_k','temperature_offset_k'):
            np.testing.assert_allclose(got[key],value,rtol=0.,atol=.002 if dtype is np.float32 else 1e-9)
        else:
            np.testing.assert_allclose(got[key],value,rtol=.1 if dtype is np.float32 else 1e-8,
                # 1e-4 kg/m2 = 0.1 micrometre water equivalent, including
                # a float32 ULP when subtracting two ~1000 kg/m2 reservoirs.
                atol=1e-4 if 'kgm2' in key and dtype is np.float32 else
                    (1e-8 if 'kgm2' in key else (.1 if dtype is np.float32 else 1e-5)))
    assert np.all(np.asarray(got['converged'])==1.)
    assert np.all(np.asarray(got['ice_kgm2'])>=0.)
    assert np.max(np.asarray(got['iterations']))<30


@pytest.mark.parametrize('inverse',[False,True])
def test_cuda_soil_thermodynamics_same_precision(inverse):
    cp=pytest.importorskip('cupy');pytest.importorskip('jax')
    from noahmp_clm.gpu.enthalpy import run_soil_thermodynamics_jax
    from noahmp_clm.gpu.cuda.enthalpy import run_soil_thermodynamics_cuda
    u,o=cases(np.float32,inverse)
    a=run_soil_thermodynamics_cuda(u,o,inverse=inverse);b=run_soil_thermodynamics_jax(u,o,inverse=inverse)
    for key in a:
        v,w=cp.asnumpy(a[key]),np.asarray(b[key])
        print('CUDA/JAX32',inverse,key,np.max(np.abs(v-w)),flush=True)
        np.testing.assert_allclose(v,w,rtol=1e-6,atol=1e-6)


def test_thermodynamics_input_contract():
    u,o=cases(np.float64)
    validate_soil_thermodynamics(u,o,False,np)
    for bad in (o.astype(float),np.full(len(o),3,np.int32)):
        with pytest.raises(ValueError):validate_soil_thermodynamics(u,bad,False,np)
    for field,value in ((0,-1.),(1,150.),(2,0.),(3,1.),(4,0.),(5,0.)):
        bad=u.copy();bad[0,field]=value
        with pytest.raises(ValueError):validate_soil_thermodynamics(bad,o,False,np)


@pytest.mark.parametrize('backend',['jax','cuda'])
def test_gpu_inverse_domain_error_is_not_clipped(backend):
    if backend=='jax':
        pytest.importorskip('jax')
        from noahmp_clm.gpu.enthalpy import run_soil_thermodynamics_jax as run
    else:
        pytest.importorskip('cupy')
        from noahmp_clm.gpu.cuda.enthalpy import run_soil_thermodynamics_cuda as run
    with pytest.raises(ValueError,match='physical domain'):
        run(np.array([[10.,1e30,.1,.4,4.,.355]],np.float32),np.array([1],np.int32),inverse=True)
