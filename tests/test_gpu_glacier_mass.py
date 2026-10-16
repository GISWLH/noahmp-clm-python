import numpy as np
import pytest
from noahmp_clm.physics.glacier import GlacierIce,exchange_glacier_vapor,add_glacier_surface_water
from noahmp_clm.gpu.glacier_mass import pack_glacier_transfer,validate_glacier_transfer
from test_gpu_glacier import host,compare_same


def cases(dtype):
    rng=np.random.default_rng(20315);rows=[]
    for row in range(512):
        n=row%5;mode=row%2
        if mode and n==0:n=1
        m=rng.uniform(.01,100.,n);h=m*2100.*rng.uniform(-30.,0.,n)
        ice=GlacierIce(m,h,rng.uniform(0.,1e-5,(n,5)))
        if mode:
            amount=float(m[0]) if row%7==0 else rng.uniform(-5.,.95*m[0]);value=rng.uniform(-60.,70.)
            tracer=np.zeros(5)
        else:
            amount=rng.uniform(0.,30.);value=amount*(334000.+4180.*rng.uniform(-73.,150.));tracer=rng.uniform(0.,1e-5,5)
        rows.append(pack_glacier_transfer(ice,mode,amount,value,impurities=tracer,dtype=dtype))
    return tuple(np.stack([r[i] for r in rows]) for i in range(4))


def budgets(args,out):
    g,n,mode,u=args;g,u=g.astype(float),u.astype(float);new=out['cells'].astype(float)
    oldmh=g[:,:,:2].sum(1);newmh=new[:,:,:2].sum(1);incoming=np.where((mode==0)[:,None],u[:,:2],0.)
    exports=np.column_stack((out['runoff_kgm2'].astype(float)+out['mass_kgm2'].astype(float),
        out['runoff_enthalpy_jm2'].astype(float)+out['advected_enthalpy_jm2'].astype(float)))
    residual=newmh-oldmh-incoming+exports
    scale=np.maximum(abs(g[:,:,:2]).sum(1)+abs(incoming)+abs(exports),1.)
    bound=64.*np.finfo(out['cells'].dtype).eps*scale
    assert np.all(abs(residual)<=bound+1e-8),(residual,bound)
    tracer=new[:,:,2:].sum(1)-g[:,:,2:].sum(1)+out['impurity_out_kgm2'].astype(float)-np.where((mode==0)[:,None],u[:,2:],0.)
    assert np.max(abs(tracer))<=64.*np.finfo(out['cells'].dtype).eps*1e-3+1e-22
    return np.max(abs(residual),axis=0),np.max(bound,axis=0)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_glacier_transfer_jax_numpy_reference(dtype):
    jax=pytest.importorskip('jax')
    if dtype is np.float64 and not jax.config.x64_enabled:pytest.skip('JAX x64 required')
    from noahmp_clm.gpu.glacier_mass import run_glacier_transfer_jax
    args=cases(dtype);raw=run_glacier_transfer_jax(*args)
    assert next(iter(raw['cells'].devices())).platform=='gpu';out=host(raw);errors={}
    for i,(g,n,mode,u) in enumerate(zip(*args)):
        g,u=g.astype(float),u.astype(float);ice=GlacierIce(g[:n,0],g[:n,1],g[:n,2:])
        if mode:
            new,d=exchange_glacier_vapor(ice,u[0],u[1]+273.15);d['impurity_out_kgm2']=d.pop('residue_kgm2')
        else:
            new,d=add_glacier_surface_water(ice,u[0],u[1],u[2:]);d.update(mass_kgm2=0.,advected_enthalpy_jm2=0.,latent_energy_jm2=0.)
            d['impurity_out_kgm2']=d.pop('runoff_impurities_kgm2')
        expected=np.zeros((4,7));expected[:len(new.mass),:2]=np.column_stack((new.mass,new.enthalpy));expected[:len(new.mass),2:]=new.impurities
        np.testing.assert_array_equal(out['counts'][i],len(new.mass))
        for key,ref in [('cells',expected)]+[(k,d[k]) for k in out if k not in ('cells','counts')]:
            value=out[key][i];errors[key]=max(errors.get(key,0.),float(np.max(abs(value-ref))))
            np.testing.assert_allclose(value,ref,rtol=.1 if dtype is np.float32 else 1e-8,
                atol=1e-4 if key in ('cells','runoff_kgm2','mass_kgm2') else .1 if dtype is np.float32 else 1e-7,err_msg=f'{key} {i}')
    res,bound=budgets(args,out);print('glacier mass reference',dtype,errors,'M/H residual/bound',res.tolist(),bound.tolist(),flush=True)


def test_glacier_transfer_cuda_same_precision():
    cp=pytest.importorskip('cupy');pytest.importorskip('jax')
    from noahmp_clm.gpu.glacier_mass import run_glacier_transfer_jax
    from noahmp_clm.gpu.cuda.glacier_mass import run_glacier_transfer_cuda
    args=cases(np.float32);a=run_glacier_transfer_cuda(*args);assert isinstance(a['cells'],cp.ndarray)
    a,b=host(a),host(run_glacier_transfer_jax(*args));res,bound=budgets(args,a)
    print('glacier mass same precision',compare_same(a,b),'M/H residual/bound',res.tolist(),bound.tolist(),flush=True)


def test_glacier_transfer_contract():
    args=cases(np.float64);validate_glacier_transfer(*args,np)
    args[3][1,0]=1e6
    with pytest.raises(ValueError,match='vapor'):validate_glacier_transfer(*args,np)
