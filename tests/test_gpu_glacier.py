"""Glacier device parity, independent extensive ledgers and annual carry."""
import numpy as np
import pytest
from noahmp_clm.physics.glacier import GlacierIce,GlacierParameters,glacier_heat_step
from noahmp_clm.physics.snowpack import Snowpack,TF
from noahmp_clm.gpu.glacier import pack_glacier_case,validate_glacier


def cases(dtype):
    rows=[];rng=np.random.default_rng(19173)
    for i in range(48):
        ns=i%6;ni=1+i%4
        mass=rng.uniform(.5,35.,ns);h=mass*rng.uniform(-2100.*20.,334000.*1.05,ns)
        snow=Snowpack(mass,h,mass/rng.uniform(120.,500.,ns),impurities=rng.uniform(0.,1e-5,(ns,5)))
        ice=GlacierIce.from_thickness(np.geomspace(.01,1.,ni),rng.uniform(250.,TF,ni))
        ice=GlacierIce(ice.mass,ice.enthalpy,rng.uniform(0.,1e-5,(ni,5)))
        rows.append(pack_glacier_case(snow,ice,rng.uniform(30.,7200.),rng.uniform(-150.,500.),
            snow_absorbed=rng.uniform(0.,40.,ns),ice_absorbed=rng.uniform(0.,3.,ni),
            top_temperature=rng.uniform(240.,290.),top_conductance=rng.uniform(0.,20.),dtype=dtype))
    for q in (-100.,0.,100.,500.,2000.):
        rows.append(pack_glacier_case(Snowpack.empty(),GlacierIce([5.],[-105000.]),1800.,q,
            parameters=GlacierParameters(geothermal_wm2=0.),dtype=dtype))
    # One joule must melt a finite amount at Tf despite a deep cold reservoir.
    rows.append(pack_glacier_case(Snowpack.empty(),GlacierIce.from_thickness([.1,100.],TF),1800.,1./1800.,
        parameters=GlacierParameters(geothermal_wm2=0.),dtype=dtype))
    return tuple(np.stack([r[i] for r in rows]) for i in range(6))


def reference(c,g,count,b,p,q):
    c,g,b,p,q=[np.asarray(v,dtype=float) for v in (c,g,b,p,q)];ns,ni=map(int,count)
    snow=Snowpack(c[:ns,0],c[:ns,1],c[:ns,2],c[:ns,3],c[:ns,4:])
    ice=GlacierIce(g[:ni,0],g[:ni,1],g[:ni,2:])
    return glacier_heat_step(snow,ice,b[0],b[1],snow_absorbed=q[:ns],ice_absorbed=q[5:5+ni],
        top_temperature=b[2]+TF,top_conductance=b[3],parameters=GlacierParameters(*p))


def host(result):
    return {k:host(v) if isinstance(v,dict) else np.asarray(v.get() if hasattr(v,'get') else v) for k,v in result.items()}


def independent_budget(inputs,out):
    c,g,count,b,p,q=inputs;c,g,b,p,q=[np.asarray(v,dtype=float) for v in (c,g,b,p,q)]
    newc=out['snow_cells'].astype(float);newg=out['ice_cells'].astype(float)
    old=c[:,:,:2].sum(1)+g[:,:,:2].sum(1);new=newc[:,:,:2].sum(1)+newg[:,:,:2].sum(1)
    inp=b[:,0]*(out['thermal']['top_flux_wm2'].astype(float)+q.sum(1)+p[:,1])
    res=new-old+np.column_stack((out['runoff_kgm2'],out['runoff_enthalpy_jm2'])).astype(float)
    res[:,1]-=inp
    bounds=np.column_stack((out['water_bound_kgm2'],out['energy_bound_jm2'])).astype(float)
    assert np.all(abs(res)<=bounds+1e-6),(res,bounds)
    tracer=newc[:,:,4:].sum(1)+newg[:,:,2:].sum(1)+out['runoff_impurities_kgm2'].astype(float)-c[:,:,4:].sum(1)-g[:,:,2:].sum(1)
    tracer_scale=np.maximum(c[:,:,4:].sum(1)+g[:,:,2:].sum(1),1e-30)
    assert np.all(abs(tracer)<=64.*np.finfo(out['ice_cells'].dtype).eps*tracer_scale+1e-22)
    return res,bounds,inp


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_glacier_jax_reference_and_independent_conservation(dtype):
    jax=pytest.importorskip('jax')
    if dtype is np.float64 and not jax.config.x64_enabled:pytest.skip('JAX x64 required')
    from noahmp_clm.gpu.glacier import run_glacier_jax
    args=cases(dtype);device=run_glacier_jax(*args)
    assert next(iter(device['ice_cells'].devices())).platform=='gpu'
    out=host(device);errors={};res,bounds,_=independent_budget(args,out)
    for row in range(len(args[0])):
        snow,ice,d=reference(*(v[row] for v in args));n=int(args[2][row].sum())
        pairs={'thermal_temperature_k':(out['thermal']['temperature_k'][row,:n],d['thermal_temperature_k']),
            'thermal_enthalpy_jm2':(out['thermal']['enthalpy_jm2'][row,:n],d['thermal_enthalpy_jm2']),
            'ice_mass_kgm2':(out['ice_cells'][row,:,0].sum(),ice.mass.sum()),
            'snow_mass_kgm2':(out['snow_cells'][row,:,0].sum(),snow.mass.sum())}
        for key in ('runoff_kgm2','runoff_enthalpy_jm2','ice_melt_kgm2','snow_drainage_kgm2','net_refreezing_kgm2'):
            pairs[key]=(out[key][row],d[key])
        for key,(got,expected) in pairs.items():
            errors[key]=max(errors.get(key,0.),float(np.max(abs(np.asarray(got)-expected))))
            if 'temperature' in key:rtol,atol=0.,.01 if dtype is np.float32 else 1e-7
            elif dtype is np.float64:rtol,atol=1e-8,1e-4 if 'enthalpy' in key else 1e-9
            else:rtol,atol=.1,.1 if 'enthalpy' in key else .001
            np.testing.assert_allclose(got,expected,rtol=rtol,atol=atol,err_msg=f'{key} row {row}')
    assert out['counts'][52,1]==0
    assert out['runoff_enthalpy_jm2'][53]==pytest.approx(1.,abs=1e-5)
    print('glacier native reference',dtype,errors,'independent M/H',np.max(abs(res),axis=0).tolist(),'bounds',np.max(bounds,axis=0).tolist(),flush=True)


def compare_same(a,b):
    maxerr=0.
    for key in a:
        if isinstance(a[key],dict):maxerr=max(maxerr,compare_same(a[key],b[key]));continue
        if a[key].dtype.kind in 'biu':np.testing.assert_array_equal(a[key],b[key])
        else:
            maxerr=max(maxerr,float(np.max(abs(a[key]-b[key]),initial=0.)))
            np.testing.assert_allclose(a[key],b[key],rtol=1e-6,atol=1e-6,err_msg=key)
    return maxerr


def test_glacier_cuda_same_precision_all_fields():
    cp=pytest.importorskip('cupy');pytest.importorskip('jax')
    from noahmp_clm.gpu.glacier import run_glacier_jax
    from noahmp_clm.gpu.cuda.glacier import run_glacier_cuda
    args=cases(np.float32);a=run_glacier_cuda(*args);b=run_glacier_jax(*args)
    assert isinstance(a['ice_cells'],cp.ndarray)
    a,b=host(a),host(b);res,bounds,_=independent_budget(args,a)
    print('glacier same precision max',compare_same(a,b),'independent M/H',np.max(abs(res),axis=0).tolist(),flush=True)


def test_glacier_365_days_two_native_paths_with_snowfall():
    cp=pytest.importorskip('cupy');pytest.importorskip('jax');import jax.numpy as j
    from noahmp_clm.gpu.glacier import run_glacier_jax
    from noahmp_clm.gpu.cuda.glacier import run_glacier_cuda
    from noahmp_clm.gpu.snow_mass import run_snow_transfer_jax
    from noahmp_clm.gpu.cuda.snow_mass import run_snow_transfer_cuda
    row=pack_glacier_case(Snowpack.empty(),GlacierIce.from_thickness([.1,.3,.6,9.],263.15),86400.,0.,dtype=np.float32)
    initial=tuple(v[None] for v in row);paths=[tuple(cp.asarray(v) for v in initial),tuple(j.asarray(v) for v in initial)]
    old=np.array([initial[1][:,:,0].astype(float).sum(),initial[1][:,:,1].astype(float).sum()])
    incoming=np.zeros((2,2));outgoing=np.zeros((2,2));bounds_sum=np.zeros((2,2));maximum=np.zeros(2);stepmax=np.zeros(2);same=0.
    for day in range(365):
        flux=np.float32(50.*np.sin(2.*np.pi*(day-90.)/365.));results=[]
        for backend,(xp,run,add) in enumerate(((cp,run_glacier_cuda,run_snow_transfer_cuda),(j,run_glacier_jax,run_snow_transfer_jax))):
            c,g,n,b,p,q=paths[backend]
            if flux<0.:
                u=xp.asarray([[.5,-10.,100.,54.5,.033,.05,.2,.02,.02,.01,.01,0.,0.,0.,0.,0.]],dtype=xp.float32)
                added=add(c,n[:,0],xp.ones(1,dtype=xp.int32),u);c=added['cells'];n=xp.stack((added['counts'],n[:,1]),axis=1)
                incoming[backend]+=[.5,-10500.]
            b=xp.asarray([[86400.,flux,0.,0.]],dtype=xp.float32);args=(c,g,n,b,p,q)
            result=run(*args);h=host(result);results.append(h)
            host_args=tuple(v.get() if hasattr(v,'get') else np.asarray(v) for v in args)
            res,bounds,energy=independent_budget(host_args,h);stepmax=np.maximum(stepmax,abs(res[0]))
            incoming[backend,1]+=energy[0];outgoing[backend]+=[h['runoff_kgm2'][0],h['runoff_enthalpy_jm2'][0]];bounds_sum[backend]+=bounds[0]
            new=h['snow_cells'][0,:,:2].astype(float).sum(0)+h['ice_cells'][0,:,:2].astype(float).sum(0)
            cumulative=new-old-incoming[backend]+outgoing[backend];maximum=np.maximum(maximum,abs(cumulative))
            assert np.all(abs(cumulative)<=bounds_sum[backend]+1e-5)
            paths[backend]=(result['snow_cells'],result['ice_cells'],result['counts'],b,p,q)
        same=max(same,compare_same(*results))
    print('glacier 365 two native paths same / max step / cumulative / bounds / runoff',same,stepmax.tolist(),maximum.tolist(),bounds_sum.max(0).tolist(),outgoing.tolist(),flush=True)


def test_glacier_packing_contract():
    args=cases(np.float64);validate_glacier(*args,np)
    args[2][0,1]=0
    with pytest.raises(ValueError,match='counts'):validate_glacier(*args,np)
    args=cases(np.float64);args[1][0,3,0]=1.
    with pytest.raises(ValueError,match='padding'):validate_glacier(*args,np)
