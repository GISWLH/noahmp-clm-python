"""Device irrigation transfer accounting, independent reference and state carry."""
import numpy as np
import pytest
from noahmp_clm.physics.irrigation import IrrigationState,IrrigationParameters,step_irrigation
from noahmp_clm.gpu.irrigation import pack_irrigation,validate_irrigation
from test_gpu_snicar import assert_science_close


def batch(dtype,n=512):
    rng=np.random.default_rng(51716);rows=[]
    for k in range(n):
        dz=np.array([.1,.3,.6,1.]);theta=rng.uniform(.02,.4,4);ice=np.zeros(4)
        if k%7==0:ice[0]=rng.uniform(.1,10.)
        check=k%3==0;state=IrrigationState() if check or k%3==1 else IrrigationState(rng.uniform(.01,100.),rng.uniform(100.,21600.))
        p=IrrigationParameters(rng.uniform(.05,2.5),rng.uniform(.2,.5),rng.uniform(.01,.19),rng.uniform(.1,1.),rng.uniform(100.,86400.))
        rows.append(pack_irrigation(state,theta*1000.*dz,ice,dz,rng.uniform(1.,7200.),0. if k%5==0 else rng.uniform(.01,7.),
            check=check,supply_kgm2=rng.uniform(0.,10.),supply_temperature_k=rng.uniform(274.,310.),parameters=p,dtype=dtype))
    return tuple(np.stack([row[k] for row in rows]) for k in range(5))


def reference(row):
    state,layers,u,flags,p=row;state=np.asarray(state,float);layers=np.asarray(layers,float);u=np.asarray(u,float)
    after,d=step_irrigation(IrrigationState(*state),layers[:,0],layers[:,1],layers[:,2],u[0],u[1],
        check=bool(flags[0]),supply_kgm2=u[2],supply_temperature_k=u[3],parameters=IrrigationParameters(*np.asarray(p,float)))
    return dict(state=np.array([after.pending_kgm2,after.remaining_seconds]),**d)


def budget(arrays,got):
    old=arrays[0].astype(float);out=got['state'].astype(float)
    actual={k:got[k].astype(float) for k in ('applied_kgm2','unmet_kgm2','cancelled_kgm2','requested_kgm2')}
    residual=out[:,0]-old[:,0]+actual['applied_kgm2']+actual['unmet_kgm2']+actual['cancelled_kgm2']-actual['requested_kgm2']
    bound=128.*np.finfo(arrays[0].dtype).eps*np.maximum(out[:,0]+old[:,0]+sum(actual.values()),1.)
    assert np.all(abs(residual)<=bound)
    expected=actual['applied_kgm2']*(334000.+4180.*(arrays[2][:,3].astype(float)-273.15))
    heat_error=abs(got['applied_enthalpy_jm2'].astype(float)-expected)
    heat_bound=16.*np.finfo(arrays[0].dtype).eps*np.maximum(abs(expected)+actual['applied_kgm2']*4180.*273.15,1.)
    assert np.all(heat_error<=heat_bound)
    return dict(request=float(abs(residual).max()),request_bound=float(bound.max()),heat=float(heat_error.max()),heat_bound=float(heat_bound.max()))


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_irrigation_native_reference_and_independent_budget(dtype):
    jax=pytest.importorskip('jax')
    if dtype is np.float64 and not jax.config.x64_enabled:pytest.skip('JAX x64 required')
    from noahmp_clm.gpu.irrigation import run_irrigation_jax
    arrays=batch(dtype);got={k:np.asarray(v) for k,v in run_irrigation_jax(*arrays).items()};maximum={}
    for k,row in enumerate(zip(*arrays)):
        expected=reference(row)
        for name,value in got.items():
            if value.dtype.kind=='b':np.testing.assert_array_equal(value[k],expected[name]);continue
            maximum[name]=max(maximum.get(name,0.),float(np.max(abs(value[k]-expected[name]))))
            if name in ('request_residual_kgm2','request_bound_kgm2'):continue
            floor,absolute=(10.,1.) if name=='applied_enthalpy_jm2' else (.001,.0001)
            assert_science_close(value[k],expected[name],relative=.1,zero_floor=floor,absolute=absolute)
    print('irrigation native',dtype,maximum,'budget',budget(arrays,got),flush=True)


def test_irrigation_cuda_same_precision_and_continuous_requests():
    cp=pytest.importorskip('cupy');pytest.importorskip('jax')
    from noahmp_clm.gpu.irrigation import run_irrigation_jax
    from noahmp_clm.gpu.cuda.irrigation import run_irrigation_cuda
    arrays=batch(np.float32);ja=run_irrigation_jax(*arrays);cu=run_irrigation_cuda(*arrays);maximum=0.
    for key in ja:
        a=cp.asnumpy(cu[key]);b=np.asarray(ja[key])
        if b.dtype.kind=='b':np.testing.assert_array_equal(a,b)
        else:
            maximum=max(maximum,float(np.max(abs(a-b))));np.testing.assert_allclose(a,b,rtol=1e-6,atol=1e-6,err_msg=key)
    print('irrigation same precision',maximum,'budget',budget(arrays,{k:cp.asnumpy(v) for k,v in cu.items()}),flush=True)
    # Prescribed soil inputs isolate the request state machine; this is not
    # presented as a physical soil/irrigation trajectory.
    arrays=list(batch(np.float32,n=8));arrays[0][:]=0.;arrays[1][:,:,1]=0.;arrays[2][:,0]=1800.;arrays[2][:,1]=3.;arrays[4][:,4]=21600.
    astate=bstate=arrays[0];total=np.zeros(8);unmet=np.zeros(8);requested=np.zeros(8);cancelled=np.zeros(8)
    for step in range(96):
        arrays[3][:]=int(step%48==0);arrays[2][:,1]=3.
        if step%48>=5:arrays[2][::2,1]=0.
        a=run_irrigation_cuda(astate,*arrays[1:]);b=run_irrigation_jax(bstate,*arrays[1:])
        for key in b:
            av=cp.asnumpy(a[key]);bv=np.asarray(b[key])
            if bv.dtype.kind=='b':np.testing.assert_array_equal(av,bv)
            else:np.testing.assert_allclose(av,bv,rtol=1e-6,atol=1e-6,err_msg=key)
        total+=cp.asnumpy(a['applied_kgm2']).astype(float);unmet+=cp.asnumpy(a['unmet_kgm2']).astype(float)
        requested+=cp.asnumpy(a['requested_kgm2']).astype(float);cancelled+=cp.asnumpy(a['cancelled_kgm2']).astype(float)
        astate=a['state'];bstate=b['state'];assert isinstance(astate,cp.ndarray)
    residual=cp.asnumpy(astate)[:,0]+total+unmet+cancelled-requested
    bound=128.*np.finfo(np.float32).eps*np.maximum(total+unmet+cancelled+requested,1.)
    assert np.all(abs(residual)<=bound)
    print('irrigation 8x96 two native paths request residual',float(abs(residual).max()),'bound',float(bound.max()),flush=True)


def test_irrigation_native_contract_rejects_fractional_events_and_pending_overwrite():
    arrays=batch(np.float64);validate_irrigation(*arrays,np)
    bad=list(arrays);bad[3]=bad[3].astype(float)
    with pytest.raises(ValueError,match='integer'):validate_irrigation(*bad,np)
    bad=[a.copy() for a in arrays];bad[0][0]=[1.,1800.];bad[3][0]=1
    with pytest.raises(ValueError,match='request'):validate_irrigation(*bad,np)
    with pytest.raises(ValueError,match='boolean'):
        pack_irrigation(IrrigationState(),np.ones(4),np.zeros(4),np.ones(4),1800.,3.,check=.5)
