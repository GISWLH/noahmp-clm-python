"""Four-layer soil water/enthalpy transfers and independently summed budgets."""
import numpy as np
import pytest
from noahmp_clm.column import SoilParams,default_soil_params
from noahmp_clm.physics.soil_enthalpy import SoilEnthalpyState,enthalpy_and_capacity,infiltrate,withdraw_liquid
from noahmp_clm.physics.soil_drainage import drain_soil
from noahmp_clm.gpu.soil_transport import pack_soil_transport,validate_soil_transport
from test_gpu_snicar import assert_science_close


def soil_state(dtype,texture='loam',temperature=280.,wetness=.7,option=1,dz=(.1,.3,.6,1.)):
    p0=default_soil_params(texture)
    p=SoilParams(smcmax=float(dtype(p0.smcmax)),bexp=float(dtype(p0.bexp)),
        smcref=float(dtype(p0.smcref)),satdk=float(dtype(p0.satdk)),
        saturated_suction_m=float(dtype(p0.saturated_suction_m)))
    z=np.asarray(dz,dtype).astype(float);capacity=1000.*p.smcmax*z
    w=np.asarray(capacity*wetness,dtype)
    # Choose representable input states within the *quantized* pore capacity.
    # This is forcing construction, before reference or device execution.
    w=np.where(w.astype(float)>capacity,np.nextafter(w,dtype(0.)),w).astype(float)
    h=enthalpy_and_capacity(w,np.broadcast_to(temperature,4),z,p,option)[0]
    return SoilEnthalpyState(w,h.astype(dtype).astype(float),z,p,option)


def cases(dtype=np.float64,random_count=90):
    rows=[]
    for texture in ('sand','loam','clay'):
        for temp in (250.,273.149,290.):
            for phase in (1,2):
                state=soil_state(dtype,texture,temp,.9,phase)
                rows.extend((pack_soil_transport(state,0,mass=50.,enthalpy=50.*(334000.+4180.*5.),dtype=dtype),
                    pack_soil_transport(state,1,demand=.7*state.liquid,dtype=dtype),
                    pack_soil_transport(state,2,permeability_option=phase,dtype=dtype)))
    for wetness in (0.,1.):
        state=soil_state(dtype,temperature=280.,wetness=wetness)
        rows.extend((pack_soil_transport(state,0,mass=0.,dtype=dtype),
            pack_soil_transport(state,0,mass=1000.,enthalpy=334e6,dtype=dtype),
            pack_soil_transport(state,1,demand=state.water,dtype=dtype),pack_soil_transport(state,2,dtype=dtype)))
    rng=np.random.default_rng(81146)
    for i in range(random_count):
        temp=rng.uniform(240.,315.,4) if i%3 else 273.15-np.exp(rng.uniform(-9.,2.,4))
        state=soil_state(dtype,('sand','loam','clay')[i%3],temp,rng.uniform(.001,.999,4),1+i%2,
            dz=10.**rng.uniform(-2.,.3,4))
        amount=10.**rng.uniform(-6.,2.7)
        rows.append(pack_soil_transport(state,i%3,mass=amount,enthalpy=amount*(334000.+4180.*rng.uniform(-10.,25.)),
            demand=rng.uniform(0.,.95,4)*state.liquid,dt=10.**rng.uniform(1.,5.),permeability_option=1+i%2,dtype=dtype))
    return tuple(np.stack([row[k] for row in rows]) for k in range(4))


def reference(c,p,o,u):
    c,p,u=[np.asarray(v,float) for v in (c,p,u)]
    params=SoilParams(smcmax=p[0],bexp=p[1],smcref=p[2],satdk=p[3],saturated_suction_m=p[4])
    state=SoilEnthalpyState(c[:,0],c[:,1],c[:,2],params,int(o[1]))
    enter=np.zeros(2);runoff=np.zeros(2);take=np.zeros(4);takeh=np.zeros(4)
    flow=np.zeros(4);flowh=np.zeros(4);bottom=np.zeros(2)
    if o[0]==0:
        new,m,h=infiltrate(state,u[0],u[1]);enter=u[:2];runoff=np.array([m,h])
    elif o[0]==1:
        new,takeh=withdraw_liquid(state,u[3:]);take=u[3:]
    else:
        new,d=drain_soil(state,u[2],permeability_option=int(o[2]))
        flow=d['interface_mass_kgm2'];flowh=d['interface_enthalpy_jm2']
        bottom=np.array([d['bottom_mass_kgm2'],d['bottom_enthalpy_jm2']])
    return dict(cells=np.column_stack((new.water,new.enthalpy,new.dz)),temperature_offset_k=new.temperature-273.15,
        temperature_k=new.temperature,liquid_kgm2=new.liquid,ice_kgm2=new.ice,incoming=enter,runoff=runoff,
        withdrawal_mass_kgm2=take,withdrawal_enthalpy_jm2=takeh,interface_mass_kgm2=flow,
        interface_enthalpy_jm2=flowh,bottom_outflow=bottom,valid=True)


def science(got,ref,dtype,dt):
    for key,expected in ref.items():
        if key=='valid':assert got[key];continue
        actual=np.asarray(got[key])
        if key in ('temperature_k','temperature_offset_k'):
            np.testing.assert_allclose(actual,expected,rtol=0.,atol=.05 if dtype==np.float32 else 2e-8)
        elif key=='cells':
            for k,floor,absolute in ((0,.001,.0001),(1,1.,.1),(2,.0001,.00001)):
                assert_science_close(actual[:,k],expected[:,k],relative=.1 if dtype==np.float32 else 1e-7,
                    absolute=absolute if dtype==np.float32 else 1e-7,zero_floor=floor)
        elif key in ('incoming','runoff','bottom_outflow'):
            # Advected energy is an integrated heat flux. Use the existing
            # V1_ACCEPTANCE near-zero H gate: 1 W/m2 below 10 W/m2, integrated
            # over this step. A 1 J gate would impose a new sub-milliwatt gate.
            for k,floor,absolute in ((0,.001,.0001),(1,10.*dt,dt)):
                assert_science_close(actual[k],expected[k],relative=.1 if dtype==np.float32 else 1e-7,
                    absolute=absolute if dtype==np.float32 else 1e-7,zero_floor=floor)
        else:
            water='kgm2' in key
            assert_science_close(actual,expected,relative=.1 if dtype==np.float32 else 1e-7,
                absolute=(.0001 if water else dt) if dtype==np.float32 else 1e-7,
                zero_floor=.001 if water else 10.*dt)


def budgets(c,result):
    old=c[:,:,:2].astype(float).sum(1);new=np.asarray(result['cells'])[:,:,:2].astype(float).sum(1)
    take=np.stack((np.asarray(result['withdrawal_mass_kgm2'],float).sum(1),
                   np.asarray(result['withdrawal_enthalpy_jm2'],float).sum(1)),axis=1)
    incoming=np.asarray(result['incoming'],float);out=np.asarray(result['runoff'],float)+np.asarray(result['bottom_outflow'],float)+take
    residual=new-old-incoming+out
    # Finite operations on extensive storage, independently accumulated in
    # double. At most eight additions/subtractions per layer, not 10% physics.
    scale=np.abs(c[:,:,:2].astype(float)).sum(1)+np.abs(np.asarray(result['cells'])[:,:,:2].astype(float)).sum(1)+abs(incoming)+abs(out)
    bound=16.*np.finfo(c.dtype).eps*np.maximum(scale,1.)
    assert np.all(abs(residual)<=bound),(residual,bound)
    return np.max(abs(residual),axis=0)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_soil_transfer_reference_and_budgets(dtype):
    jax=pytest.importorskip('jax')
    if dtype is np.float64 and not jax.config.x64_enabled:pytest.skip('JAX x64 required')
    from noahmp_clm.gpu.soil_transport import run_soil_transport_jax
    arrays=cases(dtype);result=run_soil_transport_jax(*arrays);result={k:np.asarray(v) for k,v in result.items()}
    maximum={}
    for row,inputs in enumerate(zip(*arrays)):
        ref=reference(*inputs);got={k:v[row] for k,v in result.items()};science(got,ref,dtype,float(inputs[-1][2]))
        for k,v in ref.items():
            if k!='valid':maximum[k]=max(maximum.get(k,0.),float(np.max(abs(got[k]-v))))
    print('soil transfer',str(dtype),maximum,'budget',budgets(arrays[0],result),flush=True)


def test_soil_transfer_cuda_same_precision():
    cp=pytest.importorskip('cupy');pytest.importorskip('jax')
    from noahmp_clm.gpu.soil_transport import run_soil_transport_jax
    from noahmp_clm.gpu.cuda.soil_transport import run_soil_transport_cuda
    arrays=cases(np.float32);a=run_soil_transport_cuda(*arrays);b=run_soil_transport_jax(*arrays)
    for key in a:
        x,y=cp.asnumpy(a[key]),np.asarray(b[key])
        print('soil transfer CUDA/JAX32',key,np.max(x!=y) if x.dtype==bool else np.max(abs(x-y)),flush=True)
        if x.dtype==bool:np.testing.assert_array_equal(x,y)
        else:np.testing.assert_allclose(x,y,rtol=1e-6,atol=1e-6)


def test_soil_transfer_host_contract():
    arrays=list(cases(np.float64,0));validate_soil_transport(*arrays,np)
    bad=[v.copy() for v in arrays];bad[0][0,0,0]=-1.
    with pytest.raises(ValueError):validate_soil_transport(*bad,np)
    bad=[v.copy() for v in arrays];bad[3][0,0]=0.;bad[3][0,1]=1.
    with pytest.raises(ValueError):validate_soil_transport(*bad,np)
    for row in zip(*arrays):reference(*row)


@pytest.mark.parametrize('backend',['jax','cuda'])
def test_soil_transfer_errors_and_empty(backend):
    if backend=='jax':
        pytest.importorskip('jax')
        from noahmp_clm.gpu.soil_transport import run_soil_transport_jax as run
    else:
        pytest.importorskip('cupy')
        from noahmp_clm.gpu.cuda.soil_transport import run_soil_transport_cuda as run
    arrays=[v[:1].copy() for v in cases(np.float32,0)]
    empty=run(*(v[:0] for v in arrays));assert empty['cells'].shape==(0,4,3)
    arrays[2][0,0]=1;arrays[3][0,3:]=arrays[0][0,:,0]+1.
    with pytest.raises(ValueError,match='available liquid'):run(*arrays)
    arrays[2][0,0]=0;arrays[3][0,0]=1.;arrays[3][0,1]=1e30
    with pytest.raises(ValueError,match='physical domain'):run(*arrays)
