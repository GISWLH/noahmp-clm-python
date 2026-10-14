"""Native extensive snow transfers: reference equations and seven budgets."""
import numpy as np
import pytest
import json
from pathlib import Path
from noahmp_clm.physics.snowpack import (Snowpack,regrid,add_snowfall,add_rain,
    drain_liquid,remove_sublimation,CI,CW,LF,TF)
from noahmp_clm.gpu.snow_mass import pack_snow_transfer,validate_snow_transfer


def cases(dtype=np.float64,random_count=200):
    rows=[]
    def add(p,mode,**kwargs):rows.append(pack_snow_transfer(p,mode,dtype=dtype,**kwargs))
    empty=Snowpack.empty();add(empty,0);add(empty,3);add(empty,4)
    add(empty,1,amount=2.,temperature=TF-10.)
    old=Snowpack.from_ice([120.],TF-15.,300.);add(old,1,amount=3.,temperature=TF-2.,radius=100.,impurities=np.arange(1,6)*1e-8)
    add(old,1,amount=0.);add(old,2,amount=.2,temperature=TF)
    melted=Snowpack([1.,2.,3.],np.array([1.,2.,3.])*(LF+CW*4),[.02,.04,.06],impurities=np.arange(15).reshape(3,5)*1e-9)
    add(melted,3)
    cold=Snowpack([1.,10.],[LF+CW*2,-10*CI*20],[.01,.1]);add(cold,3)
    barrier=Snowpack([11.,9.17],[10*LF,0.],[.003,.01],impurities=np.ones((2,5))*1e-8);add(barrier,3)
    expansion=Snowpack([10.,5.],[10*LF,-5*CI*20],[.02,.008]);add(expansion,3)
    sub=Snowpack([1.,2.],[-CI*10,-2*CI*20],[.01,.02],impurities=np.ones((2,5))*1e-8);add(sub,4,amount=3.)
    thin=Snowpack([300.,3e-10],[0.,-3e-10*CI*20],[1.,1e-12],[54.5,1500.],np.array([[0.]*5,[1e-10]*5]));add(thin,0)
    for fixture in json.loads((Path(__file__).parent/'fixtures/snow_melt_tracer_exhaustion.json').read_text())['cases']:
        rows.append((np.asarray(fixture['cells'],dtype),np.asarray(fixture['count'],np.int32),
            np.asarray(fixture['mode'],np.int32),np.asarray(fixture['inputs'],dtype)))
    rng=np.random.default_rng(661052)
    for row in range(random_count):
        n=1+row%5;mode=row//5%5
        z=10.**rng.uniform(-4.,-.1,n);mass=z*rng.uniform(50.,900.,n)
        h=mass*rng.uniform(-CI*40.,1.1*LF,n)
        p=Snowpack(mass,h,z,rng.uniform(30.,1500.,n),mass[:,None]*10.**rng.uniform(-12.,-4.,(n,5)))
        params={}
        if mode==1:params=dict(amount=10.**rng.uniform(-5.,1.),temperature=rng.uniform(TF-40.,TF),density=rng.uniform(50.,300.),radius=rng.uniform(54.5,204.5),impurities=10.**rng.uniform(-12.,-5.,5))
        if mode==2:params=dict(amount=10.**rng.uniform(-5.,1.),temperature=rng.uniform(TF-5.,TF+20.))
        if mode==3:params=dict(retention=rng.uniform(.01,.1),impermeable_porosity=rng.uniform(.01,.1))
        if mode==4:params=dict(amount=rng.uniform()*p.ice.sum())
        add(p,mode,**params)
    return tuple(np.stack([row[i] for row in rows]) for i in range(4))


def reference(c,n,mode,u):
    c,u=np.asarray(c,float),np.asarray(u,float);n=int(n);mode=int(mode)
    p=Snowpack(c[:n,0],c[:n,1],c[:n,2],c[:n,3],c[:n,4:])
    led=np.zeros((4,7))
    if mode==0:q=regrid(p)
    elif mode==1:
        q=add_snowfall(p,u[0],TF+u[1],u[2],u[3],u[11:])
        if u[0]>0:led[0]=np.r_[u[0],u[0]*CI*u[1],u[11:]]
    elif mode==2:
        q=add_rain(p,u[0],TF+u[1]);led[0,:2]=u[0],u[0]*(LF+CW*u[1])
    elif mode==3:
        q,d=drain_liquid(p,retention=u[4],impermeable_porosity=u[5],scavenging=u[6:11],return_ledger=True)
        for row,name in ((1,'drainage'),(2,'runoff')):led[row]=np.r_[d[name+'_kgm2'],d[name+'_enthalpy_jm2'],d[name+'_impurities_kgm2']]
    else:
        amount=min(u[0],p.ice.sum());q,heat,residue=remove_sublimation(p,amount);led[3]=np.r_[amount,heat,residue]
    cells,nn,_,_=pack_snow_transfer(q,0)
    return dict(cells=cells,counts=nn,incoming=led[0],drainage=led[1],runoff=led[2],sublimation=led[3])


def budgets(c,result):
    old=c[:,:,[0,1,4,5,6,7,8]].astype(float).sum(1)
    new=np.asarray(result['cells'])[:,:,[0,1,4,5,6,7,8]].astype(float).sum(1)
    led={k:np.asarray(result[k]).astype(float) for k in ('incoming','drainage','runoff','sublimation')}
    balance=new-old-led['incoming']+led['drainage']+led['runoff']+led['sublimation']
    scale=np.abs(c[:,:,[0,1,4,5,6,7,8]].astype(float)).sum(1)+np.abs(new)+sum(np.abs(v) for v in led.values())
    floor=np.array([1e-10,2e-7,*([1e-19]*5)])
    bound=32*np.finfo(c.dtype).eps*scale+floor
    np.testing.assert_array_less(np.abs(balance),bound)
    return np.max(np.abs(balance),axis=0)


def compare_reference(c,n,m,u,got):
    max_error=np.zeros(9);max_relative=np.zeros(9);ledger_error=np.zeros(7)
    # Scientific gates are separate from the floating-point budget bound.
    floors=np.array([.001,1.,.0001,1.,*([1e-12]*5)])
    absolute=np.array([.0001,.1,.00001,.1,*([1e-13]*5)])
    single=c.dtype==np.float32
    for row in range(len(c)):
        ref=reference(c[row],n[row],m[row],u[row]);nn=int(ref['counts'])
        assert got['counts'][row]==nn,(row,got['counts'][row],nn)
        a,b=got['cells'][row],ref['cells'];error=np.abs(a-b);max_error=np.maximum(max_error,error.max(0))
        mask=np.abs(b)>floors
        max_relative=np.maximum(max_relative,np.max(np.where(mask,error/np.maximum(np.abs(b),floors)*100.,0.),axis=0))
        if single:
            assert np.all(error<=np.where(mask,.1*np.abs(b),absolute)),(row,a,b,error)
        else:np.testing.assert_allclose(a,b,rtol=1e-9,atol=1e-7)
        for key in ('incoming','drainage','runoff','sublimation'):
            a,b=got[key][row],ref[key];err=np.abs(a-b);ledger_error=np.maximum(ledger_error,err)
            norm=np.abs(b)>floors[[0,1,4,5,6,7,8]]
            tol=np.where(norm,.1*np.abs(b),absolute[[0,1,4,5,6,7,8]])
            if single:assert np.all(err<=tol),(row,key,a,b)
            else:np.testing.assert_allclose(a,b,rtol=1e-9,atol=1e-7)
    return dict(cell_max_absolute=max_error.tolist(),cell_max_relative_percent=max_relative.tolist(),ledger_max_absolute=ledger_error.tolist())


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_snow_transfer_reference_and_budgets(dtype):
    jax=pytest.importorskip('jax')
    if dtype is np.float64 and not jax.config.x64_enabled:pytest.skip('JAX x64 required')
    from noahmp_clm.gpu.snow_mass import run_snow_transfer_jax
    inputs=cases(dtype);got={k:np.asarray(v) for k,v in run_snow_transfer_jax(*inputs).items()}
    report=compare_reference(*inputs,got);print('snow mass',dtype,report,'budget',budgets(inputs[0],got),flush=True)
    assert np.all(got['cells'][:,:,0]>=0.) and np.all(got['cells'][:,:,4:]>=0.)
    assert got['counts'][7]==0 and got['counts'][11]==0
    assert got['drainage'][9,0]==0. and got['runoff'][9,0]>8.
    assert got['runoff'][10,0]>0.
    np.testing.assert_allclose(got['drainage'][7,0]+got['runoff'][7,0],6.,rtol=1e-6)
    np.testing.assert_allclose(got['drainage'][7,1]+got['runoff'][7,1],6*(LF+CW*4),rtol=1e-6)


def test_snow_transfer_cuda_same_precision():
    cp=pytest.importorskip('cupy');pytest.importorskip('jax')
    from noahmp_clm.gpu.snow_mass import run_snow_transfer_jax
    from noahmp_clm.gpu.cuda.snow_mass import run_snow_transfer_cuda
    inputs=cases(np.float32);a=run_snow_transfer_cuda(*inputs);b=run_snow_transfer_jax(*inputs)
    host={k:cp.asnumpy(v) for k,v in a.items()}
    for key in host:
        print('snow mass CUDA/JAX32',key,np.max(np.abs(host[key]-np.asarray(b[key]))),flush=True)
        np.testing.assert_allclose(host[key],b[key],rtol=1e-6,atol=1e-6)
    print('CUDA independent budgets',budgets(inputs[0],host),flush=True)


def test_snow_transfer_contract():
    c,n,m,u=cases(np.float64,0);validate_snow_transfer(c,n,m,u,np)
    bad=c.copy();bad[0,4,1]=1.
    with pytest.raises(ValueError,match='padding'):validate_snow_transfer(bad,n,m,u,np)
    bad=u.copy();bad[11,0]=4.
    with pytest.raises(ValueError,match='supply'):validate_snow_transfer(c,n,m,bad,np)
    with pytest.raises(ValueError,match='integer'):validate_snow_transfer(c,n.astype(float),m,u,np)


def test_snow_grid_threshold_same_precision_and_conservation():
    cp=pytest.importorskip('cupy');pytest.importorskip('jax')
    from noahmp_clm.gpu.snow_mass import run_snow_transfer_jax
    from noahmp_clm.gpu.cuda.snow_mass import run_snow_transfer_cuda
    rows=[]
    for threshold in (.03,.04,.07,.12,.18,.29,.41,.64):
        for depth in (np.nextafter(np.float32(threshold),np.float32(0)),np.float32(threshold),np.nextafter(np.float32(threshold),np.float32(1))):
            p=Snowpack([300*float(depth)],[-CI*10*300*float(depth)],[float(depth)],impurities=np.full((1,5),1e-7))
            rows.append(pack_snow_transfer(p,0,dtype=np.float32))
    inputs=tuple(np.stack([row[k] for row in rows]) for k in range(4))
    a=run_snow_transfer_jax(*inputs);b=run_snow_transfer_cuda(*inputs)
    for key in a:np.testing.assert_allclose(np.asarray(a[key]),cp.asnumpy(b[key]),rtol=1e-6,atol=1e-6)
    print('Grid boundary budgets',budgets(inputs[0],{k:np.asarray(v) for k,v in a.items()}),flush=True)
