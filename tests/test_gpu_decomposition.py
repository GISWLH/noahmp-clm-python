"""Real-device C/N state, budgets and seven-day transfer trajectories."""
import numpy as np
import pytest
from noahmp_clm.physics.decomposition import DecompositionState as State,DecompositionParameters as Parameters,step_decomposition
from noahmp_clm.gpu.decomposition import pack_decomposition,validate_decomposition
from test_gpu_snicar import assert_science_close


def cases(dtype=np.float64,count=512):
    rng=np.random.default_rng(4124);rows=[]
    for k in range(count):
        carbon=rng.uniform(1e-3,5000.,7);nitrogen=carbon/rng.uniform(2.,200.,7)
        if k%7==0:carbon[k%7]=nitrogen[k%7]=0.
        state=State(carbon,nitrogen,0. if k%3==0 else 10.**rng.uniform(-7.,1.))
        p=Parameters(sand_fraction=rng.random(),wood_cellulose_fraction=rng.random(),q10=rng.uniform(1.,2.5))
        rows.append(pack_decomposition(state,rng.uniform(245.,320.),(-3.,-.3,-.002,0.)[k%4],
            (1.,1800.,3600.,86400.)[k%4],oxygen_factor=(0.,.2,1.,1.)[k%4],parameters=p,dtype=dtype,
            carbon_input=np.full(7,.01),nitrogen_input=np.full(7,.0001),mineral_n_input=.00001,
            plant_n_demand=(0.,.001,.1)[k%3]))
    return tuple(np.stack(v) for v in zip(*rows))


def reference(s,u,p):
    state=State(s[:7].astype(float),s[7:14].astype(float),float(s[14]));p=p.astype(float);u=u.astype(float)
    parameters=Parameters(tuple(p[:7]),tuple(p[7:10]),*p[10:])
    state,d=step_decomposition(state,u[0],u[1],u[2],oxygen_factor=u[3],carbon_input=u[4:11],nitrogen_input=u[11:18],
        mineral_n_input=u[18],plant_n_demand=u[19],parameters=parameters)
    return dict(state=np.r_[state.carbon,state.nitrogen,state.mineral_nitrogen],**d)


def independent_budgets(s,u,got):
    s=s.astype(float);u=u.astype(float);state=np.asarray(got['state'],float)
    co=np.asarray(got['carbon_out_gc_m2'],float);ci=np.asarray(got['carbon_in_gc_m2'],float)
    rh=(co-ci).sum(1)
    cr=state[:,:7].sum(1)-s[:,:7].sum(1)-u[:,4:11].sum(1)+rh
    nr=state[:,7:].sum(1)-s[:,7:].sum(1)-u[:,11:19].sum(1)+np.asarray(got['plant_uptake_gn_m2'],float)
    eps=np.finfo(got['state'].dtype).eps
    cb=64.*eps*np.maximum(state[:,:7].sum(1)+s[:,:7].sum(1)+u[:,4:11].sum(1)+abs(rh),1.)
    nb=64.*eps*np.maximum(state[:,7:].sum(1)+s[:,7:].sum(1)+u[:,11:19].sum(1)+abs(got['plant_uptake_gn_m2']),1.)
    assert np.all(abs(cr)<=cb),(cr,cb)
    assert np.all(abs(nr)<=nb),(nr,nb)
    return dict(carbon_gc_m2=float(abs(cr).max()),nitrogen_gn_m2=float(abs(nr).max()),
        carbon_bound_max_gc_m2=float(cb.max()),nitrogen_bound_max_gn_m2=float(nb.max()))


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_native_decomposition_reference_and_independent_budgets(dtype):
    jax=pytest.importorskip('jax')
    if dtype is np.float64 and not jax.config.x64_enabled:pytest.skip('JAX x64 required')
    from noahmp_clm.gpu.decomposition import run_decomposition_jax
    arrays=cases(dtype);got={k:np.asarray(v) for k,v in run_decomposition_jax(*arrays).items()};maximum={}
    for k,row in enumerate(zip(*arrays)):
        ref=reference(*row)
        for key,expected in ref.items():
            if key.endswith('_bound_gc_m2') or key.endswith('_bound_gn_m2') or '_residual_' in key:continue
            a=got[key][k];maximum[key]=max(maximum.get(key,0.),float(np.max(abs(a-expected))))
            # P0 scientific comparison: 10%, with unit-specific near-zero
            # absolute bounds. The separate CUDA/JAX32 gate stays at 1e-6.
            if key=='state':
                assert_science_close(a[:7],expected[:7],relative=.1,zero_floor=.001,absolute=.0001)
                assert_science_close(a[7:],expected[7:],relative=.1,zero_floor=1e-5,absolute=1e-6)
            elif '_gn_m2' in key:
                assert_science_close(a,expected,relative=.1,zero_floor=1e-6,absolute=1e-7)
            elif '_gc_m2' in key:
                assert_science_close(a,expected,relative=.1,zero_floor=1e-4,absolute=1e-5)
            elif key.endswith('_umol_m2s'):
                assert_science_close(a,expected,relative=.1,zero_floor=1e-4,absolute=1e-5)
            else:assert_science_close(a,expected,relative=.1,zero_floor=1e-5,absolute=1e-6)
    print('Decomposition native reference',str(dtype),maximum,flush=True)
    print('Decomposition native budgets',str(dtype),independent_budgets(*arrays[:2],got),flush=True)


def test_native_decomposition_same_precision():
    cp=pytest.importorskip('cupy');pytest.importorskip('jax')
    from noahmp_clm.gpu.decomposition import run_decomposition_jax
    from noahmp_clm.gpu.cuda.decomposition import run_decomposition_cuda
    arrays=cases(np.float32);a={k:np.asarray(v) for k,v in run_decomposition_jax(*arrays).items()}
    b={k:cp.asnumpy(v) for k,v in run_decomposition_cuda(*arrays).items()}
    maximum={}
    for key in a:
        maximum[key]=float(np.max(abs(a[key]-b[key])));np.testing.assert_allclose(a[key],b[key],rtol=1e-6,atol=1e-6,err_msg=key)
    print('Decomposition same precision',maximum,flush=True)
    print('Decomposition CUDA budgets',independent_budgets(*arrays[:2],b),flush=True)


def test_native_decomposition_seven_day_carry_and_global_cn_budget():
    cp=pytest.importorskip('cupy');jax=pytest.importorskip('jax')
    if not jax.config.x64_enabled:pytest.skip('JAX x64 required')
    from noahmp_clm.gpu.decomposition import run_decomposition_jax
    from noahmp_clm.gpu.cuda.decomposition import run_decomposition_cuda
    s,u,p=cases(np.float32,8);u[:,2]=1800.;u[:,3]=1.;u[:,1]=-.002
    states={'numpy64':s.astype(float),'jax64':s.astype(float),'jax32':s.copy(),'cuda32':s.copy()}
    cumulative={b:np.zeros((8,4)) for b in states};same=0.;maximum={}
    for step in range(336):
        u[:,0]=285.+10.*np.sin(2.*np.pi*(step%48)/48.)
        if step%48<12:u[:,1]=-.3
        else:u[:,1]=-.002
        results={}
        for backend,old in states.items():
            dtype=old.dtype;inputs=u.astype(dtype);parameters=p.astype(dtype)
            if backend=='numpy64':
                rows=[reference(*r) for r in zip(old,inputs,parameters)]
                got={k:np.stack([r[k] for r in rows]) for k in rows[0]}
            elif backend=='cuda32':got={k:cp.asnumpy(v) for k,v in run_decomposition_cuda(old,inputs,parameters).items()}
            else:got={k:np.asarray(v) for k,v in run_decomposition_jax(old,inputs,parameters).items()}
            independent_budgets(old,inputs,got);results[backend]=got;states[backend]=got['state']
            cumulative[backend]+=np.column_stack((inputs[:,4:11].astype(float).sum(1),inputs[:,11:19].astype(float).sum(1),
                got['heterotrophic_respiration_gc_m2'].astype(float),got['plant_uptake_gn_m2'].astype(float)))
        for key in results['jax32']:
            a,b=results['jax32'][key],results['cuda32'][key]
            np.testing.assert_allclose(a,b,rtol=1e-6,atol=1e-6,err_msg=f'{step}/{key}')
            same=max(same,float(np.max(abs(a-b))))
    for backend,now in states.items():
        inputs_c,inputs_n,rh,uptake=cumulative[backend].T;old=s.astype(float);now=now.astype(float)
        cr=now[:,:7].sum(1)-old[:,:7].sum(1)-inputs_c+rh
        nr=now[:,7:].sum(1)-old[:,7:].sum(1)-inputs_n+uptake
        # Sum of independently bounded rounding per step; no 10% allowance.
        eps=np.finfo(states[backend].dtype).eps
        cb=336*64*eps*np.maximum(2*(old[:,:7].sum(1)+inputs_c),1.)
        nb=336*64*eps*np.maximum(2*(old[:,7:].sum(1)+inputs_n),1.)
        assert np.all(abs(cr)<cb) and np.all(abs(nr)<nb)
        np.testing.assert_allclose(now,states['numpy64'],rtol=.1,atol=1e-4)
        maximum[backend]=dict(carbon_gc_m2=float(abs(cr).max()),nitrogen_gn_m2=float(abs(nr).max()),
            carbon_bound_max_gc_m2=float(cb.max()),nitrogen_bound_max_gn_m2=float(nb.max()),
            state_max_abs=float(np.max(abs(now-states['numpy64']))))
    print('Decomposition seven-day',maximum,'same precision',same,flush=True)


def test_decomposition_contract():
    s,u,p=cases();validate_decomposition(s,u,p,np)
    bad=s.copy();bad[0,0]=-1.
    with pytest.raises(ValueError):validate_decomposition(bad,u,p,np)
    bad=p.copy();bad[0,14]=bad[0,15]
    with pytest.raises(ValueError):validate_decomposition(s,u,bad,np)
    with pytest.raises(ValueError):validate_decomposition(s,u.astype(np.float32),p,np)
