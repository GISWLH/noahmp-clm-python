"""Device plant phases, independent budgets and coupled soil C/N trajectories."""
from dataclasses import replace
import numpy as np
import pytest
from noahmp_clm.physics.plant_cn import (PlantState as State,PlantParameters as Parameters,
    turnover_plant,plan_plant_allocation,apply_plant_allocation)
from noahmp_clm.gpu.plant_cn import pack_plant,validate_turnover,validate_plan,validate_allocation
from test_gpu_snicar import assert_science_close


def cases(dtype=np.float64,count=512):
    rng=np.random.default_rng(4201);rows=[]
    for k in range(count):
        c=rng.uniform(.01,3000.,3);n=c/rng.uniform(15.,500.,3)
        if k%8==0:c[0]=n[0]=0.
        p=Parameters(maintenance_q10=rng.uniform(1.,2.5),new_carbon_storage_fraction=rng.uniform(0.,.8),
            growth_respiration_ratio=rng.uniform(0.,.4))
        old=State(c,n,0. if k%7==0 else rng.uniform(0.,20.),0. if k%3==0 else rng.uniform(0.,.1))
        s,p=pack_plant(old,p,dtype=dtype);dt=(1.,1800.,3600.,86400.)[k%4]
        turn=np.array([dt,(0.,.5,1.,1.)[k%4]],dtype)
        climate=np.array([rng.uniform(0.,10.),rng.uniform(0.,.2),rng.uniform(250.,315.),rng.uniform(250.,305.),
            dt,rng.uniform(0.,3000.),int(k%7!=0)],dtype)
        rows.append((s,turn,climate,p))
    return tuple(np.stack(v) for v in zip(*rows))


def parameters(row):
    v=row.astype(float)
    # Float32 serialization rounds fractions separately. Reconstruct the
    # physical sum-one fractions for NumPy; native budgets retain/report the
    # rounding residual. Do not renormalize native tendencies on the host.
    return Parameters(tuple(v[:3]/v[:3].sum()),tuple(v[3:6]),tuple(v[6:9]),tuple(v[9:12]/v[9:12].sum()),*v[12:])


def reference(s,t,u,p,award_fraction):
    p=parameters(p);s=s.astype(float);t=t.astype(float);u=u.astype(float)
    old=State(s[:3],s[3:6],s[6],s[7]);turned,td=turnover_plant(old,t[0],leaf_retention=t[1],parameters=p)
    plan=plan_plant_allocation(turned,*u[:5],target_leaf_carbon=u[5],growing=bool(u[6]),parameters=p)
    new,ad=apply_plant_allocation(plan,plan.soil_n_demand*award_fraction)
    vec=lambda state:np.r_[state.carbon,state.nitrogen,state.reserve_carbon,state.reserve_nitrogen]
    packed_plan=np.r_[vec(turned),plan.gpp_gc_m2,plan.dt,plan.available_carbon,plan.potential_growth,plan.tissue_n_deficit,
        plan.maintenance_respiration,plan.unmet_maintenance,plan.soil_n_demand]
    return dict(state=vec(turned),**td),packed_plan,dict(state=vec(new),**ad)


def independent_budget(old,got,ci,co,ni,no):
    old=old.astype(float);now=np.asarray(got['state'],float);eps=np.finfo(got['state'].dtype).eps
    oc=old[:,:3].sum(1)+old[:,6];nc=now[:,:3].sum(1)+now[:,6]
    on=old[:,3:6].sum(1)+old[:,7];nn=now[:,3:6].sum(1)+now[:,7]
    cr=nc-oc-ci+co;nr=nn-on-ni+no
    cb=64.*eps*np.maximum(oc+nc+ci+co,1.);nb=64.*eps*np.maximum(on+nn+ni+no,1.)
    assert np.all(abs(cr)<=cb),(cr,cb)
    assert np.all(abs(nr)<=nb),(nr,nb)
    return dict(carbon=float(abs(cr).max()),nitrogen=float(abs(nr).max()),carbon_bound=float(cb.max()),nitrogen_bound=float(nb.max()))


def check_phases(s,t,u,turn,allocation):
    td=independent_budget(s,turn,0.,turn['litter_carbon_gc_m2'].astype(float).sum(1),0.,turn['litter_nitrogen_gn_m2'].astype(float).sum(1))
    ad=independent_budget(turn['state'],allocation,u[:,0].astype(float),allocation['autotrophic_respiration_gc_m2'].astype(float),
        allocation['soil_n_uptake_gn_m2'].astype(float),0.)
    return dict(turnover=td,allocation=ad)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_plant_native_science_reference_and_cn_budgets(dtype):
    jax=pytest.importorskip('jax')
    if dtype is np.float64 and not jax.config.x64_enabled:pytest.skip('JAX x64 required')
    from noahmp_clm.gpu.plant_cn import run_plant_turnover_jax,plan_plant_allocation_jax,apply_plant_allocation_jax
    s,t,u,p=cases(dtype);turn={k:np.asarray(v) for k,v in run_plant_turnover_jax(s,t,p).items()}
    plan=np.asarray(plan_plant_allocation_jax(turn['state'],u,p));fraction=np.array([0.,.25,.75,1.],dtype)[np.arange(len(s))%4]
    result={k:np.asarray(v) for k,v in apply_plant_allocation_jax(plan,(plan[:,23]*fraction)[:,None],p).items()}
    maximum={}
    for index,row in enumerate(zip(s,t,u,p)):
        rt,rp,ra=reference(*row,float(fraction[index]))
        for stage,a,b in (('turnover',turn,rt),('allocation',result,ra)):
            for key,expected in b.items():
                if '_bound_' in key or '_residual_' in key:continue
                value=a[key][index];name=stage+'/'+key;maximum[name]=max(maximum.get(name,0.),float(np.max(abs(value-expected))))
                if key=='state':
                    assert_science_close(value[[0,1,2,6]],expected[[0,1,2,6]],relative=.1,zero_floor=.001,absolute=.0001)
                    # A depleted reserve is the difference of tissue-N
                    # demands up to hundreds of gN. Use a unitful 10 ugN/m2
                    # near-zero bound; keep C/N conservation and same-precision
                    # comparisons separate and unchanged.
                    assert_science_close(value[[3,4,5,7]],expected[[3,4,5,7]],relative=.1,zero_floor=1e-4,absolute=1e-5)
                elif '_gn_m2' in key:assert_science_close(value,expected,relative=.1,zero_floor=1e-6,absolute=1e-7)
                else:assert_science_close(value,expected,relative=.1,zero_floor=1e-4,absolute=1e-5)
        maximum['plan']=max(maximum.get('plan',0.),float(np.max(abs(plan[index]-rp))))
    print('Plant native reference',str(dtype),maximum,flush=True)
    print('Plant native C/N',str(dtype),check_phases(s,t,u,turn,result),flush=True)


def test_plant_all_stages_same_precision():
    cp=pytest.importorskip('cupy');pytest.importorskip('jax')
    from noahmp_clm.gpu.plant_cn import run_plant_turnover_jax,plan_plant_allocation_jax,apply_plant_allocation_jax
    from noahmp_clm.gpu.cuda.plant_cn import run_plant_turnover_cuda,plan_plant_allocation_cuda,apply_plant_allocation_cuda
    s,t,u,p=cases(np.float32);runs=[]
    for conv,turn,plan,apply in ((np.asarray,run_plant_turnover_jax,plan_plant_allocation_jax,apply_plant_allocation_jax),
        (cp.asnumpy,run_plant_turnover_cuda,plan_plant_allocation_cuda,apply_plant_allocation_cuda)):
        td=turn(s,t,p);pd=plan(td['state'],u,p);award=pd[:,23:24]*.75;ad=apply(pd,award,p)
        runs.append(({k:conv(v) for k,v in td.items()},conv(pd),{k:conv(v) for k,v in ad.items()}))
    maximum={}
    for stage,a,b in zip(('turnover','plan','allocation'),*runs):
        if stage=='plan':a={'plan':a};b={'plan':b}
        for key in a:
            maximum[stage+'/'+key]=float(np.max(abs(a[key]-b[key])))
            np.testing.assert_allclose(a[key],b[key],rtol=1e-6,atol=1e-6,err_msg=stage+'/'+key)
    print('Plant same precision',maximum,flush=True)
    print('Plant CUDA C/N',check_phases(s,t,u,runs[1][0],runs[1][2]),flush=True)


def test_native_plant_soil_seven_day_closed_carbon_nitrogen():
    cp=pytest.importorskip('cupy');jax=pytest.importorskip('jax')
    if not jax.config.x64_enabled:pytest.skip('JAX x64 required')
    from noahmp_clm.gpu.plant_cn import run_plant_turnover_jax,plan_plant_allocation_jax,apply_plant_allocation_jax
    from noahmp_clm.gpu.cuda.plant_cn import run_plant_turnover_cuda,plan_plant_allocation_cuda,apply_plant_allocation_cuda
    from noahmp_clm.gpu.decomposition import run_decomposition_jax
    from noahmp_clm.gpu.cuda.decomposition import run_decomposition_cuda
    from test_gpu_decomposition import cases as soil_cases,reference as soil_reference,independent_budgets
    from noahmp_clm.physics.decomposition import DecompositionState,step_decomposition
    s,t,u,p=cases(np.float32,8);soil_s,soil_u,soil_p=soil_cases(np.float32,8)
    # Start from exactly representable fractions for the physical float64
    # reference; native and reference inputs then describe the same pools.
    p[:,:3]=(.25,.5,.25);p[:,9:12]=(.25,.5,.25)
    original=np.column_stack((s[:,:3].astype(float).sum(1)+s[:,6]+soil_s[:,:7].astype(float).sum(1),
        s[:,3:6].astype(float).sum(1)+s[:,7]+soil_s[:,7:].astype(float).sum(1)))
    states={b:(s.astype(dtype),soil_s.astype(dtype)) for b,dtype in (('numpy64',float),('jax64',float),('jax32',np.float32),('cuda32',np.float32))}
    ledger={b:np.zeros((8,3)) for b in states};bounds={b:np.zeros((8,2)) for b in states};max_budget={b:np.zeros(2) for b in states};same=0.
    for step in range(336):
        t[:,0]=u[:,4]=soil_u[:,2]=1800.;t[:,1]=.98 if step>=288 else 1.;u[:,6]=1.
        u[:,0]=max(0.,.2*np.sin(2*np.pi*(step%48)/48.));u[:,1]=.001
        u[:,2]=soil_u[:,0]=285.+8.*np.sin(2*np.pi*(step%48)/48.);u[:,3]=285.;soil_u[:,1]=-.03;soil_u[:,3]=1.
        results={}
        for backend,(old,old_soil) in states.items():
            dtype=old.dtype;pi=p.astype(dtype);ui=u.astype(dtype);ti=t.astype(dtype);si=soil_u.astype(dtype);sp=soil_p.astype(dtype)
            if backend=='numpy64':
                td_rows=[];ad_rows=[];sd_rows=[]
                for k in range(8):
                    pp=parameters(pi[k]);os=State(old[k,:3],old[k,3:6],old[k,6],old[k,7])
                    turned,td=turnover_plant(os,ti[k,0],leaf_retention=ti[k,1],parameters=pp)
                    plan=plan_plant_allocation(turned,*ui[k,:5],target_leaf_carbon=ui[k,5],growing=bool(ui[k,6]),parameters=pp)
                    si[k,4:11]=td['litter_carbon_gc_m2'];si[k,11:18]=td['litter_nitrogen_gn_m2'];si[k,19]=plan.soil_n_demand
                    sd=soil_reference(old_soil[k],si[k],sp[k]);new,ad=apply_plant_allocation(plan,sd['plant_uptake_gn_m2'])
                    td_rows.append(dict(state=np.r_[turned.carbon,turned.nitrogen,turned.reserve_carbon,turned.reserve_nitrogen],**td))
                    ad_rows.append(dict(state=np.r_[new.carbon,new.nitrogen,new.reserve_carbon,new.reserve_nitrogen],**ad));sd_rows.append(sd)
                collect=lambda rows:{k:np.stack([r[k] for r in rows]) for k in rows[0]}
                td,ad,sd=map(collect,(td_rows,ad_rows,sd_rows))
            else:
                if backend=='cuda32':turn,plan_fn,apply,decomp,conv=run_plant_turnover_cuda,plan_plant_allocation_cuda,apply_plant_allocation_cuda,run_decomposition_cuda,cp.asnumpy
                else:turn,plan_fn,apply,decomp,conv=run_plant_turnover_jax,plan_plant_allocation_jax,apply_plant_allocation_jax,run_decomposition_jax,np.asarray
                td=turn(old,ti,pi);plan=plan_fn(td['state'],ui,pi)
                # Only serialization/ledger inspection occurs on the host;
                # all tendencies and competition are native device kernels.
                si[:,4:11]=conv(td['litter_carbon_gc_m2']);si[:,11:18]=conv(td['litter_nitrogen_gn_m2']);si[:,19]=conv(plan[:,23])
                sd=decomp(old_soil,si,sp);ad=apply(plan,sd['plant_uptake_gn_m2'][:,None],pi)
                td={k:conv(v) for k,v in td.items()};sd={k:conv(v) for k,v in sd.items()};ad={k:conv(v) for k,v in ad.items()}
            check_phases(old,ti,ui,td,ad);independent_budgets(old_soil,si,sd)
            ledger[backend]+=np.column_stack((ui[:,0].astype(float),ad['autotrophic_respiration_gc_m2'].astype(float)+sd['heterotrophic_respiration_gc_m2'].astype(float),si[:,18].astype(float)))
            for d in (td,ad,sd):bounds[backend]+=np.column_stack((d['carbon_bound_gc_m2'],d['nitrogen_bound_gn_m2']))
            states[backend]=(ad['state'],sd['state']);results[backend]=(td,ad,sd)
            new=ad['state'].astype(float);ns=sd['state'].astype(float);gpp,resp,deposit=ledger[backend].T
            cr=new[:,:3].sum(1)+new[:,6]+ns[:,:7].sum(1)-original[:,0]-gpp+resp
            nr=new[:,3:6].sum(1)+new[:,7]+ns[:,7:].sum(1)-original[:,1]-deposit
            residual=np.column_stack((abs(cr),abs(nr)));assert np.all(residual<=bounds[backend])
            max_budget[backend]=np.maximum(max_budget[backend],residual.max(0))
        for left,right in zip(results['jax32'],results['cuda32']):
            for key in left:
                np.testing.assert_allclose(left[key],right[key],rtol=1e-6,atol=1e-6,err_msg=f'{step}/{key}')
                same=max(same,float(np.max(abs(left[key]-right[key]))))
    maximum={}
    for backend,(plant,soil) in states.items():
        for got,expected in zip((plant,soil),states['numpy64']):np.testing.assert_allclose(got,expected,rtol=.1,atol=1e-4)
        maximum[backend]=dict(carbon=float(max_budget[backend][0]),nitrogen=float(max_budget[backend][1]),
            carbon_bound=float(bounds[backend][:,0].max()),nitrogen_bound=float(bounds[backend][:,1].max()),
            plant_max_abs=float(np.max(abs(plant-states['numpy64'][0]))),soil_max_abs=float(np.max(abs(soil-states['numpy64'][1]))))
    print('Plant soil native seven-day',maximum,'same precision',same,flush=True)


def test_plant_native_contract():
    s,t,u,p=cases();validate_turnover(s,t,p,np);validate_plan(s,u,p,np)
    bad=t.copy();bad[0,1]=1.01
    with pytest.raises(ValueError):validate_turnover(s,bad,p,np)
    bad=u.copy();bad[0,6]=.5
    with pytest.raises(ValueError):validate_plan(s,bad,p,np)
    with pytest.raises(ValueError):validate_plan(s,u.astype(np.float32),p,np)
