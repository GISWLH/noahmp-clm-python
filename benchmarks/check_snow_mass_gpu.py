"""Audit snow mass/heat/tracer transfers and a repeated transfer trajectory."""
from pathlib import Path
import argparse,hashlib,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tests'))
import test_gpu_snow_mass as check
from noahmp_clm.physics.snowpack import Snowpack,CI,LF,TF
from noahmp_clm.gpu.snow_mass import pack_snow_transfer,run_snow_transfer_jax
from noahmp_clm.gpu.cuda.snow_mass import run_snow_transfer_cuda


def numpy_batch(c,n,m,u):
    rows=[check.reference(*v) for v in zip(c,n,m,u)]
    return {key:np.stack([v[key] for v in rows]) for key in rows[0]}


def transfer_series(cp,steps):
    packs=[Snowpack.from_ice([30.,70.],TF-10.,100.),
        Snowpack([20.,30.,50.],LF*np.array([1.,3.,8.]),np.array([20.,30.,50.])/300.,[54.5,100.,300.],np.ones((3,5))*1e-7),
        Snowpack([10.,20.,30.,40.,50.],LF*np.array([1.,2.,3.,4.,5.]),np.array([10.,20.,30.,40.,50.])/700.,impurities=np.ones((5,5))*1e-6)]
    rows=[pack_snow_transfer(p,0) for p in packs]
    c,n,_,_=tuple(np.stack([row[k] for row in rows]) for k in range(4))
    backends={'numpy64':numpy_batch,'jax64':run_snow_transfer_jax,'jax32':run_snow_transfer_jax,'cuda32':run_snow_transfer_cuda}
    states={name:(c.astype(np.float64 if name.endswith('64') else np.float32),n.copy()) for name in backends}
    reports={name:dict(max_step_budget=np.zeros(7),max_cumulative_budget=np.zeros(7),max_cell_absolute=np.zeros(9),max_same_precision=0.) for name in backends}
    net={name:np.zeros((3,7)) for name in backends};initial={name:st[0][:,:,[0,1,4,5,6,7,8]].astype(float).sum(1) for name,st in states.items()}
    for step in range(steps):
        mode=step%5;m=np.full(3,mode,np.int32);parameters=dict(amount=0.)
        if mode==1:parameters=dict(amount=.03,temperature=TF-4.,density=100.,impurities=np.arange(1,6)*1e-10)
        elif mode==2:parameters=dict(amount=.1,temperature=TF+10.)
        elif mode==4:parameters=dict(amount=.02)
        _,_,_,u0=pack_snow_transfer(packs[0],mode,**parameters)
        results={}
        for name,run in backends.items():
            old,nn=states[name];u=np.tile(u0,(3,1)).astype(old.dtype)
            result=run(old,nn,m,u);r={key:cp.asnumpy(v) if name=='cuda32' else np.asarray(v) for key,v in result.items()}
            reports[name]['max_step_budget']=np.maximum(reports[name]['max_step_budget'],check.budgets(old,r))
            net[name]+=r['incoming'].astype(float)-r['drainage'].astype(float)-r['runoff'].astype(float)-r['sublimation'].astype(float)
            total=r['cells'][:,:,[0,1,4,5,6,7,8]].astype(float).sum(1)
            residual=total-initial[name]-net[name]
            reports[name]['max_cumulative_budget']=np.maximum(reports[name]['max_cumulative_budget'],np.max(np.abs(residual),axis=0))
            states[name]=(r['cells'],r['counts']);results[name]=r
        for name,r in results.items():
            ref=results['numpy64'];np.testing.assert_array_equal(r['counts'],ref['counts'])
            error=np.max(np.abs(r['cells']-ref['cells']),axis=(0,1))
            reports[name]['max_cell_absolute']=np.maximum(reports[name]['max_cell_absolute'],error)
            # Independently apply the same science gate to this trajectory's
            # reference state; full timestep ledgers remain separately checked.
            floors=np.array([.001,1.,.0001,1.,*([1e-12]*5)])
            absolute=np.array([.0001,.1,.00001,.1,*([1e-13]*5)])
            b=ref['cells'];tol=np.where(np.abs(b)>floors,.1*np.abs(b),absolute)
            assert np.all(np.abs(r['cells']-b)<=tol),(step,name,error)
        for key in results['cuda32']:
            a,b=results['cuda32'][key],results['jax32'][key]
            np.testing.assert_allclose(a,b,rtol=1e-6,atol=1e-6)
            reports['cuda32']['max_same_precision']=max(reports['cuda32']['max_same_precision'],float(np.max(np.abs(a-b))))
    for name,entry in reports.items():
        for key,value in list(entry.items()):
            if isinstance(value,np.ndarray):entry[key]=value.tolist()
        entry['final_net_transfer']=net[name].tolist()
    return dict(transfers_per_column=steps,columns=3,backends=reports)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--random-count',type=int,default=2000);parser.add_argument('--steps',type=int,default=336)
    args=parser.parse_args()
    if args.output.exists():parser.error('Use a fresh report path')
    import jax,cupy as cp
    assert jax.config.x64_enabled and jax.default_backend()=='gpu'
    start=time.perf_counter();report=dict(scope='Snow transfer component; no energy fluxes, aging or complete winter column',devices=str(jax.devices()),cases=len(check.cases(np.float64,args.random_count)[0]))
    got={}
    for name,dtype,run in (('jax64',np.float64,run_snow_transfer_jax),('jax32',np.float32,run_snow_transfer_jax),('cuda32',np.float32,run_snow_transfer_cuda)):
        inputs=check.cases(dtype,args.random_count);result=run(*inputs)
        r={key:cp.asnumpy(v) if name=='cuda32' else np.asarray(v) for key,v in result.items()};got[name]=r
        report[name]=check.compare_reference(*inputs,r);report[name]['max_budget']=check.budgets(inputs[0],r).tolist()
        print(name,json.dumps(report[name]),flush=True)
    report['same_precision']={}
    for key in got['cuda32']:
        a,b=got['cuda32'][key],got['jax32'][key];np.testing.assert_allclose(a,b,rtol=1e-6,atol=1e-6)
        report['same_precision'][key]=float(np.max(np.abs(a-b)))
    report['trajectory']=transfer_series(cp,args.steps)
    report['source_sha256']=json.loads((ROOT/'SOURCE_SHA256.json').read_text()) if (ROOT/'SOURCE_SHA256.json').exists() else {
        p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src').rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.egg-info' not in str(p)}
    report['wall_seconds']=time.perf_counter()-start;args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False));print('Saved',args.output,report['wall_seconds'],flush=True)


if __name__=='__main__':main()
