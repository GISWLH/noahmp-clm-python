"""Snow vapor and microstructure audits; no full-column or site claim."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tests'))
import test_gpu_snow_processes as check
from noahmp_clm.physics.snowpack import Snowpack,TF,CI,LF
from noahmp_clm.gpu.snow_vapor import pack_snow_vapor,run_snow_vapor_jax
from noahmp_clm.gpu.cuda.snow_vapor import run_snow_vapor_cuda
from noahmp_clm.gpu.snow_evolution import pack_snow_evolution,run_snow_evolution_jax
from noahmp_clm.gpu.cuda.snow_evolution import run_snow_evolution_cuda

FLOORS=np.array([.001,1.,.0001,1.,*([1e-12]*5)])


def numpy_run(kind,c,o,u):
    rows=[check.vapor_reference(*v,c.dtype) if kind=='vapor' else check.evolution_reference(*v) for v in zip(c,o,u)]
    return {k:np.stack([r[k] for r in rows]) for k in rows[0]}


def as_host(result,name,cp):return {k:cp.asnumpy(v) if name=='cuda32' else np.asarray(v) for k,v in result.items()}


def trajectory(cp,steps):
    packs=[Snowpack([100.],[-100*CI*5],[.4],[54.5]),
        Snowpack([50.,100.],LF*np.array([1.,2.]),[.2,.4],[100.,300.]),
        Snowpack([20.,30.,40.,50.,60.],-CI*3*np.array([20.,30.,40.,50.,60.]),np.array([20.,30.,40.,50.,60.])/250.)]
    c,n,_=check.stack([pack_snow_vapor(p,0.) for p in packs]);names=('numpy64','jax64','jax32','cuda32')
    state={name:(c.astype(np.float64 if name.endswith('64') else np.float32),n.copy()) for name in names}
    initial={name:v[0][:,:,[0,1,4,5,6,7,8]].astype(float).sum(1) for name,v in state.items()}
    net={name:np.zeros((3,7)) for name in names};maxima={name:dict(cumulative=np.zeros(7),step=np.zeros(7),cells=np.zeros(9)) for name in names}
    same={};operations=0
    for step in range(steps):
        for kind in ('vapor','evolution','deposition'):
            results={};operation_kind='vapor' if kind=='vapor' else 'evolution'
            for name in names:
                cells,nn=state[name];dtype=cells.dtype
                if kind=='vapor':
                    u=np.tile(np.array([.0002 if step%2 else -.0003,-5. if step%4 else 2.,0.],dtype),(3,1));o=nn
                    run=run_snow_vapor_cuda if name=='cuda32' else run_snow_vapor_jax
                else:
                    u=np.zeros((3,23),dtype);u[:,0]=1800.;u[:,21]=-5.
                    o=np.column_stack((nn,np.full(3,3 if kind=='evolution' else 2),np.zeros((3,2),int),np.ones(3,int))).astype(np.int32)
                    if kind=='evolution':u[:,1:6]=np.where(np.arange(5)<nn[:,None],20.,0.)
                    else:u[:,16:21]=np.arange(1,6)*1e-12
                    run=run_snow_evolution_cuda if name=='cuda32' else run_snow_evolution_jax
                result=numpy_run(operation_kind,cells,o,u) if name=='numpy64' else run(cells,o,u)
                r=as_host(result,name,cp)
                addition=None if kind=='vapor' else np.where((o[:,1]==2)[:,None],u[:,16:21].astype(float)*u[:,:1].astype(float),0.)
                maxima[name]['step']=np.maximum(maxima[name]['step'],check.process_budgets(cells,r,addition))
                if kind=='vapor':net[name]-=r['outward'].astype(float)
                else:net[name][:,2:]+=addition-r['bare_deposition'].astype(float)
                total=r['cells'][:,:,[0,1,4,5,6,7,8]].astype(float).sum(1)
                maxima[name]['cumulative']=np.maximum(maxima[name]['cumulative'],np.max(np.abs(total-initial[name]-net[name]),axis=0))
                state[name]=(r['cells'],r['counts']);results[name]=r
            for name,r in results.items():
                ref=results['numpy64'];np.testing.assert_array_equal(r['counts'],ref['counts']);check.cell_science(r['cells'],ref['cells'],np.float32 if name.endswith('32') else np.float64)
                maxima[name]['cells']=np.maximum(maxima[name]['cells'],np.max(np.abs(r['cells']-ref['cells']),axis=(0,1)))
            for k in results['jax32']:
                a,b=results['jax32'][k],results['cuda32'][k];np.testing.assert_allclose(a,b,rtol=1e-6,atol=1e-6)
                value=float(np.max(a!=b)) if a.dtype==bool else float(np.max(np.abs(a-b)))
                key=kind+'.'+k;same[key]=max(same.get(key,0.),value)
            operations+=3*len(names)
    return dict(steps=steps,columns=3,operators_per_step=3,column_operations=operations,same_precision=same,
        backends={name:{k:v.tolist() for k,v in d.items()} for name,d in maxima.items()})


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--random-count',type=int,default=2000);ap.add_argument('--steps',type=int,default=336);args=ap.parse_args()
    if args.output.exists():ap.error('Use a fresh output path')
    import jax,cupy as cp
    assert jax.config.x64_enabled and jax.default_backend()=='gpu'
    start=time.perf_counter();report=dict(scope=__doc__,devices=str(jax.devices()))
    for kind in ('vapor','evolution'):
        results={};entry={}
        for name,dtype in (('jax64',np.float64),('jax32',np.float32),('cuda32',np.float32)):
            inputs=check.vapor_cases(dtype,args.random_count) if kind=='vapor' else check.evolution_cases(dtype,args.random_count,table_grid=True)
            c,o,u=inputs;entry['cases']=len(c)
            run=(run_snow_vapor_cuda if name=='cuda32' else run_snow_vapor_jax) if kind=='vapor' else (run_snow_evolution_cuda if name=='cuda32' else run_snow_evolution_jax)
            result=as_host(run(*inputs),name,cp);results[name]=result
            metrics={};maxabs=np.zeros(9);maxrel=np.zeros(9)
            for row,v in enumerate(zip(*inputs)):
                ref=check.vapor_reference(*v,dtype) if kind=='vapor' else check.evolution_reference(*v)
                assert result['counts'][row]==ref['counts'],(kind,name,row)
                check.cell_science(result['cells'][row],ref['cells'],dtype)
                error=np.abs(result['cells'][row]-ref['cells']);maxabs=np.maximum(maxabs,error.max(0))
                maxrel=np.maximum(maxrel,np.max(np.where(np.abs(ref['cells'])>FLOORS,error/np.maximum(np.abs(ref['cells']),FLOORS)*100.,0.),axis=0))
                for key in (('outward','ice_kgm2','liquid_kgm2','latent_energy_jm2') if kind=='vapor' else ('fresh_density','fresh_radius','aging_parameters')):
                    a,b=result[key][row],ref[key];err=np.abs(a-b)
                    metric=metrics.setdefault(key,dict(max_absolute=0.));metric['max_absolute']=max(metric['max_absolute'],float(np.max(err)))
                    if key=='aging_parameters':continue # table fit coefficients are diagnostics; radius is the physical gate
                    floor,absolute=(np.array([.001,1.,*([1e-12]*5)]),np.array([.0001,.1,*([1e-13]*5)])) if key=='outward' else ((.001,.0001) if 'kgm2' in key else (1.,.1))
                    check.scientific_close(a,b,floor,absolute,dtype)
            addition=None if kind=='vapor' else np.where((o[:,1]==2)[:,None],u[:,16:21].astype(float)*u[:,:1].astype(float),0.)
            entry[name]=dict(cell_max_absolute=maxabs.tolist(),cell_max_relative_percent=maxrel.tolist(),diagnostics=metrics,budget=check.process_budgets(c,result,addition).tolist())
            print(kind,name,json.dumps(entry[name]),flush=True)
        entry['same_precision']={}
        for key in results['jax32']:
            a,b=results['jax32'][key],results['cuda32'][key];np.testing.assert_allclose(a,b,rtol=1e-6,atol=1e-6)
            entry['same_precision'][key]=float(np.max(a!=b)) if a.dtype==bool else float(np.max(np.abs(a-b)))
        report[kind]=entry
    report['trajectory']=trajectory(cp,args.steps)
    report['source_sha256']=json.loads((ROOT/'SOURCE_SHA256.json').read_text())
    report['wall_seconds']=time.perf_counter()-start;args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False));print('Saved',args.output,report['wall_seconds'],flush=True)


if __name__=='__main__':main()
