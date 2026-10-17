"""Native winter surface audit; synthetic forcing, not field validation."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tests'))
import test_gpu_canopy_surface as check
from test_gpu_snicar import assert_science_close
from noahmp_clm.gpu.canopy_surface import run_canopy_surface_jax
from noahmp_clm.gpu.cuda.canopy_surface import run_canopy_surface_cuda
from noahmp_clm.physics.soil_enthalpy import SoilEnthalpyState
from noahmp_clm.column import SoilParams


def host(x,cp):
    if isinstance(x,dict):return {k:host(v,cp) for k,v in x.items()}
    return cp.asnumpy(x) if isinstance(x,cp.ndarray) else np.asarray(x)


def same(a,b):
    result={}
    for k in a:
        if isinstance(a[k],dict):result[k]=same(a[k],b[k]);continue
        if a[k].dtype.kind in 'ib':np.testing.assert_array_equal(a[k],b[k]);result[k]=int(np.max(a[k]!=b[k]))
        else:np.testing.assert_allclose(a[k],b[k],rtol=1e-6,atol=1e-6);result[k]=float(np.max(abs(a[k]-b[k])))
    return result


def maxima(old,new):
    for k,v in new.items():
        if isinstance(v,dict):maxima(old.setdefault(k,{}),v)
        else:old[k]=np.maximum(old.get(k,np.zeros_like(v)),v).tolist()


def total(arrays):
    c,g,p,o,a,u,f,v=arrays
    return np.stack((c[:,:,0].astype(float).sum(1)+g[:,:,0].astype(float).sum(1)+v[:,1],
        c[:,:,1].astype(float).sum(1)+g[:,:,1].astype(float).sum(1)+v[:,2]),axis=1)


def advance(arrays,result):
    c,g,p,o,a,u,f,v=[x.copy() for x in arrays]
    c[:]=result['snow_cells'];g[:]=result['soil_cells'];o[:,0]=result['snow_counts']
    v[:,1]=result['canopy_water_after_kgm2'];v[:,2]=result['canopy_enthalpy_after_jm2']
    return c,g,p,o,a,u,f,v


def diagnostics(r,arrays):
    p=arrays[2];g=r['soil_cells'];o=arrays[3]
    t=np.stack([SoilEnthalpyState(g[k,:,0],g[k,:,1],g[k,:,2],SoilParams(smcmax=p[k,0],bexp=p[k,1],saturated_suction_m=p[k,6]),int(o[k,1])).temperature for k in range(len(p))])
    return dict(et=r['et_kgm2'],h=r['h'],le=r['le'],gpp=r['gpp']*12e-6*arrays[7][:,0],
        runoff=r['runoff_kgm2'],swe=r['snow_cells'][:,:,0].sum(1)/1000.,temperature=t,
        water=g[:,:,0]/(1000.*g[:,:,2]))


def trajectory(cp,steps):
    base=tuple(x[[0,3,7]].copy() for x in check.cases(np.float32))
    names=('numpy64','jax64','jax32','cuda32');state={name:tuple(x.copy() if x.dtype.kind in 'iu' else x.astype(np.float64 if name.endswith('64') else np.float32) for x in base) for name in names}
    initial={name:total(arrays) for name,arrays in state.items()};net={name:np.zeros((3,2)) for name in names}
    errors={name:dict(step_budget=[0.,0.],cumulative_budget=[0.,0.]) for name in names};daily={name:{} for name in names};precision={}
    accumulated_bound={name:np.zeros(2) for name in names}
    for step in range(steps):
        results={}
        for name in names:
            arrays=state[name];u=arrays[5];angle=(step%48+.5)*2.*np.pi/48.
            # Same quantized meteorological forcing for all four histories.
            u[:,0]=np.asarray(np.array([285.,263.,278.])+7.*np.sin(angle-np.pi/2.),np.float32)
            u[:,11]=np.float32(600.*max(0.,np.sin(angle-np.pi/2.)))
            u[:,12]=np.float32(280.);u[:,1]=np.array([.006,.001,.004],np.float32)
            if name=='numpy64':
                rows=[check.reference(row) for row in zip(*arrays)]
                keys=('snow_cells','soil_cells','snow_counts','canopy_water_after_kgm2','canopy_enthalpy_after_jm2',
                    'et_kgm2','runoff_kgm2','runoff_enthalpy_jm2','h','le','rnet','gpp','advected_enthalpy_jm2',
                    'outgoing_shortwave','outgoing_longwave','water_residual_kgm2','energy_residual_jm2')
                r={k:np.stack([row[k] for row in rows]) for k in keys}
                assert np.max(abs(r['water_residual_kgm2']))<2e-9
                assert np.max(abs(r['energy_residual_jm2']))<.02
                error=dict(water_kgm2=float(np.max(abs(r['water_residual_kgm2']))),heat_jm2=float(np.max(abs(r['energy_residual_jm2']))),
                    water_bound_max_kgm2=2e-9,heat_bound_max_jm2=.02)
            else:
                r=host((run_canopy_surface_cuda if name=='cuda32' else run_canopy_surface_jax)(*arrays),cp)
                error=check.budgets(arrays,r)
            after=advance(arrays,r)
            net[name][:,0]-=r['et_kgm2'].astype(float)+r['runoff_kgm2'].astype(float)
            atm=(u[:,11].astype(float)+u[:,12]-r['outgoing_shortwave']-r['outgoing_longwave']-r['h']-r['le'])*arrays[7][:,0]
            net[name][:,1]+=atm-r['advected_enthalpy_jm2']-r['runoff_enthalpy_jm2']
            cumulative=total(after)-initial[name]-net[name]
            accumulated_bound[name]+=np.array([error['water_bound_max_kgm2'],error['heat_bound_max_jm2']])
            assert np.all(abs(cumulative)<=accumulated_bound[name])
            maxima(errors[name],dict(step_budget=[error['water_kgm2'],error['heat_jm2']],cumulative_budget=np.max(abs(cumulative),axis=0)))
            state[name]=after;results[name]=r
            obs=diagnostics(r,arrays);day=step//48
            accum=daily[name].setdefault(day,{k:np.zeros_like(x,dtype=float) for k,x in obs.items()})
            for k,x in obs.items():accum[k]+=x.astype(float) if k in ('et','gpp','runoff') else x.astype(float)/48.
        maxima(precision,same(results['cuda32'],results['jax32']))
        if step%48==47:print(json.dumps(dict(day=step//48+1,budgets=errors)),flush=True)
    comparison={}
    for name in names[1:]:
        comparison[name]={}
        for day,reference in daily['numpy64'].items():
            if (day+1)*48>steps:continue
            for key,e in reference.items():
                actual=daily[name][day][key];error=abs(actual-e)
                if key=='temperature':np.testing.assert_allclose(actual,e,rtol=0.,atol=.05);floor=np.inf
                else:
                    floor,absolute={'et':(.5,.05),'runoff':(.5,.05),'h':(10.,1.),'le':(10.,1.),'gpp':(1.,.1),
                        'swe':(.001,.0001),'water':(.02,.002)}[key]
                    assert_science_close(actual,e,relative=.1,zero_floor=floor,absolute=absolute)
                stats=dict(max_absolute=float(np.max(error)),max_relative_percent=float(np.max(np.where(abs(e)>floor,error/np.maximum(abs(e),1e-30)*100.,0.))))
                maxima(comparison[name].setdefault(key,{}),stats)
    return dict(steps=steps,columns=3,backends=4,budgets=errors,cumulative_numerical_bounds={k:v.tolist() for k,v in accumulated_bound.items()},
        daily_comparison=comparison,same_precision=precision)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--random-count',type=int,default=40);parser.add_argument('--steps',type=int,default=336);args=parser.parse_args()
    import cupy as cp
    start=time.monotonic();arrays=check.cases(np.float32,args.random_count);refs=[check.reference(row) for row in zip(*arrays)]
    report=dict(source_manifest=json.loads((ROOT/'SOURCE_SHA256.json').read_text()),cases=len(refs),single={})
    outputs={}
    for name in ('jax64','jax32','cuda32'):
        print('single backend',name,flush=True)
        x=tuple(a if a.dtype.kind in 'iu' else a.astype(np.float64 if name=='jax64' else np.float32) for a in arrays)
        got=host((run_canopy_surface_cuda if name=='cuda32' else run_canopy_surface_jax)(*x,raise_on_failure=False),cp)
        bad=np.flatnonzero(~got['converged'])
        if len(bad):
            rows=[dict(index=int(i),inputs=[a[i].tolist() for a in x],outputs={k:v[i].tolist() for k,v in got.items() if not isinstance(v,dict)}) for i in bad]
            failure=args.output.with_name(args.output.stem+'-'+name+'-failures.json')
            failure.write_text(json.dumps(rows,indent=2),encoding='utf-8')
            raise RuntimeError(f'{name} failed rows {bad.tolist()}; saved {failure}')
        outputs[name]=got;maximum={}
        for i,ref in enumerate(refs):
            one={k:v[i] for k,v in got.items() if not isinstance(v,dict)}
            maxima(maximum,check.science(one,ref,np.float64 if name=='jax64' else np.float32))
        report['single'][name]=dict(max_absolute=maximum,budget=check.budgets(x,got),max_surface_evaluations=int(np.max(got['surface_evaluations'])))
    report['same_precision']=same(outputs['cuda32'],outputs['jax32'])
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report['single']),flush=True)
    if args.steps:report['trajectory']=trajectory(cp,args.steps)
    report['seconds']=time.monotonic()-start;args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(output=str(args.output),seconds=report['seconds'])),flush=True)


if __name__=='__main__':main()
