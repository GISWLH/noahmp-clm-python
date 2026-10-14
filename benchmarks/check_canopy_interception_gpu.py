"""Native condensate audit: same-input cases and seven-day adiabatic transport."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tests'))
import test_gpu_canopy_interception as check
from test_gpu_clm_enthalpy_column import compare_same_precision
from test_gpu_snicar import assert_science_close
from noahmp_clm.gpu.canopy_interception import run_canopy_water_jax
from noahmp_clm.gpu.cuda.canopy_interception import run_canopy_water_cuda


def maxima(old,new):
    for key,value in new.items():
        if isinstance(value,dict):maxima(old.setdefault(key,{}),value)
        else:old[key]=np.maximum(old.get(key,np.zeros_like(value)),value).tolist()


def run(arrays,name,cp):
    if name=='numpy64':
        refs=[check.reference(row) for row in zip(*arrays)]
        return {key:({k:np.stack([r[key][k] for r in refs]) for k in refs[0][key]} if isinstance(refs[0][key],dict)
            else np.stack([r[key] for r in refs])) for key in refs[0]}
    fn=run_canopy_water_cuda if name=='cuda32' else run_canopy_water_jax
    return check.host(fn(*arrays),cp.asnumpy if name=='cuda32' else np.asarray)


def trajectory(steps,cp):
    base=check.cases(np.float32,0);select=[10,9,8];base=tuple(a[select].copy() for a in base)
    names=('numpy64','jax64','jax32','cuda32');states={name:base[0].astype(np.float64 if name.endswith('64') else np.float32) for name in names}
    initial={name:state.astype(float).sum(1) for name,state in states.items()};net={name:np.zeros((3,2)) for name in names}
    bounds={name:np.zeros(2) for name in names};errors={name:dict(step=[0.,0.],cumulative=[0.,0.]) for name in names}
    daily={name:{} for name in names};precision=0.
    for step in range(steps):
        angle=2.*np.pi*(step%48+.5)/48.;air=np.array([285.,265.,277.])+7.*np.sin(angle)
        precip=np.array([.06,.08,.04]) if step%12<4 else np.zeros(3)
        u=np.stack((np.where(air>273.15,precip,0.),np.where(air<=273.15,precip,0.),air,np.full(3,1800.),
            np.full(3,.8),np.full(3,3.),np.full(3,270.15)),axis=1).astype(np.float32)
        v=np.empty((3,10,2),np.float32);v[:,:,0]=.003*max(0.,np.sin(angle))-.001*max(0.,-np.sin(angle));v[:,:,1]=air[:,None]+np.linspace(-2.,2.,10)
        operation_results={name:[] for name in names}
        for name in names:
            dtype=states[name].dtype;a=base[1].astype(dtype);forcing=u.astype(dtype);vap=v.astype(dtype)
            inputs=(states[name],a,forcing,np.zeros_like(vap),np.zeros(3,np.int32));r=run(inputs,name,cp)
            for mode in (0,1):
                if mode:
                    inputs=(states[name],np.zeros_like(a),np.zeros_like(forcing),vap,np.ones(3,np.int32));r=run(inputs,name,cp)
                b=check.budgets(inputs,r);maxima(errors[name],dict(step=b['max_residual']))
                bounds[name]+=np.asarray(b['max_numerical_bound']);states[name]=r['cells'];operation_results[name].append(r)
                if mode==0:
                    f=forcing.astype(float);pheat=f[:,0]*(334000.+4180.*(f[:,2]-273.15))+f[:,1]*2100.*(np.minimum(f[:,2],273.15)-273.15)
                    net[name][:,0]+=f[:,0]+f[:,1]-r['ledger']['ground_liquid_kgm2'].astype(float)-r['ledger']['ground_ice_kgm2'].astype(float)
                    net[name][:,1]+=pheat-r['ledger']['ground_liquid_enthalpy_jm2'].astype(float)-r['ledger']['ground_ice_enthalpy_jm2'].astype(float)
                else:
                    net[name][:,0]-=r['mass_kgm2'].astype(float).sum(1);net[name][:,1]-=r['advected_enthalpy_jm2'].astype(float).sum(1)
                cumulative=states[name].astype(float).sum(1)-initial[name]-net[name]
                assert np.all(abs(cumulative)<=bounds[name]),(name,step,mode,cumulative,bounds[name])
                maxima(errors[name],dict(cumulative=np.max(abs(cumulative),axis=0)))
            first,last=operation_results[name]
            observation=dict(et=last['mass_kgm2'].sum(1),le=last['latent_energy_jm2'].sum(1)/1800.,
                liquid_out=first['ledger']['ground_liquid_kgm2'],ice_out=first['ledger']['ground_ice_kgm2'],
                canopy_water=last['cells'][:,:,0].sum(1),temperature=last['temperature'])
            accum=daily[name].setdefault(step//48,{k:np.zeros_like(value,dtype=float) for k,value in observation.items()})
            for key,value in observation.items():accum[key]+=value.astype(float)/48. if key in ('temperature','le','canopy_water') else value.astype(float)
        for mode in (0,1):precision=max(precision,compare_same_precision(operation_results['cuda32'][mode],operation_results['jax32'][mode]))
        if step%48==47:print(json.dumps(dict(day=step//48+1,budgets=errors,same_precision=precision)),flush=True)
    comparisons={}
    for name in names[1:]:
        comparisons[name]={}
        for day,reference in daily['numpy64'].items():
            if (day+1)*48>steps:continue
            for key,expected in reference.items():
                actual=daily[name][day][key];error=abs(actual-expected)
                if key=='temperature':np.testing.assert_allclose(actual,expected,rtol=0.,atol=.05);floor=np.inf
                else:
                    floor,absolute=(10.,1.) if key=='le' else (.5,.05)
                    assert_science_close(actual,expected,relative=.1,zero_floor=floor,absolute=absolute)
                maxima(comparisons[name].setdefault(key,{}),dict(max_absolute=float(np.max(error)),
                    max_relative_percent=float(np.max(np.where(abs(expected)>floor,100.*error/np.maximum(abs(expected),1e-30),0.)))))
    return dict(steps=steps,columns=3,operations_per_step=2,backends=4,same_precision_max_absolute=precision,
        budgets=errors,cumulative_numerical_bounds={k:v.tolist() for k,v in bounds.items()},daily_comparison=comparisons)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--random-count',type=int,default=400);parser.add_argument('--steps',type=int,default=336);args=parser.parse_args()
    import cupy as cp
    start=time.monotonic();arrays=check.cases(np.float32,args.random_count);refs=[check.reference(row) for row in zip(*arrays)]
    report=dict(source_manifest=json.loads((ROOT/'SOURCE_SHA256.json').read_text()),cases=len(refs),single={});outputs={}
    for name in ('jax64','jax32','cuda32'):
        dtype=np.float64 if name=='jax64' else np.float32;x=tuple(a if a.dtype.kind in 'iu' else a.astype(dtype) for a in arrays)
        result=run(x,name,cp);maximum={}
        for k,ref in enumerate(refs):
            one={key:({a:b[k] for a,b in value.items()} if isinstance(value,dict) else value[k]) for key,value in result.items()}
            maxima(maximum,check.science(one,ref,dtype))
        report['single'][name]=dict(max_absolute=maximum,budgets=check.budgets(x,result));outputs[name]=result
    report['same_precision_max_absolute']=compare_same_precision(outputs['cuda32'],outputs['jax32'])
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report['single']),flush=True)
    if args.steps:report['trajectory']=trajectory(args.steps,cp)
    report['seconds']=time.monotonic()-start;args.output.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(dict(seconds=report['seconds'])),flush=True)


if __name__=='__main__':main()
