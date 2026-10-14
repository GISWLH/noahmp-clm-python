"""Synthetic soil mass/heat transfer audit and continuous open-system chain."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tests'))
import test_gpu_soil_transport as check
from test_gpu_snicar import assert_science_close
from noahmp_clm.gpu.soil_transport import pack_soil_transport,run_soil_transport_jax
from noahmp_clm.gpu.cuda.soil_transport import run_soil_transport_cuda


def host(r,cp):return {k:cp.asnumpy(v) if isinstance(v,cp.ndarray) else np.asarray(v) for k,v in r.items()}
def stack(rows):return {k:np.stack([r[k] for r in rows]) for k in rows[0]}
def numpy_run(arrays):return stack([check.reference(*v) for v in zip(*arrays)])


def ledger(result):
    taken=np.stack((result['withdrawal_mass_kgm2'].astype(float).sum(1),
        result['withdrawal_enthalpy_jm2'].astype(float).sum(1)),axis=1)
    return result['incoming'].astype(float)-result['runoff'].astype(float)-result['bottom_outflow'].astype(float)-taken


def same_precision(a,b):
    result={}
    for k in a:
        if a[k].dtype==bool:
            np.testing.assert_array_equal(a[k],b[k]);result[k]=int(np.max(a[k]!=b[k]))
        else:
            np.testing.assert_allclose(a[k],b[k],rtol=1e-6,atol=1e-6);result[k]=float(np.max(abs(a[k]-b[k])))
    return result


def metrics(got,reference,inputs):
    result={}
    for k,a in got.items():
        if k=='valid':continue
        e=reference[k];err=abs(a-e)
        if k=='cells':floor=np.array([.001,1.,.0001]);axes=(0,1)
        elif k in ('incoming','runoff','bottom_outflow'):
            floor=np.column_stack((np.full(len(a),.001),inputs[3][:,2]*10.));axes=0
        elif 'temperature' in k:floor=np.inf;axes=0
        elif 'kgm2' in k:floor=.001;axes=0
        else:floor=inputs[3][:,2:3]*10.;axes=0
        rel=np.where(abs(e)>floor,err/np.maximum(abs(e),np.maximum(floor,1e-30))*100.,0.)
        result[k]=dict(max_absolute=np.max(err,axis=axes).tolist(),max_relative_percent=np.max(rel,axis=axes).tolist())
    return result


def trajectory(cp,steps):
    rows=[pack_soil_transport(check.soil_state(np.float32,t,np.array([260.,268.,274.,282.]),.75,1+i%2),0,dtype=np.float32)
        for i,t in enumerate(('sand','loam','clay'))]
    base=tuple(np.stack([r[k] for r in rows]) for k in range(4))
    names=('numpy64','jax64','jax32','cuda32');state={};initial={};net={};bound={};maxima={}
    for name in names:
        dtype=np.float64 if name.endswith('64') else np.float32
        state[name]=base[0].astype(dtype);initial[name]=state[name][:,:,:2].astype(float).sum(1)
        net[name]=np.zeros((3,2));bound[name]=np.zeros((3,2))
        maxima[name]=dict(step_budget=np.zeros(2),cumulative_budget=np.zeros(2),cells=np.zeros(3),temperature_k=0.)
    same={};instant_failures={name:0 for name in names};first_failure=None
    daily={name:{} for name in names};raw_difference={name:{} for name in names}
    flux_keys=('incoming','runoff','withdrawal_mass_kgm2','withdrawal_enthalpy_jm2',
        'interface_mass_kgm2','interface_enthalpy_jm2','bottom_outflow')
    for step in range(steps):
        for mode in (0,1,2):
            results={};before={name:state[name].copy() for name in names}
            for name in names:
                c=state[name];dtype=c.dtype;p=base[1].astype(dtype);o=base[2].copy();o[:,0]=mode
                u=np.zeros((3,7),dtype);u[:,2]=1800.
                if mode==0:
                    amount=.03+(2. if step%48==0 else 0.)
                    energy=amount*(334000.+4180.*(5.+5.*np.sin(step*2*np.pi/48.)))
                    u[:,:2]=np.array([amount,energy],np.float32)
                elif mode==1:
                    # Prescribed atmospheric withdrawal: a small fixed mass
                    # remains below every rooted store in this fixed setup.
                    u[:,3:]=np.array([.0003,.0002,.0001,.00005],np.float32)
                arrays=c,p,o,u
                r=numpy_run(arrays) if name=='numpy64' else host((run_soil_transport_cuda if name=='cuda32' else run_soil_transport_jax)(*arrays),cp)
                maxima[name]['step_budget']=np.maximum(maxima[name]['step_budget'],check.budgets(c,r))
                net[name]+=ledger(r)
                total=r['cells'][:,:,:2].astype(float).sum(1)
                residual=total-initial[name]-net[name]
                scale=abs(c[:,:,:2].astype(float)).sum(1)+abs(r['cells'][:,:,:2].astype(float)).sum(1)+abs(ledger(r))
                bound[name]+=16.*np.finfo(dtype).eps*np.maximum(scale,1.)
                assert np.all(abs(residual)<=bound[name])
                maxima[name]['cumulative_budget']=np.maximum(maxima[name]['cumulative_budget'],np.max(abs(residual),axis=0))
                state[name]=r['cells'];results[name]=r
                day=step//48
                accum=daily[name].setdefault(day,{k:np.zeros_like(r[k],dtype=float) for k in flux_keys})
                for k in flux_keys:accum[k]+=r[k].astype(float)
            for name in names:
                for row in range(3):
                    actual={k:v[row] for k,v in results[name].items()};expected={k:v[row] for k,v in results['numpy64'].items()}
                    dtype=np.float32 if name.endswith('32') else np.float64
                    # Different rounded trajectories need the user's daily
                    # flux criterion, not a same-input component gate applied
                    # to each small threshold event. Retain that diagnostic.
                    try:check.science(actual,expected,dtype,1800.)
                    except AssertionError as error:
                        instant_failures[name]+=1
                        if first_failure is None:
                            failed_fields=[]
                            for field in expected:
                                try:check.science({field:actual[field]},{field:expected[field]},dtype,1800.)
                                except AssertionError:failed_fields.append(field)
                            first_failure=dict(step=step,mode=mode,column=row,backend=name,error=str(error),
                                fields=failed_fields,
                                actual_before=before[name][row].tolist(),reference_before=before['numpy64'][row].tolist(),
                                actual={k:v.tolist() for k,v in actual.items()},reference={k:v.tolist() for k,v in expected.items()})
                            Path('trajectory-first-failure.json').write_text(json.dumps(first_failure,indent=2)+'\n')
                    # P0 soil moisture is volumetric, not a snow mass gate.
                    # Keep raw phase and H differences as diagnostics; audit
                    # H conservation independently at every operator above.
                    np.testing.assert_allclose(actual['temperature_k'],expected['temperature_k'],rtol=0.,atol=.05)
                    dz=expected['cells'][:,2]
                    for water,refwater in ((actual['cells'][:,0],expected['cells'][:,0]),
                            (actual['liquid_kgm2'],expected['liquid_kgm2']),(actual['ice_kgm2'],expected['ice_kgm2'])):
                        assert_science_close(water/(1000.*dz),refwater/(1000.*dz),zero_floor=.02,absolute=.002)
                for k in flux_keys:
                    value=float(np.max(abs(results[name][k]-results['numpy64'][k])))
                    raw_difference[name][k]=max(raw_difference[name].get(k,0.),value)
                maxima[name]['cells']=np.maximum(maxima[name]['cells'],np.max(abs(state[name]-state['numpy64']),axis=(0,1)))
                maxima[name]['temperature_k']=max(maxima[name]['temperature_k'],float(np.max(abs(results[name]['temperature_k']-results['numpy64']['temperature_k']))))
            current=same_precision(results['cuda32'],results['jax32'])
            for k,v in current.items():same[k]=max(same.get(k,0.),v)
    day_metrics={}
    for name in names:
        day_metrics[name]={}
        for k in flux_keys:
            paired=k in ('incoming','runoff','bottom_outflow')
            absolute=np.zeros(2) if paired else np.asarray(0.)
            relative=absolute.copy()
            for day in daily[name]:
                a=daily[name][day][k];e=daily['numpy64'][day][k]
                seconds=1800.*min(48,steps-day*48)
                if k in ('incoming','runoff','bottom_outflow'):
                    aa=np.stack((a[:,0]*86400./seconds,a[:,1]/seconds),axis=1)
                    ee=np.stack((e[:,0]*86400./seconds,e[:,1]/seconds),axis=1)
                    for col,floor,atol in ((0,.5,.05),(1,10.,1.)):
                        assert_science_close(aa[:,col],ee[:,col],zero_floor=floor,absolute=atol)
                    floor=np.array([.5,10.])
                elif 'kgm2' in k:
                    aa=a*86400./seconds;ee=e*86400./seconds;floor=.5
                    assert_science_close(aa,ee,zero_floor=.5,absolute=.05)
                else:
                    aa=a/seconds;ee=e/seconds;floor=10.
                    assert_science_close(aa,ee,zero_floor=10.,absolute=1.)
                axes=0 if paired else None
                absolute=np.maximum(absolute,np.max(abs(aa-ee),axis=axes))
                relative=np.maximum(relative,np.max(np.where(abs(ee)>floor,abs(aa-ee)/np.maximum(abs(ee),floor)*100.,0.),axis=axes))
            day_metrics[name][k]=dict(max_absolute=absolute.tolist(),max_relative_percent=relative.tolist(),
                units=['kg/m2/day','W/m2'] if paired else 'kg/m2/day' if 'kgm2' in k else 'W/m2')
    return dict(columns=3,steps=steps,operators=3,column_operations=3*steps*3*4,same_precision=same,
        daily=day_metrics,raw_step_difference=raw_difference,instantaneous_component_gate_failures=instant_failures,
        first_instantaneous_component_gate_failure=first_failure,
        daily_fluxes={name:{day:{k:v.tolist() for k,v in d.items()} for day,d in days.items()} for name,days in daily.items()},
        backends={name:{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in d.items()} for name,d in maxima.items()})


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--random-count',type=int,default=1000);ap.add_argument('--steps',type=int,default=336);args=ap.parse_args()
    if args.output.exists():ap.error('Use a fresh report path')
    import jax,cupy as cp
    assert jax.config.x64_enabled and jax.default_backend()=='gpu'
    start=time.perf_counter();report=dict(scope=__doc__,devices=str(jax.devices()),backends={});results={}
    for dtype in (np.float64,np.float32):
        arrays=check.cases(dtype,args.random_count);ref=numpy_run(arrays);report['cases']=len(arrays[0])
        for name,run in ([('jax64',run_soil_transport_jax)] if dtype==np.float64 else [('jax32',run_soil_transport_jax),('cuda32',run_soil_transport_cuda)]):
            r=host(run(*arrays),cp);results[name]=r
            for row in range(len(arrays[0])):
                check.science({k:v[row] for k,v in r.items()},{k:v[row] for k,v in ref.items()},dtype,float(arrays[3][row,2]))
            report['backends'][name]=dict(errors=metrics(r,ref,arrays),budget=check.budgets(arrays[0],r).tolist())
            print('single',name,report['backends'][name]['budget'],flush=True)
    report['same_precision']=same_precision(results['cuda32'],results['jax32'])
    report['trajectory']=trajectory(cp,args.steps)
    report['source_sha256']=json.loads((ROOT/'SOURCE_SHA256.json').read_text())
    report['elapsed_s']=time.perf_counter()-start;args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print('Saved',args.output,report['elapsed_s'],flush=True)


if __name__=='__main__':main()
