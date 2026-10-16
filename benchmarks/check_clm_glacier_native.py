"""Four-backend meteorological glacier continuity audit; synthetic, not observations."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from noahmp_clm.column import Forcing
from noahmp_clm.clm_glacier import GlacierColumnState,step_clm_glacier_column
from noahmp_clm.physics.glacier import GlacierIce
from noahmp_clm.physics.canopy_energy import _saturation,_ice_saturation
from noahmp_clm.gpu.clm_glacier_column import pack_clm_glacier_columns,next_glacier_arrays
from test_gpu_clm_glacier_column import budgets
from test_gpu_glacier import host,compare_same


def stores(arrays):
    c,g=tuple(np.asarray(v.get() if hasattr(v,'get') else v,dtype=float) for v in arrays[:2])
    return (c[:,:,:2].sum(1)+g[:,:,:2].sum(1))[0]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--days',type=int,default=365);parser.add_argument('--steps-per-day',type=int,default=4)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.days<1 or args.steps_per_day<1:parser.error('Positive duration required')
    import jax
    import cupy as cp
    from noahmp_clm.gpu.clm_glacier_column import step_clm_glacier_columns_jax
    from noahmp_clm.gpu.cuda.clm_glacier_column import step_clm_glacier_columns_cuda
    assert jax.config.x64_enabled and jax.devices()[0].platform=='gpu'
    root=Path(__file__).resolve().parents[1]
    paths=[p for folder in ('src','tests','benchmarks') for p in (root/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    manifest={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    state=GlacierColumnState(GlacierIce.from_thickness([.1,.3,.6,99.],263.15))
    arrays={name:pack_clm_glacier_columns([state],dtype=dtype) for name,dtype in
        (('jax64',np.float64),('jax32',np.float32),('cuda32',np.float32))}
    names=('numpy64',*arrays);initial={k:stores(v) for k,v in arrays.items()};initial['numpy64']=np.array([state.mass_kgm2,state.enthalpy_jm2])
    audit={k:{field:np.zeros(2) for field in ('input','output','bound','step','cumulative')} for k in names}
    tracer_in={k:np.zeros(5) for k in names};tracer_out={k:np.zeros(5) for k in names}
    dep=np.array([1.,.1,.2,.3,.4])*1e-13;histories={k:[] for k in names};maxsame=0.;completed=0;start=time.perf_counter()
    report=dict(status='running',days_requested=args.days,steps_per_day=args.steps_per_day,source_manifest=manifest,
        reference='Synthetic forcing, NumPy versus native GPU. No CLM/FLUXNET observation acceptance.',daily={k:[] for k in names})
    keys=('et_mm','runoff_mm','prcp_mm','h','le','rnet')
    def save():
        report.update(steps_completed=completed,seconds=time.perf_counter()-start,same_precision=maxsame,
            audits={k:{field:v.tolist() for field,v in a.items()} for k,a in audit.items()})
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    try:
        for day in range(args.days):
            daily={k:{key:0. for key in keys} for k in names};seasonal=np.sin(2.*np.pi*(day-80.)/365.)
            for hour in range(args.steps_per_day):
                phase=(hour+.5)/args.steps_per_day;solar=max(0.,np.sin(2.*np.pi*(phase-.25)))
                air=260.+18.*seasonal+3.*np.sin(2.*np.pi*(phase-.35));pressure=75000.
                q=float((_ice_saturation if air<273.15 else _saturation)(air,pressure)[0])*.75
                f=Forcing(86400./args.steps_per_day,air,q,3.,pressure,(350.+150.*seasonal)*solar,
                    .8*5.670374419e-8*air**4,1e-3/86400. if day%4==0 else 0.)
                old=np.array([state.mass_kgm2,state.enthalpy_jm2]);state,d=step_clm_glacier_column(state,f,cosine_zenith=max(.05,solar),deposition_rates=dep)
                results={};values={'numpy64':d};newstores={'numpy64':np.array([state.mass_kgm2,state.enthalpy_jm2])}
                oldstores={'numpy64':old};forcings={'numpy64':np.array(list(vars(f).values()),float)}
                for name in arrays:
                    dtype=np.float64 if name=='jax64' else np.float32
                    a=list(arrays[name]);a[4]=np.asarray(a[4].get() if hasattr(a[4],'get') else a[4]).copy();a[4][:,4]=max(.05,solar);a=tuple(a)
                    ff=np.array([list(vars(f).values())],dtype);dd=dep[None,:].astype(dtype);oldstores[name]=stores(a)
                    fun=step_clm_glacier_columns_cuda if name=='cuda32' else step_clm_glacier_columns_jax
                    result=fun(*a,ff,deposition=dd);out=host(result);budgets(a,ff,dd,out)
                    if name=='cuda32':assert isinstance(result['ice_cells'],cp.ndarray)
                    else:assert next(iter(result['ice_cells'].devices())).platform=='gpu'
                    arrays[name]=next_glacier_arrays(a,result);newstores[name]=stores(arrays[name]);results[name]=out
                    values[name]={key:v[0] for key,v in out['diagnostics'].items()};values[name]['impurity_out_kgm2']=out['impurity_out_kgm2'][0]
                    forcings[name]=ff[0].astype(float)
                maxsame=max(maxsame,compare_same(results['jax32'],results['cuda32']))
                for name in names:
                    d=values[name];ff=forcings[name];a=audit[name]
                    pgeo=state.parameters.geothermal_wm2 if name=='numpy64' else float(np.asarray(arrays[name][2])[0,1])
                    inp=np.array([ff[0]*ff[7]*1000.,float(d['precipitation_enthalpy_jm2'])+(float(d['rnet'])-float(d['h'])-float(d['le']))*ff[0]+pgeo*ff[0]])
                    out=np.array([float(d['et_mm'])+float(d['runoff_mm']),float(d['advected_enthalpy_jm2'])+float(d['runoff_enthalpy_jm2'])])
                    bound=np.array([d['water_bound_kgm2'],d['energy_bound_jm2']],float)
                    residual=newstores[name]-oldstores[name]-inp+out;assert np.all(abs(residual)<=bound),(name,day,hour,residual,bound)
                    a['input']+=inp;a['output']+=out;a['bound']+=bound;a['step']=np.maximum(a['step'],abs(residual))
                    cumulative=newstores[name]-initial[name]-a['input']+a['output'];a['cumulative']=np.maximum(a['cumulative'],abs(cumulative))
                    assert np.all(abs(cumulative)<=a['bound']+1e-6),(name,cumulative,a['bound'])
                    dtype=np.float64 if name in ('numpy64','jax64') else np.float32
                    tracer_in[name]+=dep.astype(dtype).astype(float)*ff[0];tracer_out[name]+=np.asarray(d['impurity_out_kgm2'],float)
                    current=state.snow.impurities.sum(0)+state.ice.impurities.sum(0) if name=='numpy64' else \
                        results[name]['snow_cells'][0,:,4:].astype(float).sum(0)+results[name]['ice_cells'][0,:,2:].astype(float).sum(0)
                    error=current+tracer_out[name]-tracer_in[name]
                    assert np.all(abs(error)<=128.*np.finfo(dtype).eps*(completed+1)*np.maximum(tracer_in[name],1e-30)+1e-20)
                    for key in keys:daily[name][key]+=float(d[key]) if key.endswith('_mm') else float(d[key])/args.steps_per_day
                    histories[name].append({key:float(d[key]) for key in ('h','le','ts','swe')})
                completed+=1
            for name in names:report['daily'][name].append(dict(day=day+1,**daily[name],**{k:histories[name][-1][k] for k in ('ts','swe')}))
            save()
            if day%10==9 or day==0:print(json.dumps(dict(day=day+1,same_precision=maxsame,audits=report['audits'])),flush=True)
        metrics={}
        for name in names[1:]:
            metrics[name]={}
            for key in keys:
                ref=np.array([v[key] for v in report['daily']['numpy64']]);val=np.array([v[key] for v in report['daily'][name]])
                atol=.01 if key.endswith('_mm') else 1.;nonzero=abs(ref)>atol;difference=abs(val-ref)
                assert np.all(difference<=np.maximum(atol,.1*abs(ref))),(name,key,difference)
                metrics[name][key]=dict(max_absolute=float(difference.max()),max_bias_percent=float(np.max(difference[nonzero]/abs(ref[nonzero])*100.,initial=0.)))
            for key in ('h','le','ts','swe'):
                ref=np.array([v[key] for v in histories['numpy64']]);val=np.array([v[key] for v in histories[name]])
                r=float(np.corrcoef(ref,val)[0,1]) if ref.std()>1e-12 and val.std()>1e-12 else None
                if r is not None:assert r>.85
                if key=='ts':assert np.max(abs(val-ref))<.1
                metrics[name]['instant_'+key]=dict(max_absolute=float(np.max(abs(val-ref))),r=r)
        report.update(status='complete',metrics=metrics)
    except Exception as exc:report.update(status='failed',error=repr(exc));raise
    finally:
        save();print(json.dumps(dict(status=report['status'],steps=completed,seconds=report['seconds'],output=str(args.output))),flush=True)


if __name__=='__main__':main()
