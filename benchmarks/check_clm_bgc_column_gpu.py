"""Continuous four-backend BGC feedback and independent M/H/C/N audit.

Synthetic weather and explicit seasonal calendar, never labelled FLUXNET or
official validation. Every day's totals and closure bounds are retained.
"""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tests'))
import test_gpu_clm_bgc_column as check
from test_gpu_clm_enthalpy_column import soil_temperature,SoilParams
from noahmp_clm.gpu.clm_enthalpy_column import DIAGNOSTICS
from test_gpu_snicar import assert_science_close
from noahmp_clm.gpu.clm_bgc_column import step_clm_bgc_columns_jax,next_bgc_arrays
from noahmp_clm.gpu.cuda.clm_bgc_column import step_clm_bgc_columns_cuda
from noahmp_clm.physics.precip import partition_precipitation

CARBON=('gpp_gc_m2','npp_gc_m2','nee_gc_m2','autotrophic_respiration_gc_m2','heterotrophic_respiration_gc_m2','soil_respiration_gc_m2')


def totals(a):
    c,g,_,_,_,_,_,cc=a[:8];plant=a[11].astype(float);soil=a[13].astype(float)
    return np.column_stack((c[:,:,0].astype(float).sum(1)+g[:,:,0].astype(float).sum(1)+cc[:,:,0].astype(float).sum(1),
        c[:,:,1].astype(float).sum(1)+g[:,:,1].astype(float).sum(1)+cc[:,:,1].astype(float).sum(1),
        plant[:,:3].sum(1)+plant[:,6]+soil[:,:7].sum(1),plant[:,3:6].sum(1)+plant[:,7]+soil[:,7:].sum(1)))


def run(a,f,dep,env,cal,nitrogen,backend,cp):
    if backend!='numpy64':
        fn=step_clm_bgc_columns_cuda if backend=='cuda32' else step_clm_bgc_columns_jax
        return check.host(fn(*a,f,environment=env,calendar=cal,nitrogen_deposition_gn_m2=nitrogen,
            deposition=dep,return_surface=True),cp.asnumpy if backend=='cuda32' else np.asarray)
    refs=[check.reference(row,one,dd,e,c,nn,1.) for row,one,dd,e,c,nn in zip(zip(*a),f,dep,env,cal,nitrogen)]
    result={key:np.stack([r[key] for r in refs]) for key in ('snow_cells','soil_cells','options','storage','impurity_output_kgm2',
        'condensate','site','plant_state','soil_cn_state','phenology_state','phenology_flags')}
    result['diagnostics']={key:np.array([r['diagnostics'][key] for r in refs]) for key in (*DIAGNOSTICS,*CARBON)}
    result['surface']={key:np.stack([r['surface'][key] for r in refs]) for key in ('outgoing_shortwave','outgoing_longwave','thermal_layer_tolerance','layer_residual')}
    return result


def maximum(old,new):
    for key,value in new.items():old[key]=max(old.get(key,0.),float(value))


def main():
    import cupy as cp
    parser=argparse.ArgumentParser();parser.add_argument('--steps',type=int,default=336);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    if args.steps<=0:parser.error('Positive step count required')
    started=time.monotonic();names=('numpy64','jax64','jax32','cuda32');selected=[0,2,4]
    states={};initial={};net={};bound={};errors={};daily={};diurnal={};base_env={};calendar={}
    for name in names:
        dtype=np.float64 if name.endswith('64') else np.float32
        a,_,_,e,c=check.cases(dtype);states[name]=tuple(v[selected].copy() for v in a)
        # Each backend starts from the same physical configuration represented
        # in its own dtype, including valid fraction and aggregate aliases.
        initial[name]=totals(states[name]);net[name]=np.zeros((3,4));bound[name]=np.zeros(4)
        errors[name]={'step':np.zeros(4),'cumulative':np.zeros(4)};daily[name]={};diurnal[name]={}
        base_env[name]=e[selected].copy();calendar[name]=c[selected].copy()
    same=0.;report={'source_manifest':json.loads((ROOT/'SOURCE_SHA256.json').read_text()) if (ROOT/'SOURCE_SHA256.json').exists() else None,
        'steps_requested':args.steps,'columns':3,'backends':4,'most':False,'wetbulb':5,'runoff':6,'status':'running'}
    def save(status,step):
        report.update(status=status,steps_completed=step,seconds=time.monotonic()-started,same_precision_max_absolute=same,
            budgets={n:{k:v.tolist() for k,v in d.items()} for n,d in errors.items()},cumulative_numerical_bounds={n:v.tolist() for n,v in bound.items()})
        args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    save('running',0)
    try:
        for step in range(args.steps):
            angle=(step%48+.5)*2*np.pi/48.;solar=max(0.,np.sin(angle-np.pi/2.));results={}
            for name in names:
                a=states[name];dtype=a[0].dtype;f=np.empty((3,8),dtype)
                f[:,0]=1800.;f[:,1]=np.array([285.,285.,267.+1.5*(step//48)])+7.*np.sin(angle-np.pi/2.)
                f[:,2]=[.006,.006,.002];f[:,3]=3.;f[:,4]=101325.;f[:,5]=600.*solar;f[:,6]=280.
                f[:,7]=np.array([2e-7,1e-7,1e-7]) if 8<=step%48<16 else 0.
                dep=np.broadcast_to(np.array([1e-12,2e-11,1e-11,5e-12,2e-12],dtype),(3,5)).copy();nitrogen=np.full(3,.00001,dtype)
                r=run(a,f,dep,base_env[name],calendar[name],nitrogen,name,cp);results[name]=r
                b=check.budgets(a[:8],f,dep,r);cn=check.cn_budgets(a,r,nitrogen);after=next_bgc_arrays(a,r)
                ff=f.astype(float);d=r['diagnostics'];sf=r['surface'];ph=[]
                for one,option in zip(ff,a[3]):
                    rain,snow=partition_precipitation(one[7],one[0],one[1],int(option[2]),qair=one[2],pressure=one[4])
                    ph.append(1000.*(rain*(334000.+4180.*(one[1]-273.15))+snow*2100.*(min(one[1],273.15)-273.15)))
                energy=(ff[:,5]+ff[:,6]-sf['outgoing_shortwave'].astype(float)-sf['outgoing_longwave'].astype(float)-d['h'].astype(float)-d['le'].astype(float))*ff[:,0]
                net[name][:,0]+=ff[:,7]*ff[:,0]*1000.-d['et_mm'].astype(float)-d['runoff_mm'].astype(float)
                net[name][:,1]+=np.array(ph)+energy-d['advected_enthalpy_jm2'].astype(float)-d['runoff_enthalpy_jm2'].astype(float)
                net[name][:,2]+=d['gpp_gc_m2'].astype(float)-d['autotrophic_respiration_gc_m2'].astype(float)-d['heterotrophic_respiration_gc_m2'].astype(float)
                net[name][:,3]+=nitrogen.astype(float)
                bound[name]+=np.array([b['water_bound_max_kgm2'],b['heat_bound_max_jm2'],cn['carbon_bound_max'],cn['nitrogen_bound_max']])
                cumulative=totals(after)-initial[name]-net[name]
                assert np.all(abs(cumulative)<=bound[name]),(name,step,cumulative,bound[name])
                errors[name]['step']=np.maximum(errors[name]['step'],[b['water_kgm2'],b['heat_jm2'],cn['carbon'],cn['nitrogen']])
                errors[name]['cumulative']=np.maximum(errors[name]['cumulative'],abs(cumulative).max(0))
                obs={key:d[key] for key in ('et_mm','runoff_mm','h','le','swe',*CARBON)}
                obs['lai']=r['site'][:,0];obs['water']=r['soil_cells'][:,:,0]/(1000.*r['soil_cells'][:,:,2])
                obs['temperature']=np.stack([soil_temperature(after[1][k],SoilParams(smcmax=float(after[2][k,0]),bexp=float(after[2][k,1])),int(after[3][k,1]),dtype) for k in range(3)])
                sums=daily[name].setdefault(step//48,{k:np.zeros_like(v,dtype=float) for k,v in obs.items()})
                for key,value in obs.items():sums[key]+=value.astype(float) if key in ('et_mm','runoff_mm',*CARBON) else value.astype(float)/48.
                for key in ('h','le','gpp_gc_m2','nee_gc_m2'):diurnal[name].setdefault(key,[]).append(np.asarray(d[key],float))
                states[name]=after
            same=max(same,check.compare_same_precision(results['cuda32'],results['jax32']))
            if step%48==47:
                save('running',step+1);print(json.dumps({'day':step//48+1,'budgets':report['budgets'],'same_precision':same}),flush=True)
        comparisons={};correlation={}
        floors={'et_mm':(.5,.05),'runoff_mm':(.5,.05),'h':(10.,1.),'le':(10.,1.),'swe':(.001,.0001),'lai':(.01,.001),'water':(.02,.002)}
        floors.update({k:(.1,.01) for k in CARBON})
        for name in names[1:]:
            comparisons[name]={};correlation[name]={}
            for day,ref in daily['numpy64'].items():
                if (day+1)*48>args.steps:continue
                for key,expected in ref.items():
                    actual=daily[name][day][key];err=abs(actual-expected)
                    if key=='temperature':np.testing.assert_allclose(actual,expected,rtol=0.,atol=.05);floor=np.inf
                    else:
                        floor,absolute=floors[key];assert_science_close(actual,expected,relative=.1,zero_floor=floor,absolute=absolute)
                    maximum(comparisons[name].setdefault(key,{}),dict(max_absolute=err.max(),max_relative_percent=np.where(abs(expected)>floor,100.*err/np.maximum(abs(expected),1e-30),0.).max()))
            for key,records in diurnal['numpy64'].items():
                ref=np.asarray(records);value=np.asarray(diurnal[name][key]);rs=[]
                for k in range(3):
                    if np.std(ref[:,k])<1e-9:rs.append(None);continue
                    r=float(np.corrcoef(ref[:,k],value[:,k])[0,1]);assert np.isfinite(r) and r>.85;rs.append(r)
                correlation[name][key]=rs
        report.update(daily_comparison=comparisons,instantaneous_correlation=correlation)
        save('complete',args.steps);print(json.dumps({'output':str(args.output),'seconds':report['seconds'],'status':'complete'}),flush=True)
    except Exception as error:
        for name,a in states.items():np.savez_compressed(str(args.output)+'.'+name+'.failed.npz',**{f'a{k}':v for k,v in enumerate(a)})
        report['failure']=repr(error);report['failure_step']=step;save('failed',step);raise


if __name__=='__main__':main()
