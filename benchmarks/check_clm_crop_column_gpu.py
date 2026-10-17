"""Four-backend continuous crop/BGC/physical audit with real state feedback.

Synthetic seven-day management exercise (seed, emergence, harvest, replant),
not a full growing season, yield validation, FLUXNET or official CLM output.
"""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tests'))
import test_gpu_clm_crop_column as check
from test_gpu_clm_bgc_column import reference
from test_gpu_clm_enthalpy_column import soil_temperature,SoilParams
from noahmp_clm.gpu.clm_enthalpy_column import DIAGNOSTICS
from noahmp_clm.gpu.clm_crop_column import step_clm_crop_columns_jax,next_crop_arrays
from noahmp_clm.gpu.cuda.clm_crop_column import step_clm_crop_columns_cuda
from test_gpu_snicar import assert_science_close
from check_clm_bgc_column_gpu import CARBON,totals as bgc_totals,maximum


def totals(arrays):
    result=bgc_totals(arrays);result[:,2:4]+=arrays[16][:,3:5].astype(float)
    return result


def run(a,f,dep,events,nitrogen,backend,cp):
    if backend!='numpy64':
        fn=step_clm_crop_columns_cuda if backend=='cuda32' else step_clm_crop_columns_jax
        return check.host(fn(*a,f,management=events,nitrogen_deposition_gn_m2=nitrogen,
            deposition=dep,return_surface=True),cp.asnumpy if backend=='cuda32' else np.asarray)
    refs=[reference(row[:16],one,dd,np.zeros(2),np.zeros(2,int),nn,1.,crop_data=(*row[16:],ev))
        for row,one,dd,nn,ev in zip(zip(*a),f,dep,nitrogen,events)]
    result={key:np.stack([r[key] for r in refs]) for key in ('snow_cells','soil_cells','options','storage','impurity_output_kgm2',
        'condensate','site','plant_state','soil_cn_state','phenology_state','phenology_flags','crop_state','crop_phase')}
    result['diagnostics']={key:np.array([r['diagnostics'][key] for r in refs]) for key in (*DIAGNOSTICS,*CARBON)}
    result['surface']={key:np.stack([r['surface'][key] for r in refs]) for key in ('outgoing_shortwave','outgoing_longwave','thermal_layer_tolerance','layer_residual')}
    result['crop']={key:np.stack([r['crop'][key] for r in refs]) for key in refs[0]['crop']}
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--steps',type=int,default=336);parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--numpy-only',action='store_true',help='Run the same physical/management trajectory on NumPy only')
    args=parser.parse_args()
    if args.steps<=0:parser.error('Positive step count required')
    cp=None
    if not args.numpy_only:import cupy as cp
    started=time.monotonic();names=('numpy64',) if args.numpy_only else ('numpy64','jax64','jax32','cuda32');selected=[0,3,4]
    states={};initial={};net={};bound={};errors={};daily={};diurnal={};events_seen={}
    for name in names:
        dtype=np.float64 if name.endswith('64') else np.float32
        a,_,_=check.cases(dtype);a=tuple(v[selected].copy() for v in a)
        a[16][2,0]=1690.;a[16][2,1]=1690.;a[16][2,2]=90.
        # Start the second crop N-limited with matching actual leaf-N alias.
        a[11][1,3:6]*=.25;a[4][1,11]=.25;a[11][1,7]=0.;a[13][1,14]=0.
        states[name]=a;initial[name]=totals(a);net[name]=np.zeros((3,4));bound[name]=np.zeros(4)
        errors[name]={'step':np.zeros(4),'cumulative':np.zeros(4)};daily[name]={};diurnal[name]={};events_seen[name]=[]
    same=0.;report={'source_manifest':json.loads((ROOT/'SOURCE_SHA256.json').read_text()) if (ROOT/'SOURCE_SHA256.json').exists() else None,
        'steps_requested':args.steps,'columns':3,'backends':len(names),'most':False,'wetbulb':5,'runoff':6,'status':'running'}
    def save(status,step):
        report.update(status=status,steps_completed=step,seconds=time.monotonic()-started,same_precision_max_absolute=same,
            budgets={n:{k:v.tolist() for k,v in d.items()} for n,d in errors.items()},
            cumulative_numerical_bounds={n:v.tolist() for n,v in bound.items()},management_events=events_seen)
        report['daily_totals_and_means']={n:{str(day):{k:v.tolist() for k,v in values.items()}
            for day,values in days.items()} for n,days in daily.items()}
        args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    save('running',0)
    try:
        for step in range(args.steps):
            angle=(step%48+.5)*2*np.pi/48.;solar=max(0.,np.sin(angle-np.pi/2.));results={}
            for name in names:
                a=states[name];dtype=a[0].dtype;f=np.empty((3,8),dtype)
                f[:,0]=1800.;f[:,1]=293.15+7.*np.sin(angle-np.pi/2.);f[:,2]=.008;f[:,3]=3.;f[:,4]=101325.
                f[:,5]=600.*solar;f[:,6]=320.;f[:,7]=2e-7 if 8<=step%48<16 else 0.
                dep=np.zeros((3,5),dtype);nitrogen=np.full(3,.00001,dtype);events=np.zeros((3,2),np.int32)
                if step==0:events[0,0]=1
                if step==192:events[2,0]=1
                r=run(a,f,dep,events,nitrogen,name,cp);results[name]=r
                b=check.budgets(a[:8],f,dep,r);cn=check.cn_budgets(a,r,nitrogen);after=next_crop_arrays(a,r)
                ff=f.astype(float);d=r['diagnostics'];sf=r['surface'];cd=r['crop']
                ph=1000.*ff[:,7]*ff[:,0]*(334000.+4180.*(ff[:,1]-273.15))
                energy=(ff[:,5]+ff[:,6]-sf['outgoing_shortwave'].astype(float)-sf['outgoing_longwave'].astype(float)-d['h'].astype(float)-d['le'].astype(float))*ff[:,0]
                net[name][:,0]+=ff[:,7]*ff[:,0]*1000.-d['et_mm'].astype(float)-d['runoff_mm'].astype(float)
                net[name][:,1]+=ph+energy-d['advected_enthalpy_jm2'].astype(float)-d['runoff_enthalpy_jm2'].astype(float)
                net[name][:,2]+=d['gpp_gc_m2'].astype(float)-d['autotrophic_respiration_gc_m2'].astype(float)-d['heterotrophic_respiration_gc_m2'].astype(float)
                net[name][:,2]+=cd['seed_input_gc_m2'].astype(float)-cd['harvest_gc_m2'].astype(float)
                net[name][:,3]+=nitrogen.astype(float)+cd['seed_input_gn_m2'].astype(float)-cd['harvest_gn_m2'].astype(float)
                bound[name]+=np.array([b['water_bound_max_kgm2'],b['heat_bound_max_jm2'],cn['carbon_bound_max'],cn['nitrogen_bound_max']])
                cumulative=totals(after)-initial[name]-net[name]
                assert np.all(abs(cumulative)<=bound[name]),(name,step,cumulative,bound[name])
                errors[name]['step']=np.maximum(errors[name]['step'],[b['water_kgm2'],b['heat_jm2'],cn['carbon'],cn['nitrogen']])
                errors[name]['cumulative']=np.maximum(errors[name]['cumulative'],abs(cumulative).max(0))
                for key in ('planted','emerged','harvested'):
                    for k in np.flatnonzero(cd[key]):events_seen[name].append([step,int(k),key])
                obs={key:d[key] for key in ('et_mm','runoff_mm','h','le','swe',*CARBON)}
                obs.update(lai=r['site'][:,0],water=r['soil_cells'][:,:,0]/(1000.*r['soil_cells'][:,:,2]))
                obs['temperature']=np.stack([soil_temperature(after[1][k],SoilParams(smcmax=float(after[2][k,0]),bexp=float(after[2][k,1]),saturated_suction_m=float(after[2][k,6])),int(after[3][k,1]),dtype) for k in range(3)])
                sums=daily[name].setdefault(step//48,{k:np.zeros_like(v,dtype=float) for k,v in obs.items()})
                for key,value in obs.items():sums[key]+=value.astype(float) if key in ('et_mm','runoff_mm',*CARBON) else value.astype(float)/48.
                for key in ('h','le','gpp_gc_m2','nee_gc_m2'):diurnal[name].setdefault(key,[]).append(np.asarray(d[key],float))
                states[name]=after
            if not args.numpy_only:same=max(same,check.compare_same_precision(results['cuda32'],results['jax32']))
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
        if not args.numpy_only:assert events_seen['cuda32']==events_seen['jax32']
        report.update(daily_comparison=comparisons,instantaneous_correlation=correlation)
        save('complete',args.steps);print(json.dumps({'output':str(args.output),'seconds':report['seconds'],'status':'complete'}),flush=True)
    except Exception as error:
        for name,a in states.items():np.savez_compressed(str(args.output)+'.'+name+'.failed.npz',**{f'a{k}':v for k,v in enumerate(a)})
        report['failure']=repr(error);report['failure_step']=step;save('failed',step);raise


if __name__=='__main__':main()
