"""Real crop/irrigation/wetland BGC and physical feedback over repeated days.

Three synthetic columns, four backends. This is a conservation/backend audit,
not a wetland observation or yield benchmark. Supply is externally accounted;
CO2 NEE, CH4, harvested products and seed inputs have separate carbon ledgers.
"""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tests'))
from test_clm_wetland_column import case
from test_gpu_clm_bgc_column import reference
from test_gpu_clm_wetland_column import independent_budgets
from test_gpu_clm_enthalpy_column import host,compare_same_precision,soil_temperature,SoilParams
from test_gpu_snicar import assert_science_close
from noahmp_clm.gpu.clm_wetland_column import pack_clm_wetland_columns,step_clm_wetland_columns_jax,next_wetland_arrays
from noahmp_clm.gpu.cuda.clm_wetland_column import step_clm_wetland_columns_cuda
from noahmp_clm.gpu.clm_enthalpy_column import DIAGNOSTICS
from check_clm_crop_column_gpu import totals as crop_totals
from check_clm_bgc_column_gpu import CARBON,maximum

EXTRA=('irrigation_mm','irrigation_enthalpy_jm2','decomposed_carbon_gc_m2',
    'methane_emission_gc_m2','methane_emission_mgch4_m2','total_atmospheric_carbon_gc_m2')


def totals(arrays):
    result=crop_totals(arrays[:19]);result[:,2]+=arrays[-2][:,0].astype(float)
    return result


def stack_dict(values):
    return {key:stack_dict([v[key] for v in values]) if isinstance(values[0][key],dict)
        else np.stack([v[key] for v in values]) for key in values[0]}


def run(a,f,kwargs,name,cp):
    if name!='numpy64':
        fn=step_clm_wetland_columns_cuda if name=='cuda32' else step_clm_wetland_columns_jax
        return host(fn(a,f,return_surface=True,nitrogen_deposition_gn_m2=.0001,**kwargs),cp.asnumpy if name=='cuda32' else np.asarray)
    refs=[reference(row[:16],one,np.zeros(5),np.zeros(2),np.zeros(2,int),.0001,1.,
        crop_data=(*row[16:19],kwargs['management'][k]),irrigation_data=(*row[19:21],kwargs['inputs'][k],kwargs['checks'][k]),
        wetland_data=row[21:]) for k,(row,one) in enumerate(zip(zip(*a),f))]
    result={key:np.stack([r[key] for r in refs]) for key in ('snow_cells','soil_cells','options','storage','impurity_output_kgm2',
        'condensate','site','plant_state','soil_cn_state','phenology_state','phenology_flags','crop_state','crop_phase','irrigation_state','wetland_state')}
    result['diagnostics']={key:np.array([r['diagnostics'][key] for r in refs]) for key in (*DIAGNOSTICS,*CARBON,*EXTRA)}
    result['surface']={key:np.stack([r['surface'][key] for r in refs]) for key in ('outgoing_shortwave','outgoing_longwave','thermal_layer_tolerance','layer_residual')}
    for key in ('crop','wetland','irrigation'):result[key]=stack_dict([r[key] for r in refs])
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--steps',type=int,default=336)
    parser.add_argument('--numpy-only',action='store_true');parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
    if args.steps<=0:parser.error('Positive step count required')
    cp=None
    if not args.numpy_only:import cupy as cp
    started=time.monotonic();names=('numpy64',) if args.numpy_only else ('numpy64','jax64','jax32','cuda32')
    initial_states=[case(kind)[0].with_irrigation() for kind in ('crop','irrigated','harvest')]
    states={};initial={};net={};bounds={};errors={};daily={};diurnal={};management={}
    for name in names:
        dtype=np.float64 if name.endswith('64') else np.float32
        a=pack_clm_wetland_columns(initial_states,dtype=dtype);states[name]=a;initial[name]=totals(a);net[name]=np.zeros((3,4));bounds[name]=np.zeros(4)
        errors[name]={'step':np.zeros(4),'cumulative':np.zeros(4)};daily[name]={};diurnal[name]={};management[name]=[]
    same=0.;report=dict(source_manifest=json.loads((ROOT/'SOURCE_SHA256.json').read_text()) if (ROOT/'SOURCE_SHA256.json').exists() else None,
        steps_requested=args.steps,columns=3,backends=len(names),most=False,irrigation=True,wetland=True,status='running')
    def save(status,step):
        report.update(status=status,steps_completed=step,seconds=time.monotonic()-started,same_precision_max_absolute=same,
            budgets={n:{k:v.tolist() for k,v in d.items()} for n,d in errors.items()},
            cumulative_numerical_bounds={n:v.tolist() for n,v in bounds.items()},management_events=management,
            daily_totals_and_means={n:{str(day):{k:v.tolist() for k,v in values.items()} for day,values in days.items()} for n,days in daily.items()})
        args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    save('running',0)
    try:
        for step in range(args.steps):
            angle=(step%48+.5)*2.*np.pi/48.;solar=max(0.,np.sin(angle-np.pi/2.));results={}
            for name in names:
                a=states[name];dtype=a[0].dtype
                f=np.broadcast_to(np.array([1800.,293.15+7.*np.sin(angle-np.pi/2.),.008,3.,101325.,600.*solar,320.,
                    2e-7 if 8<=step%48<16 else 0.],dtype),(3,8)).copy()
                events=np.zeros((3,2),np.int32)
                if step==0:events[2,1]=1
                if step==48:events[2,0]=1
                inputs=np.array([[1.,290.],[2.,290.],[0.,290.]],dtype);checks=np.full((3,1),int(step%48==12),np.int32)
                kwargs=dict(management=events,inputs=inputs,checks=checks)
                r=run(a,f,kwargs,name,cp);results[name]=r;b=independent_budgets(a,f,kwargs,r);after=next_wetland_arrays(a,r)
                ff=f.astype(float);d=r['diagnostics'];sf=r['surface'];crop=r['crop']
                rain=1000.*ff[:,7]*ff[:,0];rain_h=rain*(334000.+4180.*(ff[:,1]-273.15))
                water=d['irrigation_mm'].astype(float);water_h=water*(334000.+4180.*(inputs[:,1].astype(float)-273.15))
                energy=(ff[:,5]+ff[:,6]-sf['outgoing_shortwave'].astype(float)-sf['outgoing_longwave'].astype(float)-d['h'].astype(float)-d['le'].astype(float))*ff[:,0]
                net[name][:,0]+=rain+water-d['et_mm'].astype(float)-d['runoff_mm'].astype(float)
                net[name][:,1]+=rain_h+water_h+energy-d['advected_enthalpy_jm2'].astype(float)-d['runoff_enthalpy_jm2'].astype(float)
                net[name][:,2]+=-d['nee_gc_m2'].astype(float)-d['methane_emission_gc_m2'].astype(float)+crop['seed_input_gc_m2'].astype(float)-crop['harvest_gc_m2'].astype(float)
                net[name][:,3]+=.0001+crop['seed_input_gn_m2'].astype(float)-crop['harvest_gn_m2'].astype(float)
                bounds[name]+=np.array([b['water_bound_max_kgm2'],b['heat_bound_max_jm2'],b['carbon_bound_max'],b['nitrogen_bound_max']])
                cumulative=totals(after)-initial[name]-net[name]
                assert np.all(abs(cumulative)<=bounds[name]),(name,step,cumulative,bounds[name])
                errors[name]['step']=np.maximum(errors[name]['step'],[b['water_kgm2'],b['heat_jm2'],b['carbon'],b['nitrogen']])
                errors[name]['cumulative']=np.maximum(errors[name]['cumulative'],abs(cumulative).max(0))
                for key in ('planted','emerged','harvested'):
                    for k in np.flatnonzero(crop[key]):management[name].append([step,int(k),key])
                obs={key:d[key] for key in ('et_mm','runoff_mm','h','le','swe','irrigation_mm','methane_emission_gc_m2','total_atmospheric_carbon_gc_m2',*CARBON)}
                obs.update(lai=r['site'][:,0],water=r['soil_cells'][:,:,0]/(1000.*r['soil_cells'][:,:,2]),methane_storage=r['wetland_state'][:,0])
                obs['temperature']=np.stack([soil_temperature(after[1][k],SoilParams(smcmax=float(after[2][k,0]),bexp=float(after[2][k,1]),saturated_suction_m=float(after[2][k,6])),int(after[3][k,1]),dtype) for k in range(3)])
                sums=daily[name].setdefault(step//48,{k:np.zeros_like(v,dtype=float) for k,v in obs.items()})
                amounts=('et_mm','runoff_mm','irrigation_mm','methane_emission_gc_m2','total_atmospheric_carbon_gc_m2',*CARBON)
                for key,value in obs.items():sums[key]+=value.astype(float) if key in amounts else value.astype(float)/48.
                for key in ('h','le','gpp_gc_m2','nee_gc_m2','methane_emission_gc_m2'):diurnal[name].setdefault(key,[]).append(np.asarray(d[key],float))
                states[name]=after
            if not args.numpy_only:same=max(same,compare_same_precision(results['cuda32'],results['jax32']))
            if step%48==47:
                save('running',step+1);print(json.dumps(dict(day=step//48+1,budgets=report['budgets'],same_precision=same)),flush=True)
        comparison={};correlation={}
        floors={'et_mm':(.5,.05),'runoff_mm':(.5,.05),'irrigation_mm':(.5,.05),'h':(10.,1.),'le':(10.,1.),
            'swe':(.001,.0001),'lai':(.01,.001),'water':(.02,.002),'methane_storage':(.0001,.00001),'methane_emission_gc_m2':(.001,.0001),
            'total_atmospheric_carbon_gc_m2':(.1,.01)}
        floors.update({k:(.1,.01) for k in CARBON})
        for name in names[1:]:
            comparison[name]={};correlation[name]={}
            for day,ref in daily['numpy64'].items():
                if (day+1)*48>args.steps:continue
                for key,expected in ref.items():
                    value=daily[name][day][key];delta=abs(value-expected)
                    if key=='temperature':np.testing.assert_allclose(value,expected,rtol=0.,atol=.05);floor=np.inf
                    else:
                        floor,absolute=floors[key];assert_science_close(value,expected,relative=.1,zero_floor=floor,absolute=absolute)
                    maximum(comparison[name].setdefault(key,{}),dict(max_absolute=delta.max(),max_relative_percent=np.where(abs(expected)>floor,100.*delta/np.maximum(abs(expected),1e-30),0.).max()))
            for key,records in diurnal['numpy64'].items():
                ref=np.asarray(records);values=np.asarray(diurnal[name][key]);rs=[]
                for k in range(3):
                    if len(ref)<3 or np.std(ref[:,k])<1e-9:rs.append(None);continue
                    r=float(np.corrcoef(ref[:,k],values[:,k])[0,1]);assert np.isfinite(r) and r>.85;rs.append(r)
                correlation[name][key]=rs
        report.update(daily_comparison=comparison,instantaneous_correlation=correlation)
        save('complete',args.steps);print(json.dumps(dict(status='complete',seconds=report['seconds'])),flush=True)
    except Exception as error:
        for name,a in states.items():np.savez_compressed(str(args.output)+'.'+name+'.failed.npz',**{f'a{k}':v for k,v in enumerate(a)})
        report.update(failure=repr(error),failure_step=step);save('failed',step);raise


if __name__=='__main__':main()
