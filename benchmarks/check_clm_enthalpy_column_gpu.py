"""Synthetic four-backend precipitation-to-runoff extensive column audit."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tests'))
import test_gpu_clm_enthalpy_column as check
from test_gpu_snicar import assert_science_close
from noahmp_clm.gpu.clm_enthalpy_column import step_clm_enthalpy_columns_jax,DIAGNOSTICS
from noahmp_clm.gpu.cuda.clm_enthalpy_column import step_clm_enthalpy_columns_cuda
from noahmp_clm.physics.precip import partition_precipitation
from noahmp_clm.physics.canopy_most import MOSTGeometry
from dataclasses import astuple

MOST=None
VIC=False
WETBULB=False


def cast_column_precision(arrays,dtype):
    """Preserve quantized physical stores; regenerate redundant layer aliases.

    A float32 sum promoted to float64 is not the float64 sum of the promoted
    layers. This only repairs packing metadata, never rescales layer M/H.
    """
    result=[a.copy() if a.dtype.kind in 'iu' else a.astype(dtype) for a in arrays]
    if len(result)==8:result[5][:,:2]=result[7].sum(axis=1)
    return tuple(result)


def maxima(old,new):
    for key,value in new.items():
        if isinstance(value,dict):maxima(old.setdefault(key,{}),value)
        else:old[key]=np.maximum(old.get(key,np.zeros_like(value)),value).tolist()


def run(arrays,forcing,dep,backend,cp):
    if backend!='numpy64':
        runner=step_clm_enthalpy_columns_cuda if backend=='cuda32' else step_clm_enthalpy_columns_jax
        options={} if MOST is None else dict(most_geometry=np.broadcast_to(np.array(astuple(MOST),arrays[0].dtype),(len(forcing),9)).copy())
        return check.host(runner(*arrays,forcing,deposition=dep,return_surface=True,**options),cp.asnumpy if backend=='cuda32' else np.asarray)
    options={} if MOST is None else dict(most_geometry=MOST)
    refs=[check.reference(row,f,d,**options) for row,f,d in zip(zip(*arrays),forcing,dep)]
    result={key:np.stack([r[key] for r in refs]) for key in ('snow_cells','soil_cells','options','storage','impurity_output_kgm2')}
    if len(arrays)==8:result['condensate']=np.stack([r['condensate'] for r in refs])
    result['diagnostics']={key:np.array([r['diagnostics'][key] for r in refs]) for key in DIAGNOSTICS}
    result['surface']={key:np.stack([r['surface'][key] for r in refs]) for key in
        ('outgoing_shortwave','outgoing_longwave','thermal_layer_tolerance','layer_residual')}
    return result


def total(arrays):
    c,g,_,_,_,store,_=arrays[:7]
    if len(arrays)==8:
        store=store.astype(float).copy();store[:,:2]=arrays[7].astype(float).sum(1)
    return np.stack((c[:,:,0].astype(float).sum(1)+g[:,:,0].astype(float).sum(1)+store[:,0],
        c[:,:,1].astype(float).sum(1)+g[:,:,1].astype(float).sum(1)+store[:,1]),axis=1)


def advance(arrays,r):
    base=r['snow_cells'],r['soil_cells'],arrays[2],r['options'],arrays[4],r['storage'],arrays[6]
    return (*base,r['condensate']) if len(arrays)==8 else base


def trajectory(steps,cp):
    base,_,deposition=check.cases(np.float32);selected=[0,9,4]
    if MOST is not None:base[3][:,5]=1
    if VIC:base[3][:,6:8]=6
    if WETBULB:base[3][:,2]=5
    base=tuple(x[selected].copy() for x in base);deposition=deposition[selected]
    names=('numpy64','jax64','jax32','cuda32')
    states={name:cast_column_precision(base,np.float64 if name.endswith('64') else np.float32) for name in names}
    initial={name:total(a) for name,a in states.items()};net={name:np.zeros((3,2)) for name in names}
    errors={name:dict(step=[0.,0.],cumulative=[0.,0.]) for name in names};bounds={name:np.zeros(2) for name in names}
    daily={name:{} for name in names};same_precision=0.
    for step in range(steps):
        angle=(step%48+.5)*2.*np.pi/48.;sun=max(0.,np.sin(angle-np.pi/2.))
        f=np.empty((3,8),np.float32);f[:,0]=1800.;f[:,1]=np.array([285.,267.+1.5*(step//48),280.])+7.*np.sin(angle-np.pi/2.)
        f[:,2]=[.006,.002,.004];f[:,3]=3.;f[:,4]=101325.;f[:,5]=600.*sun;f[:,6]=280.
        f[:,7]=np.array([2e-7,1e-7,1e-7]) if 8<=step%48<16 else 0.
        results={}
        for name in names:
            arrays=states[name];dtype=arrays[0].dtype;forcing=f.astype(dtype);dep=deposition.astype(dtype)
            r=run(arrays,forcing,dep,name,cp);b=check.budgets(arrays,forcing,dep,r);after=advance(arrays,r)
            diag=r['diagnostics'];surface=r['surface'];ff=forcing.astype(float)
            pm=ff[:,0]*ff[:,7]*1000.;ph=[]
            for one,opt in zip(ff,arrays[3]):
                rain,snow=partition_precipitation(one[7],one[0],one[1],int(opt[2]),qair=one[2],pressure=one[4])
                ph.append(1000.*(rain*(334000.+4180.*(one[1]-273.15))+snow*2100.*(min(one[1],273.15)-273.15)))
            atmo=(ff[:,5]+ff[:,6]-surface['outgoing_shortwave'].astype(float)-surface['outgoing_longwave'].astype(float)-
                diag['h'].astype(float)-diag['le'].astype(float))*ff[:,0]
            net[name][:,0]+=pm-diag['et_mm'].astype(float)-diag['runoff_mm'].astype(float)
            net[name][:,1]+=np.array(ph)+atmo-diag['advected_enthalpy_jm2'].astype(float)-diag['runoff_enthalpy_jm2'].astype(float)
            cumulative=total(after)-initial[name]-net[name]
            bounds[name]+=np.array([b['water_bound_max_kgm2'],b['heat_bound_max_jm2']])
            assert np.all(abs(cumulative)<=bounds[name]),(name,step,cumulative,bounds[name])
            maxima(errors[name],dict(step=[b['water_kgm2'],b['heat_jm2']],cumulative=np.max(abs(cumulative),axis=0)))
            obs={key:diag[key] for key in ('et_mm','runoff_mm','gpp_gcm2','h','le','swe')}
            obs['temperature']=np.stack([check.soil_temperature(after[1][k],
                check.SoilParams(smcmax=float(after[2][k,0]),bexp=float(after[2][k,1])),int(after[3][k,1]),dtype) for k in range(3)])
            obs['water']=r['soil_cells'][:,:,0]/(1000.*r['soil_cells'][:,:,2])
            accum=daily[name].setdefault(step//48,{k:np.zeros_like(v,dtype=float) for k,v in obs.items()})
            for key,value in obs.items():accum[key]+=value.astype(float) if key in ('et_mm','runoff_mm','gpp_gcm2') else value.astype(float)/48.
            states[name]=after;results[name]=r
        same_precision=max(same_precision,check.compare_same_precision(results['cuda32'],results['jax32']))
        if step%48==47:print(json.dumps(dict(day=step//48+1,budgets=errors,same_precision=same_precision)),flush=True)
    comparison={}
    for name in names[1:]:
        comparison[name]={}
        for day,ref in daily['numpy64'].items():
            if (day+1)*48>steps:continue
            for key,expected in ref.items():
                actual=daily[name][day][key];error=abs(actual-expected)
                if key=='temperature':np.testing.assert_allclose(actual,expected,rtol=0.,atol=.05);floor=np.inf
                else:
                    floor,absolute={'et_mm':(.5,.05),'runoff_mm':(.5,.05),'gpp_gcm2':(1.,.1),
                        'h':(10.,1.),'le':(10.,1.),'swe':(.001,.0001),'water':(.02,.002)}[key]
                    assert_science_close(actual,expected,relative=.1,zero_floor=floor,absolute=absolute)
                maxima(comparison[name].setdefault(key,{}),dict(max_absolute=float(np.max(error)),
                    max_relative_percent=float(np.max(np.where(abs(expected)>floor,100.*error/np.maximum(abs(expected),1e-30),0.)))))
    return dict(steps=steps,columns=3,backends=4,budgets=errors,cumulative_numerical_bounds={k:v.tolist() for k,v in bounds.items()},
        daily_comparison=comparison,same_precision_max_absolute=same_precision)


def main():
    global check,step_clm_enthalpy_columns_jax,step_clm_enthalpy_columns_cuda,MOST,VIC,WETBULB
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--steps',type=int,default=336)
    parser.add_argument('--layered-canopy',action='store_true')
    parser.add_argument('--most',action='store_true',help='Actual H/E feedback with explicit synthetic tower heights; requires --layered-canopy')
    parser.add_argument('--vic',action='store_true',help='VIC admission and ARNO baseflow with explicit documented defaults; requires --layered-canopy')
    parser.add_argument('--wetbulb',action='store_true',help='Wang wet-bulb partition, including independent precipitation enthalpy')
    args=parser.parse_args()
    VIC=args.vic;WETBULB=args.wetbulb
    if VIC and not args.layered_canopy:parser.error('--vic requires --layered-canopy')
    if args.most:
        if not args.layered_canopy:parser.error('--most requires --layered-canopy')
        MOST=MOSTGeometry(wind_height=15.,temperature_height=12.,humidity_height=10.,displacement=2.)
    if args.layered_canopy:
        import test_gpu_clm_layered_column as check
        from noahmp_clm.gpu.clm_layered_column import step_clm_layered_columns_jax as step_clm_enthalpy_columns_jax
        from noahmp_clm.gpu.cuda.clm_layered_column import step_clm_layered_columns_cuda as step_clm_enthalpy_columns_cuda
    import cupy as cp
    start=time.monotonic();arrays,f,dep=check.cases(np.float32)
    if MOST is not None:arrays[3][:,5]=1
    if VIC:arrays[3][:,6:8]=6
    if WETBULB:arrays[3][:,2]=5
    options={} if MOST is None else dict(most_geometry=MOST)
    refs=[check.reference(row,one,d,**options) for row,one,d in zip(zip(*arrays),f,dep)]
    report=dict(source_manifest=json.loads((ROOT/'SOURCE_SHA256.json').read_text()),layered_canopy=args.layered_canopy,
        most_geometry=None if MOST is None else vars(MOST),vic=VIC,wetbulb=WETBULB,cases=len(refs),single={});results={}
    for name in ('jax64','jax32','cuda32'):
        dtype=np.float64 if name=='jax64' else np.float32
        x=cast_column_precision(arrays,dtype)
        print('single backend',name,flush=True);r=run(x,f.astype(dtype),dep.astype(dtype),name,cp);maximum={}
        for k,ref in enumerate(refs):
            one={key:({a:b[k] for a,b in v.items()} if isinstance(v,dict) else v[k]) for key,v in r.items()
                if key not in ('surface','canopy_interception','canopy_post_interception','most')}
            maxima(maximum,check.science(one,ref,dtype))
        report['single'][name]=dict(max_absolute=maximum,budget=check.budgets(x,f.astype(dtype),dep.astype(dtype),r));results[name]=r
    report['same_precision_max_absolute']=check.compare_same_precision(results['cuda32'],results['jax32'])
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report['single']),flush=True)
    if args.steps:report['trajectory']=trajectory(args.steps,cp)
    report['seconds']=time.monotonic()-start;args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(output=str(args.output),seconds=report['seconds'])),flush=True)


if __name__=='__main__':main()
