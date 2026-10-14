"""Seven-day fixed-mass snow/soil thermal stress test, not a hydrology year."""
from pathlib import Path
from dataclasses import replace
import argparse,hashlib,importlib.util,json,time
import numpy as np
import cupy as cp
import jax
from noahmp_clm.column import ColumnState,default_soil_params
from noahmp_clm.physics.snowpack import Snowpack,TF
from noahmp_clm.physics.soil_enthalpy import SoilEnthalpyState
from noahmp_clm.gpu.column_heat import pack_joint_heat_case,run_joint_heat_jax
from noahmp_clm.gpu.cuda.column_heat import run_joint_heat_cuda

ROOT=Path(__file__).resolve().parents[1]


def host(result): return {k:cp.asnumpy(v) if isinstance(v,cp.ndarray) else np.asarray(v) for k,v in result.items()}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path);args=ap.parse_args()
    start=time.perf_counter();assert jax.config.x64_enabled and any(d.platform=='gpu' for d in jax.devices())
    spec=importlib.util.spec_from_file_location('heat_reference',ROOT/'tests/test_gpu_column_heat.py')
    ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
    rows=[]
    for case,mass in enumerate(([],[.01],[2.,5.,10.,20.,40.])):
        snow=Snowpack.from_ice(mass,TF-4.) if mass else Snowpack.empty()
        col=ColumnState(soil=default_soil_params(('sand','loam','clay')[case]),tsoil=np.full(4,TF-4.),smc=np.full(4,.25))
        col.options=replace(col.options,OptSoilSupercoolWater=1+case%2)
        rows.append(pack_joint_heat_case(snow,SoilEnthalpyState.from_column(col),1800.,-30.,
            top_temperature=TF-4.,top_conductance=8.,dtype=np.float32))
    c,p,o,b=tuple(np.stack([row[i] for row in rows]) for i in range(4))
    states={'numpy64':c.astype(float),'jax64':c.astype(float),'jax32':c.copy(),'cuda32':c.copy()}
    initial=c[:,:,1].astype(float).sum(1)
    incoming={k:np.zeros(3) for k in states};gross={k:np.zeros(3) for k in states}
    maxima={k:np.zeros(4) for k in states};daily=[];same_max=0.;max_evaluations=0.
    accumulated_allowance={k:np.zeros(3) for k in states}
    for step in range(336):
        hour=(step%48)*.5;day=step/48.
        sun=max(0.,np.sin(np.pi*(hour-6.)/12.))
        air=TF-7.+12.*np.sin(np.pi*day/7.)+3.*np.sin(2.*np.pi*(hour-8.)/24.)
        b[:,1]=-30.;b[:,2]=np.float32(air-TF)
        diagnostic={}
        for backend,old in states.items():
            old[:,:,4]=0.;old[:,0,4]=np.float32(150.*sun)
            before=old[:,:,1].copy()
            if backend=='numpy64':
                temperature=np.zeros((3,9));top=np.zeros(3);step_tol=np.full(3,1e-5)
                for row in range(3):
                    n=int(o[row,0])+4;h,t,d=ref.numpy_reference(old[row],p[row],o[row],b[row])
                    old[row,:n,1]=h;temperature[row,:n]=t;top[row]=d['top_flux_wm2']
                output={'enthalpy_jm2':old[:,:,1].copy(),'temperature_k':temperature,'top_flux_wm2':top}
            else:
                precision=np.float64 if backend=='jax64' else np.float32
                function=run_joint_heat_cuda if backend=='cuda32' else run_joint_heat_jax
                try:
                    output=host(function(old,p.astype(precision),o,b.astype(precision)))
                except RuntimeError:
                    failure=args.output.with_suffix(f'.failed-{backend}-{step}.npz')
                    np.savez(failure,cells=old,props=p.astype(precision),options=o,boundary=b.astype(precision))
                    print('Failed thermal step',step,backend,'saved to',failure,flush=True)
                    if backend.startswith('jax'):
                        from noahmp_clm.gpu.column_heat import _compiled,unpack_joint_heat
                        raw=host(unpack_joint_heat(_compiled()(old,p.astype(precision),o,b.astype(precision))))
                        print(json.dumps({k:v.tolist() for k,v in raw.items()}),flush=True)
                    raise
                old[:,:,1]=output['enthalpy_jm2'];top=output['top_flux_wm2'].astype(float)
                step_tol=output['layer_tolerance_jm2'].astype(float).sum(1)+1e-5
                max_evaluations=max(max_evaluations,float(output['iterations'].max()))
            step_input=b[:,0].astype(float)*(top+old[:,:,4].astype(float).sum(1))
            incoming[backend]+=step_input;gross[backend]+=np.abs(step_input)
            budget=old[:,:,1].astype(float).sum(1)-initial-incoming[backend]
            step_budget=(old[:,:,1].astype(float)-before.astype(float)).sum(1)-step_input
            accumulated_allowance[backend]+=step_tol+16.*np.finfo(old.dtype).eps*np.maximum(1.,np.abs(step_input))
            maxima[backend][2]=max(maxima[backend][2],float(np.max(np.abs(step_budget))))
            maxima[backend][3]=max(maxima[backend][3],float(np.max(np.abs(budget))))
            assert np.all(np.abs(budget)<=accumulated_allowance[backend]+1e-5)
            diagnostic[backend]=dict(temperature=output['temperature_k'],enthalpy=old[:,:,1].copy(),budget=budget)
            if backend!='numpy64':
                tdiff=np.max(np.abs(output['temperature_k']-diagnostic['numpy64']['temperature']))
                hdiff=np.max(np.abs(old[:,:,1].astype(float)-diagnostic['numpy64']['enthalpy']))
                maxima[backend][:2]=np.maximum(maxima[backend][:2],[tdiff,hdiff]);assert tdiff<.05
            if backend=='cuda32':
                np.testing.assert_allclose(old[:,:,1],states['jax32'][:,:,1],rtol=1e-6,atol=1e-6)
                np.testing.assert_allclose(output['temperature_k'],diagnostic['jax32']['temperature'],rtol=1e-6,atol=1e-6)
                same_max=max(same_max,float(np.max(np.abs(old[:,:,1]-states['jax32'][:,:,1]))))
        if step%48==47:
            daily.append(dict(day=step//48+1,backends={k:dict(temperature_k=v['temperature'].tolist(),energy_residual_jm2=v['budget'].tolist()) for k,v in diagnostic.items()}))
            print('day',step//48+1,'max budget J/m2',{k:float(v[3]) for k,v in maxima.items()},flush=True)
    report=dict(scope=__doc__,steps_per_column=336,columns=3,same_precision_max_enthalpy_difference_jm2=same_max,
        max_iterations=max_evaluations,backends={k:dict(max_temperature_difference_k=v[0],max_enthalpy_difference_jm2=v[1],
            max_step_energy_residual_jm2=v[2],max_cumulative_energy_residual_jm2=v[3],
            final_budget_jm2=diagnostic[k]['budget'].tolist(),gross_input_energy_jm2=gross[k].tolist()) for k,v in maxima.items()},daily=daily)
    report['source_sha256']={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [Path(__file__),ROOT/'tests/test_gpu_column_heat.py']+list((ROOT/'src/noahmp_clm').rglob('*.py'))+list((ROOT/'src/noahmp_clm/gpu/cuda').glob('*.cu'))}
    report['wall_seconds']=time.perf_counter()-start
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps({k:v for k,v in report.items() if k not in ('source_sha256','daily')},indent=2),flush=True)


if __name__=='__main__': main()
