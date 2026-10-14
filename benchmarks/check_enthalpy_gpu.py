"""Device phase/heat audit; thermal columns only, no canopy or snow hydrology."""
from dataclasses import replace
from pathlib import Path
import argparse,hashlib,importlib.util,json,time
import numpy as np
import cupy as cp
import jax,jax.numpy as jnp
from noahmp_clm.column import ColumnState,default_soil_params
from noahmp_clm.physics.snowpack import Snowpack,TF,LF
from noahmp_clm.physics.soil_enthalpy import SoilEnthalpyState
from noahmp_clm.gpu.enthalpy import run_soil_thermodynamics_jax
from noahmp_clm.gpu.cuda.enthalpy import run_soil_thermodynamics_cuda
from noahmp_clm.gpu.column_heat import pack_joint_heat_case,run_joint_heat_jax
from noahmp_clm.gpu.cuda.column_heat import run_joint_heat_cuda

ROOT=Path(__file__).resolve().parents[1]


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path);module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);return module


def host(result):
    return {k:cp.asnumpy(v) if isinstance(v,cp.ndarray) else np.asarray(v) for k,v in result.items()}


def compare_same(a,b):
    maximum={}
    for key in a:
        np.testing.assert_allclose(a[key],b[key],rtol=1e-6,atol=1e-6)
        maximum[key]=float(np.max(a[key]!=b[key]) if a[key].dtype==bool else np.max(np.abs(a[key]-b[key])))
    return maximum


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path);args=ap.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    assert jax.config.x64_enabled and any(d.platform=='gpu' for d in jax.devices())
    phase=load('phase_reference','tests/test_gpu_enthalpy.py');heat=load('heat_reference','tests/test_gpu_column_heat.py')
    report={'scope':__doc__,'phase':{},'thermal':{}}
    for inverse in (False,True):
        u,o=phase.cases(np.float32,inverse);j=host(run_soil_thermodynamics_jax(u,o,inverse=inverse))
        c=host(run_soil_thermodynamics_cuda(u,o,inverse=inverse));ref=phase.reference(u,o,inverse)
        metrics={}
        for key,value in ref.items():
            abs_error=np.abs(c[key]-value)
            threshold=1e-3 if 'kgm2' in key else 1.
            mask=np.abs(value)>threshold
            metrics[key]={'max_absolute':float(abs_error.max())}
            if 'temperature' not in key:
                metrics[key]['max_relative_percent_away_from_zero']=float(np.max(abs_error[mask]/np.abs(value[mask]),initial=0.)*100.)
                metrics[key]['relative_reference_floor']=threshold
        report['phase']['inverse' if inverse else 'forward']=dict(cases=len(u),same_precision=compare_same(c,j),numpy64=metrics)
    base=heat.cases(np.float64);rows=[tuple(v[i] for v in base) for i in range(len(base[0]))]
    rng=np.random.default_rng(8928)
    for case in range(192):
        ns=case%6;m=10.**rng.uniform(-3.,1.7,ns)
        snow=Snowpack(m,m*rng.uniform(-2100*20,LF*1.2,ns),m/rng.uniform(80.,500.,ns))
        column=ColumnState(soil=default_soil_params(('sand','loam','clay')[(case//2)%3]),
            tsoil=rng.uniform(TF-15.,TF+12.,4),smc=rng.uniform(.05,.35,4))
        column.options=replace(column.options,OptSoilSupercoolWater=1+case%2)
        ground=SoilEnthalpyState.from_column(column)
        rows.append(pack_joint_heat_case(snow,ground,rng.uniform(60.,3600.),rng.uniform(-150.,200.),
            snow_absorbed=rng.uniform(0.,100.,ns),soil_absorbed=rng.uniform(0.,20.,4),
            top_temperature=rng.uniform(TF-25.,TF+20.),top_conductance=rng.uniform(1.,20.)))
    inputs=tuple(np.stack([row[i] for row in rows]) for i in range(4))
    # Repack float32 states through the directed pore-boundary quantization in
    # the input contract, as in normal calls, rather than blindly casting a
    # float64 packed array with independently rounded saturated parameters.
    c,p,o,b=inputs;quant=[v.astype(np.float32) if i!=2 else v for i,v in enumerate(inputs)]
    for row in range(len(c)):
        ns=int(o[row,0]);n=ns+4;pore=1000.*quant[0][row,ns:n,2].astype(float)*float(quant[1][row,0])
        limit=pore.astype(np.float32);limit=np.where(limit.astype(float)>pore,np.nextafter(limit,np.float32(-np.inf)),limit)
        quant[0][row,ns:n,0]=np.minimum(quant[0][row,ns:n,0],limit)
    results={'jax64':host(run_joint_heat_jax(*inputs)), 'jax32':host(run_joint_heat_jax(*quant)),
        'cuda32':host(run_joint_heat_cuda(*quant))}
    report['thermal']['same_precision']=compare_same(results['cuda32'],results['jax32'])
    for backend,result in results.items():
        data=inputs if backend=='jax64' else quant
        errors=np.zeros(5);rel=np.zeros(2);max_iter=0.
        for row in range(len(c)):
            n=int(o[row,0])+4;href,tref,d=heat.numpy_reference(*(v[row] for v in data))
            hh=result['enthalpy_jm2'][row,:n].astype(float);tt=result['temperature_k'][row,:n].astype(float)
            mass_before=data[0][row,:n,0].astype(float)
            budget=(hh-data[0][row,:n,1].astype(float)).sum()-float(data[3][row,0])*(float(result['top_flux_wm2'][row])+data[0][row,:n,4].astype(float).sum())
            change_error=abs(float(result['top_flux_wm2'][row])-d['top_flux_wm2'])
            errors=np.maximum(errors,[np.max(np.abs(tt-tref)),np.max(np.abs(hh-href)),change_error,abs(budget),
                np.max(np.abs(result['liquid_kgm2'][row,:n].astype(float)+result['ice_kgm2'][row,:n].astype(float)-mass_before))])
            for ix,aa,ee,floor in ((0,hh,href,1.),(1,np.array([result['top_flux_wm2'][row]]),np.array([d['top_flux_wm2']]),1.)):
                mask=np.abs(ee)>floor;rel[ix]=max(rel[ix],np.max(np.abs(aa[mask]-ee[mask])/np.abs(ee[mask]),initial=0.)*100.)
            max_iter=max(max_iter,float(result['iterations'][row]))
            np.testing.assert_allclose(tt,tref,rtol=0,atol=.01)
            np.testing.assert_allclose(hh,href,rtol=.1,atol=.1)
            assert abs(budget)<=float(result['layer_tolerance_jm2'][row,:n].sum())+16.*np.finfo(data[0].dtype).eps*max(1.,abs(float(result['input_energy_jm2'][row])))+1e-5
        report['thermal'][backend]=dict(cases=len(c),max_temperature_difference_k=errors[0],max_enthalpy_difference_jm2=errors[1],
            max_top_flux_difference_wm2=errors[2],max_independent_energy_residual_jm2=errors[3],max_phase_mass_roundoff_kgm2=errors[4],
            max_enthalpy_relative_percent_above_1_jm2=rel[0],max_flux_relative_percent_above_1_wm2=rel[1],max_iterations=max_iter)
    report['source_sha256']={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [Path(__file__),ROOT/'tests/test_gpu_column_heat.py',ROOT/'tests/test_gpu_enthalpy.py']+
        list((ROOT/'src/noahmp_clm').rglob('*.py'))+list((ROOT/'src/noahmp_clm/gpu/cuda').glob('*.cu'))}
    report['wall_seconds']=time.perf_counter()-start
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'},indent=2),flush=True)


if __name__=='__main__': main()
