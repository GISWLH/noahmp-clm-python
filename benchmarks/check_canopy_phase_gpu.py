"""Synthetic native phase-canopy science, precision and full parcel ledgers."""
import argparse,hashlib,json,math,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tests'))
from test_canopy_phase_energy import cases
from check_canopy_phase_energy_reference import random_cases
from test_gpu_canopy_phase_energy import reference,host
from test_gpu_snicar import assert_science_close
from noahmp_clm.gpu.canopy_phase_energy import pack_canopy_phase,run_canopy_phase_jax
from noahmp_clm.physics.canopy_energy import CanopyExchange
from noahmp_clm.physics.canopy_interception import CanopyCondensate,intercept_precipitation
from noahmp_clm.physics.canopy_phase_energy import solve_canopy_phase_energy
from noahmp_clm.physics.soil_uptake import SoilWaterSupply


def packed(rows,dtype):return tuple(np.stack([row[k] for row in rows]).astype(np.int32 if k==8 else dtype) for k in range(9))


def same_precision(a,b):
    maximum={};failures=[]
    def compare(x,y,path=''):
        for k in x:
            name=path+k
            if isinstance(x[k],dict):compare(x[k],y[k],name+'.');continue
            maximum[name]=float(np.max(abs(np.asarray(x[k],float)-np.asarray(y[k],float))))
            good=np.array_equal(x[k],y[k]) if x[k].dtype.kind in 'ibu' else np.allclose(x[k],y[k],rtol=1e-6,atol=1e-6)
            if not good:failures.append(name)
    compare(a,b);return dict(max_absolute=maximum,failures=failures)


def science(got,refs):
    errors={}
    for key in ('leaf_temperature','ground_temperature','canopy_temperature','canopy_humidity','h','le','gpp','ground_heat','wet_evaporation','transpiration','ground_evaporation','root_uptake'):
        actual=np.asarray(got[key],float);expected=np.array([r[key] for r in refs]);error=actual-expected
        tiny=key in ('wet_evaporation','transpiration','ground_evaporation','root_uptake','canopy_humidity');floor=1e-6 if tiny else .1 if key=='gpp' else 10.;absolute=1e-7 if tiny else .01 if key=='gpp' else 1.
        mask=abs(expected)>floor
        errors[key]=dict(max_absolute=float(np.max(abs(error))),nonzero_n=int(mask.sum()),zero_floor=floor,
            max_relative_percent=None if 'temperature' in key or not np.any(mask) else float(np.max(100.*abs(error[mask]/expected[mask]))),
            rmse=float(np.sqrt(np.mean(error**2))),mean_bias=float(error.mean()))
        if 'temperature' in key:np.testing.assert_allclose(actual,expected,rtol=0.,atol=.05)
        else:assert_science_close(actual,expected,relative=.1,zero_floor=floor,absolute=absolute)
    for key in ('mass','enthalpy'):
        actual=np.asarray(got[key],float);expected=np.asarray([getattr(r['condensate'],key) for r in refs]);errors[key]=dict(max_absolute=float(np.max(abs(actual-expected))))
        assert_science_close(actual,expected,relative=.1,zero_floor=.001 if key=='mass' else 100.,absolute=1e-4 if key=='mass' else 10.)
    return errors


def ledgers(got,arrays):
    a,u,f,soil,stem,c,b,snow,n=arrays;old=c.astype(float);dt=b[:,4].astype(float);eps=np.finfo(c.dtype).eps
    water=np.asarray(got['mass'],float)-old[:,:,0]+dt[:,None]*np.asarray(got['wet_evaporation'],float)
    heat=np.asarray(got['enthalpy'],float)-old[:,:,1]+np.asarray(got['vapor_enthalpy'],float)-(np.asarray(got['thermal_enthalpy'],float)-old[:,:,1])
    thermal_magnitude=abs(np.asarray(got['enthalpy'],float))+abs(old[:,:,1])+abs(np.asarray(got['vapor_enthalpy'],float))+abs(np.asarray(got['thermal_enthalpy'],float))
    assert np.all(abs(water)<=64.*eps*np.maximum(old[:,:,0],1.)+1e-12)
    assert np.all(abs(heat)<=64.*eps*np.maximum(thermal_magnitude,1.)+1e-6)
    root=np.sum(np.asarray(got['root_uptake'],float)*b[:,5:9].astype(float),axis=1)
    trheat=np.sum(np.asarray(got['transpiration'],float)*(334000.+4180.*(np.asarray(got['leaf_temperature'],float)-273.15)),axis=1)-root
    storage=np.sum(np.asarray(got['enthalpy'],float)-old[:,:,1],axis=1)/dt
    boundary=u[:,11].astype(float)+u[:,12].astype(float)-np.asarray(got['outgoing_shortwave'],float)-np.asarray(got['outgoing_longwave'],float)-np.asarray(got['h'],float)-np.asarray(got['le'],float)-np.asarray(got['ground_heat'],float)
    residual=dt*(boundary-storage-np.sum(np.asarray(got['vapor_enthalpy'],float),axis=1)/dt-trheat)
    tol=.005 if c.dtype==np.float32 else 5e-5
    scale=abs(u[:,11].astype(float))+abs(u[:,12].astype(float))+abs(np.asarray(got['le'],float))+abs(np.asarray(got['h'],float))+abs(boundary)+abs(storage)+abs(trheat)
    bound=dt*(10.*tol+128.*eps*np.maximum(scale,1.))+1e-6
    assert np.all(abs(residual)<=bound),(residual,bound)
    return dict(max_layer_water_kgm2=float(np.max(abs(water))),max_layer_parcel_heat_jm2=float(np.max(abs(heat))),
        max_total_heat_jm2=float(np.max(abs(residual))),max_total_heat_bound_jm2=float(np.max(bound)),max_layer_residual_wm2=float(np.max(abs(got['layer_residual']))))


def main():
    import cupy as cp
    from noahmp_clm.gpu.cuda.canopy_phase_energy import run_canopy_phase_cuda
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--random-count',type=int,default=60);args=parser.parse_args()
    if args.output.exists():parser.error('Use a fresh result path')
    started=time.monotonic();report=dict(kind='native synthetic equations, not observations',backends={},failures=[])
    raw=[*cases(),*random_cases(args.random_count)];report['cases']=len(raw)
    report['sources']={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'src/noahmp_clm/gpu/canopy_phase_energy.py',ROOT/'src/noahmp_clm/gpu/cuda/canopy_phase_energy_kernel.cu',ROOT/'src/noahmp_clm/gpu/cuda/canopy_phase_energy.py',Path(__file__)]}
    def save():args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    for dtype in (np.float64,np.float32):
        arrays=packed([pack_canopy_phase(a,sw,lw,e,c,1800.,dtype=dtype,**kw) for _,a,sw,lw,e,c,kw in raw],dtype)
        refs=[reference(row) for row in zip(*arrays)]
        for name,run in ([('jax64',run_canopy_phase_jax)] if dtype==np.float64 else [('jax32',run_canopy_phase_jax),('cuda32',run_canopy_phase_cuda)]):
            got=host(run(*arrays,raise_on_failure=False),lambda x:cp.asnumpy(x) if isinstance(x,cp.ndarray) else np.asarray(x))
            failed=np.where(~got['converged'])[0];result=dict(converged=int(got['converged'].sum()),max_iterations=int(got['iterations'].max()),
                failed_cases=[dict(index=int(k),name=raw[k][0],iterations=int(got['iterations'][k]),residual=float(got['max_solver_residual'][k])) for k in failed])
            report['backends'][name]=result;save()
            if len(failed):report['failures'].append(name+' convergence');save()
            else:
                try:result.update(errors=science(got,refs),budgets=ledgers(got,arrays))
                except AssertionError as error:report['failures'].append(name+': '+str(error));save()
            if name=='jax32':j32=got
            if name=='cuda32':report['same_precision']=same_precision(got,j32)
            print(name,json.dumps(result),flush=True);save()
    report['seconds']=time.monotonic()-started;save()
    if report['failures'] or report['same_precision']['failures']:raise RuntimeError('Native audit failed; full report retained')


if __name__=='__main__':main()
