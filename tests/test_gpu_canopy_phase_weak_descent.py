"""Saved actual surface input: native reference, original budgets and all-field parity."""
import numpy as np
import pytest
from test_canopy_phase_weak_descent import load_case
from test_gpu_canopy_phase_surface import budgets
from test_gpu_snicar import assert_science_close


def run(backend):
    pytest.importorskip('jax');cp=pytest.importorskip('cupy');arrays,controls,expected=load_case()
    if backend=='cuda32':
        from noahmp_clm.gpu.cuda.canopy_phase_surface import run_canopy_phase_surface_cuda as solve
    else:
        from noahmp_clm.gpu.canopy_phase_surface import run_canopy_phase_surface_jax as solve
    result=solve(*arrays,**controls)
    def host(v):return {k:host(x) for k,x in v.items()} if isinstance(v,dict) else cp.asnumpy(v) if isinstance(v,cp.ndarray) else np.asarray(v)
    return arrays,host(result),expected


@pytest.mark.parametrize('backend',['jax32','cuda32'])
def test_actual_weak_phase_descent_native_reference(backend):
    arrays,got,expected=run(backend);assert np.all(got['converged']) and np.all(got['canopy_converged'])
    maximum={}
    for key in ('leaf_temperature','ground_temperature','canopy_temperature','surface_temperature'):
        maximum[key]=float(np.max(abs(got[key][0]-expected[key])))
        np.testing.assert_allclose(got[key][0],expected[key],rtol=0.,atol=.05)
    for key in ('h','le','ground_heat','gpp','et_kgm2','runoff_kgm2','snow_drainage_kgm2','canopy_humidity'):
        floor,absolute=(1e-6,1e-7) if key=='canopy_humidity' else (.001,.0001) if 'kgm2' in key else (.1,.01) if key=='gpp' else (10.,1.)
        assert_science_close(got[key][0],expected[key],relative=.1,zero_floor=floor,absolute=absolute)
        maximum[key]=float(np.max(abs(got[key][0]-expected[key])))
    for key in ('soil_cells','snow_cells','condensate'):
        a,b=got[key][0],expected[key]
        for k in range(a.shape[1]):
            floor,absolute=(18000.,1800.) if k==1 else (1.,.1) if k==3 else (1e-10,1e-11) if k>=4 else (.001,.0001)
            assert_science_close(a[:,k],b[:,k],relative=.1,zero_floor=floor,absolute=absolute)
    np.testing.assert_array_equal(got['snow_counts'][0],expected['snow_counts'])
    properties=arrays[2][0,[0,1,6]]
    inputs=np.column_stack((got['soil_cells'][0],np.broadcast_to(properties,(4,3)))).astype(np.float32)
    options=np.full(4,arrays[3][0,1],np.int32)
    if backend=='cuda32':
        import cupy as cp
        from noahmp_clm.gpu.cuda.enthalpy import run_soil_thermodynamics_cuda
        temperature=cp.asnumpy(run_soil_thermodynamics_cuda(inputs,options,inverse=True)['temperature_k'])
    else:
        from noahmp_clm.gpu.enthalpy import run_soil_thermodynamics_jax
        temperature=np.asarray(run_soil_thermodynamics_jax(inputs,options,inverse=True)['temperature_k'])
    np.testing.assert_allclose(temperature,expected['soil_temperature'],rtol=0.,atol=.05)
    maximum['soil_temperature']=float(np.max(abs(temperature-expected['soil_temperature'])))
    print('actual phase edge native',backend,got['iterations'].tolist(),got['restarts'].tolist(),
        got['max_solver_residual'].tolist(),maximum,'original budgets',budgets(arrays,got),flush=True)


def test_actual_weak_phase_descent_all_raw_fields_same_precision():
    from run_fluxnet_native import compare_same_precision
    _,cuda,_=run('cuda32');_,jax,_=run('jax32')
    maximum=compare_same_precision(cuda,jax)
    print('actual phase edge all raw fields same precision max',maximum,flush=True)
