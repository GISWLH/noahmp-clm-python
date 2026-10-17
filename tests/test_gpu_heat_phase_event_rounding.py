"""Native phase-event progress with unchanged final residual/energy bounds."""
import json
import numpy as np
import pytest
from test_heat_phase_event_rounding import saved

def check(inputs,got,reference,label):
    cells,props,options,boundary=inputs
    h=np.asarray(got['enthalpy_jm2']);temperature=np.asarray(got['temperature_k'])
    residual=np.asarray(got['layer_residual_jm2']);tolerance=np.asarray(got['layer_tolerance_jm2'])
    assert np.all(got['converged'])
    norm=float(np.max(abs(residual)/tolerance));assert norm<=1.
    differences=[];energy=[];bounds=[]
    for k,o in enumerate(options):
        n=int(o[0])+4
        np.testing.assert_allclose(h[k,:n],reference['reference_enthalpy'][k],rtol=.1,atol=.1)
        differences.append(float(np.max(abs(temperature[k,:n]-reference['reference_temperature'][k]))))
        assert differences[-1]<=.5
        incoming=float(boundary[k,0])*(float(got['top_flux_wm2'][k])+float(cells[k,:n,4].astype(float).sum()))
        energy.append(float((h[k,:n].astype(float)-cells[k,:n,1].astype(float)).sum()-incoming))
        bounds.append(float(tolerance[k,:n].astype(float).sum())+16*np.finfo(cells.dtype).eps*max(abs(incoming),1.)+1e-5)
        assert abs(energy[-1])<=bounds[-1]
    print('Qia heat phase events',json.dumps(dict(backend=label,norm=norm,maximum_temperature_difference=max(differences),
        energy_residual=energy,energy_bound=[float(v) for v in bounds],iterations=np.asarray(got['iterations']).tolist())),flush=True)

@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_qia_phase_event_jax_reference_and_unchanged_bounds(dtype):
    jax=pytest.importorskip('jax')
    if dtype is np.float64 and not jax.config.x64_enabled:pytest.skip('JAX x64 required')
    from noahmp_clm.gpu.column_heat import run_joint_heat_jax
    inputs,reference=saved(dtype);got=run_joint_heat_jax(*inputs)
    check(inputs,{k:np.asarray(v) for k,v in got.items()},reference,'jax'+str(np.dtype(dtype).itemsize*8))

def test_qia_phase_event_cuda_jax_all_fields_same_precision():
    cp=pytest.importorskip('cupy');pytest.importorskip('jax')
    from noahmp_clm.gpu.column_heat import run_joint_heat_jax
    from noahmp_clm.gpu.cuda.column_heat import run_joint_heat_cuda
    inputs,reference=saved();a=run_joint_heat_cuda(*inputs);b=run_joint_heat_jax(*inputs);maximum=0.
    for key,value in a.items():
        left,right=cp.asnumpy(value),np.asarray(b[key])
        if left.dtype.kind in 'iub':np.testing.assert_array_equal(left,right)
        else:
            np.testing.assert_allclose(left,right,rtol=1e-6,atol=1e-6,err_msg=key)
            maximum=max(maximum,float(abs(left.astype(float)-right.astype(float)).max()))
    check(inputs,{k:cp.asnumpy(v) for k,v in a.items()},reference,'cuda32')
    print('Qia phase event all fields same precision',maximum,flush=True)
