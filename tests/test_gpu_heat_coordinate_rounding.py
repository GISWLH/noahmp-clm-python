"""Preserve every original thermal bound while closing a saved local equation."""
import json
import numpy as np
import pytest
from test_heat_coordinate_rounding import saved

def check(inputs,got,expected,label):
    c,p,o,b=inputs;n=int(o[0,0])+4
    h=np.asarray(got['enthalpy_jm2']);temperature=np.asarray(got['temperature_k'])
    residual=np.asarray(got['layer_residual_jm2']);tolerance=np.asarray(got['layer_tolerance_jm2'])
    assert np.all(got['converged'])
    norm=float(np.max(abs(residual)/tolerance));assert norm<=1.
    np.testing.assert_allclose(h[0,:n],expected['enthalpy_jm2'],rtol=.1,atol=.1)
    delta_t=float(np.max(abs(temperature[0,:n]-expected['temperature_k'])));assert delta_t<=.5
    incoming=float(b[0,0])*(float(got['top_flux_wm2'][0])+float(c[0,:n,4].astype(float).sum()))
    energy_residual=float((h[0,:n].astype(float)-c[0,:n,1].astype(float)).sum()-incoming)
    bound=float(tolerance[0,:n].astype(float).sum())+16*np.finfo(c.dtype).eps*max(abs(incoming),1.)+1e-5
    assert abs(energy_residual)<=bound
    print('Qia coordinate heat',json.dumps(dict(backend=label,norm=norm,maximum_temperature_difference=delta_t,
        maximum_enthalpy_difference=float(np.max(abs(h[0,:n]-expected['enthalpy_jm2']))),
        independent_energy_residual=energy_residual,independent_energy_bound=float(bound),
        iterations=np.asarray(got['iterations']).tolist(),layer_residual=residual.tolist(),layer_tolerance=tolerance.tolist())),flush=True)

@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_qia_coordinate_jax_reference_and_original_bounds(dtype):
    jax=pytest.importorskip('jax')
    if dtype is np.float64 and not jax.config.x64_enabled:pytest.skip('JAX x64 required')
    from noahmp_clm.gpu.column_heat import run_joint_heat_jax
    inputs,reference=saved(dtype);got=run_joint_heat_jax(*inputs)
    check(inputs,{k:np.asarray(v) for k,v in got.items()},reference,'jax'+str(np.dtype(dtype).itemsize*8))

def test_qia_coordinate_cuda_jax_same_precision():
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
    print('Qia heat all fields same precision',maximum,flush=True)
