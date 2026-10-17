"""Preserve original convergence and precision gates on an actual narrow bracket."""
import numpy as np
import pytest
from test_surface_secant_rounding import saved_case,numpy_reference
from test_gpu_canopy_phase_surface import budgets
from noahmp_clm.validation import compare_series

def test_actual_cng_secant_rounding_same_precision_and_budgets():
    cp=pytest.importorskip('cupy');pytest.importorskip('jax')
    from noahmp_clm.gpu.cuda.canopy_phase_surface import run_canopy_phase_surface_cuda
    from noahmp_clm.gpu.canopy_phase_surface import run_canopy_phase_surface_jax
    arrays,extension,vic=saved_case();kwargs=dict(exchange_extension=extension,vic=vic)
    cuda=run_canopy_phase_surface_cuda(*arrays,**kwargs)
    jax=run_canopy_phase_surface_jax(*arrays,**kwargs)
    maximum=0.
    for key in jax:
        for field in (jax[key] if isinstance(jax[key],dict) else [None]):
            aa,bb=(cuda[key][field],jax[key][field]) if field else (cuda[key],jax[key])
            a,b=cp.asnumpy(aa),np.asarray(bb)
            if a.dtype.kind in 'iub':np.testing.assert_array_equal(a,b)
            else:
                np.testing.assert_allclose(a,b,rtol=1e-6,atol=1e-6,err_msg=str((key,field)))
                maximum=max(maximum,float(np.max(abs(a.astype(float)-b.astype(float)),initial=0.)))
    _,_,_,ref=numpy_reference(arrays,extension,vic)
    flat={k:cp.asnumpy(v) for k,v in cuda.items() if not isinstance(v,dict)}
    assert np.all(flat['converged']) and np.all(flat['surface_evaluations']<60)
    tol=2*np.finfo(np.float32).eps*273.15
    width=np.diff(flat['surface_root_bracket_k'],axis=1)[:,0]
    assert np.all((abs(flat['surface_temperature_residual'])<=tol)|(width<=tol))
    for key,var in [('h','h_wm2'),('le','le_wm2')]:
        assert compare_series(flat[key],[ref[key]],var)['bias_pass']
    assert abs(float(flat['surface_temperature'][0])-ref['surface_temperature'])<=.5
    ledger=budgets(arrays,flat)
    print('Actual Cng surface',dict(same_precision_max_absolute=maximum,
        surface_evaluations=flat['surface_evaluations'].tolist(),bracket_width=width.tolist(),
        temperature_residual=flat['surface_temperature_residual'].tolist(),
        h_absolute_difference=abs(float(flat['h'][0])-ref['h']),
        le_absolute_difference=abs(float(flat['le'][0])-ref['le']),budgets=ledger),flush=True)
