"""Actual stable Qia state: original tolerances, native parity and four budgets."""
import numpy as np
import pytest
from test_most_stagnation import qia_case
from test_gpu_clm_bgc_column import reference,cn_budgets
from test_gpu_clm_layered_column import host,compare_same_precision,budgets
from noahmp_clm.physics.canopy_most import MOSTGeometry


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_qia_stagnation_jax_reference_and_conservation(dtype):
    jax=pytest.importorskip('jax')
    if dtype is np.float64 and not jax.config.x64_enabled:pytest.skip('JAX x64 required')
    from noahmp_clm.gpu.clm_bgc_column import step_clm_bgc_columns_jax
    a,v=qia_case(dtype)
    out=host(step_clm_bgc_columns_jax(*a,v['forcing'],environment=v['environment'],
        calendar=v['calendar'],most_geometry=v['geometry'],return_surface=True))
    ref=reference(tuple(x[0] for x in a),v['forcing'][0],np.zeros(5),v['environment'][0],
        v['calendar'][0],0.,1.,most_geometry=MOSTGeometry(*map(float,v['geometry'][0])))
    errors={}
    for key in ('h','le','gpp_gc_m2','nee_gc_m2'):
        actual=float(out['diagnostics'][key][0]);expected=ref['diagnostics'][key]
        errors[key]=abs(actual-expected)
        assert errors[key]<=max(.01 if key.endswith('gc_m2') else .05,.1*abs(expected))
    most=out['surface']['most']
    assert most['coefficient_residual'][0]<=1e-4 and most['iterations'][0]<=50
    if dtype is np.float32:assert most['bisection_restarts'][0]==1
    print('Qia native reference',str(dtype),errors,most,flush=True)
    print('Qia native M/H',str(dtype),budgets(a[:8],v['forcing'],np.zeros((1,5),dtype),out),flush=True)
    print('Qia native C/N',str(dtype),cn_budgets(a,out,np.zeros(1,dtype)),flush=True)


def test_qia_stagnation_cuda_jax_same_precision():
    cp=pytest.importorskip('cupy');pytest.importorskip('jax')
    from noahmp_clm.gpu.clm_bgc_column import step_clm_bgc_columns_jax
    from noahmp_clm.gpu.cuda.clm_bgc_column import step_clm_bgc_columns_cuda
    a,v=qia_case();kw=dict(environment=v['environment'],calendar=v['calendar'],
        most_geometry=v['geometry'],return_surface=True)
    left=host(step_clm_bgc_columns_cuda(*a,v['forcing'],**kw),cp.asnumpy)
    right=host(step_clm_bgc_columns_jax(*a,v['forcing'],**kw))
    assert left['surface']['most']['bisection_restarts'][0]==1
    print('Qia CUDA/JAX same precision',compare_same_precision(left,right),flush=True)
    print('Qia CUDA M/H',budgets(a[:8],v['forcing'],np.zeros((1,5),np.float32),left),flush=True)
    print('Qia CUDA C/N',cn_budgets(a,left,np.zeros(1,np.float32)),flush=True)
