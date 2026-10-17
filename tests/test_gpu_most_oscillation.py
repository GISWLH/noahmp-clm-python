"""Original I4016 unstable feedback: reference, same precision and four budgets."""
import numpy as np
import pytest
from test_most_oscillation import recorded_case
from test_gpu_clm_bgc_column import reference,cn_budgets
from test_gpu_clm_layered_column import host,compare_same_precision,budgets
from noahmp_clm.physics.canopy_most import MOSTGeometry


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_i4016_jax_reference_and_original_convergence_bounds(dtype):
    jax=pytest.importorskip('jax')
    if dtype is np.float64 and not jax.config.x64_enabled:pytest.skip('JAX x64 required')
    from noahmp_clm.gpu.clm_bgc_column import step_clm_bgc_columns_jax
    a,v=recorded_case(dtype)
    got=host(step_clm_bgc_columns_jax(*a,v['forcing'],environment=v['environment'],calendar=v['calendar'],
        nitrogen_deposition_gn_m2=v['nitrogen'],deposition=v['deposition'],most_geometry=v['geometry'],return_surface=True))
    maximum={}
    for k,row in enumerate(zip(*a)):
        ref=reference(row,v['forcing'][k],v['deposition'][k],v['environment'][k],v['calendar'][k],v['nitrogen'][k],1.,most_geometry=MOSTGeometry(*map(float,v['geometry'][k])))
        for key in ['h','le','gpp_gc_m2','nee_gc_m2']:
            expected=ref['diagnostics'][key];error=abs(float(got['diagnostics'][key][k])-expected)
            assert error<=max(.0001 if key.endswith('gc_m2') else .05,.1*abs(expected))
            maximum[key]=max(maximum.get(key,0.),error)
        assert abs(got['soil_cells'][k,:,0]-ref['soil_cells'][:,0]).max()<.01
    m=got['surface']['most']
    assert np.all(m['iterations']<=50) and np.all(m['coefficient_residual']<=1e-4) and np.all(m['heat_change_wm2']<=.05) and np.all(m['vapor_change_kgm2s']<=2e-8) and np.all(m['leaf_temperature_change_k']<=.01)
    print('I4016 JAX NumPy differences',str(dtype),maximum,flush=True)
    print('I4016 JAX M/H',str(dtype),budgets(a[:8],v['forcing'],v['deposition'],got),flush=True)
    print('I4016 JAX C/N',str(dtype),cn_budgets(a,got,v['nitrogen']),flush=True)


def test_i4016_cuda_jax_all_fields_same_precision_and_original_budgets():
    cp=pytest.importorskip('cupy');pytest.importorskip('jax')
    from noahmp_clm.gpu.cuda.clm_bgc_column import step_clm_bgc_columns_cuda
    from noahmp_clm.gpu.clm_bgc_column import step_clm_bgc_columns_jax
    a,v=recorded_case();kwargs=dict(environment=v['environment'],calendar=v['calendar'],nitrogen_deposition_gn_m2=v['nitrogen'],deposition=v['deposition'],most_geometry=v['geometry'],return_surface=True)
    cuda=host(step_clm_bgc_columns_cuda(*a,v['forcing'],**kwargs),cp.asnumpy);jax=host(step_clm_bgc_columns_jax(*a,v['forcing'],**kwargs))
    print('I4016 CUDA/JAX all fields',compare_same_precision(cuda,jax),flush=True)
    print('I4016 CUDA M/H',budgets(a[:8],v['forcing'],v['deposition'],cuda),flush=True)
    print('I4016 CUDA C/N',cn_budgets(a,cuda,v['nitrogen']),flush=True)
