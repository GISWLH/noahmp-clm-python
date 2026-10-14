"""Wet-bulb option in native full columns, retaining all conservation checks."""
import numpy as np
import pytest
from test_gpu_clm_layered_column import cases as base_cases,reference,budgets,science,host,compare_same_precision


def cases(dtype):
    from test_wetbulb import humidity
    arrays,f,dep=base_cases(dtype);select=[0,1,8,15]
    arrays=tuple(a[select].copy() for a in arrays);f=f[select].copy();dep=dep[select].copy()
    arrays[3][:,2]=5
    tc=np.array([1.,3.,8.,-10.]);rh=np.array([.5,.9,.2,.8])
    f[:,1]=273.15+tc;f[:,2]=humidity(tc,rh,f[:,4]);f[:,7]=2e-6
    return arrays,f,dep


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_wetbulb_full_column_native_reference(dtype):
    jax=pytest.importorskip('jax')
    if dtype is np.float64 and not jax.config.x64_enabled:pytest.skip('JAX x64 required')
    from noahmp_clm.gpu.clm_layered_column import step_clm_layered_columns_jax
    arrays,f,dep=cases(dtype)
    got=host(step_clm_layered_columns_jax(*arrays,f,deposition=dep,return_surface=True))
    for k,row in enumerate(zip(*arrays)):
        ref=reference(row,f[k],dep[k])
        one={key:({a:b[k] for a,b in v.items()} if key=='diagnostics' else v[k]) for key,v in got.items()
            if key not in ('surface','canopy_interception','canopy_post_interception')}
        error=science(one,ref,dtype)
        np.testing.assert_allclose(one['condensate'][:,0],ref['condensate'][:,0],rtol=.1,atol=1e-4)
        np.testing.assert_allclose(one['condensate'][:,1],ref['condensate'][:,1],rtol=.1,atol=1800.)
        print('wetbulb full native',str(dtype),k,error,flush=True)
    print('wetbulb full budgets',str(dtype),budgets(arrays,f,dep,got),flush=True)


def test_wetbulb_full_column_native_same_precision():
    cp=pytest.importorskip('cupy');pytest.importorskip('jax')
    from noahmp_clm.gpu.clm_layered_column import step_clm_layered_columns_jax
    from noahmp_clm.gpu.cuda.clm_layered_column import step_clm_layered_columns_cuda
    arrays,f,dep=cases(np.float32)
    a=host(step_clm_layered_columns_cuda(*arrays,f,deposition=dep,return_surface=True),cp.asnumpy)
    b=host(step_clm_layered_columns_jax(*arrays,f,deposition=dep,return_surface=True))
    print('wetbulb full same precision',compare_same_precision(a,b),flush=True)
    print('wetbulb full CUDA budgets',budgets(arrays,f,dep,a),flush=True)
