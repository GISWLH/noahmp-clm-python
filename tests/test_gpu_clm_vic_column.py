"""VIC through precipitation, layered snowmelt, soil admission and bottom flow."""
import numpy as np
import pytest
from test_gpu_clm_layered_column import cases as base_cases,reference,budgets,science,host,compare_same_precision,validate_layered_column
from noahmp_clm.physics.vic import VICParameters
from noahmp_clm.gpu.vic import pack_vic,column_vic


def cases(dtype):
    arrays,f,dep=base_cases(dtype);select=[0,1,4,8,9,11,14,15]
    arrays=tuple(a[select].copy() for a in arrays);f=f[select].copy();dep=dep[select].copy()
    arrays[3][:,6:8]=np.array([(6,6),(6,3),(3,6),(6,6),(6,6),(6,6),(6,6),(3,3)])
    f[:,7]=8e-6
    parameters=np.stack([pack_vic(VICParameters(infiltration_shape=(0.,.2,.4)[k%3],
        maximum_baseflow_mm_day=(0.,10.,80.)[k%3],residual_pore_fraction=.05),dtype=dtype)[2:] for k in range(len(f))])
    return arrays,f,dep,parameters


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_vic_full_native_reference_and_conservation(dtype):
    jax=pytest.importorskip('jax')
    if dtype is np.float64 and not jax.config.x64_enabled:pytest.skip('JAX x64 required')
    from noahmp_clm.gpu.clm_layered_column import step_clm_layered_columns_jax
    arrays,f,dep,parameters=cases(dtype)
    got=host(step_clm_layered_columns_jax(*arrays,f,deposition=dep,vic_parameters=parameters,return_surface=True))
    for k,row in enumerate(zip(*arrays)):
        param=VICParameters(*parameters[k].astype(float)) if np.any(row[3][6:8]==6) else None
        ref=reference(row,f[k],dep[k],vic_parameters=param)
        one={key:({a:b[k] for a,b in v.items()} if key=='diagnostics' else v[k]) for key,v in got.items()
             if key not in ('surface','canopy_interception','canopy_post_interception')}
        err=science(one,ref,dtype)
        np.testing.assert_allclose(one['condensate'][:,0],ref['condensate'][:,0],rtol=.1,atol=1e-4)
        np.testing.assert_allclose(one['condensate'][:,1],ref['condensate'][:,1],rtol=.1,atol=1800.)
        print('VIC full native',str(dtype),k,err,flush=True)
    print('VIC full budgets',str(dtype),budgets(arrays,f,dep,got),flush=True)


def test_vic_full_native_same_precision():
    cp=pytest.importorskip('cupy');pytest.importorskip('jax')
    from noahmp_clm.gpu.clm_layered_column import step_clm_layered_columns_jax
    from noahmp_clm.gpu.cuda.clm_layered_column import step_clm_layered_columns_cuda
    arrays,f,dep,parameters=cases(np.float32)
    a=host(step_clm_layered_columns_cuda(*arrays,f,deposition=dep,vic_parameters=parameters,return_surface=True),cp.asnumpy)
    b=host(step_clm_layered_columns_jax(*arrays,f,deposition=dep,vic_parameters=parameters,return_surface=True))
    print('VIC full same precision',compare_same_precision(a,b),flush=True)
    print('VIC full CUDA budgets',budgets(arrays,f,dep,a),flush=True)


def test_vic_full_configuration_contract():
    arrays,f,dep,parameters=cases(np.float64)
    validate_layered_column(arrays,f,dep,np)
    config=column_vic(arrays[3],arrays[0].dtype,np,parameters)
    np.testing.assert_array_equal(config[:,:2],arrays[3][:,6:8]==6)
    np.testing.assert_array_equal(config[:,2:],parameters)
    bad=[a.copy() for a in arrays];bad[3][0,7]=1
    with pytest.raises(NotImplementedError):validate_layered_column(bad,f,dep,np)
    bad[3][0,7]=5
    with pytest.raises(NotImplementedError):validate_layered_column(bad,f,dep,np)
    with pytest.raises(ValueError):column_vic(arrays[3],arrays[0].dtype,np,parameters[:1])
