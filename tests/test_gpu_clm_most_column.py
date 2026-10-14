"""Device MOST feedback in full columns, with independent extensive budgets."""
from dataclasses import astuple
import numpy as np
import pytest
from noahmp_clm.physics.canopy_most import MOSTGeometry
from test_gpu_clm_layered_column import cases as base_cases,reference,budgets,science,host,compare_same_precision


def cases(dtype):
    arrays,f,dep=base_cases(dtype);select=[0,1,2,4,6,8,9,15,0]
    arrays=tuple(a[select].copy() for a in arrays);f=f[select].copy();dep=dep[select].copy()
    arrays[3][:,5]=1;f[-1,3]=0.
    from test_clm_most_column import stable_trajectory_case
    from noahmp_clm.gpu.clm_layered_column import pack_clm_layered_columns
    old,forcing=stable_trajectory_case()
    extra=pack_clm_layered_columns([old],dtype=dtype,cosine_zenith=.05)
    arrays=tuple(np.concatenate((a,b)) for a,b in zip(arrays,extra))
    f=np.concatenate((f,np.array([astuple(forcing)],dtype)))
    dep=np.concatenate((dep,np.array([[1.,2.,1.,.5,.25]],dtype)*1e-12))
    geometry=np.broadcast_to(np.array(astuple(MOSTGeometry(wind_height=15.,temperature_height=12.,humidity_height=10.,displacement=2.)),dtype),(len(f),9)).copy()
    return arrays,f,dep,geometry


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_full_most_column_reference(dtype):
    jax=pytest.importorskip('jax')
    if dtype is np.float64 and not jax.config.x64_enabled:pytest.skip('JAX x64 required')
    from noahmp_clm.gpu.clm_layered_column import step_clm_layered_columns_jax
    arrays,f,dep,geometry=cases(dtype)
    got=host(step_clm_layered_columns_jax(*arrays,f,deposition=dep,return_surface=True,most_geometry=geometry))
    for k,row in enumerate(zip(*arrays)):
        ref=reference(row,f[k],dep[k],most_geometry=MOSTGeometry(*geometry[k].astype(float)))
        one={key:({a:b[k] for a,b in v.items()} if key=='diagnostics' else v[k]) for key,v in got.items()
            if key not in ('surface','canopy_interception','canopy_post_interception','most')}
        error=science(one,ref,dtype)
        np.testing.assert_allclose(one['condensate'][:,0],ref['condensate'][:,0],rtol=.1,atol=1e-4)
        np.testing.assert_allclose(one['condensate'][:,1],ref['condensate'][:,1],rtol=.1,atol=1800.)
        error['canopy_layer_mass']=float(abs(one['condensate'][:,0]-ref['condensate'][:,0]).max())
        error['canopy_layer_enthalpy']=float(abs(one['condensate'][:,1]-ref['condensate'][:,1]).max())
        error['most_iterations']=int(got['most']['iterations'][k]);error['reference_iterations']=ref['diagnostics']['most']['iterations']
        print('MOST full native',str(dtype),k,error,flush=True)
    assert np.all(got['most']['coefficient_residual']<=1e-4)
    print('MOST full budgets',str(dtype),budgets(arrays,f,dep,got),flush=True)


def test_full_most_column_same_precision():
    cp=pytest.importorskip('cupy');pytest.importorskip('jax')
    from noahmp_clm.gpu.clm_layered_column import step_clm_layered_columns_jax
    from noahmp_clm.gpu.cuda.clm_layered_column import step_clm_layered_columns_cuda
    arrays,f,dep,geometry=cases(np.float32)
    a=host(step_clm_layered_columns_cuda(*arrays,f,deposition=dep,return_surface=True,most_geometry=geometry),cp.asnumpy)
    b=host(step_clm_layered_columns_jax(*arrays,f,deposition=dep,return_surface=True,most_geometry=geometry))
    print('MOST full same precision',compare_same_precision(a,b),flush=True)
    print('MOST full CUDA budgets',budgets(arrays,f,dep,a),flush=True)
