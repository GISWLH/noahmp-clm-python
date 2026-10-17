"""A representationally exhausted bracket is not an accepted fixed point."""
from pathlib import Path
import sys
import numpy as np
import pytest
from clm_soil_oracle import fixture_suction


def qia_case(dtype=np.float32):
    with np.load(Path(__file__).parent/'fixtures/qia_most_float32_stagnation.npz') as z:
        values={k:(v.astype(dtype) if v.dtype.kind=='f' else v.copy()) for k,v in z.items()}
    if dtype is np.float64:
        # This is a float64 interpretation, not a bitwise float32 checkpoint.
        # Normalize rounded probability vectors and rebuild derived aliases
        # at destination precision. All authoritative M/H/C/N stores stay
        # exactly equal to their recorded float32 values.
        values['a6']/=values['a6'].sum(1,keepdims=True)
        p=values['a12']
        p[:,:3]/=p[:,:3].sum(1,keepdims=True)
        p[:,9:12]/=p[:,9:12].sum(1,keepdims=True)
        plant,site=values['a11'],values['a4']
        site[:,0]=plant[:,0]*p[:,19]/site[:,1]
        site[:,11]=np.minimum(1.,plant[:,3]*p[:,3]/plant[:,0])
        values['a5'][:,:2]=values['a7'].sum(1)
    values['a2']=fixture_suction(values['a2'],.355)
    return tuple(values[f'a{k}'] for k in range(16)), values


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_fixture_precision_has_valid_parameters_without_changing_stores(dtype):
    from noahmp_clm.gpu.clm_bgc_column import validate_bgc_column
    a,v=qia_case(dtype)
    validate_bgc_column(a,v['forcing'],v['environment'],v['calendar'],np.zeros(1,dtype),np.ones(1,dtype),np)
    with np.load(Path(__file__).parent/'fixtures/qia_most_float32_stagnation.npz') as z:
        for k in (0,1,7,11,13):np.testing.assert_array_equal(a[k],z[f'a{k}'].astype(dtype))
        if dtype is np.float32:
            for k in range(16):np.testing.assert_array_equal(a[k][:,:6] if k==2 else a[k],z[f'a{k}'])
            np.testing.assert_array_equal(a[2][:,6],np.full(len(a[2]),.355,dtype))


@pytest.mark.parametrize('has_root',[True,False])
def test_native_schedule_recovers_only_if_original_residual_converges(monkeypatch,has_root):
    # Execute the scheduling arithmetic with float32 host arrays, and a
    # manufactured discontinuous inner map. This is not a GPU physics test.
    monkeypatch.setitem(sys.modules,'cupy',np)
    import noahmp_clm.gpu.cuda.canopy_most as coefficients_module
    import noahmp_clm.gpu.cuda.canopy_phase_surface as surface_module
    from noahmp_clm.gpu.canopy_most_surface import run_most_surface
    names=('friction_velocity','heat_velocity','water_velocity','leaf_velocity','ground_velocity')
    def coefficients(geometry,atmosphere,state):
        zeta=state[:,0]
        target=zeta+np.where(zeta<np.float32(1.875),np.float32(.01),np.float32(-.01))
        if has_root:
            target=np.where(abs(zeta-np.float32(1.87))<np.float32(1e-6),np.float32(1.87),target)
        return dict(**{k:np.float32(1)/(np.float32(1)+zeta) for k in names},
            zeta=target,raw_zeta=target,effective_wind=np.ones(len(zeta),np.float32),
            potential_temperature=atmosphere[:,0],buoyancy=np.zeros(len(zeta),np.float32),
            convective_wind=np.zeros(len(zeta),np.float32))
    def surface(*args,**kwargs):
        return dict(h=np.zeros(1,np.float32),et_kgm2=np.zeros(1,np.float32),
                    leaf_temperature=np.zeros((1,10),np.float32))
    monkeypatch.setattr(coefficients_module,'run_canopy_most_cuda',coefficients)
    monkeypatch.setattr(surface_module,'run_canopy_phase_surface_cuda',surface)
    arrays=[np.zeros((1,1),np.float32) for _ in range(9)]
    arrays[5]=np.array([[280.,.005,100000.]+[1.]*20],np.float32);arrays[7][:]=1800.
    def run():
        return run_most_surface(arrays,np.zeros((1,9),np.float32),np.ones(1,np.float32),.04,3.,backend='cuda')
    if not has_root:
        with pytest.raises(RuntimeError,match='50 iterations'):run()
    else:
        most=run()['most']
        assert most['bisection_restarts'][0]==1 and most['iterations'][0]==27
        assert most['coefficient_residual'][0]==0.
        assert most['used_zeta'][0]==np.float32(1.87)


def test_failed_float32_state_has_a_converged_numpy_reference():
    from test_gpu_clm_bgc_column import reference
    from noahmp_clm.physics.canopy_most import MOSTGeometry
    a,v=qia_case()
    ref=reference(tuple(x[0] for x in a),v['forcing'][0],np.zeros(5),v['environment'][0],
        v['calendar'][0],0.,1.,most_geometry=MOSTGeometry(*map(float,v['geometry'][0])))
    d=ref['diagnostics'];m=d['most']
    assert m['coefficient_residual']<=1e-4 and m['iterations']<=50
    assert abs(d['h']-(-.488049424337917))<1e-8
    assert abs(d['le']-3.732525052048051)<1e-8
    assert m['bisection_restarts']==0
    print('Qia same-state NumPy',dict(h=d['h'],le=d['le'],iterations=m['iterations'],
        coefficient_residual=m['coefficient_residual']),flush=True)
