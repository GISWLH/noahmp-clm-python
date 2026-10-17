"""Device mixed deposition against a prescribed analytic phase equilibrium."""
import numpy as np
import pytest
from test_dry_condensation_phase import analytic_mixed_deposition
from test_gpu_canopy_phase_energy import host
from noahmp_clm.gpu.canopy_phase_energy import pack_canopy_phase,run_canopy_phase_jax
from noahmp_clm.physics.canopy_interception import CanopyCondensate
from noahmp_clm.physics.snowpack import LF


def batch(dtype):
    rows=[];references=[]
    for fraction in (.1,.5,.9):
        args,kw,ref=analytic_mixed_deposition(fraction);leaf,sw,lw,exchange,_,dt=args
        kw.pop('tolerance')
        row=pack_canopy_phase(np.r_[leaf,np.zeros(9)],sw,lw,exchange,
            CanopyCondensate(np.zeros(10),np.zeros(10)),dt,dtype=dtype,**kw)
        rows.append(row);references.append(ref)
    return tuple(np.stack([r[k] for r in rows]) for k in range(9)),references


def check(out,arrays,refs,dtype):
    assert np.all(out['converged'])
    actual=out['enthalpy'][:,0].astype(float)/(LF*out['mass'][:,0].astype(float))
    wanted=np.array([r['fraction'] for r in refs]);assert np.all(abs(actual-wanted)<=.1*wanted)
    for k,r in enumerate(refs):
        assert out['mass'][k,0]==pytest.approx(r['mass'],rel=.01)
        assert out['enthalpy'][k,0]==pytest.approx(r['enthalpy'],rel=.1)
    water=out['mass'].astype(float)+arrays[6][:,4,None].astype(float)*out['wet_evaporation'].astype(float)
    heat=out['enthalpy'].astype(float)+out['vapor_enthalpy'].astype(float)-out['thermal_enthalpy'].astype(float)
    eps=np.finfo(dtype).eps
    assert np.all(abs(water)<=64*eps*np.maximum(out['mass'],1.)+1e-12)
    assert np.all(abs(heat)<=64*eps*np.maximum(abs(out['enthalpy'])+abs(out['vapor_enthalpy'])+abs(out['thermal_enthalpy']),1.)+1e-6)
    print('dry mixed deposition',np.dtype(dtype).name,dict(fractions=actual.tolist(),
        maximum_fraction_relative_error=float(np.max(abs(actual-wanted)/wanted)),
        max_mass_residual=float(abs(water).max()),max_heat_residual=float(abs(heat).max()),
        max_solver_residual=float(out['max_solver_residual'].max())),flush=True)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_new_mixed_condensate_jax_analytic_and_budgets(dtype):
    jax=pytest.importorskip('jax')
    if dtype is np.float64 and not jax.config.x64_enabled:pytest.skip('JAX x64 required')
    arrays,refs=batch(dtype);out=host(run_canopy_phase_jax(*arrays));check(out,arrays,refs,dtype)


def test_new_mixed_condensate_cuda_jax_same_precision():
    cp=pytest.importorskip('cupy');pytest.importorskip('jax')
    from noahmp_clm.gpu.cuda.canopy_phase_energy import run_canopy_phase_cuda
    from noahmp_clm.gpu.canopy_phase_energy import run_canopy_phase_jax
    from test_gpu_clm_layered_column import compare_same_precision
    arrays,refs=batch(np.float32);cu=host(run_canopy_phase_cuda(*arrays),cp.asnumpy);ja=host(run_canopy_phase_jax(*arrays))
    check(cu,arrays,refs,np.float32)
    print('dry mixed deposition same precision',compare_same_precision(cu,ja),flush=True)
