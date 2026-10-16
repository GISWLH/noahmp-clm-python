"""Native phase solve against independently validated NumPy physics."""
import numpy as np
import pytest
import json
from pathlib import Path
from test_canopy_phase_energy import cases
from test_gpu_snicar import assert_science_close
from noahmp_clm.gpu.canopy_phase_energy import pack_canopy_phase,run_canopy_phase_jax,validate_phase
from noahmp_clm.physics.canopy_phase_energy import solve_canopy_phase_energy


def reference(row,**solver_controls):
    from noahmp_clm.physics.canopy_energy import CanopyExchange
    from noahmp_clm.physics.canopy_interception import CanopyCondensate
    from noahmp_clm.physics.soil_uptake import SoilWaterSupply
    from noahmp_clm.physics.snowpack import Snowpack
    from noahmp_clm.physics.photosynthesis import LeafParameters
    from noahmp_clm.gpu.photosynthesis import pack_leaf_parameters
    dtype=row[0].dtype
    a,u,f,soil,stem,c,b,snow,n=[np.asarray(v,float) for v in row];n=int(n)
    pack=Snowpack(snow[:n,0],np.zeros(n),snow[:n,0]/250.,snow[:n,1],snow[:n,2:]) if n else None
    supply=SoilWaterSupply(soil[1:5],soil[5:9],soil[9]) if soil[0]>0. else None
    kw=dict(stem_area=stem,soil_supply=supply,root_enthalpy=b[5:9] if supply is not None else None,vegetated_fraction=u[13],
        transpiration_limit=u[16],ground_evaporation_limit=u[17],ground_emissivity=u[18],
        ground_albedo=u[19:21],diffuse_fraction=u[21],cosine_zenith=u[22],
        nitrogen_factor=f[:,0],water_stress=f[:,1],fixed_ground_temperature=b[1] if b[0]>0. else None,
        ground_is_ice=bool(b[2]),ground_shares_soil=bool(b[3]),snowpack=pack,
        leaf_parameters=LeafParameters(*pack_leaf_parameters(dtype=dtype).astype(float)))
    return solve_canopy_phase_energy(a,u[11],u[12],CanopyExchange(*u[:11]),CanopyCondensate(c[:,0],c[:,1]),b[4],**kw,**solver_controls)


def batch(dtype):
    rows=[pack_canopy_phase(a,sw,lw,e,c,1800.,dtype=dtype,**kw) for _,a,sw,lw,e,c,kw in cases()]
    saved=json.loads((Path(__file__).parent/'fixtures/canopy_phase_native_regressions.json').read_text())
    rows.extend(tuple(np.asarray(v,np.int32 if k==8 else dtype) for k,v in enumerate(row['arrays'])) for row in saved['cases'])
    saved=json.loads((Path(__file__).parent/'fixtures/canopy_phase_humidity_bound.json').read_text())
    rows.append(tuple(np.asarray(v,np.int32 if k==8 else dtype) for k,v in enumerate(saved['arrays'])))
    return tuple(np.stack([r[k] for r in rows]) for k in range(9))


def test_humidity_bound_restart_closes_original_coupled_equations():
    saved=json.loads((Path(__file__).parent/'fixtures/canopy_phase_humidity_bound.json').read_text())
    arrays=tuple(np.asarray(v,np.int32 if k==8 else np.float64) for k,v in enumerate(saved['arrays']))
    result=reference(arrays,tolerance=1e-7)
    assert result['restarts']>0 and 0.<result['canopy_humidity']<.19
    assert abs(result['layer_residual']).max()<=1.01e-7
    assert abs(result['solver_residual']).max()<=1e-7
    assert abs(result['canopy_water_residual'])<1e-12
    print('humidity-bound restart',result['iterations'],result['restarts'],result['canopy_humidity'],
        abs(result['layer_residual']).max(),result['canopy_water_residual'],flush=True)


def host(value,convert=np.asarray):
    return {k:host(v,convert) if isinstance(v,dict) else convert(v) for k,v in value.items()}


@pytest.mark.parametrize('dtype',[np.float64,np.float32])
def test_phase_native_reference(dtype):
    jax=pytest.importorskip('jax')
    if dtype==np.float64 and not jax.config.x64_enabled:pytest.skip('x64 required')
    a=batch(dtype);got=host(run_canopy_phase_jax(*a,raise_on_failure=False));maximum={}
    print('phase convergence',dtype,got['converged'].tolist(),got['iterations'].tolist(),got['max_solver_residual'].tolist(),flush=True)
    assert np.all(got['converged'])
    for k,row in enumerate(zip(*a)):
        ref=reference(row)
        for key in ('leaf_temperature','ground_temperature','canopy_temperature','h','le','gpp','ground_heat'):
            error=float(np.max(abs(np.asarray(got[key][k],float)-ref[key])));maximum[key]=max(maximum.get(key,0.),error)
            if 'temperature' in key:np.testing.assert_allclose(got[key][k],ref[key],rtol=0.,atol=.05)
            else:assert_science_close(got[key][k],ref[key],relative=.1,zero_floor=.1 if key=='gpp' else 10.,absolute=.01 if key=='gpp' else 1.)
        assert_science_close(got['mass'][k],ref['condensate'].mass,relative=.1,zero_floor=.001,absolute=1e-4)
        assert_science_close(got['enthalpy'][k],ref['condensate'].enthalpy,relative=.1,zero_floor=100.,absolute=10.)
    mass=a[5][:,:,0].astype(float);heat=a[5][:,:,1].astype(float);dt=a[6][:,4].astype(float)
    water=got['mass'].astype(float)-mass+dt[:,None]*got['wet_evaporation'].astype(float)
    energy=(got['enthalpy'].astype(float)-heat+got['vapor_enthalpy'].astype(float))-(got['thermal_enthalpy'].astype(float)-heat)
    magnitude=abs(got['enthalpy'].astype(float))+abs(heat)+abs(got['vapor_enthalpy'].astype(float))+abs(got['thermal_enthalpy'].astype(float))
    assert np.all(abs(water)<=64.*np.finfo(dtype).eps*np.maximum(mass,1.)+1e-12)
    assert np.all(abs(energy)<=64.*np.finfo(dtype).eps*np.maximum(magnitude,1.)+1e-6)
    print('phase science',dtype,maximum,'parcel budgets',np.max(abs(water)),np.max(abs(energy)),flush=True)


def test_phase_input_contract():
    a=list(batch(np.float64));validate_phase(a,np)
    a[6]=a[6].copy();a[6][0,4]=0.
    with pytest.raises(ValueError):validate_phase(a,np)


def test_phase_cuda_same_precision():
    cp=pytest.importorskip('cupy');pytest.importorskip('jax')
    from noahmp_clm.gpu.cuda.canopy_phase_energy import run_canopy_phase_cuda
    arrays=batch(np.float32);ja=host(run_canopy_phase_jax(*arrays,raise_on_failure=False));cu=host(run_canopy_phase_cuda(*arrays,raise_on_failure=False),cp.asnumpy)
    print('CUDA phase convergence',cu['converged'].tolist(),cu['iterations'].tolist(),cu['max_solver_residual'].tolist(),flush=True)
    assert np.all(cu['converged']) and np.all(ja['converged'])
    maximum={};failures=[]
    def compare(a,b,path=''):
        for key in a:
            name=path+key
            if isinstance(a[key],dict):compare(a[key],b[key],name+'.');continue
            av,bv=a[key],b[key]
            maximum[name]=float(np.max(abs(av.astype(float)-bv.astype(float))))
            try:
                if av.dtype.kind in 'biu':np.testing.assert_array_equal(av,bv)
                else:np.testing.assert_allclose(av,bv,rtol=1e-6,atol=1e-6)
            except AssertionError as error:failures.append((name,str(error)))
    compare(cu,ja);print('phase same precision',maximum,flush=True)
    assert not failures,failures
