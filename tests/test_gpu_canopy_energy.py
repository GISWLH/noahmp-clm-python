from dataclasses import astuple
import numpy as np
import pytest
from noahmp_clm.physics.canopy_energy import CanopyExchange,solve_canopy_energy
from noahmp_clm.physics.soil_uptake import SoilWaterSupply
from noahmp_clm.physics.photosynthesis import LeafParameters
from noahmp_clm.gpu.canopy_energy import pack_canopy_case
from noahmp_clm.gpu.photosynthesis import pack_leaf_parameters
from test_canopy_energy import CASES


def cases(dtype):
    packed=[]
    for name,area,sw,lw,e,kw in CASES:
        area=np.pad(area,(0,10-len(area)))
        packed.append(pack_canopy_case(area,sw,lw,e,dtype=dtype,**kw))
    rng=np.random.default_rng(191)
    for j in range(32):
        e=CanopyExchange(rng.uniform(278.,304.),rng.uniform(.004,.012),rng.uniform(80000.,102000.),
            rng.uniform(.008,.04),rng.uniform(.002,.008),rng.uniform(.02,.05),rng.uniform(280.,300.),
            rng.uniform(5.,30.),ground_resistance=rng.uniform(0.,400.))
        kw={"vegetated_fraction":rng.uniform(.1,1.),"wet_fraction":rng.uniform(0.,.8),
            "nitrogen_factor":rng.uniform(.4,1.,10),"water_stress":rng.uniform(.1,1.,10),
            "canopy_evaporation_limit":rng.uniform(0.,1e-5),"cosine_zenith":rng.uniform(.1,1.)}
        if j%2==0:
            roots=rng.uniform(0.,2e-5,4)
            kw["soil_supply"]=SoilWaterSupply(roots,rng.uniform(0.,1.,4),roots[0]+1e-5)
        packed.append(pack_canopy_case(rng.uniform(0.,.8,10),rng.uniform(0.,950.),rng.uniform(270.,380.),e,dtype=dtype,**kw))
    return tuple(np.stack([x[k] for x in packed]) for k in range(5))


def numpy_reference(arrays):
    dtype=arrays[0].dtype
    p=LeafParameters(*pack_leaf_parameters(dtype=dtype).astype(float))
    references=[]
    for area,u,factors,soil,initial in zip(*arrays):
        area,u,factors,soil,initial=[x.astype(float) for x in (area,u,factors,soil,initial)]
        supply=SoilWaterSupply(soil[1:5],soil[5:9],soil[9]) if soil[0] else None
        references.append(solve_canopy_energy(area,u[11],u[12],CanopyExchange(*u[:11]),
            vegetated_fraction=u[13],wet_fraction=u[14],canopy_evaporation_limit=u[15],
            transpiration_limit=u[16],ground_evaporation_limit=u[17],ground_emissivity=u[18],
            ground_albedo=u[19:21],diffuse_fraction=u[21],cosine_zenith=u[22],
            nitrogen_factor=factors[:,0],water_stress=factors[:,1],soil_supply=supply,
            initial_temperatures=initial,leaf_parameters=p))
    return references


@pytest.mark.parametrize("dtype",[np.float32,np.float64])
def test_jax_canopy_energy_against_numpy(dtype):
    jax=pytest.importorskip("jax")
    if dtype is np.float64 and not jax.config.x64_enabled:
        pytest.skip("JAX_ENABLE_X64=true required")
    from noahmp_clm.gpu.canopy_energy import run_canopy_energy_jax
    inputs=cases(dtype)
    actual=run_canopy_energy_jax(*inputs)
    reference=numpy_reference(inputs)
    for key in ("leaf_temperature","ground_temperature","canopy_temperature","canopy_humidity",
                "gpp","leaf_respiration","h","le","ground_heat","rnet","wet_evaporation",
                "transpiration","ground_evaporation","outgoing_shortwave","outgoing_longwave"):
        a=np.asarray(actual[key]);e=np.asarray([r[key] for r in reference])
        print("canopy JAX",dtype,key,"max_abs",np.max(abs(a-e)),flush=True)
        if dtype is np.float64:
            np.testing.assert_allclose(a,e,atol=1e-6,rtol=1e-7)
        elif "temperature" in key:
            np.testing.assert_allclose(a,e,atol=.05,rtol=0.)
        else:
            absolute=.1 if key in ("gpp","leaf_respiration") else 1.
            if key in ("canopy_humidity","wet_evaporation","transpiration","ground_evaporation"):
                absolute=1e-7
            np.testing.assert_allclose(a,e,rtol=.1,atol=absolute)
    threshold=.06 if dtype is np.float32 else .0006
    assert np.max(abs(np.asarray(actual["residual"])))<threshold
    assert np.max(abs(np.asarray(actual["air_water_residual"])))<(1e-8 if dtype is np.float32 else 1e-12)
    assert np.all(np.asarray(actual["converged"]))


def test_cuda_canopy_energy_matches_jax_same_precision():
    cp=pytest.importorskip("cupy")
    pytest.importorskip("jax")
    from noahmp_clm.gpu.canopy_energy import run_canopy_energy_jax
    from noahmp_clm.gpu.cuda.canopy_energy import run_canopy_energy_cuda
    inputs=cases(np.float32)
    j=run_canopy_energy_jax(*inputs);c=run_canopy_energy_cuda(*inputs)
    for key in j:
        if key=="gas_exchange":
            for field in j[key]:
                a,e=cp.asnumpy(c[key][field]),np.asarray(j[key][field])
                print("canopy gas CUDA/JAX32",field,"max_abs",np.max(abs(a-e)),flush=True)
                np.testing.assert_allclose(a,e,rtol=1e-6,atol=1e-6)
        elif key in ("iterations","converged"):
            np.testing.assert_array_equal(cp.asnumpy(c[key]),np.asarray(j[key]))
        else:
            a,e=cp.asnumpy(c[key]),np.asarray(j[key])
            print("canopy CUDA/JAX32",key,"max_abs",np.max(abs(a-e)),flush=True)
            np.testing.assert_allclose(a,e,rtol=1e-6,atol=1e-6)


def test_canopy_batch_validation_without_gpu():
    from noahmp_clm.gpu.canopy_energy import validate_canopy_batch
    inputs=list(cases(np.float64))
    validate_canopy_batch(*inputs,np)
    inputs[1][0,15]=-1.
    with pytest.raises(ValueError):
        validate_canopy_batch(*inputs,np)


def test_gpu_canopy_failure_is_explicit():
    cp=pytest.importorskip("cupy")
    pytest.importorskip("jax")
    from noahmp_clm.gpu.canopy_energy import run_canopy_energy_jax
    from noahmp_clm.gpu.cuda.canopy_energy import run_canopy_energy_cuda
    inputs=tuple(a[:1] for a in cases(np.float32))
    for run in (run_canopy_energy_jax,run_canopy_energy_cuda):
        result=run(*inputs,max_iterations=1,raise_on_failure=False)
        assert not bool(result["converged"][0])
        assert int(result["iterations"][0])==1
        with pytest.raises(RuntimeError,match="did not converge"):
            run(*inputs,max_iterations=1)
