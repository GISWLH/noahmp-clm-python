import json
from pathlib import Path
import numpy as np
import pytest
from noahmp_clm.column import ColumnState,SoilParams,default_soil_params
from noahmp_clm.physics.snowpack import Snowpack,TF,LF
from noahmp_clm.physics.soil_enthalpy import SoilEnthalpyState
from noahmp_clm.physics.column_heat import coupled_heat_step
from noahmp_clm.gpu.column_heat import pack_joint_heat_case,validate_joint_heat
from clm_soil_oracle import fixture_suction


def cases(dtype):
    rng=np.random.default_rng(3908);rows=[]
    for case in range(48):
        ns=case%6;m=rng.uniform(.1,40.,ns)
        snow=Snowpack(m,m*rng.uniform(-2100.*15.,LF*1.1,ns),m/rng.uniform(80.,550.,ns))
        col=ColumnState(soil=default_soil_params(('sand','loam','clay')[(case//2)%3]),
            tsoil=rng.uniform(TF-12.,TF+8.,4),smc=rng.uniform(.1,.35,4))
        from dataclasses import replace
        col.options=replace(col.options,OptSoilSupercoolWater=1+case%2)
        earth=SoilEnthalpyState.from_column(col)
        rows.append(pack_joint_heat_case(snow,earth,rng.uniform(30.,3600.),rng.uniform(-100.,150.),
            snow_absorbed=rng.uniform(0.,50.,ns),soil_absorbed=rng.uniform(0.,20.,4),
            top_temperature=rng.uniform(TF-20.,TF+15.),top_conductance=rng.uniform(0.,15.),dtype=dtype,
            thermal_conductivity=rng.uniform(.05,1.5,ns+4) if case%2 else None))
    snow=Snowpack.from_ice([.1],TF)
    earth=SoilEnthalpyState.from_column(ColumnState(tsoil=np.full(4,TF-1.),smc=np.full(4,.25)))
    rows.append(pack_joint_heat_case(snow,earth,1800.,300.,dtype=dtype))
    earth=SoilEnthalpyState.from_column(ColumnState(tsoil=np.full(4,TF),smc=np.full(4,.25)))
    rows.append(pack_joint_heat_case(Snowpack.from_ice([50.],TF),earth,1800.,1./1800.,dtype=dtype))
    fixture=json.loads((Path(__file__).parent/'fixtures/column_phase_boundary.json').read_text())
    p=fixture['soil'].pop('params');earth=SoilEnthalpyState(**fixture['soil'],params=SoilParams(**p))
    rows.append(pack_joint_heat_case(Snowpack(**fixture['snow']),earth,fixture['dt'],fixture['flux'],**fixture['kwargs'],dtype=dtype))
    failures=json.loads((Path(__file__).parent/'fixtures/heat_merit_phase_events.json').read_text())
    for row in failures['cases']:
        c,p,o,b=(np.asarray(row[key],np.int32 if key=='options' else dtype)
                 for key in ('cells','props','options','boundary'))
        rows.append((c,fixture_suction(p,.355),o,b))
    return tuple(np.stack([row[i] for row in rows]) for i in range(4))


def numpy_reference(c,p,o,b):
    ns=int(o[0]);n=ns+4;c,p,b=c.astype(float),p.astype(float),b.astype(float)
    snow=Snowpack(c[:ns,0],c[:ns,1],c[:ns,2]) if ns else Snowpack.empty()
    earth=SoilEnthalpyState(c[ns:n,0],c[ns:n,1],c[ns:n,2],SoilParams(smcmax=p[0],bexp=p[1],quartz=p[2],saturated_suction_m=p[3]),int(o[1]))
    from noahmp_clm.physics.snow_heat import conductivity
    from noahmp_clm.physics.soil import thermal_conductivity
    native_k=np.r_[conductivity(snow),thermal_conductivity(earth.water/(1000.*earth.dz),earth.params)]
    k=np.where(c[:n,3]>0,c[:n,3],native_k)
    pack,soil,d=coupled_heat_step(snow,earth,b[0],b[1],snow_absorbed=c[:ns,4],soil_absorbed=c[ns:n,4],
        top_temperature=b[2]+273.15,top_conductance=b[3],thermal_conductivity=k)
    return np.r_[pack.enthalpy,soil.enthalpy],np.r_[pack.temperature,soil.temperature],d


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_jax_joint_heat_against_numpy_and_conservation(dtype):
    jax=pytest.importorskip('jax')
    if dtype is np.float64 and not jax.config.x64_enabled: pytest.skip('JAX x64 required')
    from noahmp_clm.gpu.column_heat import run_joint_heat_jax
    c,p,o,b=cases(dtype);got=run_joint_heat_jax(c,p,o,b)
    max_t=max_h=max_budget=0.
    for row in range(len(c)):
        n=int(o[row,0])+4;h,t,d=numpy_reference(c[row],p[row],o[row],b[row])
        hg=np.asarray(got['enthalpy_jm2'][row,:n]);tg=np.asarray(got['temperature_k'][row,:n])
        max_t=max(max_t,np.max(np.abs(tg-t)));max_h=max(max_h,np.max(np.abs(hg-h)))
        np.testing.assert_allclose(tg,t,rtol=0.,atol=.01 if dtype is np.float32 else 1e-7)
        np.testing.assert_allclose(hg,h,rtol=.1 if dtype is np.float32 else 1e-8,atol=.1 if dtype is np.float32 else .001)
        budget=float(np.sum(hg.astype(float)-c[row,:n,1].astype(float))-float(b[row,0])*
            (float(got['top_flux_wm2'][row])+float(c[row,:n,4].astype(float).sum())))
        max_budget=max(max_budget,abs(budget))
        allowance=float(np.sum(np.asarray(got['layer_tolerance_jm2'][row,:n])))+16.*np.finfo(dtype).eps*max(1.,abs(float(got['input_energy_jm2'][row])))
        assert abs(budget)<=allowance+1e-5
        assert np.max(np.abs(np.asarray(got['layer_residual_jm2'][row]))/np.asarray(got['layer_tolerance_jm2'][row]))<=1.
    print('joint heat',dtype,'T max K',max_t,'H max J/m2',max_h,'independent budget J/m2',max_budget,flush=True)
    assert float(got['temperature_k'][48,0])>TF and float(got['ice_kgm2'][48,0])==0.
    # One joule into a melting snow surface must not be ignored because deep
    # soil stores O(1e8) J/m2. This is a conservation requirement, not accuracy.
    assert float(got['enthalpy_jm2'][49,0])==pytest.approx(1.,abs=1e-6)
    assert float(got['temperature_k'][50,0])>TF


def test_cuda_joint_heat_same_precision():
    cp=pytest.importorskip('cupy');pytest.importorskip('jax')
    from noahmp_clm.gpu.column_heat import run_joint_heat_jax
    from noahmp_clm.gpu.cuda.column_heat import run_joint_heat_cuda
    inputs=cases(np.float32);a=run_joint_heat_cuda(*inputs);b=run_joint_heat_jax(*inputs)
    for key in a:
        u,v=cp.asnumpy(a[key]),np.asarray(b[key])
        error=np.max(u!=v) if u.dtype==bool else np.max(np.abs(u-v))
        print('joint CUDA/JAX32',key,error,flush=True)
        np.testing.assert_allclose(u,v,rtol=1e-6,atol=1e-6)


def test_heat_batch_contract_rejects_bad_padding():
    c,p,o,b=cases(np.float64);validate_joint_heat(c,p,o,b,np)
    c[0,8,0]=1.
    with pytest.raises(ValueError,match='padding'):validate_joint_heat(c,p,o,b,np)
