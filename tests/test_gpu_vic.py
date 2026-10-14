"""Independent NumPy VIC transfers versus both native GPU backends."""
from dataclasses import replace
import numpy as np
import pytest
from test_gpu_soil_transport import cases as base_cases,reference as old_reference,science,budgets
from noahmp_clm.column import SoilParams
from noahmp_clm.physics.soil_enthalpy import SoilEnthalpyState
from noahmp_clm.physics.vic import VICParameters,infiltrate_vic
from noahmp_clm.physics.soil_drainage import drain_soil
from noahmp_clm.gpu.vic import pack_vic,validate_vic


def cases(dtype):
    c,p,o,u=base_cases(dtype,90)
    values=[]
    for i in range(len(c)):
        param=VICParameters(infiltration_shape=(0.,1e-5,.2,.4)[i%4],baseflow_fraction=(0.,.1,.8)[i%3],
            maximum_baseflow_mm_day=(0.,10.,1000.)[i%3],residual_pore_fraction=(0.,.05,.2)[i%3])
        values.append(pack_vic(param,surface=i%7!=0,bottom=i%5!=0,dtype=dtype))
    return c,p,o,u,np.stack(values)


def reference(c,p,o,u,v):
    c,p,u,v=[np.asarray(a,float) for a in (c,p,u,v)]
    answer=old_reference(c,p,o,u)
    if o[0]==1 or (o[0]==0 and v[0]==0.) or (o[0]==2 and v[1]==0.):return answer
    state=SoilEnthalpyState(c[:,0],c[:,1],c[:,2],SoilParams(smcmax=p[0],bexp=p[1],smcref=p[2],satdk=p[3]),int(o[1]))
    param=VICParameters(*v[2:])
    if o[0]==0:
        new,m,h=infiltrate_vic(state,u[0],u[1],param);answer['runoff']=np.array([m,h])
    else:
        new,d=drain_soil(state,u[2],permeability_option=int(o[2]),bottom_parameters=param)
        answer.update(interface_mass_kgm2=d['interface_mass_kgm2'],interface_enthalpy_jm2=d['interface_enthalpy_jm2'],
                      bottom_outflow=np.array([d['bottom_mass_kgm2'],d['bottom_enthalpy_jm2']]))
    answer.update(cells=np.column_stack((new.water,new.enthalpy,new.dz)),temperature_offset_k=new.temperature-273.15,
        temperature_k=new.temperature,liquid_kgm2=new.liquid,ice_kgm2=new.ice)
    return answer


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_vic_native_reference_and_conservation(dtype):
    jax=pytest.importorskip('jax')
    if dtype is np.float64 and not jax.config.x64_enabled:pytest.skip('JAX x64 required')
    from noahmp_clm.gpu.vic import run_vic_transport_jax
    arrays=cases(dtype);result={k:np.asarray(v) for k,v in run_vic_transport_jax(*arrays).items()};maximum={}
    for i,row in enumerate(zip(*arrays)):
        expected=reference(*row);actual={k:v[i] for k,v in result.items()};science(actual,expected,dtype,float(row[3][2]))
        for key,val in expected.items():
            if key!='valid':maximum[key]=max(maximum.get(key,0.),float(np.max(abs(actual[key]-val))))
    print('VIC native',str(dtype),maximum,'water/heat budget',budgets(arrays[0],result),flush=True)


def test_vic_native_same_precision():
    cp=pytest.importorskip('cupy');pytest.importorskip('jax')
    from noahmp_clm.gpu.vic import run_vic_transport_jax
    from noahmp_clm.gpu.cuda.vic import run_vic_transport_cuda
    arrays=cases(np.float32);a=run_vic_transport_cuda(*arrays);b=run_vic_transport_jax(*arrays)
    for key in a:
        x,y=cp.asnumpy(a[key]),np.asarray(b[key])
        print('VIC same precision',key,np.max(x!=y) if x.dtype==bool else np.max(abs(x-y)),flush=True)
        if x.dtype==bool:np.testing.assert_array_equal(x,y)
        else:np.testing.assert_allclose(x,y,rtol=1e-6,atol=1e-6)
    print('VIC CUDA water/heat budget',budgets(arrays[0],{k:cp.asnumpy(v) for k,v in a.items()}),flush=True)


def test_vic_host_configuration_and_reference():
    arrays=cases(np.float64);validate_vic(arrays[-1],len(arrays[0]),np)
    for row in zip(*arrays):reference(*row)
    bad=arrays[-1].copy();bad[0,5]=1.
    with pytest.raises(ValueError):validate_vic(bad,len(bad),np)
    bad=arrays[-1].copy();bad[0,0]=.5
    with pytest.raises(ValueError):validate_vic(bad,len(bad),np)
