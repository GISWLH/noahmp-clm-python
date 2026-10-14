"""Native layered phase surface against NumPy and independent M/H sums."""
import numpy as np
import pytest
from noahmp_clm.column import SoilParams
from noahmp_clm.physics.snowpack import Snowpack,TF
from noahmp_clm.physics.soil_enthalpy import SoilEnthalpyState
from noahmp_clm.physics.canopy_interception import CanopyCondensate
from noahmp_clm.physics.canopy_energy import CanopyExchange
from noahmp_clm.physics.canopy_column import surface_step
from noahmp_clm.gpu.canopy_phase_surface import validate_phase_surface
from test_gpu_canopy_surface import cases as base_cases,science,budgets as surface_budgets


def cases(dtype=np.float64,random_count=15):
    base=list(base_cases(dtype,random_count));condensate=[]
    for k,(a,u) in enumerate(zip(base[4],base[5])):
        mass=np.full(10,.03 if u[0]>TF else .1)
        if not np.any(a):mass[:]=0.
        state=CanopyCondensate.from_temperature(mass,float(u[0]))
        condensate.append(np.column_stack((state.mass,state.enthalpy)))
        base[7][k,1:3]=state.mass.sum(),state.enthalpy.sum()
    return (*base,np.asarray(condensate,dtype))


def reference(row):
    c,g,p,o,a,u,f,v,condensate=row;n=int(o[0])
    c,g,p,a,u,f,v,condensate=[np.asarray(x,float) for x in (c,g,p,a,u,f,v,condensate)]
    pack=Snowpack(c[:n,0],c[:n,1],c[:n,2],c[:n,3],c[:n,4:])
    props=SoilParams(smcmax=p[0],bexp=p[1],smcref=p[2],satdk=p[3],quartz=p[4],smcwlt=p[5])
    soil=SoilEnthalpyState(g[:,0],g[:,1],g[:,2],props,int(o[1]))
    pack,soil,water,d=surface_step(pack,soil,v[0],a,u[11],u[12],CanopyExchange(*u[:11]),
        canopy_condensate=CanopyCondensate(condensate[:,0],condensate[:,1]),root_weights=v[3:],
        vegetated_fraction=u[13],transpiration_limit=u[16],ground_emissivity=u[18],ground_albedo=u[19:21],
        diffuse_fraction=u[21],cosine_zenith=u[22],nitrogen_factor=f[:,0],water_stress=f[:,1])
    d['snow_cells']=np.zeros((5,9));d['snow_cells'][:,3]=54.5;n=len(pack.mass)
    d['snow_cells'][:n]=np.column_stack((pack.mass,pack.enthalpy,pack.thickness,pack.radius_um,pack.impurities))
    d['snow_counts']=n;d['soil_cells']=np.column_stack((soil.water,soil.enthalpy,soil.dz))
    d['soil_temperature']=soil.temperature;d['soil_parameters']=props;d['freezing_option']=int(o[1])
    d['condensate']=np.column_stack((d['canopy_condensate'].mass,d['canopy_condensate'].enthalpy))
    return d


def budgets(arrays,got):
    *base,old=arrays;old=old.astype(float);dt=base[7][:,0].astype(float)
    base[7]=base[7].astype(float);base[7][:,1:3]=old.sum(1)
    got=dict(got);final=np.asarray(got['condensate'],float)
    got['canopy_water_after_kgm2']=final[:,:,0].sum(1);got['canopy_enthalpy_after_jm2']=final[:,:,1].sum(1)
    result=surface_budgets(base,got);eps=np.finfo(arrays[0].dtype).eps
    water=final[:,:,0]-old[:,:,0]+np.asarray(got['ice_vapor'],float)+np.asarray(got['liquid_vapor'],float)
    thermal=np.asarray(got['thermal_enthalpy'],float);advected=np.asarray(got['vapor_enthalpy'],float)
    heat=thermal-final[:,:,1]-advected
    assert np.all(abs(water)<=64.*eps*np.maximum(abs(old[:,:,0])+abs(final[:,:,0]),1e-12))
    assert np.all(abs(heat)<=64.*eps*np.maximum(abs(thermal)+abs(final[:,:,1])+abs(advected),1e-8))
    # Root inflow remains internal; only actual vapor and leaf transpiration
    # enthalpy leave the whole snow/soil/canopy column.
    tr=np.asarray(got['transpiration'],float);leaf=np.asarray(got['leaf_temperature'],float)
    transp=(tr*(334000.+4180.*(leaf-TF))).sum(1)
    observed=np.asarray(got['transpiration_advected_heat_wm2'],float)
    np.testing.assert_allclose(observed,transp,rtol=64.*eps,atol=1e-7)
    result.update(layer_water_kgm2=float(abs(water).max()),layer_parcel_heat_jm2=float(abs(heat).max()))
    return result


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_phase_surface_reference(dtype):
    jax=pytest.importorskip('jax')
    if dtype is np.float64 and not jax.config.x64_enabled:pytest.skip('JAX x64 required')
    from noahmp_clm.gpu.canopy_phase_surface import run_canopy_phase_surface_jax
    arrays=cases(dtype);got=run_canopy_phase_surface_jax(*arrays)
    for k,row in enumerate(zip(*arrays)):
        ref=reference(row);one={key:np.asarray(v[k]) for key,v in got.items() if not isinstance(v,dict)}
        errors=science(one,ref,dtype)
        np.testing.assert_allclose(one['condensate'][:,0],ref['condensate'][:,0],rtol=.1,atol=1e-4)
        np.testing.assert_allclose(one['condensate'][:,1],ref['condensate'][:,1],rtol=.1,atol=1800.)
        errors['canopy_layer_mass']=float(abs(one['condensate'][:,0]-ref['condensate'][:,0]).max())
        errors['canopy_layer_enthalpy']=float(abs(one['condensate'][:,1]-ref['condensate'][:,1]).max())
        print('phase surface',str(dtype),k,errors,flush=True)
    print('phase surface budgets',budgets(arrays,got),flush=True)


def test_phase_surface_same_precision():
    cp=pytest.importorskip('cupy');pytest.importorskip('jax')
    from noahmp_clm.gpu.canopy_phase_surface import run_canopy_phase_surface_jax
    from noahmp_clm.gpu.cuda.canopy_phase_surface import run_canopy_phase_surface_cuda
    arrays=cases(np.float32);a=run_canopy_phase_surface_cuda(*arrays);b=run_canopy_phase_surface_jax(*arrays)
    for key in b:
        for field in (b[key] if isinstance(b[key],dict) else [None]):
            aa,bb=(a[key][field],b[key][field]) if field else (a[key],b[key])
            aa,bb=cp.asnumpy(aa),np.asarray(bb)
            print('phase surface same',key,field,np.max(aa!=bb) if aa.dtype.kind in 'ib' else np.max(abs(aa-bb)),flush=True)
            if aa.dtype.kind in 'ib':np.testing.assert_array_equal(aa,bb)
            else:np.testing.assert_allclose(aa,bb,rtol=1e-6,atol=1e-6)
    print('CUDA phase surface budgets',budgets(arrays,{k:cp.asnumpy(v) for k,v in a.items() if not isinstance(v,dict)}),flush=True)


def test_phase_surface_contract():
    arrays=cases();validate_phase_surface(arrays,np)
    bad=[v.copy() for v in arrays];bad[-1][0,0,0]=-1.
    with pytest.raises(ValueError):validate_phase_surface(bad,np)
    bad=[v.copy() for v in arrays];bad[7][0,1]+=.1
    with pytest.raises(ValueError,match='aliases'):validate_phase_surface(bad,np)
