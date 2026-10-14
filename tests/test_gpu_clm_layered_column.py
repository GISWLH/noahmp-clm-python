"""Full native layered column, including precipitation and unloading ledgers."""
from dataclasses import replace
import numpy as np
import pytest
from noahmp_clm.column import Forcing
from noahmp_clm.physics.canopy_interception import CanopyCondensate
from noahmp_clm.clm_enthalpy_driver import step_clm_enthalpy_column
from noahmp_clm.gpu.clm_layered_column import pack_clm_layered_columns,validate_layered_column
from test_gpu_clm_enthalpy_column import (cases as base_cases,state_from_row,SITE_NAMES,
    science,budgets as base_budgets,host,compare_same_precision,soil_temperature,SoilParams)


def cases(dtype=np.float64):
    arrays,f,dep=base_cases(dtype)
    condensate=np.broadcast_to(arrays[5][:,None,:2]/10.,(len(f),10,2)).copy()
    return (*arrays,condensate),f,dep


def reference(row,forcing,deposition,*,most_geometry=None,vic_parameters=None):
    state=state_from_row(row[:7]);roots=state.root_frac
    cc=np.asarray(row[7],float);state=replace(state,canopy_condensate=CanopyCondensate(cc[:,0],cc[:,1]))
    object.__setattr__(state,'root_frac',roots);site=np.asarray(row[4],float)
    final,d=step_clm_enthalpy_column(state,Forcing(*np.asarray(forcing,float)),
        **dict(zip(SITE_NAMES[7:],site[7:])),deposition_rates=np.asarray(deposition,float),most_geometry=most_geometry,vic_parameters=vic_parameters)
    cells=np.zeros((5,9));cells[:,3]=54.5;n=len(final.snow.mass)
    cells[:n]=np.column_stack((final.snow.mass,final.snow.enthalpy,final.snow.thickness,final.snow.radius_um,final.snow.impurities))
    options=row[3].copy();options[0]=n;d['leaf_respiration_umol_m2s']=d['leaf_respiration'];d['valid']=1.
    d['thermal_layer_tolerance']=np.zeros(9)
    d['thermal_layer_tolerance'][:len(d['thermal']['layer_residual_jm2'])]=d['thermal']['convergence_tolerance_jm2']
    return dict(snow_cells=cells,soil_cells=np.column_stack((final.ground.water,final.ground.enthalpy,final.ground.dz)),
        options=options,storage=np.array([final.canopy_water_kgm2,final.canopy_enthalpy_jm2,final.surface_temperature]),
        diagnostics=d,impurity_output_kgm2=d['impurity_output_kgm2'],surface=d,
        condensate=np.column_stack((final.canopy_condensate.mass,final.canopy_condensate.enthalpy)),
        soil_temperature=final.ground.temperature,soil_parameters=final.ground.params,freezing_option=final.ground.freezing_option)


def budgets(arrays,f,dep,got):
    base=list(arrays[:7]);base[5]=base[5].astype(float);base[5][:,:2]=arrays[7].astype(float).sum(1)
    corrected=dict(got);corrected['storage']=got['storage'].astype(float).copy()
    corrected['storage'][:,:2]=got['condensate'].astype(float).sum(1)
    result=base_budgets(base,f,dep,corrected)
    eps=np.finfo(arrays[0].dtype).eps
    for key in ('canopy_interception','canopy_post_interception'):
        if key not in got:continue # NumPy trajectory audits its global M/H; component ledgers have separate independent tests.
        d=got[key]['ledger'];w=abs(d['water_residual_kgm2']);h=abs(d['energy_residual_jm2'])
        scale=abs(d['precipitation_enthalpy_jm2'])+abs(d['ground_liquid_enthalpy_jm2'])+abs(d['ground_ice_enthalpy_jm2'])
        scale+=abs(arrays[7][:,:,1]).sum(1)+abs(got[key]['cells'][:,:,1]).sum(1)
        assert np.all(h<128.*eps*np.maximum(scale,1.))
        assert np.all(w<128.*eps*np.maximum(arrays[7][:,:,0].sum(1)+f[:,0]*f[:,7]*1000.,1.))
        result[key+'_heat_jm2']=float(h.max());result[key+'_water_kgm2']=float(w.max())
    return result


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_layered_column_reference_and_conservation(dtype):
    jax=pytest.importorskip('jax')
    if dtype is np.float64 and not jax.config.x64_enabled:pytest.skip('JAX x64 required')
    from noahmp_clm.gpu.clm_layered_column import step_clm_layered_columns_jax
    arrays,f,dep=cases(dtype);got=host(step_clm_layered_columns_jax(*arrays,f,deposition=dep,return_surface=True))
    for k,row in enumerate(zip(*arrays)):
        ref=reference(row,f[k],dep[k])
        one={key:({a:b[k] for a,b in v.items()} if key=='diagnostics' else v[k]) for key,v in got.items()
            if key not in ('surface','canopy_interception','canopy_post_interception')}
        err=science(one,ref,dtype)
        np.testing.assert_allclose(one['condensate'][:,0],ref['condensate'][:,0],rtol=.1,atol=1e-4)
        np.testing.assert_allclose(one['condensate'][:,1],ref['condensate'][:,1],rtol=.1,atol=1800.)
        err['canopy_layer_mass']=float(abs(one['condensate'][:,0]-ref['condensate'][:,0]).max())
        err['canopy_layer_enthalpy']=float(abs(one['condensate'][:,1]-ref['condensate'][:,1]).max())
        print('layered full column',str(dtype),k,err,flush=True)
    print('layered full budgets',budgets(arrays,f,dep,got),flush=True)


def test_layered_column_same_precision():
    cp=pytest.importorskip('cupy');pytest.importorskip('jax')
    from noahmp_clm.gpu.clm_layered_column import step_clm_layered_columns_jax
    from noahmp_clm.gpu.cuda.clm_layered_column import step_clm_layered_columns_cuda
    arrays,f,dep=cases(np.float32)
    a=host(step_clm_layered_columns_cuda(*arrays,f,deposition=dep,return_surface=True),cp.asnumpy)
    b=host(step_clm_layered_columns_jax(*arrays,f,deposition=dep,return_surface=True))
    print('layered full same precision',compare_same_precision(a,b),flush=True)
    print('layered full CUDA budgets',budgets(arrays,f,dep,a),flush=True)


def test_layered_column_contract():
    arrays,f,dep=cases();validate_layered_column(arrays,f,dep,np)
    state=state_from_row(tuple(v[0] for v in arrays[:7]))
    with pytest.raises(ValueError,match='ten canopy'):pack_clm_layered_columns([state])
    packed=pack_clm_layered_columns([state.with_layered_canopy()]);validate_layered_column(packed,f[:1],dep[:1],np)
    bad=[v.copy() for v in arrays];bad[7][0,0,0]=-1.
    with pytest.raises(ValueError):validate_layered_column(bad,f,dep,np)


def test_quantized_state_precision_conversion_rebuilds_only_aliases():
    from benchmarks.check_clm_enthalpy_column_gpu import cast_column_precision
    arrays,f,dep=cases(np.float32)
    naive=tuple(a.copy() if a.dtype.kind in 'iu' else a.astype(float) for a in arrays)
    with pytest.raises(ValueError,match='aliases'):validate_layered_column(naive,f.astype(float),dep.astype(float),np)
    promoted=cast_column_precision(arrays,np.float64)
    validate_layered_column(promoted,f.astype(float),dep.astype(float),np)
    np.testing.assert_array_equal(promoted[7],arrays[7].astype(float))
    np.testing.assert_array_equal(promoted[0],arrays[0].astype(float))
    np.testing.assert_array_equal(promoted[1],arrays[1].astype(float))
    np.testing.assert_array_equal(promoted[5][:,:2],promoted[7].sum(1))
