"""Meteorological glacier: native state carry, independent ledgers and MOST."""
from dataclasses import replace
import numpy as np
import pytest
from noahmp_clm.column import Forcing
from noahmp_clm.clm_glacier import GlacierColumnState,step_clm_glacier_column
from noahmp_clm.physics.glacier import GlacierIce,GlacierParameters
from noahmp_clm.physics.snowpack import Snowpack,TF
from noahmp_clm.gpu.clm_glacier_column import pack_clm_glacier_columns,validate_glacier_column,next_glacier_arrays
from test_gpu_glacier import host,compare_same


def cases(dtype,drag):
    states=[];forcing=[];deposition=[]
    kinds=('cold','melt','snow_cold','snow_melt','rain','snowfall','frost','dirty') if drag==2 else ('stable','unstable')
    for kind in kinds:
        temp=TF if 'melt' in kind or kind=='rain' else 263.15
        ice=GlacierIce.from_thickness([.1,.3,.6,9.],temp)
        ice=GlacierIce(ice.mass,ice.enthalpy,np.ones((4,5))*1e-8)
        snow=Snowpack.from_ice([2.,5.,11.,23.,30.],temp,300.) if kind.startswith('snow_') or kind=='dirty' else Snowpack.empty()
        state=GlacierColumnState(ice,snow,humidity_height=4.)
        state=replace(state,options=replace(state.options,OptSurfaceDrag=drag));states.append(state)
        air=285. if 'melt' in kind or kind=='rain' else 250. if kind=='stable' else 268. if kind=='unstable' else 260.
        humidity=.008 if kind=='frost' else .001;rate=2e-6 if kind in ('rain','snowfall') else 0.
        forcing.append([1800.,air,humidity,3.,80000.,600. if air>TF else 400. if kind=='unstable' else 0. if kind=='stable' else 150.,350. if air>TF else 240.,rate])
        deposition.append(np.arange(1,6)*1e-12 if kind=='dirty' else np.zeros(5))
    return pack_clm_glacier_columns(states,dtype=dtype),np.asarray(forcing,dtype),np.asarray(deposition,dtype)


def unpack_states(arrays):
    states=[]
    for c,g,p,o,site in zip(*arrays):
        c,g,p,site=[v.astype(float) for v in (c,g,p,site)];ns,ni=o[:2]
        snow=Snowpack(c[:ns,0],c[:ns,1],c[:ns,2],c[:ns,3],c[:ns,4:])
        ice=GlacierIce(g[:ni,0],g[:ni,1],g[:ni,2:])
        s=GlacierColumnState(ice,snow,GlacierParameters(*p),z0m=site[0],wind_height=site[1],temperature_height=site[2],humidity_height=site[3])
        states.append(replace(s,options=replace(s.options,OptRainSnowPartition=int(o[2]),OptSurfaceDrag=int(o[3]))))
    return states


def budgets(arrays,f,dep,out):
    c,g,p,o,site=tuple(np.asarray(v.get() if hasattr(v,'get') else v) for v in arrays)
    c,g,p,f,dep=[v.astype(float) for v in (c,g,p,f,dep)]
    nc,ng=out['snow_cells'].astype(float),out['ice_cells'].astype(float);d=out['diagnostics']
    old=c[:,:,:2].sum(1)+g[:,:,:2].sum(1);new=nc[:,:,:2].sum(1)+ng[:,:,:2].sum(1)
    incoming=np.column_stack((f[:,0]*f[:,7]*1000.,d['precipitation_enthalpy_jm2'].astype(float)+
        (d['rnet'].astype(float)-d['h'].astype(float)-d['le'].astype(float))*f[:,0]+p[:,1]*f[:,0]))
    exports=np.column_stack((d['et_mm'].astype(float)+d['runoff_mm'].astype(float),
        d['advected_enthalpy_jm2'].astype(float)+d['runoff_enthalpy_jm2'].astype(float)))
    residual=new-old-incoming+exports
    bounds=np.column_stack((d['water_bound_kgm2'],d['energy_bound_jm2'])).astype(float)
    assert np.all(abs(residual)<=bounds),(residual,bounds)
    tracer=nc[:,:,4:].sum(1)+ng[:,:,2:].sum(1)-c[:,:,4:].sum(1)-g[:,:,2:].sum(1)+out['impurity_out_kgm2'].astype(float)-dep*f[:,:1]
    scale=c[:,:,4:].sum(1)+g[:,:,2:].sum(1)+dep*f[:,:1]
    bound=128.*np.finfo(out['ice_cells'].dtype).eps*np.maximum(scale,1e-30)
    assert np.all(abs(tracer)<=bound+1e-22),(tracer,bound)
    return np.max(abs(residual),axis=0),np.max(bounds,axis=0)


def compare_reference(out,states,diagnostics,dtype):
    expected=pack_clm_glacier_columns(states,dtype=np.float64);errors={}
    np.testing.assert_array_equal(out['options'],expected[3])
    # Units are explicit; temperature always uses an absolute K tolerance.
    for key,value,ref,atol in (
        ('snow_mass_kgm2',out['snow_cells'][:,:,0],expected[0][:,:,0],1e-4),
        ('snow_heat_jm2',out['snow_cells'][:,:,1],expected[0][:,:,1],.1),
        ('snow_depth_m',out['snow_cells'][:,:,2],expected[0][:,:,2],1e-6),
        ('snow_radius_um',out['snow_cells'][:,:,3],expected[0][:,:,3],.01),
        ('snow_impurity_kgm2',out['snow_cells'][:,:,4:],expected[0][:,:,4:],1e-12),
        ('ice_mass_kgm2',out['ice_cells'][:,:,0],expected[1][:,:,0],1e-4),
        ('ice_heat_jm2',out['ice_cells'][:,:,1],expected[1][:,:,1],.1),
        ('ice_impurity_kgm2',out['ice_cells'][:,:,2:],expected[1][:,:,2:],1e-12)):
        errors[key]=float(np.max(abs(value-ref)))
        np.testing.assert_allclose(value,ref,rtol=.1 if dtype is np.float32 else 1e-7,
            atol=atol if dtype is np.float32 else min(atol,1e-4),err_msg=key)
    for key in ('ts','flux_surface_temperature','h','le','rnet','g','et_mm','runoff_mm','swe',
                'precipitation_enthalpy_jm2','runoff_enthalpy_jm2','advected_enthalpy_jm2'):
        ref=np.asarray([d[key] for d in diagnostics]);value=out['diagnostics'][key];errors[key]=float(np.max(abs(value-ref)))
        temperature=key in ('ts','flux_surface_temperature')
        atol=.01 if temperature else .05 if key in ('h','le','rnet','g') else .1 if key.endswith('jm2') else 1e-4
        if dtype is np.float64:atol=3e-7 if temperature else min(atol,1e-4)
        np.testing.assert_allclose(value,ref,rtol=0. if temperature else .1 if dtype is np.float32 else 1e-7,atol=atol,err_msg=key)
    return errors


@pytest.mark.parametrize('drag',[2,1])
@pytest.mark.parametrize('dtype',[np.float64,np.float32])
def test_glacier_column_jax_reference_state_carry(dtype,drag):
    jax=pytest.importorskip('jax')
    if dtype is np.float64 and not jax.config.x64_enabled:pytest.skip('JAX x64 required')
    from noahmp_clm.gpu.clm_glacier_column import step_clm_glacier_columns_jax
    args,f,dep=cases(dtype,drag);states=unpack_states(args)
    for step in range(2):
        result=step_clm_glacier_columns_jax(*args,f,deposition=dep)
        assert next(iter(result['ice_cells'].devices())).platform=='gpu';out=host(result)
        res,bound=budgets(args,f,dep,out);updated=[];diags=[]
        for i,state in enumerate(states):
            new,d=step_clm_glacier_column(state,Forcing(*map(float,f[i])),deposition_rates=dep[i].astype(float))
            updated.append(new);diags.append(d)
        print('glacier whole reference',str(dtype),'drag',drag,'step',step,compare_reference(out,updated,diags,dtype),
            'independent M/H',res.tolist(),'bounds',bound.tolist(),flush=True)
        if drag==1:assert np.all(out['most']['coefficient_residual']<=1e-4)
        args=next_glacier_arrays(args,result);states=updated


@pytest.mark.parametrize('drag',[2,1])
def test_glacier_column_cuda_same_precision_state_carry(drag):
    cp=pytest.importorskip('cupy');pytest.importorskip('jax')
    from noahmp_clm.gpu.clm_glacier_column import step_clm_glacier_columns_jax
    from noahmp_clm.gpu.cuda.clm_glacier_column import step_clm_glacier_columns_cuda
    a,f,dep=cases(np.float32,drag);b=a
    for step in range(2):
        ja=step_clm_glacier_columns_jax(*a,f,deposition=dep);cu=step_clm_glacier_columns_cuda(*b,f,deposition=dep)
        assert isinstance(cu['ice_cells'],cp.ndarray)
        res,bound=budgets(b,f,dep,host(cu))
        print('glacier whole same precision drag',drag,'step',step,compare_same(host(ja),host(cu)),
            'independent M/H',res.tolist(),'bounds',bound.tolist(),flush=True)
        a=next_glacier_arrays(a,ja);b=next_glacier_arrays(b,cu)


def test_glacier_column_contract():
    args,f,dep=cases(np.float64,2);validate_glacier_column(args,f,dep,np)
    args[3][0,3]=1
    with pytest.raises(ValueError,match='Group'):validate_glacier_column(args,f,dep,np)
    args[3][0,3]=2;f[0,0]=0.
    with pytest.raises(ValueError):validate_glacier_column(args,f,dep,np)


@pytest.mark.parametrize('backend',['jax','cuda'])
def test_exhausted_vapor_surface_rejects_without_zero_cell_heat(backend):
    pytest.importorskip('jax' if backend=='jax' else 'cupy')
    from noahmp_clm.gpu.clm_glacier_column import step_clm_glacier_columns_jax
    from noahmp_clm.gpu.cuda.clm_glacier_column import step_clm_glacier_columns_cuda
    state=GlacierColumnState(GlacierIce([.001],[-21.]))
    f=Forcing(86400.,330.,0.,50.,80000.,1500.,1000.,0.)
    with pytest.raises(RuntimeError,match='exhausted'):step_clm_glacier_column(state,f)
    arrays=pack_clm_glacier_columns([state],dtype=np.float32)
    fun=step_clm_glacier_columns_jax if backend=='jax' else step_clm_glacier_columns_cuda
    with pytest.raises(RuntimeError,match='surface root failed'):fun(*arrays,np.array([list(vars(f).values())],np.float32))
