"""Full native closed-soil columns with continuous state and offline oracles."""
from pathlib import Path
import json,os
import numpy as np
import pytest
from richards_column_cases import KINDS


def fixture(precision,index):
    with np.load(Path(__file__).parent/'fixtures/clm_richards_reference.npz',allow_pickle=False) as z:
        prefix=precision+'/'+str(index)+'/'
        return {k[len(prefix):]:v for k,v in z.items() if k.startswith(prefix)}


def host(x):
    if isinstance(x,dict):return {k:host(v) for k,v in x.items()}
    return x.get() if hasattr(x,'get') else np.asarray(x)


def advance(arrays,forcing,deposition,kind,backend,geometry):
    from noahmp_clm.physics.richards import RichardsHydrology
    kwargs=dict(deposition=deposition,return_surface=True,soil_hydrology=None if kind=='legacy' else RichardsHydrology())
    if geometry is not None:kwargs['most_geometry']=geometry
    if kind in ('bgc','structure'):
        dtype=arrays[0].dtype;n=len(arrays[0])
        kwargs.update(environment=np.zeros((n,2),dtype),calendar=np.zeros((n,2),np.int32),nitrogen_deposition_gn_m2=np.full(n,1e-5,dtype))
    if kind=='structure':
        from noahmp_clm.gpu.clm_structure_column import step_clm_structure_columns_jax,step_clm_structure_columns_cuda
        run=step_clm_structure_columns_cuda if backend=='cuda32' else step_clm_structure_columns_jax
        return run(arrays,forcing,kind='bgc',**kwargs)
    if kind=='bgc':
        if backend=='cuda32':from noahmp_clm.gpu.cuda.clm_bgc_column import step_clm_bgc_columns_cuda as run
        else:from noahmp_clm.gpu.clm_bgc_column import step_clm_bgc_columns_jax as run
    elif kind in ('layered','snow','phase2','most'):
        if backend=='cuda32':from noahmp_clm.gpu.cuda.clm_layered_column import step_clm_layered_columns_cuda as run
        else:from noahmp_clm.gpu.clm_layered_column import step_clm_layered_columns_jax as run
    else:
        if backend=='cuda32':from noahmp_clm.gpu.cuda.clm_enthalpy_column import step_clm_enthalpy_columns_cuda as run
        else:from noahmp_clm.gpu.clm_enthalpy_column import step_clm_enthalpy_columns_jax as run
    return run(*arrays,forcing,**kwargs)


def carry(a,r,kind):
    if kind=='structure':
        from noahmp_clm.gpu.clm_structure_column import next_structure_arrays
        return next_structure_arrays(a,r,kind='bgc')
    if kind=='bgc':
        from noahmp_clm.gpu.clm_bgc_column import next_bgc_arrays
        return next_bgc_arrays(a,r)
    base=(r['snow_cells'],r['soil_cells'],a[2],r['options'],a[4],r['storage'],a[6])
    return (*base,r['condensate']) if len(a)==8 else base


def flatten(d,prefix=''):
    for key,value in d.items():
        name=prefix+key
        if isinstance(value,dict):yield from flatten(value,name+'/')
        else:yield name,np.asarray(value)


def forbid_numpy(monkeypatch):
    import noahmp_clm.physics.richards as physics
    import noahmp_clm.clm_enthalpy_driver as driver
    def forbidden(*a,**kw):raise AssertionError('Native column cannot call NumPy physics')
    monkeypatch.setattr(physics,'redistribute_soil_enthalpy',forbidden)
    monkeypatch.setattr(driver,'step_clm_enthalpy_column',forbidden)


def audit(a,r,f,dep,c,prefix,kind):
    """Original independent numerical bounds; forcing phase came from EHPC."""
    dtype=a[0].dtype;eps=np.finfo(dtype).eps;d=r['diagnostics'];sf=r['surface']
    snow=np.asarray(r['snow_cells'],float);soil=np.asarray(r['soil_cells'],float)
    oldcan=a[5][:,:2].astype(float) if kind in ('legacy','aggregate','five_layers') else a[7].astype(float).sum(1)
    newcan=np.asarray(r['storage'],float)[:,:2] if kind in ('legacy','aggregate','five_layers') else np.asarray(r['condensate'],float).sum(1)
    oldm=a[0][:,:,0].astype(float).sum(1)+a[1][:,:,0].astype(float).sum(1)+oldcan[:,0]
    newm=snow[:,:,0].sum(1)+soil[:,:,0].sum(1)+newcan[:,0]
    oldh=a[0][:,:,1].astype(float).sum(1)+a[1][:,:,1].astype(float).sum(1)+oldcan[:,1]
    newh=snow[:,:,1].sum(1)+soil[:,:,1].sum(1)+newcan[:,1]
    dt=f[:,0].astype(float);precip=c[prefix+'diagnostics/prcp_mm'];ph=c[prefix+'diagnostics/precipitation_enthalpy_jm2']
    atmo=(f[:,5].astype(float)+f[:,6]-sf['outgoing_shortwave'].astype(float)-sf['outgoing_longwave'].astype(float)-d['h'].astype(float)-d['le'].astype(float))*dt
    et=d['et_mm'].astype(float);runoff=d['runoff_mm'].astype(float);adv=d['advected_enthalpy_jm2'].astype(float);rh=d['runoff_enthalpy_jm2'].astype(float)
    water=newm-oldm-precip+et+runoff;heat=newh-oldh-ph-atmo+adv+rh
    mb=32.*eps*np.maximum(abs(oldm)+abs(newm)+abs(precip)+abs(et)+abs(runoff),1.)
    scale=abs(a[0][:,:,1].astype(float)).sum(1)+abs(a[1][:,:,1].astype(float)).sum(1)+abs(oldcan[:,1])+abs(snow[:,:,1]).sum(1)+abs(soil[:,:,1]).sum(1)+abs(newcan[:,1])
    hb=sf['thermal_layer_tolerance'].astype(float).sum(1)+dt*abs(sf['layer_residual'].astype(float)).sum(1)+32.*eps*np.maximum(scale+abs(ph)+abs(atmo)+abs(adv)+abs(rh),1.)+1e-5
    assert np.all(abs(water)<=mb),(water,mb)
    assert np.all(abs(heat)<=hb),(heat,hb)
    tracer=snow[:,:,4:].sum(1)-a[0][:,:,4:].astype(float).sum(1)-dep.astype(float)*dt[:,None]+r['impurity_output_kgm2'].astype(float)
    tb=64.*eps*np.maximum(a[0][:,:,4:].astype(float).sum(1)+dep.astype(float)*dt[:,None],1e-20)
    assert np.all(abs(tracer)<=tb)
    result=dict(water=float(abs(water).max()),heat=float(abs(heat).max()),water_bound=float(mb.max()),heat_bound=float(hb.max()),impurity=float(abs(tracer).max()))
    if kind in ('bgc','structure'):
        from test_gpu_clm_bgc_column import cn_budgets
        result.update(cn_budgets(a[:16],r,c['nitrogen']))
    return result


@pytest.mark.parametrize('backend',['jax32','jax64','cuda32'])
def test_continuous_columns_against_ehpc_references(backend,monkeypatch):
    jax=pytest.importorskip('jax');assert any(d.platform=='gpu' for d in jax.devices())
    if backend=='cuda32':pytest.importorskip('cupy')
    assert backend!='jax64' or jax.config.x64_enabled
    forbid_numpy(monkeypatch)
    for index,kind in enumerate(KINDS):
        c=fixture(backend[-2:],index);a=tuple(c['initial/'+str(i)] for i in range(int(c['arrays'])))
        geometry=c.get('geometry')
        for step in range(2):
            result=advance(a,c['forcing'][step],c['deposition'][step],kind,backend,geometry);r=host(result)
            prefix='expected/'+str(step)+'/';expected=c[prefix+'soil_cells']
            budget=audit(tuple(np.asarray(v.get() if hasattr(v,'get') else v) for v in a),r,c['forcing'][step],c['deposition'][step],c,prefix,kind)
            water=float(np.max(abs(r['soil_cells'][:,:,0]-expected[:,:,0])/(1000.*expected[:,:,2])))
            assert water<=.002
            if kind!='legacy':
                temperature=float(np.max(abs(r['soil_hydrology']['temperature_k']-c[prefix+'temperature_k'])))
                assert temperature<=.05
                np.testing.assert_array_equal(r['soil_hydrology']['lower_boundary_condition'],2)
            else:temperature=None
            maximum={}
            for key in ['h','le','gpp_gcm2','et_mm','runoff_mm','swe']:
                got=r['diagnostics'][key];ref=c[prefix+'diagnostics/'+key];diff=abs(got-ref)
                floor,absolute=(10.,1.) if key in ('h','le') else ((1e-6,1e-7) if key=='swe' else (.001,.0001))
                assert np.all(diff<=np.where(abs(ref)<=floor,absolute,.1*abs(ref))),(kind,step,key,got,ref)
                maximum[key]=float(diff.max())
            if prefix+'diagnostics/nee_gc_m2' in c:
                ref=c[prefix+'diagnostics/nee_gc_m2'];diff=abs(r['diagnostics']['nee_gc_m2']-ref)
                assert np.all(diff<=np.where(abs(ref)<=.001,.0001,.1*abs(ref)))
                maximum['nee_gc_m2']=float(diff.max())
            np.testing.assert_array_equal(r['options'],c[prefix+'options'])
            if kind!='legacy':assert np.all(r['diagnostics']['runoff_deep_mm']==0.)
            print('Richards whole native reference',json.dumps(dict(backend=backend,kind=kind,step=step,max_theta_error=water,max_temperature_error_k=temperature,maximum_absolute=maximum,original_budgets=budget)),flush=True)
            a=carry(a,result,kind)


def test_all_raw_outputs_and_carried_arrays_same_precision(monkeypatch):
    pytest.importorskip('cupy');pytest.importorskip('jax');forbid_numpy(monkeypatch)
    for index,kind in enumerate(KINDS):
        c=fixture('32',index);a=b=tuple(c['initial/'+str(i)] for i in range(int(c['arrays'])))
        for step in range(2):
            x=advance(a,c['forcing'][step],c['deposition'][step],kind,'cuda32',c.get('geometry'))
            y=advance(b,c['forcing'][step],c['deposition'][step],kind,'jax32',c.get('geometry'))
            hx,hy=host(x),host(y);left,right=dict(flatten(hx)),dict(flatten(hy));assert left.keys()==right.keys()
            audit(tuple(np.asarray(v.get() if hasattr(v,'get') else v) for v in a),hx,c['forcing'][step],c['deposition'][step],c,'expected/'+str(step)+'/',kind)
            audit(tuple(np.asarray(v.get() if hasattr(v,'get') else v) for v in b),hy,c['forcing'][step],c['deposition'][step],c,'expected/'+str(step)+'/',kind)
            if os.environ.get('NOAHMP_RICHARDS_COLUMN_DIAGNOSTICS'):
                out=Path(os.environ['NOAHMP_RICHARDS_COLUMN_DIAGNOSTICS']);out.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(out/(kind+'-'+str(step)+'.npz'),**{'cuda/'+k:v for k,v in left.items()},**{'jax/'+k:v for k,v in right.items()})
            for k in left:
                assert left[k].dtype==right[k].dtype,k
                if left[k].dtype.kind in 'biu':np.testing.assert_array_equal(left[k],right[k],err_msg=k)
                else:np.testing.assert_allclose(left[k],right[k],rtol=1e-6,atol=1e-6,err_msg=k)
            a,b=carry(a,x,kind),carry(b,y,kind)
            for i,(aa,bb) in enumerate(zip(host(dict(enumerate(a))).values(),host(dict(enumerate(b))).values())):
                if aa.dtype.kind in 'biu':np.testing.assert_array_equal(aa,bb,err_msg=str(i))
                else:np.testing.assert_allclose(aa,bb,rtol=1e-6,atol=1e-6,err_msg=str(i))


def test_requested_arno_and_untyped_hydrology_rejected():
    pytest.importorskip('jax')
    from noahmp_clm.physics.richards import RichardsHydrology
    from noahmp_clm.gpu.clm_layered_column import step_clm_layered_columns_jax
    c=fixture('32',3);a=[c['initial/'+str(i)] for i in range(8)];f=c['forcing'][0]
    with pytest.raises(TypeError,match='RichardsHydrology'):step_clm_layered_columns_jax(*a,f,soil_hydrology={})
    a[3]=a[3].copy();a[3][:,7]=6
    with pytest.raises(ValueError,match='bottom outlets'):step_clm_layered_columns_jax(*a,f,soil_hydrology=RichardsHydrology())
