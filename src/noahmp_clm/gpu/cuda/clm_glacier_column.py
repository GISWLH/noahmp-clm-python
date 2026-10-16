"""CUDA meteorological glacier column with native surface and MOST iteration."""
from functools import lru_cache
from pathlib import Path
import numpy as np
from ..clm_glacier_column import validate_glacier_column,unpack_glacier_column,run_glacier_most


@lru_cache(maxsize=1)
def _stages():
    import cupy as cp
    names=('column_kernel.cu','enthalpy_kernel.cu','column_heat_kernel.cu','glacier_heat_kernel.cu',
        'snow_mass_kernel.cu','snow_evolution_kernel.cu','snow_vapor_kernel.cu','glacier_mass_kernel.cu',
        'glacier_kernel.cu','clm_glacier_column_kernel.cu')
    source='\n'.join(Path(__file__).with_name(n).read_text(encoding='utf-8') for n in names)
    exports=('prepare_glacier_column','solve_glacier_surface','finish_glacier_column')
    module=cp.RawModule(code=source,options=('--std=c++14','--fmad=false','--prec-div=false'),name_expressions=exports)
    return tuple(module.get_function(n) for n in exports)


def step_clm_glacier_columns_cuda(c,g,p,o,site,forcing,*,deposition=None):
    import cupy as cp
    from .snicar import run_snow_shortwave_cuda
    from .snow_evolution import _table as aging
    if cp.asarray(o).dtype.kind not in 'iu':raise ValueError('Glacier options must be integers')
    c,g,p,site,f=[cp.ascontiguousarray(cp.asarray(v,cp.float32)) for v in (c,g,p,site,forcing)]
    o=cp.ascontiguousarray(cp.asarray(o,cp.int32));n=len(c)
    dep=cp.zeros((n,5),cp.float32) if deposition is None else cp.ascontiguousarray(cp.asarray(deposition,cp.float32))
    validate_glacier_column((c,g,p,o,site),f,dep,cp)
    new=lambda *shape:cp.empty((n,*shape),cp.float32)
    pc=new(5,9);pg=new(4,7);po=cp.empty_like(o);prep=new(9);vel=new(3)
    launch=((min((n+31)//32,4096),),(32,))
    prepare,surface,finish=_stages()
    if n:prepare(*launch,(c,g,p,o,site,f,pc,pg,po,prep,vel,np.int32(n)))
    if bool(cp.any(po[:,1]==0)):raise RuntimeError('Precipitation exhausted glacier; reduce time step')
    opt=pc[:,:,[0,3,4,5,6,7,8]];beam=.5*f[:,5]*(1.-site[:,5]);sky=.5*f[:,5]*site[:,5]
    boundary=cp.stack((beam,beam,sky,sky,p[:,2],p[:,2],site[:,4]),axis=1)
    rad=run_snow_shortwave_cuda(opt,po[:,0],boundary)
    q=cp.ascontiguousarray(cp.concatenate((rad['absorbed'],rad['ground_absorbed'][:,None],cp.zeros((n,3),cp.float32)),axis=1))
    reflected=cp.ascontiguousarray(rad['reflected'])
    def solve(v):
        raw=new(188)
        if n:surface(*launch,(pc,pg,p,po,f,cp.ascontiguousarray(v),q,reflected,raw,np.int32(n)))
        return raw
    if n and int(o[0,3])==1:raw,most=run_glacier_most(solve,site,f,backend='cuda')
    else:raw=solve(vel);most=None
    if not bool(cp.all(cp.isfinite(raw))) or not bool(cp.all(raw[:,187]>0.)):raise RuntimeError('CUDA glacier surface root failed')
    nc=new(5,9);ng=new(4,7);no=cp.empty_like(o);diag=new(22);tracer=new(5)
    if n:finish(*launch,(c,g,p,o,f,dep,prep,raw,aging(),nc,ng,no,diag,tracer,np.int32(n)))
    result=(nc,ng,no,diag,tracer)
    if not all(bool(cp.all(cp.isfinite(v))) for v in result):raise RuntimeError('Nonfinite CUDA glacier column output')
    out=unpack_glacier_column(result,raw);d=out['diagnostics']
    if bool(cp.any((cp.abs(d['water_residual_kgm2'])>d['water_bound_kgm2'])|(cp.abs(d['energy_residual_jm2'])>d['energy_bound_jm2']))):
        raise RuntimeError('CUDA glacier column conservation failed')
    if most is not None:out['most']=most
    return out
