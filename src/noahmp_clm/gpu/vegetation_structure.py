"""Native total/exposed canopy structure; separate from the released state ABI.

Inputs contain previous SAI/LAI and the actual updated LAI. Callers own the
phenology/CN operator ordering; this component neither creates C nor advances it.
"""
from functools import lru_cache
import numpy as np
from ..physics.vegetation_structure import CanopyStructureParameters

FIELDS=('sai_before','sai_after','lai_before','lai_after','leaf_area_loss',
        'decay_factor','minimum_sai','total_lai','total_sai','exposed_lai',
        'exposed_sai','exposed_plant_area','exposed_fraction','snow_depth_m')


def pack_structure(sai,previous_lai,lai,dt,snow_depth_m,parameters=CanopyStructureParameters(),*,
                   prescribed_sai=None,harvested=False,fallow=False,newly_planted=False,dtype=np.float64):
    """Pack one row: six inputs, two geometric parameters, six integer flags."""
    from ..physics.vegetation_structure import step_stem_area,exposed_canopy_geometry
    result,_=step_stem_area(sai,previous_lai,lai,dt,parameters,prescribed_sai=prescribed_sai,
        harvested=harvested,fallow=fallow,newly_planted=newly_planted)
    exposed_canopy_geometry(lai,result,snow_depth_m,parameters)
    return (np.asarray([sai,previous_lai,lai,dt,snow_depth_m,0. if prescribed_sai is None else prescribed_sai],dtype),
            np.asarray([parameters.canopy_top_m,parameters.canopy_bottom_m],dtype),
            np.asarray([int(parameters.scheme),parameters.short_vegetation,prescribed_sai is not None,
                        harvested,fallow,newly_planted],np.int32))


def validate_structure(c,p,o,xp):
    n=len(c)
    if c.shape!=(n,6) or p.shape!=(n,2) or o.shape!=(n,6):raise ValueError('Structure shapes require n by6/2/6')
    if c.dtype not in (xp.float32,xp.float64) or p.dtype!=c.dtype or o.dtype.kind not in 'iu':
        raise ValueError('Structure requires a common floating precision and integer flags')
    if not bool(xp.all(xp.isfinite(c))) or not bool(xp.all(xp.isfinite(p))):raise ValueError('Nonfinite structure input')
    if bool(xp.any(c<0.)) or bool(xp.any((c[:,3]<=0.)|(c[:,3]>86400.))):raise ValueError('Invalid structure area/time/depth')
    if bool(xp.any((p[:,1]<0.)|(p[:,0]<=p[:,1]))):raise ValueError('Invalid canopy top/bottom')
    if bool(xp.any((o[:,0]<0)|(o[:,0]>3))) or bool(xp.any((o[:,1:]<0)|(o[:,1:]>1))):
        raise ValueError('Invalid structure scheme or boolean flags')
    kind=o[:,0];harvest=o[:,3]!=0;fallow=o[:,4]!=0;plant=o[:,5]!=0
    if bool(xp.any(((kind!=0)&(o[:,2]!=0))|((kind!=3)&(harvest|fallow|plant))|
                   (harvest&plant)|((harvest|fallow)&(c[:,2]!=0.)))):
        raise ValueError('Inconsistent prescribed/dynamic/crop structure inputs')


def structure_kernel_jax(c,p,o):
    import jax.numpy as j
    from jax import lax
    s=lax.optimization_barrier
    kind=o[0];loss=j.maximum(0.,s(c[1]-c[2]));cn=(kind==1)|(kind==2)
    rate=j.where(kind==1,.5,1.);minimum=j.where(cn,j.where(kind==1,.5,.05),0.)
    factor=j.where(cn,s(1.-s(s(rate*c[3])/2592000.)),1.)
    relaxed=j.maximum(minimum,s(s(factor*c[0])+loss))
    crop=s(.2*c[2]);crop=j.where((o[3]!=0)|((o[4]!=0)&(c[0]>0.)),.25,crop)
    crop=j.where((o[5]!=0)&(c[2]==0.),0.,crop)
    sai=j.where(kind==0,j.where(o[2]!=0,c[5],c[0]),j.where(kind==3,crop,relaxed))
    height=s(p[0]-p[1]);hidden=j.minimum(j.maximum(s(c[4]-p[1]),0.),height)
    # GPU reciprocal division need not give exactly1 for x/x. Complete
    # burial is a geometric boundary, and must leave exactly zero exposed area.
    short=j.where(c[4]>=.2,0.,s(1.-s(c[4]/.2)))
    tall=j.where(c[4]>=p[0],0.,j.where(c[4]<=p[1],1.,s(1.-s(hidden/height))))
    fraction=j.where(o[1]!=0,short,tall)
    leaf=s(c[2]*fraction);stem=s(sai*fraction);plant=s(s(c[2]+sai)*fraction)
    return j.asarray([c[0],sai,c[1],c[2],loss,factor,minimum,c[2],sai,leaf,stem,plant,fraction,c[4]],c.dtype)


@lru_cache(maxsize=1)
def _kernel():
    import jax
    return jax.jit(jax.vmap(structure_kernel_jax))


def unpack_structure(out):return {key:out[:,k] for k,key in enumerate(FIELDS)}


def run_structure_jax(c,p,o):
    import jax.numpy as j
    arrays=tuple(map(j.asarray,(c,p,o)));validate_structure(*arrays,j)
    result=_kernel()(*arrays)
    if not bool(j.all(j.isfinite(result))):raise RuntimeError('Nonfinite native canopy structure')
    return unpack_structure(result)
