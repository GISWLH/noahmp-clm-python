"""Native two-tissue optics; the first584 outputs retain the snow ABI.

The extra40 values retain visible/NIR leaf and stem absorption separately.
Every input parameter remains a dynamic device argument.
"""
from functools import lru_cache
import numpy as np
from ..physics.canopy_optics import CanopyOptics


def pack_canopy_optics(optics,dtype=np.float64):
    if not isinstance(optics,CanopyOptics):raise ValueError('Expected explicit CanopyOptics')
    return np.asarray([v for name in optics.__dataclass_fields__ for v in getattr(optics,name)],dtype)


def validate_canopy_optics(area,leaf,parameters,xp):
    n=len(area)
    if leaf.shape!=(n,10) or parameters.shape!=(n,8):raise ValueError('PFT optics require leaf(n,10) and parameters(n,8)')
    if leaf.dtype!=area.dtype or parameters.dtype!=area.dtype:raise ValueError('PFT optical floats must match area precision')
    if not all(bool(xp.all(xp.isfinite(v))) for v in (leaf,parameters)):
        raise ValueError('Nonfinite leaf/stem optical input')
    if bool(xp.any((leaf<0.)|(leaf>area))) or bool(xp.any((parameters<0.)|(parameters>1.))):
        raise ValueError('Unphysical leaf/stem optical input')
    if bool(xp.any((parameters[:,:2]+parameters[:,2:4]>1.)|(parameters[:,4:6]+parameters[:,6:8]>1.))):
        raise ValueError('Leaf and stem optical properties must each be passive')


def mixture_kernel_jax(area,leaf,parameters):
    import jax.numpy as j
    from jax import lax
    s=lax.optimization_barrier
    lr,lt,sr,st=parameters.reshape(4,2)[:,:,None]
    stem=s(area-leaf);den=j.where(area>0.,area,1.)
    rho=j.where(area>0.,s(s(lr*leaf)+s(sr*stem))/den,lr)
    tau=j.where(area>0.,s(s(lt*leaf)+s(st*stem))/den,lt)
    rho=j.minimum(rho,1.)
    tau=j.minimum(tau,s(1.-rho))
    lw=s(leaf*s(1.-s(lr+lt)));sw=s(stem*s(1.-s(sr+st)));weight=s(lw+sw)
    fraction=j.where(weight>0.,lw/j.where(weight>0.,weight,1.),0.)
    return rho,tau,fraction


def unpack_canopy_snow_optics(out):
    from .canopy_snow_radiation import unpack_canopy_snow
    result=unpack_canopy_snow(out[:,:584])
    result.update(leaf_absorbed_bands=out[:,584:604].reshape(-1,2,10),
                  stem_absorbed_bands=out[:,604:624].reshape(-1,2,10))
    return result


def attach_optical_absorption(result,radiation):
    if radiation is not None:
        result.update(leaf_absorbed_shortwave=radiation[:,584:604].reshape(-1,2,10),
                      stem_absorbed_shortwave=radiation[:,604:624].reshape(-1,2,10))
    return result


@lru_cache(maxsize=1)
def _compiled():
    import jax
    from .canopy_snow_radiation import canopy_snow_kernel_jax
    def kernel(area,cells,count,inputs,leaf,p,ice,impurity):
        return canopy_snow_kernel_jax(area,cells,count,inputs,ice,impurity,track_absorption=True,
            photosynthetic_leaf_area=leaf,canopy_optics=p)
    return jax.jit(jax.vmap(kernel,in_axes=(0,0,0,0,0,0,None,None)))


def canopy_snow_optics_raw_jax(area,cells,count,inputs,leaf,parameters):
    import jax.numpy as j
    from .canopy_snow_radiation import validate_canopy_snow
    from .snicar import _jax_tables
    area,cells,count,inputs,leaf,parameters=map(j.asarray,(area,cells,count,inputs,leaf,parameters))
    validate_canopy_snow(area,cells,count,inputs,j);validate_canopy_optics(area,leaf,parameters,j)
    out=_compiled()(area,cells,count,inputs,leaf,parameters,*_jax_tables(str(area.dtype)))
    if not bool(j.all(j.isfinite(out))):raise RuntimeError('Nonfinite PFT canopy optics')
    return out


def run_canopy_snow_optics_jax(area,cells,count,inputs,leaf,parameters):
    return unpack_canopy_snow_optics(canopy_snow_optics_raw_jax(area,cells,count,inputs,leaf,parameters))
