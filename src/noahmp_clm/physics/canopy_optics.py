"""Explicit two-band leaf/stem optics for a horizontally mixed canopy.

Each layer retains the existing spherical leaf-angle assumption. Leaf and
stem extinction area mix their scattering coefficients; their absorptivities
partition absorbed photons. Stem absorption heats the canopy but supplies no
photosynthetic PAR. Fluxes here are per vegetated ground area.
"""
from dataclasses import dataclass
import numpy as np
from .canopy_radiation import canopy_shortwave


@dataclass(frozen=True)
class CanopyOptics:
    """Visible/NIR reflectance and transmittance, explicitly for both tissues.

    The defaults reproduce the existing generic optics. Site/PFT values are
    supplied explicitly, without selecting them from a flux score.
    """
    leaf_reflectance: tuple = (.1, .45)
    leaf_transmittance: tuple = (.05, .25)
    stem_reflectance: tuple = (.1, .45)
    stem_transmittance: tuple = (.05, .25)

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            a=np.asarray(getattr(self,name),float)
            if a.shape!=(2,) or not np.all(np.isfinite(a)) or np.any((a<0.)|(a>1.)):
                raise ValueError('Canopy optical properties require two finite fractions in [0,1]')
            object.__setattr__(self,name,tuple(float(v) for v in a))
        for tissue in ('leaf','stem'):
            if np.any(np.asarray(getattr(self,tissue+'_reflectance'))+
                      np.asarray(getattr(self,tissue+'_transmittance'))>1.):
                raise ValueError('Leaf and stem optical properties must each be passive')


def mixed_optical_properties(leaf_area,stem_area,optics):
    """Area-weight scattering, and absorption-weight the leaf photon fraction.

    Zero-area/zero-absorption entries have a zero leaf absorption fraction;
    no artificial minimum LAI, absorption or reflectance is introduced.
    """
    if not isinstance(optics,CanopyOptics):raise ValueError('Explicit CanopyOptics required')
    leaf=np.asarray(leaf_area,float)
    if leaf.ndim!=1 or not leaf.size:raise ValueError('Leaf area must be a nonempty vector')
    stem=np.broadcast_to(stem_area,leaf.shape).astype(float)
    if not np.all(np.isfinite(leaf)) or not np.all(np.isfinite(stem)) or np.any((leaf<0.)|(stem<0.)):
        raise ValueError('Leaf and stem areas must be finite and nonnegative')
    area=leaf+stem
    lr,lt,sr,st=(np.asarray(getattr(optics,n))[:,None] for n in optics.__dataclass_fields__)
    refl=np.divide(lr*leaf+sr*stem,area,out=np.broadcast_to(lr,(2,len(leaf))).copy(),where=area>0.)
    tran=np.divide(lt*leaf+st*stem,area,out=np.broadcast_to(lt,(2,len(leaf))).copy(),where=area>0.)
    # A convex mixture is passive analytically. Division can round a fully
    # scattering layer one ULP outside that domain; constrain coefficients
    # before solving transport, while retaining every raw flux diagnostic.
    refl=np.minimum(refl,1.)
    tran=np.minimum(tran,1.-refl)
    leaf_weight=leaf*(1.-(lr+lt));stem_weight=stem*(1.-(sr+st))
    denominator=leaf_weight+stem_weight
    fraction=np.divide(leaf_weight,denominator,out=np.zeros_like(denominator),where=denominator>0.)
    return area,refl,tran,fraction


def canopy_shortwave_pft(leaf_area,stem_area,swdown,optics,**kwargs):
    """Conservative two-stream fluxes with photosynthetic absorption separated.

    ``absorbed`` remains total canopy heating. ``absorbed_par`` is per leaf
    area and excludes stem absorption. Band absorption and both tissue totals
    are retained as diagnostics, including floating-point residuals.
    """
    area,refl,tran,fraction=mixed_optical_properties(leaf_area,stem_area,optics)
    result=canopy_shortwave(area,swdown,leaf_reflectance=refl,leaf_transmittance=tran,**kwargs)
    bands=np.asarray([b['absorbed'] for b in result['bands']])
    leaf_bands=bands*fraction;stem_bands=bands-leaf_bands
    result.update(leaf_absorbed_bands=leaf_bands,stem_absorbed_bands=stem_bands,
                  leaf_absorbed=leaf_bands.sum(axis=0),stem_absorbed=stem_bands.sum(axis=0),
                  absorbed_par_area=result['absorbed_par'])
    leaf=np.asarray(leaf_area,float)
    result['absorbed_par']=np.divide(leaf_bands[0]*4.6,leaf,out=np.zeros_like(leaf),where=leaf>0.)
    return result
