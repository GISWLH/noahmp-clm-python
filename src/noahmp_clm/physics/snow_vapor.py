"""Snow surface vapour mass and enthalpy exchange (CLM5 2.5.145--148).

Loss is partitioned by the top layer's ice/liquid mass fractions; deposition
is frost below freezing and liquid dew at/above freezing. Latent heat and
transported condensed-phase enthalpy are separately reported, never discarded.
"""
import numpy as np
from .snowpack import Snowpack,TF,CI,CW,LF,RHO_ICE,regrid,equilibrium

LV=2.5e6


def exchange_vapor(pack,amount_kgm2,*,deposition_temperature=None,remap=True):
    """Return new pack and signed outward mass/heat ledger for top exchange.

    Positive amount removes water proportionally from the top layer only.
    Negative amount deposits frost/dew. The surface solver must enforce the
    top-layer supply before calling this function; no post-solve clipping is
    used. Empty-pack frost is supported; empty-pack dew belongs in soil.
    """
    amount=float(amount_kgm2)
    if not np.isfinite(amount): raise ValueError('Nonfinite snow vapour exchange')
    if amount==0.:
        return pack,dict(mass_kgm2=0.,ice_kgm2=0.,liquid_kgm2=0.,advected_enthalpy_jm2=0.,latent_energy_jm2=0.,residue_kgm2=np.zeros(5))
    if amount>0.:
        # Two independently limited surface branches and the rate-to-mass
        # conversion can accumulate several ULPs at exact exhaustion.
        roundoff=64.*np.finfo(float).eps*max(amount,float(pack.mass[0]) if len(pack.mass) else 0.)
        if not len(pack.mass) or amount>pack.mass[0]+roundoff:
            raise ValueError('Snow vapour removal exceeds top-layer water supply')
        # Permit only representational roundoff at an exactly exhausted layer.
        if abs(amount-pack.mass[0])<=roundoff: amount=float(pack.mass[0])
        fraction=amount/pack.mass[0]
        ice=amount*pack.ice[0]/pack.mass[0];liquid=amount-ice
        heat=fraction*pack.enthalpy[0]
        m,h,z,a=pack.mass.copy(),pack.enthalpy.copy(),pack.thickness.copy(),pack.impurities.copy()
        m[0]-=amount;h[0]-=heat;z[0]*=1.-fraction
        residue=np.zeros(5)
        if m[0]==0.:
            if len(m)>1: a[1]+=a[0]
            else: residue=a[0].copy()
        keep=m>0.
        result=Snowpack(m[keep],h[keep],z[keep],pack.radius_um[keep],a[keep]) if np.any(keep) else Snowpack.empty()
    else:
        temperature=(float(pack.temperature[0]) if deposition_temperature is None and len(pack.mass)
                     else deposition_temperature)
        if temperature is None or not np.isfinite(temperature) or not 200.<=temperature<=350.:
            raise ValueError('Deposition needs a valid surface temperature')
        ice=amount if temperature<TF else 0.;liquid=amount-ice
        heat=ice*CI*(temperature-TF)+liquid*(LF+CW*(temperature-TF))
        residue=np.zeros(5)
        if len(pack.mass):
            m,h,z=pack.mass.copy(),pack.enthalpy.copy(),pack.thickness.copy()
            m[0]-=amount;h[0]-=heat
            # Deposited frost adds porous volume at 200 kg/m3; dew occupies
            # existing pores until drainage. Include any freezing expansion.
            z[0]-=ice/200.
            _,frozen,_=equilibrium(m[0],h[0]);z[0]=max(z[0],float(frozen)/RHO_ICE)
            result=Snowpack(m,h,z,pack.radius_um,pack.impurities)
        elif ice<0.:
            result=Snowpack.from_ice([-ice],temperature,200.)
        else: raise ValueError('Dew on an empty pack must be routed to soil')
    if remap and len(result.mass): result=regrid(result)
    return result,dict(mass_kgm2=amount,ice_kgm2=float(ice),liquid_kgm2=float(liquid),
        advected_enthalpy_jm2=float(heat),latent_energy_jm2=float(LV*amount+LF*ice),residue_kgm2=residue)
