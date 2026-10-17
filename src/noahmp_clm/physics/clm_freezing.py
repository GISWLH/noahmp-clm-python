"""CLM5 Eq2.6.52 equilibrium liquid water with explicit saturated suction.

The positive suction head is in metres. A compact4% cap retains the accepted
C1 freeze-onset regularization; its physical change relative to min is about1%.
This NumPy reference is staged before integration into the extensive heat ABI.
"""
from __future__ import annotations
import numpy as np

FREEZING_TEMPERATURE=273.15
LATENT_HEAT_FUSION=334000.
GRAVITY=9.80616

def equilibrium_liquid_water(total_water,temperature,porosity,bexp,saturated_suction_m):
    """Return liquid m3/m3 and analytic temperature/total-water derivatives.

    Below freezing, cap theta_sat*[Lf*(Tf-T)/(g*T*psi_sat)]**(-1/B)
    by the available total water. No common suction or air-vapour scale is
    substituted for the supplied soil texture parameter. T is150--400 K.
    """
    total,t,sat,b,psi=np.broadcast_arrays(*[np.asarray(v,float) for v in
        [total_water,temperature,porosity,bexp,saturated_suction_m]])
    if (not all(np.all(np.isfinite(v)) for v in [total,t,sat,b,psi]) or
        np.any((t<150.)|(t>400.)) or np.any((sat<=0.)|(sat>=1.)) or
        np.any((total<0.)|(total>sat)) or np.any(b<=0.) or np.any((psi<1e-12)|(psi>1e6))):
        raise ValueError('Finite physical water/porosity,150--400K,positive B and suction head in metres required')
    deficit=np.maximum(FREEZING_TEMPERATURE-t,0.)
    coefficient=LATENT_HEAT_FUSION/(GRAVITY*psi)
    raw=coefficient*deficit/t;ratio=np.maximum(raw,1e-12)
    # Above2*porosity the available-water cap is identically inactive. This
    # mathematically equivalent bound prevents a warm-side power overflow.
    potential=np.exp(np.minimum(np.log(sat)-np.log(ratio)/b,np.log(2.*sat)))
    dpotential=np.where(raw>1e-12,potential*coefficient*FREEZING_TEMPERATURE/(b*t*t*ratio),0.)
    width=np.maximum(.04*total,1e-12);dw=np.where(.04*total>1e-12,.04,0.)
    difference=total-potential;sign=np.sign(difference)
    rounded=np.maximum(width-np.abs(difference),0.)
    liquid=np.maximum(np.minimum(total,potential)-rounded*rounded/(4.*width),0.)
    dtotal=np.where(difference<0.,1.,np.where(difference>0.,0.,.5))
    dtotal-=rounded*(dw-sign)/(2.*width)
    dtotal+=rounded*rounded*dw/(4.*width*width)
    dcap=np.where(difference>0.,1.,np.where(difference<0.,0.,.5))-rounded*sign/(2.*width)
    dtemperature=np.where(liquid>0.,dcap*dpotential,0.)
    dtotal=np.where(liquid>0.,dtotal,0.)
    unfrozen=t>=FREEZING_TEMPERATURE
    return dict(liquid=np.where(unfrozen,total,liquid),
        d_liquid_d_temperature=np.where(unfrozen,0.,dtemperature),
        d_liquid_d_total=np.where(unfrozen,1.,dtotal))
