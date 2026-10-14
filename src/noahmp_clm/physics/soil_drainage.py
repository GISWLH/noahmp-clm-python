"""Liquid gravity drainage with explicitly advected enthalpy.

Retains the Campbell conductivity of the earlier soil-water scheme. Transfers
cannot consume ice or overfill the receiving pore volume. Drainage is an
operator-split step; no lateral exchange or river routing is represented.
"""
import numpy as np
from .soil_enthalpy import SoilEnthalpyState
from .snowpack import TF,LF,CW


def drain_soil(state,dt,*,permeability_option=1):
    """Return state and bottom-outflow/per-interface mass/enthalpy ledger."""
    if not np.isfinite(dt) or dt<=0. or permeability_option not in (1,2):
        raise ValueError('Invalid drainage time step or frozen permeability option')
    p=state.params;current=state;n=len(state.water);mass=np.zeros(n);heat=np.zeros(n)
    for j in range(n):
        w=current.water.copy();h=current.enthalpy.copy()
        liquid=current.liquid;ice=w-liquid;temperature=current.temperature
        wet=np.clip(w[j]/(1000.*p.smcmax*state.dz[j]),0.,1.)
        frozen=1. if permeability_option==1 else np.exp(-7.*ice[j]/(1000.*p.smcmax*state.dz[j]))
        hydraulic=1000.*dt*p.satdk*frozen*wet**(2.*p.bexp+3.)
        excess=max(0.,w[j]-1000.*p.smcref*state.dz[j])
        transfer=min(hydraulic,excess,liquid[j])
        if j+1<n: transfer=min(transfer,max(0.,1000.*p.smcmax*state.dz[j+1]-w[j+1]))
        energy=transfer*(LF+CW*(temperature[j]-TF))
        w[j]-=transfer;h[j]-=energy
        if j+1<n: w[j+1]+=transfer;h[j+1]+=energy
        mass[j]=transfer;heat[j]=energy
        current=SoilEnthalpyState(w,h,state.dz,p,state.freezing_option)
    return current,dict(bottom_mass_kgm2=float(mass[-1]),bottom_enthalpy_jm2=float(heat[-1]),
        interface_mass_kgm2=mass,interface_enthalpy_jm2=heat,
        water_residual_kgm2=float(current.water.sum()+mass[-1]-state.water.sum()),
        energy_residual_jm2=float(current.enthalpy.sum()+heat[-1]-state.enthalpy.sum()))
