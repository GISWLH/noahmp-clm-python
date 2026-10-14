"""Implicit, mass-based snow enthalpy conduction with an explicit energy ledger.

Finite-volume Fourier heat flow, CLM5 technical note 2.6.7--9 and 2.6.85.
The time integrator is backward Euler in enthalpy (an original implementation),
not CTSM's temperature-first Crank--Nicolson/phase adjustment algorithm. Each
interface has one flux shared by both cells, including the bottom boundary.
"""
import numpy as np

from .snowpack import Snowpack, equilibrium, TF, CI, CW, LF


def conductivity(pack):
    """CLM/Jordan density-dependent snow conductivity, W m-1 K-1.

    Limit bulk density to pure ice for transient saturated cells. This routine
    describes snow, not a permanent liquid-water thermal layer.
    """
    rho = np.minimum(pack.density, 917.)
    return .023 + (7.75e-5*rho + 1.105e-6*rho*rho)*(2.29-.023)


def heat_step(pack, dt, top_flux, *, absorbed=None, bottom_temperature=TF,
              bottom_conductance=0., thermal_conductivity=None):
    """Advance a nonempty pack; return (new_pack, diagnostic dictionary).

    ``top_flux`` and per-layer ``absorbed`` are W/m2, positive into snow.
    ``bottom_conductance`` is the full center-to-center conductance in W/m2/K;
    a coupled caller must apply the returned bottom flux to its soil reservoir.
    Conductivities and geometry are frozen within this time step. All ice,
    mixed-phase and warm-water energy is retained; call ``drain_liquid`` next
    to route meltwater and its enthalpy. No excess energy is temperature-clipped.
    """
    n = pack.mass.size
    q = np.zeros(n) if absorbed is None else np.asarray(absorbed, float).copy()
    k = conductivity(pack) if thermal_conductivity is None else np.asarray(
        np.broadcast_to(thermal_conductivity, (n,)), float)
    if (not n or q.shape != (n,) or not np.all(np.isfinite(q))
            or not np.all(np.isfinite(k)) or np.any(k <= 0.)
            or not np.all(np.isfinite([dt,top_flux,bottom_temperature,bottom_conductance]))
            or dt <= 0. or bottom_conductance < 0.):
        raise ValueError('Invalid snow heat boundary, conductivity or time step')
    z, m = pack.thickness, pack.mass
    conduct = 1./(.5*z[:-1]/k[:-1] + .5*z[1:]/k[1:])
    operator = np.zeros((n,n))
    for j, g in enumerate(conduct):
        operator[j,j] -= g; operator[j+1,j+1] -= g
        operator[j,j+1] += g; operator[j+1,j] += g
    operator[-1,-1] -= bottom_conductance
    q[0] += top_flux
    q[-1] += bottom_conductance*(bottom_temperature-TF)
    h0 = pack.enthalpy
    # Work with temperature relative to the enthalpy origin, not absolute K.
    def offset(h):
        return np.where(h<0., h/(CI*m), np.where(h>LF*m,(h-LF*m)/(CW*m),0.))
    def residual(h):
        return h-h0-dt*(q+operator@offset(h))
    h = h0.copy()
    tolerance = max(2e-8, 64.*np.finfo(float).eps*max(
        np.max(np.abs(h0)), np.max(np.abs(dt*q)), 1.))
    iterations = 0
    for iterations in range(1,97):
        r = residual(h)
        norm = np.max(np.abs(r))
        if norm <= tolerance: break
        derivative = np.where(h<0., 1./(CI*m), np.where(h>LF*m,1./(CW*m),0.))
        jacobian = np.eye(n)-dt*operator*derivative[None,:]
        delta = np.linalg.solve(jacobian,-r)
        # A phase boundary makes the Jacobian discontinuous. Backtracking
        # globalizes the semismooth Newton step without altering the ledger.
        scale = 1.
        for _ in range(40):
            candidate = h+scale*delta
            if np.max(np.abs(residual(candidate))) < norm or scale < 2.**-35:
                break
            scale *= .5
        h = candidate
    r = residual(h)
    if np.max(np.abs(r)) > tolerance:
        raise RuntimeError(f'Snow enthalpy solve failed: {np.max(np.abs(r)):.9g} J/m2')
    temp = TF+offset(h)
    internal = conduct*(temp[:-1]-temp[1:])
    bottom = bottom_conductance*(temp[-1]-bottom_temperature)
    total_input = dt*(top_flux+(0. if absorbed is None else np.sum(absorbed))-bottom)
    balance = float(np.sum(h-h0)-total_input)
    return pack.with_enthalpy(h), dict(
        bottom_flux_wm2=float(bottom), interface_flux_wm2=internal,
        enthalpy_change_jm2=float(np.sum(h-h0)), input_energy_jm2=float(total_input),
        residual_jm2=balance, layer_residual_jm2=r, iterations=iterations,
        convergence_tolerance_jm2=float(tolerance))
