"""Layered, conservative two-stream radiation with direct-beam scattering.

Diffuse equations follow CLM5 technical note 2.3.1--2.3.2. An original
adding/doubling solver approaches the continuous solution from passive thin
slabs; no CTSM source is copied. Fluxes are W/m² ground. The caller supplies
one-sided leaf area per layer, ordered top to bottom.
"""
from __future__ import annotations
import numpy as np


def layer_coefficients(leaf_area, reflectance, transmittance, *,
                       cosine_zenith=.5, diffuse_extinction=1., doublings=10):
    """Return r, t, beam-up, beam-down, direct transmittance for each layer.

    Spherical leaf angles: projected area 0.5, diffuse upscatter 5/8 of
    reflectance + 3/8 of transmittance. Beam upscatter uses the spherical
    analytic integral in CLM5 eqs. 2.3.15--16. Ten doublings (1024 thin
    slabs/layer) give a convergent numerical approximation, tested against
    an independent matrix-exponential boundary-value solution.
    """
    lai, refl, tran = np.broadcast_arrays(np.asarray(leaf_area, float),
        np.asarray(reflectance, float), np.asarray(transmittance, float))
    if lai.ndim != 1 or not lai.size:
        raise ValueError("Leaf area must be a nonempty one-dimensional array")
    if not all(np.all(np.isfinite(x)) for x in (lai, refl, tran)):
        raise ValueError("Nonfinite canopy optical input")
    if np.any((lai < 0) | (refl < 0) | (tran < 0) | (refl+tran > 1)):
        raise ValueError("Negative area or non-passive leaf optical properties")
    if not (0 < cosine_zenith <= 1 and 0 < diffuse_extinction <= 10):
        raise ValueError("Invalid zenith cosine or diffuse extinction")
    if not isinstance(doublings, int) or not 0 <= doublings <= 16:
        raise ValueError("doublings must be an integer in [0,16]")
    k = .5/cosine_zenith
    omega = refl+tran
    backscatter = .625*refl + .375*tran
    beta_beam = .5*(1.+k)/k * (1.-cosine_zenith*np.log1p(1./cosine_zenith))
    a = (1.-omega+backscatter)*diffuse_extinction
    b = backscatter*diffuse_extinction
    width = lai/(2**doublings)
    t = 1./(1.+a*width)
    r = b*width*t
    intercepted = -np.expm1(-k*width)
    direct = 1.-intercepted
    up = omega*beta_beam*intercepted
    down = omega*(1.-beta_beam)*intercepted
    for _ in range(doublings):
        den = 1.-r*r
        r, t, up, down, direct = (r+t*t*r/den, t*t/den,
            up+t*(r*down+up*direct)/den,
            down*direct+t*(down+r*up*direct)/den, direct*direct)
        # Exact non-absorbing limit: prevent roundoff accumulated by doubling
        # from inventing absorption in a purely scattering slab.
        t = np.where(omega == 1., 1.-r, t)
        down = np.where(omega == 1., 1.-direct-up, down)
    return r, t, up, down, direct


def _solve_layers(r, t, source_up, source_down, top_diffuse, bottom_albedo, bottom_source):
    """Solve layer interface fluxes by eliminating boundaries bottom-up."""
    n = len(r)
    reflect = np.zeros(n+1)
    source = np.zeros(n+1)
    reflect[n], source[n] = bottom_albedo, bottom_source
    for j in range(n-1, -1, -1):
        den = 1.-r[j]*reflect[j+1]
        reflect[j] = r[j]+t[j]*t[j]*reflect[j+1]/den
        source[j] = source_up[j]+t[j]*(source[j+1]+reflect[j+1]*source_down[j])/den
    diffuse = np.zeros(n+1)
    diffuse[0] = top_diffuse
    for j in range(n):
        diffuse[j+1] = (t[j]*diffuse[j]+r[j]*source[j+1]+source_down[j])/(1.-r[j]*reflect[j+1])
    upward = reflect*diffuse+source
    return diffuse, upward


def two_stream_shortwave(leaf_area, direct_flux, diffuse_flux, *,
                         leaf_reflectance=.1, leaf_transmittance=.05,
                         ground_albedo=.2, cosine_zenith=.5, doublings=10):
    """Single spectral band; all reflected and transmitted streams are explicit."""
    if not (np.isfinite(direct_flux) and np.isfinite(diffuse_flux)
            and direct_flux >= 0 and diffuse_flux >= 0 and 0 <= ground_albedo <= 1):
        raise ValueError("Invalid radiation boundary conditions")
    r, t, beam_up, beam_down, beam_t = layer_coefficients(leaf_area,
        leaf_reflectance, leaf_transmittance, cosine_zenith=cosine_zenith, doublings=doublings)
    direct = np.concatenate(([direct_flux], direct_flux*np.cumprod(beam_t)))
    diffuse, upward = _solve_layers(r, t, beam_up*direct[:-1],
        beam_down*direct[:-1], diffuse_flux, ground_albedo, ground_albedo*direct[-1])
    absorbed = direct[:-1]+diffuse[:-1]+upward[1:]-direct[1:]-diffuse[1:]-upward[:-1]
    ground = (1.-ground_albedo)*(direct[-1]+diffuse[-1])
    residual = direct_flux+diffuse_flux-upward[0]-np.sum(absorbed)-ground
    return {"absorbed": absorbed, "direct": direct, "diffuse_down": diffuse,
        "diffuse_up": upward, "reflected": upward[0], "ground_absorbed": ground,
        "residual": residual}


def canopy_shortwave(leaf_area, swdown, *, diffuse_fraction=.5, cosine_zenith=.5,
                     visible_fraction=.5, ground_albedo=(.15,.3),
                     leaf_reflectance=(.1,.45), leaf_transmittance=(.05,.25)):
    """Visible/NIR bands; PAR conversion is 4.6 µmol photon/J of visible light.

    Fractions and zenith are explicit inputs, to be mapped from site forcing;
    defaults are assumptions and must not be described as measured radiation.
    """
    if not (0 <= diffuse_fraction <= 1 and 0 <= visible_fraction <= 1):
        raise ValueError("Radiation fractions must lie in [0,1]")
    bands = []
    for i, fraction in enumerate((visible_fraction, 1.-visible_fraction)):
        bands.append(two_stream_shortwave(leaf_area, swdown*fraction*(1.-diffuse_fraction),
            swdown*fraction*diffuse_fraction, leaf_reflectance=leaf_reflectance[i],
            leaf_transmittance=leaf_transmittance[i], ground_albedo=ground_albedo[i],
            cosine_zenith=cosine_zenith))
    lai = np.asarray(leaf_area, float)
    par = np.divide(bands[0]["absorbed"]*4.6, lai, out=np.zeros_like(lai), where=lai>0)
    return {"bands": bands, "absorbed": bands[0]["absorbed"]+bands[1]["absorbed"],
        "absorbed_par": par, "ground_absorbed": sum(x["ground_absorbed"] for x in bands),
        "reflected": sum(x["reflected"] for x in bands),
        "residual": sum(x["residual"] for x in bands)}


def canopy_longwave(leaf_area, leaf_temperature, sky_flux, ground_temperature, *,
                    ground_emissivity=.97, extinction=1.):
    """Grey, non-scattering thermal canopy with both leaf faces emitting.

    Effective leaves are black in the thermal band. Ground reflection is
    included, and returned net absorptions can be negative (radiative cooling).
    """
    lai, temp = np.broadcast_arrays(np.asarray(leaf_area, float), np.asarray(leaf_temperature, float))
    if lai.ndim != 1 or not lai.size or np.any(lai < 0) or not np.all(np.isfinite(lai)):
        raise ValueError("Invalid longwave leaf area")
    if not (np.all(np.isfinite(temp)) and np.all((temp>=200)&(temp<=350))
            and 200 <= ground_temperature <= 350 and sky_flux >= 0 and np.isfinite(sky_flux)
            and 0 < ground_emissivity <= 1 and extinction > 0 and np.isfinite(extinction)):
        raise ValueError("Invalid longwave input")
    sigma = 5.670374419e-8
    t = np.exp(-extinction*lai)
    emission = (1.-t)*sigma*temp**4
    diffuse, upward = _solve_layers(np.zeros_like(t), t, emission, emission,
        sky_flux, 1.-ground_emissivity, ground_emissivity*sigma*ground_temperature**4)
    net = diffuse[:-1]+upward[1:]-diffuse[1:]-upward[:-1]
    ground = diffuse[-1]-upward[-1]
    return {"net_absorbed": net, "down": diffuse, "up": upward,
        "ground_net": ground, "outgoing": upward[0],
        "residual": sky_flux-upward[0]-net.sum()-ground}
