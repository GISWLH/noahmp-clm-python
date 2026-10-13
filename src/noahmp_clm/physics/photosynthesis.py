"""C3 Farquhar leaf biochemistry and coupled Ball--Berry gas exchange.

Original implementation of published equations, not a translation of CTSM.
References and scope: docs/CLM_PHOTOSYNTHESIS.md. Rates are micromol CO2 per
m² leaf per second; CO2 is ppm (micromol/mol), PAR is absorbed micromol photons
per m² leaf per second, conductance is mol H2O per m² leaf per second.
This leaf component is not yet connected to the column driver.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

R_GAS = 8.314462618
T_REFERENCE = 298.15


@dataclass(frozen=True)
class LeafParameters:
    vcmax25: float = 60.0
    jmax25: float = 100.0
    tpu25: float = 10.0
    respiration25: float = 0.9
    quantum_yield_psii: float = 0.85
    curvature: float = 0.7
    ball_berry_slope: float = 9.0
    minimum_conductance: float = 0.01
    growth_temperature: float = 298.15
    oxygen_ppm: float = 210000.0

    def __post_init__(self):
        values = np.array(list(vars(self).values()), float)
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("Leaf parameters must be finite and nonnegative")
        if not (0.0 < self.curvature <= 1.0 and 0.0 < self.quantum_yield_psii <= 1.0
                and self.minimum_conductance > 0.0 and 250.0 <= self.growth_temperature <= 320.0):
            raise ValueError("Invalid leaf efficiency, conductance, or growth temperature")


def _arrhenius(temperature, activation):
    return np.exp(activation / (R_GAS * T_REFERENCE) * (1.0 - T_REFERENCE / temperature))


def _peaked(temperature, activation, deactivation, entropy):
    numerator = 1.0 + np.exp((T_REFERENCE * entropy - deactivation) / (R_GAS * T_REFERENCE))
    denominator = 1.0 + np.exp((temperature * entropy - deactivation) / (R_GAS * temperature))
    return _arrhenius(temperature, activation) * numerator / denominator


def farquhar_rates(ci_ppm, absorbed_par, leaf_temperature, parameters=None, *, nitrogen_factor=1.0):
    """Rubisco/RuBP/TPU limitation, gross uptake and net uptake at specified ci.

    Use the requested minimum of the three limitations. Electron transport
    uses the stable small quadratic root, including exactly zero light.
    Nitrogen scales biochemical capacities, not the separately supplied Rd.
    """
    p = parameters or LeafParameters()
    ci, par, temperature, nitrogen = np.broadcast_arrays(
        np.asarray(ci_ppm, float), np.asarray(absorbed_par, float),
        np.asarray(leaf_temperature, float), np.asarray(nitrogen_factor, float))
    if not all(np.all(np.isfinite(x)) for x in (ci, par, temperature, nitrogen)):
        raise ValueError("Nonfinite photosynthesis input")
    if np.any((ci < 0) | (par < 0) | (temperature < 200) | (temperature > 350)
              | (nitrogen < 0) | (nitrogen > 1)):
        raise ValueError("Unphysical photosynthesis input")
    return _rates_at_ci(ci, _biochemical_coefficients(par, temperature, nitrogen, p))


def _biochemical_coefficients(par, temperature, nitrogen, p):
    """Coefficients fixed during the ci root solve; inputs validated by caller."""
    growth_c = np.clip(p.growth_temperature - 273.15, 11.0, 35.0)
    vc_factor = _peaked(temperature, 72000., 200000., 668.39 - 1.07 * growth_c)
    vcmax = p.vcmax25 * nitrogen * vc_factor
    jmax = p.jmax25 * nitrogen * _peaked(temperature, 50000., 200000., 659.70 - .75 * growth_c)
    tpu = p.tpu25 * nitrogen * vc_factor
    rd = p.respiration25 * _peaked(temperature, 46390., 150650., 490.)
    kc = 404.9 * _arrhenius(temperature, 79430.)
    ko = 278400. * _arrhenius(temperature, 36380.)
    gamma = 42.75 * _arrhenius(temperature, 37830.)
    light = .5 * p.quantum_yield_psii * par
    total = light + jmax
    disc = np.maximum(total * total - 4.0 * p.curvature * light * jmax, 0.0)
    denominator = total + np.sqrt(disc)
    electron_transport = np.divide(2.0 * light * jmax, denominator,
        out=np.zeros_like(denominator), where=denominator > 0.0)
    return vcmax, kc*(1.+p.oxygen_ppm/ko), gamma, 3.*tpu, rd, electron_transport


def _rates_at_ci(ci, coefficients):
    vcmax, michaelis, gamma, tpu, rd, electron_transport = coefficients
    carbon = np.maximum(ci - gamma, 0.0)
    rubisco = vcmax * carbon / (ci + michaelis)
    rubp = electron_transport * carbon / (4.0 * ci + 8.0 * gamma)
    product = tpu + np.zeros_like(ci)
    gross = np.minimum(np.minimum(rubisco, rubp), product)
    return {"gross": gross, "net": gross-rd, "rubisco": rubisco,
            "rubp": rubp, "tpu": product, "respiration": rd, "electron_transport": electron_transport}


def ball_berry_conductance(net_assimilation, surface_co2_ppm, relative_humidity,
                           boundary_conductance, parameters=None, *, water_stress=1.0):
    """Positive root of Ball--Berry with leaf-surface humidity feedback.

    ``relative_humidity`` is ambient vapor pressure / leaf saturation pressure.
    The boundary conductance includes the leaf boundary layer, not stomata.
    """
    p = parameters or LeafParameters()
    a, cs, rh, gb, stress = np.broadcast_arrays(np.asarray(net_assimilation, float),
        np.asarray(surface_co2_ppm, float), np.asarray(relative_humidity, float),
        np.asarray(boundary_conductance, float), np.asarray(water_stress, float))
    if not all(np.all(np.isfinite(x)) for x in (a, cs, rh, gb, stress)):
        raise ValueError("Nonfinite gas-exchange input")
    if np.any((cs <= 0) | (gb <= 0) | (rh < 0) | (stress < 0) | (stress > 1)):
        raise ValueError("Unphysical gas-exchange input")
    return _ball_berry(a,cs,rh,gb,stress,p)


def _ball_berry(a,cs,rh,gb,stress,p):
    x = p.ball_berry_slope * np.maximum(a, 0.0) * stress / cs
    b = gb - p.minimum_conductance - x
    c = gb * (p.minimum_conductance + x * np.minimum(rh, 1.0))
    root = np.sqrt(b*b + 4.0*c)
    # Both forms are algebraically identical; choose the non-cancelling one.
    positive = 2.0*c / np.maximum(b + root, 1e-30)
    return np.where(b >= 0.0, positive, .5*(root-b))


def leaf_gas_exchange(absorbed_par, leaf_temperature, *, co2_ppm=420.0,
                      relative_humidity=0.7, boundary_conductance=1.0,
                      parameters=None, nitrogen_factor=1.0, water_stress=1.0):
    """Solve biochemical demand = diffusive supply for ci with a fixed bracket.

    The night-time bracket allows ci above ambient (outward respiration).
    Return the actual diffusion residual, not a convergence flag alone.
    """
    p = parameters or LeafParameters()
    par, temp, ca, rh, gb, nitrogen, stress = np.broadcast_arrays(np.asarray(absorbed_par, float),
        np.asarray(leaf_temperature, float), np.asarray(co2_ppm, float),
        np.asarray(relative_humidity, float), np.asarray(boundary_conductance, float),
        np.asarray(nitrogen_factor,float),np.asarray(water_stress,float))
    if not all(np.all(np.isfinite(x)) for x in (par,temp,ca,rh,gb,nitrogen,stress)):
        raise ValueError("Nonfinite leaf input")
    if np.any((par<0)|(temp<200)|(temp>350)|(ca<=0)|(rh<0)|(gb<=0)
              |(nitrogen<0)|(nitrogen>1)|(stress<0)|(stress>1)):
        raise ValueError("Unphysical leaf input")
    coefficients = _biochemical_coefficients(par,temp,nitrogen,p)
    rd = coefficients[4]
    lo = np.zeros_like(ca)
    hi = ca + rd * (1.4 / gb + 1.6 / p.minimum_conductance) + 1.0

    def evaluate(ci):
        rates = _rates_at_ci(ci,coefficients)
        cs = np.maximum(ca - 1.4 * rates["net"] / gb, 1e-9)
        gs = _ball_berry(rates["net"],cs,rh,gb,stress,p)
        residual = rates["net"] - (ca-ci) / (1.4/gb + 1.6/gs)
        return rates, cs, gs, residual

    for _ in range(48):
        ci = .5 * (lo+hi)
        _, _, _, residual = evaluate(ci)
        lo = np.where(residual < 0.0, ci, lo)
        hi = np.where(residual >= 0.0, ci, hi)
    ci = .5 * (lo+hi)
    rates, cs, gs, residual = evaluate(ci)
    return {**rates, "ci_ppm": ci, "cs_ppm": cs, "gs_mol": gs,
        "surface_relative_humidity": (gb * np.minimum(rh, 1.0) + gs) / (gb+gs),
        "diffusion_residual": residual}
