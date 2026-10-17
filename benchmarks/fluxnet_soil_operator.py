"""Explicit depth and time operators for FLUXNET soil states.

The FULLSET table defines TS in deg C and SWC in percent. It does not specify
how each sensor responds to ice, so liquid and total model water are both
reported. No choice is made from their agreement with observations.
"""
from __future__ import annotations
import numpy as np


def depth_weights(dz, depth):
    dz = np.asarray(dz, float)
    if dz.ndim != 1 or not dz.size or not np.all(np.isfinite(dz)) or np.any(dz <= 0):
        raise ValueError('Finite positive soil layer thicknesses required')
    if not np.isfinite(depth) or not 0 <= depth <= dz.sum():
        raise ValueError('Sensor depth is outside the model soil column')
    centres = np.cumsum(dz)-.5*dz
    weights = np.zeros_like(dz)
    if depth <= centres[0]: weights[0] = 1.
    elif depth >= centres[-1]: weights[-1] = 1.
    else:
        right = int(np.searchsorted(centres, depth)); left = right-1
        weights[right] = (depth-centres[left])/(centres[right]-centres[left])
        weights[left] = 1.-weights[right]
    return weights


def initial_soil_from_manifest(manifest):
    from noahmp_clm.column import SoilParams
    from noahmp_clm.physics.soil_enthalpy import SoilEnthalpyState
    raw = manifest['initial_state']['ground'].copy()
    if 'saturated_suction_m' not in raw['params']:
        raise ValueError('Historical soil enthalpy requires its original frozen source; saturated_suction_m is absent')
    raw['params'] = SoilParams(**raw['params'])
    state = SoilEnthalpyState(**raw)
    return {'dz': state.dz, 'tsoil_k': state.temperature,
            'smc_m3m3': state.water/(1000.*state.dz),
            'liquid_m3m3': state.liquid/(1000.*state.dz)}


def soil_products(model, observed, initial, operator):
    """Map end-of-step layer states to estimated interval-mean sensor values.

    Linear interpolation between layer centres with constant outer half cells
    is a declared approximation, not a claim to resolve within-layer gradients.
    Trapezoidal means use the initial state before the first interval.
    """
    if operator.get('spatial_method') != 'linear_centres_constant_outer_half_cell':
        raise ValueError('Explicit supported spatial soil operator required')
    if operator.get('temporal_method') != 'trapezoid_endpoints':
        raise ValueError('Explicit supported temporal soil operator required')
    if not operator.get('source'): raise ValueError('Sensor depth source required')
    dz = np.asarray(initial['dz'], float); n = len(model['time_start'])
    wt = depth_weights(dz, float(operator['temperature_depth_m']))
    ww = depth_weights(dz, float(operator['water_depth_m']))
    mapped = {}
    for key, weights in [('tsoil_k', wt), ('smc_m3m3', ww), ('liquid_m3m3', ww)]:
        data = np.asarray(model[key], float); first = np.asarray(initial[key], float)
        if data.shape != (n, len(dz)) or first.shape != dz.shape:
            raise ValueError(f'Initial and endpoint soil profile shapes disagree: {key}')
        if not np.all(np.isfinite(data)) or not np.all(np.isfinite(first)):
            raise ValueError(f'Nonfinite soil state: {key}')
        endpoints = np.r_[first @ weights, data @ weights]
        mapped[key] = .5*(endpoints[:-1]+endpoints[1:])
    products = {}
    for label, field, key, variable, factor, offset in [
        ('TS_F_MDS_1', 'TS_F_MDS_1', 'tsoil_k', 'tsoil_c', 1., -273.15),
        ('SWC_F_MDS_1_liquid', 'SWC_F_MDS_1', 'liquid_m3m3', 'smc', .01, 0.),
        ('SWC_F_MDS_1_total', 'SWC_F_MDS_1', 'smc_m3m3', 'smc', .01, 0.)]:
        obs = np.asarray(observed['obs_'+field], float)
        qc = np.asarray(observed['qc_'+field], float)
        if obs.shape != (n,) or qc.shape != (n,):
            raise ValueError(f'Observation/QC shape mismatch: {field}')
        products[label] = (mapped[key]+offset, obs*factor, qc, variable, False)
    metadata = dict(operator)
    metadata.update(layer_thickness_m=dz.tolist(), temperature_weights=wt.tolist(),
        water_weights=ww.tolist(), initial_state_included=True,
        soil_water_interpretation='Both liquid and total water retained; sensor ice response unverified',
        maximum_liquid_total_difference_m3m3=float(np.max(abs(mapped['smc_m3m3']-mapped['liquid_m3m3']), initial=0.)),
        units={'TS_F_MDS_1': 'deg C', 'SWC_F_MDS_1_liquid': 'm3/m3', 'SWC_F_MDS_1_total': 'm3/m3'},
        required_criterion='daily/monthly mean bias; state correlation is descriptive')
    return products, metadata
