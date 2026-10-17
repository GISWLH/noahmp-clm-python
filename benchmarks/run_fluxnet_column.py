"""Reproducible NumPy CLM station trajectory with explicit configuration.

Every state is carried continuously. Periodic generation checkpoints bind the
trace, state, forcing, configuration and model hashes. A failed or partial run
is never marked complete for observation evaluation. Pickle checkpoints are
internal artifacts, not a format for untrusted external model inputs.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import pickle
import time
import traceback

import numpy as np
from noahmp_clm.column import ColumnState, SoilParams, Forcing
from noahmp_clm.options import PhysicsOptions
from noahmp_clm.clm_state import CLMColumnState
from noahmp_clm.clm_enthalpy_driver import step_clm_enthalpy_column
from noahmp_clm.physics.biogeochemistry import BGCState, BGCParameters
from noahmp_clm.physics.plant_cn import PlantState, PlantParameters
from noahmp_clm.physics.decomposition import DecompositionState, DecompositionParameters
from noahmp_clm.physics.phenology import PhenologyState, PhenologyParameters, PhenologyEnvironment
from noahmp_clm.physics.canopy_most import MOSTGeometry
from noahmp_clm.physics.photosynthesis import LeafParameters
from fluxnet_solar import solar_geometry
from fluxnet_deposition import DepositionSchedule

ROOT = Path(__file__).resolve().parents[1]
TRACE = {'h_wm2': 'h', 'le_wm2': 'le', 'gpp_gc_m2': 'gpp_gc_m2',
         'nee_gc_m2': 'nee_gc_m2', 'et_mm': 'et_mm', 'runoff_mm': 'runoff_mm',
         'autotrophic_respiration_gc_m2': 'autotrophic_respiration_gc_m2',
         'heterotrophic_respiration_gc_m2': 'heterotrophic_respiration_gc_m2'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    paths = sorted((ROOT/'src/noahmp_clm').rglob('*.py'))
    paths += [Path(__file__), ROOT/'benchmarks/fluxnet_solar.py', ROOT/'benchmarks/fluxnet_deposition.py']
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}


def encode(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    raise TypeError(type(value).__name__)


def atomic_json(path, value):
    path = Path(path); temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=encode, allow_nan=False)+'\n', encoding='utf-8')
    os.replace(temporary, path)


def build_state(config):
    """All site choices are supplied explicitly; unknown fields fail loudly."""
    c = config['column'].copy()
    c['soil'] = SoilParams(**c['soil']); c['options'] = PhysicsOptions(**c['options'])
    for key in ('dz', 'tsoil', 'smc', 'root_frac'):
        c[key] = np.asarray(c[key], float)
    state = CLMColumnState.from_column(ColumnState(**c), options=c['options']).with_layered_canopy()
    seasonal = config['phenology']
    if seasonal is not None:
        state = state.with_phenology(PhenologyParameters(**seasonal['parameters']), PhenologyState(**seasonal['initial']))
    b = config['biogeochemistry']
    p = b['plant_state']; s = b['soil_state']
    plant = PlantState(np.array(p['carbon']), np.array(p['nitrogen']), p['reserve_carbon'], p['reserve_nitrogen'])
    soil = DecompositionState(np.array(s['carbon']), np.array(s['nitrogen']), s['mineral_nitrogen'])
    parameters = BGCParameters(PlantParameters(**b['plant_parameters']), DecompositionParameters(**b['decomposition_parameters']),
                              b['maximum_lai'], b['saturated_potential_mpa'])
    return state.with_biogeochemistry(BGCState(plant, soil), parameters)


def calendar(stamp, metadata, seasonal):
    if not seasonal: return None
    def length(day):
        return solar_geometry(day.replace(hour=12, minute=0), metadata['latitude'], metadata['longitude'],
                              metadata['utc_offset_hours'])['daylength_seconds']
    current, previous, before = [length(stamp-timedelta(days=k)) for k in range(3)]
    increasing = current > previous
    crossing = stamp.hour == 0 and stamp.minute == 0 and increasing and previous <= before
    return PhenologyEnvironment(current, bool(increasing), bool(crossing), metadata['latitude'])


def totals(state):
    return np.array([state.water_kgm2, state.enthalpy_jm2, state.carbon_gc_m2, state.nitrogen_gn_m2])


def audit_step(old, new, forcing, diagnostic, nitrogen):
    d = diagnostic; f = forcing
    incoming = np.array([f.prcp*f.dt*1000.-d['et_mm']-d['runoff_mm'],
        d['precipitation_enthalpy_jm2']+(f.swdown+f.lwdown-d['outgoing_shortwave']-d['outgoing_longwave']-d['h']-d['le'])*f.dt
        -d['advected_enthalpy_jm2']-d['runoff_enthalpy_jm2'], -d['nee_gc_m2'], nitrogen])
    residual = totals(new)-totals(old)-incoming
    eps = np.finfo(float).eps
    mb = 32*eps*max(old.water_kgm2+new.water_kgm2+f.prcp*f.dt*1000.+abs(d['et_mm'])+d['runoff_mm'], 1.)
    storage = sum(float(abs(a).sum()) for s in (old, new) for a in (s.ground.enthalpy, s.snow.enthalpy, s.canopy_condensate.enthalpy))
    hb = float(np.sum(d['thermal']['convergence_tolerance_jm2']))+f.dt*float(np.sum(abs(d['layer_residual'])))
    hb += 32*eps*max(storage+abs(d['precipitation_enthalpy_jm2'])+abs(d['atmospheric_energy_jm2'])+
                    abs(d['advected_enthalpy_jm2'])+abs(d['runoff_enthalpy_jm2']), 1.)+1e-5
    cn = d['biogeochemistry']
    bound = np.array([mb, hb, cn['carbon_bound_gc_m2'], cn['nitrogen_bound_gn_m2']])
    if not np.all(np.isfinite(residual)) or not np.all(abs(residual) <= bound):
        raise RuntimeError(f'Independent M/H/C/N step budget: residual={residual}, bound={bound}')
    return incoming, residual, bound


def validate_input(data, metadata, input_path):
    if metadata['prepared_npz_sha256'] != sha(input_path): raise ValueError('Prepared forcing hash mismatch')
    starts, stops = data['time_start'], data['time_end']; n = len(starts)
    if not n or starts.ndim != 1 or stops.shape != starts.shape: raise ValueError('Invalid time arrays')
    if data['forcing'].shape != (n, 8) or data['environment'].shape != (n, 3): raise ValueError('Invalid forcing shape')
    if not np.all(np.isfinite(data['forcing'])) or not np.all(np.isfinite(data['environment'])): raise ValueError('Nonfinite forcing')
    begin = [datetime.strptime(str(t), '%Y%m%d%H%M') for t in starts]
    end = [datetime.strptime(str(t), '%Y%m%d%H%M') for t in stops]
    dt = float(data['forcing'][0, 0])
    if dt not in (1800., 3600.) or np.any(data['forcing'][:, 0] != dt): raise ValueError('Require constant HH/HR forcing')
    if any((b-a).total_seconds() != dt for a, b in zip(begin, end)) or any(a != b for a, b in zip(end[:-1], begin[1:])):
        raise ValueError('Forcing time gap or duration mismatch')
    return begin


def run(input_path, input_manifest, config_path, output, *, resume=False, stop_after=None):
    input_path, input_manifest, config_path, output = map(Path, (input_path, input_manifest, config_path, output))
    metadata = json.loads(input_manifest.read_text(encoding='utf-8')); config = json.loads(config_path.read_text(encoding='utf-8'))
    with np.load(input_path, allow_pickle=False) as archive: data = dict(archive)
    stamps = validate_input(data, metadata, input_path); n = len(stamps)
    for key in ('latitude', 'longitude', 'utc_offset_hours'):
        if config[key] != metadata[key]: raise ValueError(f'Configuration/forcing geometry disagreement: {key}')
    identity = dict(forcing_npz_sha256=sha(input_path), forcing_manifest_sha256=sha(input_manifest),
                    configuration_sha256=sha(config_path), source_sha256=source_hashes())
    geometry = MOSTGeometry(**config['most_geometry']); leaf = LeafParameters(**config['leaf_parameters'])
    deposition = DepositionSchedule(config)
    output.mkdir(parents=True, exist_ok=True); report_path = output/'report.json'
    started = time.monotonic(); elapsed_before = 0.
    if resume:
        report = json.loads(report_path.read_text(encoding='utf-8'))
        for key, value in identity.items():
            if report[key] != value: raise ValueError(f'Resume identity mismatch: {key}')
        if report['status'] == 'complete': return report
        for filename, expected in ((report['checkpoint_file'], report['checkpoint_sha256']), (report['model_npz'], report['model_npz_sha256'])):
            p = output/filename
            if p.parent != output or sha(p) != expected: raise ValueError('Checkpoint generation hash mismatch')
        saved = pickle.loads((output/report['checkpoint_file']).read_bytes())
        state, initial, net, bounds, maximum, stepmax, index = saved
        if index != report['steps_completed']: raise ValueError('Checkpoint index disagreement')
        with np.load(output/report['model_npz'], allow_pickle=False) as old: records = {k:list(v) for k,v in old.items() if k not in ('time_start','time_end')}
        if any(len(v) != index for v in records.values()): raise ValueError('Checkpoint trace length mismatch')
        elapsed_before = report['seconds']
    else:
        if any(output.iterdir()): raise ValueError('Use an empty run directory or explicit --resume')
        state = build_state(config); initial = totals(state); net = np.zeros(4); bounds = np.zeros(4)
        maximum = np.zeros(4); stepmax = np.zeros(4); index = 0
        records = {key:[] for key in (*TRACE, 'swe_m', 'lai', 'tsoil_k', 'smc_m3m3', 'liquid_m3m3',
                                      'step_residual_m_h_c_n', 'step_bound_m_h_c_n', 'cumulative_residual_m_h_c_n')}
        report = dict(**identity, status='running', site=config['site'], scope=config['scope'], configuration=config,
                      time_convention='local standard time', utc_offset_hours=metadata['utc_offset_hours'],
                      steps_requested=n, initial_state=asdict(state))
    def save(status):
        if source_hashes() != identity['source_sha256']: raise RuntimeError('Model/instrument source changed during trajectory')
        generation = f'{index:07d}'
        trace_path = output/f'trace-{generation}.npz'; state_path = output/f'state-{generation}.pkl'
        arrays = {key:np.asarray(value) for key,value in records.items()}
        arrays.update(time_start=data['time_start'][:index], time_end=data['time_end'][:index])
        with trace_path.with_suffix('.npz.tmp').open('wb') as stream: np.savez_compressed(stream, **arrays)
        os.replace(trace_path.with_suffix('.npz.tmp'), trace_path)
        state_path.with_suffix('.pkl.tmp').write_bytes(pickle.dumps((state, initial, net, bounds, maximum, stepmax, index)))
        os.replace(state_path.with_suffix('.pkl.tmp'), state_path)
        report.update(status=status, steps_completed=index, seconds=elapsed_before+time.monotonic()-started,
                      model_npz=trace_path.name, model_npz_sha256=sha(trace_path), checkpoint_file=state_path.name,
                      checkpoint_sha256=sha(state_path), maximum_cumulative_residual_m_h_c_n=maximum,
                      maximum_step_residual_m_h_c_n=stepmax, cumulative_numerical_bounds_m_h_c_n=bounds,
                      source_unchanged=True)
        if status == 'complete': report['final_state'] = asdict(state)
        atomic_json(report_path, report)
    save('running')
    end_index = n if stop_after is None else min(n, index+stop_after)
    if end_index <= index: raise ValueError('stop_after must advance at least one interval')
    try:
        while index < end_index:
            f = Forcing(*data['forcing'][index]); co2, mu, diffuse = data['environment'][index]
            environment = calendar(stamps[index], metadata, config['phenology'] is not None)
            dep = deposition.aerosol(stamps[index], f.dt)
            nitrogen = deposition.nitrogen_rate*f.dt
            after, d = step_clm_enthalpy_column(state, f, co2_ppm=float(co2), cosine_zenith=float(mu), diffuse_fraction=float(diffuse),
                leaf_dimension=config['leaf_dimension_m'], leaf_parameters=leaf, most_geometry=geometry,
                phenology_environment=environment, deposition_rates=dep, nitrogen_deposition_gn_m2=nitrogen)
            incoming, residual, bound = audit_step(state, after, f, d, nitrogen)
            next_net = net+incoming; next_bounds = bounds+bound; cumulative = totals(after)-initial-next_net
            if not np.all(abs(cumulative) <= next_bounds): raise RuntimeError(f'Independent cumulative M/H/C/N budget: {cumulative}, {next_bounds}')
            values = {key:float(d[field]) for key, field in TRACE.items()}
            values.update(swe_m=after.swe, lai=after.lai, tsoil_k=after.tsoil.copy(), smc_m3m3=after.smc.copy(),
                          liquid_m3m3=after.ground.liquid/(1000.*after.ground.dz), step_residual_m_h_c_n=residual,
                          step_bound_m_h_c_n=bound, cumulative_residual_m_h_c_n=cumulative)
            if not all(np.all(np.isfinite(v)) for v in values.values()): raise RuntimeError('Nonfinite trajectory output')
            for key, value in values.items(): records[key].append(value)
            state = after; net = next_net; bounds = next_bounds; index += 1
            maximum = np.maximum(maximum, abs(cumulative)); stepmax = np.maximum(stepmax, abs(residual))
            stop = datetime.strptime(str(data['time_end'][index-1]), '%Y%m%d%H%M')
            if stop.hour == 0 and stop.minute == 0:
                save('running')
                print(json.dumps({'site':config['site'], 'case':config['case'], 'steps':index, 'seconds':report['seconds'],
                                  'max_cumulative_m_h_c_n':maximum.tolist()}), flush=True)
        save('complete' if index == n else 'running')
    except Exception:
        report['failure'] = traceback.format_exc(); report['failed_index'] = index
        save('failed'); raise
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('input','input-manifest','config','output'): parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--resume', action='store_true'); parser.add_argument('--stop-after', type=int)
    args = parser.parse_args()
    result = run(args.input, args.input_manifest, args.config, args.output, resume=args.resume, stop_after=args.stop_after)
    print(json.dumps({'status':result['status'], 'steps':result['steps_completed'], 'seconds':result['seconds']}), flush=True)


if __name__ == '__main__': main()
