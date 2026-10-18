"""Continuous native BGC/MOST trajectories for independent FLUXNET columns.

This validation runner consumes the same explicit configurations and prepared
forcing as run_fluxnet_column. Batch axes never exchange state. CUDA32 can
carry an independent JAX32 shadow and compare every returned field each step.
Checkpoints retain sixteen base arrays plus three explicit structure arrays
when configured; cold starts are not called spinup.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, astuple
import json
import os
from pathlib import Path
import re
import time
import traceback
from uuid import uuid4
import numpy as np

from run_fluxnet_column import (ROOT, TRACE, sha, atomic_json, build_state,
                                calendar, validate_input,structure_configuration,structure_trace_keys)
from fluxnet_deposition import DepositionSchedule
from noahmp_clm.column import SoilParams
from noahmp_clm.physics.canopy_most import MOSTGeometry
from noahmp_clm.physics.photosynthesis import LeafParameters
from noahmp_clm.physics.precip import partition_precipitation
from noahmp_clm.physics.soil_enthalpy import SoilEnthalpyState
from noahmp_clm.gpu.clm_bgc_column import (pack_clm_bgc_columns,
    pack_bgc_environment, next_bgc_arrays)
from noahmp_clm.gpu.clm_structure_column import pack_clm_structure_columns,next_structure_arrays


def native_state_count(cases):
    enabled=[structure_configuration(c['config']) is not None for c in cases]
    if len(set(enabled))!=1:raise ValueError('Split structured and leaf-only station batches')
    return 19 if enabled[0] else 16


def checkpoint_arrays(saved,identity,*,shadow=False):
    count=identity.get('native_state_arrays',16)
    if type(count) is not int or count not in (16,19):raise ValueError('Unsupported native checkpoint state count')
    prefix='shadow' if shadow else 'a';expected={f'{prefix}{k}' for k in range(count)}
    actual={key for key in saved if re.fullmatch(prefix+r'\d+',key)}
    if actual!=expected:raise ValueError('Native checkpoint must retain every declared state array')
    return tuple(saved[f'{prefix}{k}'] for k in range(count))


def next_station_arrays(previous,result):
    if len(previous)==16:return next_bgc_arrays(previous,result)
    if len(previous)==19:return next_structure_arrays(previous,result,kind='bgc')
    raise ValueError('Unsupported native station state count')


def source_hashes():
    paths = sorted(p for p in (ROOT/'src/noahmp_clm').rglob('*')
                   if p.suffix in ('.py', '.cu', '.npz'))
    paths += [Path(__file__), ROOT/'benchmarks/run_fluxnet_column.py',
              ROOT/'benchmarks/fluxnet_solar.py', ROOT/'benchmarks/fluxnet_deposition.py']
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}


def totals(a):
    snow, soil, canopy = (np.asarray(a[k], float) for k in (0, 1, 7))
    plant, earth = (np.asarray(a[k], float) for k in (11, 13))
    return np.column_stack((snow[:, :, 0].sum(1)+soil[:, :, 0].sum(1)+canopy[:, :, 0].sum(1),
        snow[:, :, 1].sum(1)+soil[:, :, 1].sum(1)+canopy[:, :, 1].sum(1),
        plant[:, :3].sum(1)+plant[:, 6]+earth[:, :7].sum(1),
        plant[:, 3:6].sum(1)+plant[:, 7]+earth[:, 7:].sum(1)))


def audit_step(before, after, forcing, deposition, nitrogen, out):
    """Independent float64 accounting, with a separate bound for each column.

    Authoritative canopy cells are summed, never the rounded aggregate alias.
    The precipitation phase is recomputed independently from the quantized
    forcing. Bounds follow existing native tests; no batch maximum is used.
    """
    f = np.asarray(forcing, float); dt = f[:, 0]
    d, sf = out['diagnostics'], out['surface']
    old, new = totals(before), totals(after)
    eps = np.finfo(before[0].dtype).eps
    ph = []
    for one, options in zip(f, before[3]):
        rain, snow = partition_precipitation(one[7], one[0], one[1], int(options[2]),
                                             qair=one[2], pressure=one[4])
        ph.append(1000.*(rain*(334000.+4180.*(one[1]-273.15))
                        +snow*2100.*(min(one[1], 273.15)-273.15)))
    ph = np.asarray(ph); precipitation = f[:, 7]*dt*1000.
    et, runoff, adv, runh = (np.asarray(d[k], float) for k in
        ('et_mm', 'runoff_mm', 'advected_enthalpy_jm2', 'runoff_enthalpy_jm2'))
    atmo = (f[:, 5]+f[:, 6]-np.asarray(sf['outgoing_shortwave'], float)
        -np.asarray(sf['outgoing_longwave'], float)-np.asarray(d['h'], float)
        -np.asarray(d['le'], float))*dt
    gpp, ra, rh = (np.asarray(d[k], float) for k in
        ('gpp_gc_m2', 'autotrophic_respiration_gc_m2', 'heterotrophic_respiration_gc_m2'))
    nitrogen = np.asarray(nitrogen, float)
    incoming = np.column_stack((precipitation-et-runoff, ph+atmo-adv-runh,
                                gpp-ra-rh, nitrogen))
    residual = new-old-incoming
    storage = sum(abs(np.asarray(a[k][:, :, 1], float)).sum(1)
                  for a in (before, after) for k in (0, 1, 7))
    hb = np.asarray(sf['thermal_layer_tolerance'], float).sum(1)
    hb += dt*abs(np.asarray(sf['layer_residual'], float)).sum(1)
    hb += 32.*eps*np.maximum(storage+abs(ph)+abs(atmo)+abs(adv)+abs(runh), 1.)+1e-5
    bound = np.column_stack((32.*eps*np.maximum(abs(old[:, 0])+abs(new[:, 0])+
        precipitation+abs(et)+abs(runoff), 1.), hb,
        128.*eps*np.maximum(old[:, 2]+new[:, 2]+gpp+ra+rh, 1.),
        128.*eps*np.maximum(old[:, 3]+new[:, 3]+nitrogen, 1.)))
    if not np.all(np.isfinite(residual)) or not np.all(abs(residual) <= bound):
        raise RuntimeError(f'Independent native M/H/C/N budget: {residual}, bounds {bound}')
    old_impurity = np.asarray(before[0][:, :, 4:], float).sum(1)
    dep = np.asarray(deposition, float)*dt[:, None]
    tracer = np.asarray(after[0][:, :, 4:], float).sum(1)-old_impurity-dep
    tracer += np.asarray(out['impurity_output_kgm2'], float)
    tracer_bound = 64.*eps*np.maximum(old_impurity+dep, 1e-20)
    if not np.all(np.isfinite(tracer)) or not np.all(abs(tracer) <= tracer_bound):
        raise RuntimeError(f'Independent native impurity budget: {tracer}, bounds {tracer_bound}')
    return incoming, residual, bound, abs(tracer)


def compare_same_precision(left, right, prefix=''):
    if left.keys() != right.keys(): raise ValueError(f'Backend fields differ: {prefix}')
    maximum = 0.
    for key, a in left.items():
        b = right[key]; name = f'{prefix}/{key}'
        if isinstance(a, dict):
            maximum = max(maximum, compare_same_precision(a, b, name))
        elif a.dtype.kind in 'iub':
            np.testing.assert_array_equal(a, b, err_msg=name)
        else:
            if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
                raise ValueError(f'Nonfinite backend field: {name}')
            np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6, err_msg=name)
            maximum = max(maximum, float(np.max(abs(a.astype(float)-b.astype(float)), initial=0.)))
    return maximum


def load_batch(path):
    path = Path(path); spec = json.loads(path.read_text(encoding='utf-8'))
    loaded = []; names = set()
    for row in spec['cases']:
        name = row['case']
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name) or name in names:
            raise ValueError('Unique safe case names required')
        names.add(name)
        paths = {k: (path.parent/row[k]).resolve() for k in ('input', 'input_manifest', 'configuration')}
        meta = json.loads(paths['input_manifest'].read_text(encoding='utf-8'))
        config = json.loads(paths['configuration'].read_text(encoding='utf-8'))
        with np.load(paths['input'], allow_pickle=False) as archive: data = dict(archive)
        stamps = validate_input(data, meta, paths['input'])
        for key in ('latitude', 'longitude', 'utc_offset_hours'):
            if config[key] != meta[key]: raise ValueError(f'Configuration/forcing disagreement: {key}')
        # Older prepared manifests identify the site through the original
        # FULLSET archive/member, without a separate `site` key.
        recorded_site = meta.get('site')
        if recorded_site is not None:
            if recorded_site != config['site']: raise ValueError('Configuration/forcing disagreement: site')
        elif '_'+config['site']+'_' not in meta['archive_name']:
            raise ValueError('Configuration site is absent from forcing archive identity')
        DepositionSchedule(config)
        structure_configuration(config)
        loaded.append(dict(case=name, paths=paths, metadata=meta, config=config, data=data, stamps=stamps))
    if not loaded: raise ValueError('Empty native batch')
    first = loaded[0]
    if any(len(v['stamps']) != len(first['stamps']) for v in loaded):
        raise ValueError('Batch columns need equal interval counts, with independent local dates')
    if any(v['config']['leaf_parameters'] != first['config']['leaf_parameters'] for v in loaded):
        raise ValueError('Native leaf parameters are shared within a batch; split differing parameter sets')
    return loaded


def initial_ground(arrays, column):
    g, p, o = (arrays[k][column] for k in (1, 2, 3))
    p = np.asarray(p, float)
    params = SoilParams(smcmax=p[0], bexp=p[1], smcref=p[2], satdk=p[3], quartz=p[4], smcwlt=p[5], saturated_suction_m=p[6])
    return asdict(SoilEnthalpyState(g[:, 0].astype(float), g[:, 1].astype(float),
                                  g[:, 2].astype(float), params, int(o[1])))


class Backend:
    def __init__(self, name):
        self.name = name
        if name == 'cuda32':
            import cupy as cp
            from noahmp_clm.gpu.cuda.clm_bgc_column import step_clm_bgc_columns_cuda
            from noahmp_clm.gpu.cuda.enthalpy import run_soil_thermodynamics_cuda
            self.xp, self.convert = cp, cp.asnumpy
            from noahmp_clm.gpu.clm_structure_column import step_clm_structure_columns_cuda
            self._plain_step,self._structure_step=step_clm_bgc_columns_cuda,step_clm_structure_columns_cuda
            self.inverse = run_soil_thermodynamics_cuda
        else:
            import jax
            import jax.numpy as j
            if name == 'jax64' and not jax.config.x64_enabled: raise ValueError('JAX x64 must be enabled')
            if not any(d.platform == 'gpu' for d in jax.devices()): raise ValueError('Native validation requires a JAX GPU')
            from noahmp_clm.gpu.clm_bgc_column import step_clm_bgc_columns_jax
            from noahmp_clm.gpu.enthalpy import run_soil_thermodynamics_jax
            self.xp, self.convert = j, np.asarray
            from noahmp_clm.gpu.clm_structure_column import step_clm_structure_columns_jax
            self._plain_step,self._structure_step=step_clm_bgc_columns_jax,step_clm_structure_columns_jax
            self.inverse = run_soil_thermodynamics_jax

    def step(self,*args,**kwargs):
        arrays,forcing=args[:-1],args[-1]
        if len(arrays)==16:return self._plain_step(*arrays,forcing,**kwargs)
        if len(arrays)==19:return self._structure_step(arrays,forcing,kind='bgc',**kwargs)
        raise ValueError('Native station step requires sixteen or nineteen arrays')

    def host(self, value):
        if isinstance(value, dict): return {k: self.host(v) for k, v in value.items()}
        return self.convert(value)

    def update_forcing(self, arrays, environment):
        values = list(arrays); site = values[4]
        if self.name == 'cuda32':
            site = site.copy(); site[:, 7:10] = self.xp.asarray(environment, site.dtype)
        else: site = site.at[:, 7:10].set(self.xp.asarray(environment, site.dtype))
        values[4] = site
        return tuple(values)

    def soil(self, arrays):
        xp = self.xp; g, p, o = arrays[1:4]; n = len(g)
        thermodynamic_parameters = p[:, xp.asarray([0, 1, 6])]
        inputs = xp.concatenate((g, xp.broadcast_to(thermodynamic_parameters[:, None, :], (n, 4, 3))), axis=2).reshape(-1, 6)
        out = self.inverse(inputs, xp.repeat(o[:, 1], 4), inverse=True)
        return self.host(out['temperature_k']).reshape(n, 4), self.host(out['liquid_kgm2']).reshape(n, 4)


def write_npz(path, **arrays):
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('wb') as stream: np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def read_generation(folder):
    """Read the generation committed by batch.json, never mutable aliases.

    A crash while writing the next generation leaves this pointer intact.
    Older outputs without generation reports remain readable but retain their
    original strict mixed-generation rejection, rather than guessing a state.
    """
    folder = Path(folder)
    report = json.loads((folder/'batch.json').read_text(encoding='utf-8'))
    checkpoint = folder/report['checkpoint_file']
    if checkpoint.parent != folder or sha(checkpoint) != report['checkpoint_sha256']:
        raise ValueError('Native checkpoint hash mismatch')
    with np.load(checkpoint, allow_pickle=False) as z: state = dict(z)
    checkpoint_arrays(state,report['identity'])
    if report['identity']['compare_jax32']:checkpoint_arrays(state,report['identity'],shadow=True)
    elif any(re.fullmatch(r'shadow\d+',key) for key in state):raise ValueError('Undeclared native shadow state')
    index = report['steps_completed']
    if int(state['index']) != index: raise ValueError('Native checkpoint generation mismatch')
    cases = []
    for identity in report['identity']['cases']:
        name = identity['case']
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name):
            raise ValueError('Unsafe checkpoint case name')
        cp = folder/name
        descriptor = report.get('case_reports', {}).get(name)
        p = cp/descriptor['file'] if descriptor else cp/'report.json'
        if p.parent != cp or (descriptor and sha(p) != descriptor['sha256']):
            raise ValueError('Native case report hash mismatch')
        r = json.loads(p.read_text(encoding='utf-8')); trace = cp/r['model_npz']
        if (r['steps_completed'] != index or trace.parent != cp or sha(trace) != r['model_npz_sha256']
                or r['batch_checkpoint_sha256'] != report['checkpoint_sha256']):
            raise ValueError('Native trace generation/hash mismatch')
        with np.load(trace, allow_pickle=False) as z: data = dict(z)
        if any(len(v) != index for v in data.values()): raise ValueError('Native trace length mismatch')
        cases.append((r, data))
    return report, state, cases


def carry_generation(folder, mode, cases, identity):
    """Carry every state and the original conservation ledger across segments.

    Configuration and source hashes must be identical. Continuing requires
    contiguous dates. Cycling permits only the identical full-calendar-year
    forcing block. Neither operation asserts that spinup has converged.
    """
    if mode not in ('continue', 'cycle'): raise ValueError('Initial mode must be continue or cycle')
    report, state, previous = read_generation(folder)
    if report['status'] != 'complete' or report['steps_completed'] != report['steps_requested']:
        raise ValueError('Initial run must be complete')
    old = report['identity']
    for key in ('backend', 'compare_jax32', 'source_sha256'):
        if old[key] != identity[key]: raise ValueError(f'Initial run identity mismatch: {key}')
    if old.get('native_state_arrays',16)!=identity.get('native_state_arrays',16):
        raise ValueError('Initial run structure state mismatch')
    if len(old['cases']) != len(cases): raise ValueError('Initial run column count mismatch')
    for c, new, former, (_, data) in zip(cases, identity['cases'], old['cases'], previous):
        for key in ('case', 'configuration_sha256'):
            if former[key] != new[key]: raise ValueError(f'Initial run identity mismatch: {key}')
        if mode == 'continue':
            if str(data['time_end'][-1]) != str(c['data']['time_start'][0]):
                raise ValueError('Continuation forcing must be contiguous')
        else:
            for key in ('forcing_npz_sha256', 'forcing_manifest_sha256'):
                if former[key] != new[key]: raise ValueError('Cycle requires identical forcing')
            start, end = str(data['time_start'][0]), str(data['time_end'][-1])
            if start[4:] != '01010000' or end[4:] != '01010000' or int(end[:4]) <= int(start[:4]):
                raise ValueError('Cycle requires complete calendar years')
    link = dict(mode=mode, batch_report_sha256=sha(Path(folder)/'batch.json'),
                checkpoint_sha256=report['checkpoint_sha256'],
                previous_ledger_steps=report.get('ledger_steps', report['steps_completed']),
                previous_segments=report.get('ledger_segments', 1))
    return link, report, state


def run(batch_path, output, *, backend='cuda32', compare_jax32=False, resume=False, stop_after=None,
        initial_run=None, initial_mode=None):
    batch_path, output = Path(batch_path), Path(output)
    if compare_jax32 and backend != 'cuda32': raise ValueError('JAX32 shadow requires CUDA32 primary')
    if stop_after is not None and stop_after <= 0: raise ValueError('stop_after must be positive')
    if (initial_run is None) != (initial_mode is None):
        raise ValueError('Supply both initial_run and initial_mode')
    cases = load_batch(batch_path); count = len(cases); n = len(cases[0]['stamps'])
    state_count=native_state_count(cases)
    dtype = np.float64 if backend == 'jax64' else np.float32
    sources = source_hashes()
    identities = [dict(case=c['case'], forcing_npz_sha256=sha(c['paths']['input']),
        forcing_manifest_sha256=sha(c['paths']['input_manifest']), configuration_sha256=sha(c['paths']['configuration'])) for c in cases]
    identity = dict(batch_sha256=sha(batch_path), cases=identities, source_sha256=sources,
                    backend=backend, compare_jax32=compare_jax32)
    if state_count==19:identity['native_state_arrays']=19
    carried = None
    if initial_run is not None:
        link, previous_report, carried = carry_generation(initial_run, initial_mode, cases, identity)
        identity['initial_run'] = link
    previous_steps = identity.get('initial_run', {}).get('previous_ledger_steps', 0)
    previous_segments = identity.get('initial_run', {}).get('previous_segments', 0)
    leaf = LeafParameters(**cases[0]['config']['leaf_parameters'])
    geometry = np.asarray([astuple(MOSTGeometry(**c['config']['most_geometry'])) for c in cases], dtype)
    deposition_schedules = [DepositionSchedule(c['config']) for c in cases]
    nrates = np.asarray([d.nitrogen_rate for d in deposition_schedules], dtype)
    output.mkdir(parents=True, exist_ok=True); path = output/'batch.json'
    records = []; reports = []; elapsed = 0.; started = time.monotonic()
    if resume:
        report, saved, prior_cases = read_generation(output)
        if report['identity'] != identity: raise ValueError('Native resume identity mismatch')
        if report['status'] == 'complete': return report
        a = checkpoint_arrays(saved,identity)
        shadow = checkpoint_arrays(saved,identity,shadow=True) if compare_jax32 else None
        initial, net, bounds, maximum, stepmax, tracermax = (saved[k] for k in
            ('initial', 'net', 'bounds', 'maximum', 'stepmax', 'tracermax'))
        index = int(saved['index'])
        same = report['same_precision_max_absolute'] or 0.; elapsed = report['seconds']
        for r, data in prior_cases:
            values = {k: list(v) for k, v in data.items() if k not in ('time_start', 'time_end')}
            records.append(values); reports.append(r)
    else:
        if any(output.iterdir()): raise ValueError('Use an empty output directory or --resume')
        if carried is None:
            states = [build_state(c['config']) for c in cases]
            env = np.asarray([c['data']['environment'][0] for c in cases], dtype)
            pack=pack_clm_bgc_columns if state_count==16 else lambda values,**kw:pack_clm_structure_columns(values,kind='bgc',**kw)
            a = pack(states, dtype=dtype, co2_ppm=env[:, 0], cosine_zenith=env[:, 1],
                diffuse_fraction=env[:, 2], leaf_dimension=np.asarray([c['config']['leaf_dimension_m'] for c in cases], dtype))
            shadow = tuple(v.copy() for v in a) if compare_jax32 else None
            initial = totals(a); net = np.zeros((count, 4)); bounds = net.copy()
            maximum = net.copy(); stepmax = net.copy(); tracermax = np.zeros((count, 5)); same = 0.
        else:
            a = checkpoint_arrays(carried,identity)
            shadow = checkpoint_arrays(carried,identity,shadow=True) if compare_jax32 else None
            initial, net, bounds, maximum, stepmax, tracermax = (carried[k] for k in
                ('initial', 'net', 'bounds', 'maximum', 'stepmax', 'tracermax'))
            same = previous_report['same_precision_max_absolute'] or 0.
        index = 0
        report = dict(identity=identity, steps_requested=n, columns=count)
        for k, c in enumerate(cases):
            records.append({key: [] for key in (*TRACE,*structure_trace_keys(c['config']), 'swe_m', 'lai', 'tsoil_k', 'smc_m3m3', 'liquid_m3m3',
                'step_residual_m_h_c_n', 'step_bound_m_h_c_n', 'cumulative_residual_m_h_c_n')})
            reports.append(dict(**identities[k], source_sha256=sources, backend=backend,
                site=c['config']['site'], scope=c['config']['scope'], configuration=c['config'],
                time_convention='local standard time', utc_offset_hours=c['metadata']['utc_offset_hours'],
                steps_requested=n, initial_state={'ground': initial_ground(a, k)},
                state_authority=f'All {state_count} native arrays in batch checkpoint, with the case index below',
                batch_column=k, forcing_quantization=np.dtype(dtype).name))
    engine = Backend(backend); checker = Backend('jax32') if compare_jax32 else None
    # This shadow has its own carried state and cumulative accounting. It is
    # not reset to the CUDA output at each step.
    if checker and (resume or carried is not None):
        saved_shadow = saved if resume else carried
        shadow_net, shadow_bounds, shadow_maximum = (saved_shadow[k] for k in
            ('shadow_net', 'shadow_bounds', 'shadow_maximum'))
    elif checker:
        shadow_net = np.zeros_like(net); shadow_bounds = np.zeros_like(bounds)
        shadow_maximum = np.zeros_like(maximum)
    device = tuple(engine.xp.asarray(v) for v in a)
    shadow_device = tuple(checker.xp.asarray(v) for v in shadow) if checker else None

    def save(status):
        if source_hashes() != sources: raise RuntimeError('Native model/instrument source changed during run')
        generation = f'{index:07d}-{uuid4().hex}'; state_path = output/f'state-{generation}.npz'
        state = {f'a{k}': v for k, v in enumerate(a)}
        if checker:
            state.update({f'shadow{k}': v for k, v in enumerate(shadow)})
            state.update(shadow_net=shadow_net, shadow_bounds=shadow_bounds, shadow_maximum=shadow_maximum)
        write_npz(state_path, **state, index=np.array(index), initial=initial, net=net, bounds=bounds,
                  maximum=maximum, stepmax=stepmax, tracermax=tracermax)
        state_sha = sha(state_path)
        descriptors = {}
        for k, c in enumerate(cases):
            folder = output/c['case']; folder.mkdir(exist_ok=True); trace = folder/f'trace-{generation}.npz'
            values = {key: np.asarray(value) for key, value in records[k].items()}
            write_npz(trace, **values, time_start=c['data']['time_start'][:index], time_end=c['data']['time_end'][:index])
            reports[k].update(status=status, source_unchanged=True, steps_completed=index,
                seconds=elapsed+time.monotonic()-started, model_npz=trace.name, model_npz_sha256=sha(trace),
                batch_checkpoint_file='../'+state_path.name, batch_checkpoint_sha256=state_sha,
                maximum_cumulative_residual_m_h_c_n=maximum[k], maximum_step_residual_m_h_c_n=stepmax[k],
                cumulative_numerical_bounds_m_h_c_n=bounds[k], maximum_step_impurity_residual_kgm2=tracermax[k],
                same_precision_max_absolute=same if checker else None,
                initial_run=identity.get('initial_run'), ledger_steps=previous_steps+index,
                ledger_segments=previous_segments+1)
            generation_report = folder/f'report-{generation}.json'
            atomic_json(generation_report, reports[k])
            descriptors[c['case']] = dict(file=generation_report.name, sha256=sha(generation_report))
        report.update(status=status, steps_completed=index, seconds=elapsed+time.monotonic()-started,
            checkpoint_file=state_path.name, checkpoint_sha256=state_sha, source_unchanged=True,
            maximum_cumulative_residual_m_h_c_n=maximum, cumulative_numerical_bounds_m_h_c_n=bounds,
            same_precision_max_absolute=same if checker else None, case_reports=descriptors,
            ledger_steps=previous_steps+index, ledger_segments=previous_segments+1)
        if checker:
            report.update(shadow_maximum_cumulative_residual_m_h_c_n=shadow_maximum,
                          shadow_cumulative_numerical_bounds_m_h_c_n=shadow_bounds)
        atomic_json(path, report)
        # Convenience aliases are published only after the transaction commits.
        # Resume uses the immutable descriptors above, even if alias publication
        # is interrupted. Scientific evaluation must bind the chosen report hash.
        for k, c in enumerate(cases): atomic_json(output/c['case']/'report.json', reports[k])
    save('running')
    end = n if stop_after is None else min(n, index+stop_after)
    try:
        while index < end:
            f = np.asarray([c['data']['forcing'][index] for c in cases], dtype)
            env = np.asarray([c['data']['environment'][index] for c in cases], dtype)
            e, cal = pack_bgc_environment([calendar(c['stamps'][index], c['metadata'], c['config']['phenology'] is not None)
                                          for c in cases], dtype=dtype)
            nitrogen = nrates*f[:, 0]
            deposition = np.asarray([d.aerosol(c['stamps'][index], f[k, 0])
                for k, (c, d) in enumerate(zip(cases, deposition_schedules))], dtype)
            kwargs = dict(environment=e, calendar=cal, nitrogen_deposition_gn_m2=nitrogen,
                deposition=deposition, most_geometry=geometry, leaf_parameters=leaf, return_surface=True)
            before_device = engine.update_forcing(device, env)
            out_device = engine.step(*before_device, f, **kwargs); out = engine.host(out_device)
            next_device = next_station_arrays(before_device, out_device)
            after = tuple(engine.host(v) for v in next_device)
            incoming, residual, bound, tracer = audit_step(a, after, f, deposition, nitrogen, out)
            next_net = net+incoming; next_bounds = bounds+bound; cumulative = totals(after)-initial-next_net
            if not np.all(abs(cumulative) <= next_bounds):
                raise RuntimeError(f'Native cumulative M/H/C/N budget: {cumulative}, {next_bounds}')
            temperature, liquid = engine.soil(next_device)
            if checker:
                sb = checker.update_forcing(shadow_device, env)
                so_device = checker.step(*sb, f, **kwargs); so = checker.host(so_device)
                same = max(same, compare_same_precision(out, so))
                next_shadow_device = next_station_arrays(sb, so_device)
                next_shadow = tuple(checker.host(v) for v in next_shadow_device)
                si, sr, bound_shadow, _ = audit_step(shadow, next_shadow, f, deposition, nitrogen, so)
                snet = shadow_net+si; sbounds = shadow_bounds+bound_shadow
                scumulative = totals(next_shadow)-initial-snet
                if not np.all(abs(scumulative) <= sbounds):
                    raise RuntimeError(f'JAX32 shadow cumulative M/H/C/N budget: {scumulative}, {sbounds}')
                st, sl = checker.soil(next_shadow_device)
                same = max(same, compare_same_precision({'temperature': temperature, 'liquid': liquid},
                                                       {'temperature': st, 'liquid': sl}))
            d = out['diagnostics']; g = after[1].astype(float)
            values = {key: d[field] for key, field in TRACE.items()}
            values.update(swe_m=after[0][:, :, 0].astype(float).sum(1)/1000., lai=after[4][:, 0],
                tsoil_k=temperature, smc_m3m3=g[:, :, 0]/(1000.*g[:, :, 2]), liquid_m3m3=liquid/(1000.*g[:, :, 2]),
                step_residual_m_h_c_n=residual, step_bound_m_h_c_n=bound, cumulative_residual_m_h_c_n=cumulative)
            if state_count==19:
                used=out['canopy_geometry']
                values.update(sai_m2m2=after[16][:,0],canopy_sai_used_m2m2=used['total_sai'],
                    canopy_exposed_lai_m2m2=used['exposed_lai'],canopy_exposed_sai_m2m2=used['exposed_sai'])
            if not all(np.all(np.isfinite(v)) for v in values.values()): raise RuntimeError('Nonfinite native trace')
            for k in range(count):
                for key, value in values.items(): records[k][key].append(np.array(value[k], copy=True))
            a, device = after, next_device
            if checker:
                shadow, shadow_device = next_shadow, next_shadow_device
                shadow_net, shadow_bounds = snet, sbounds
                shadow_maximum = np.maximum(shadow_maximum, abs(scumulative))
            net, bounds = next_net, next_bounds
            maximum = np.maximum(maximum, abs(cumulative)); stepmax = np.maximum(stepmax, abs(residual))
            tracermax = np.maximum(tracermax, tracer); index += 1
            if index % 48 == 0:
                save('running')
                print(json.dumps(dict(steps=index, columns=count, seconds=report['seconds'],
                    max_cumulative_m_h_c_n=maximum.max(0).tolist(),
                    same_precision_max_absolute=same if checker else None)), flush=True)
        save('complete' if index == n else 'running')
    except BaseException:
        report['failure'] = traceback.format_exc(); report['failed_step'] = index
        save('failed'); raise
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--backend', choices=('cuda32', 'jax32', 'jax64'), default='cuda32')
    parser.add_argument('--compare-jax32', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after', type=int)
    parser.add_argument('--initial-run', type=Path, help='Completed native run whose full state/ledger is carried')
    parser.add_argument('--initial-mode', choices=('continue', 'cycle'))
    args = parser.parse_args()
    run(args.batch, args.output, backend=args.backend, compare_jax32=args.compare_jax32,
        resume=args.resume, stop_after=args.stop_after, initial_run=args.initial_run, initial_mode=args.initial_mode)
