"""Freeze a controlled height experiment, not calibrated station parameters.

Only warm-season sensitivity is claimed. Nominal hydraulics/plant pools are
explicit in every JSON; real site validation requires separate parameter and
spinup evidence. The immutable protocol determines all 27 heights and dates.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
from noahmp_clm.column import SoilParams
from noahmp_clm.clm_state import clm_options
from noahmp_clm.physics.plant_cn import PlantParameters
from noahmp_clm.physics.decomposition import DecompositionParameters
from noahmp_clm.physics.phenology import PhenologyParameters, PhenologyState, PhenologyPhase
from noahmp_clm.physics.photosynthesis import LeafParameters
from run_fluxnet_column import atomic_json, build_state


def prepare(protocol_path, inputs, output):
    protocol_path, inputs, output = map(Path, (protocol_path, inputs, output))
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for site, geometry in protocol['sites'].items():
        metadata = json.loads((inputs/(site+'.json')).read_text(encoding='utf-8'))
        if [metadata['start'], metadata['end_exclusive']] != protocol['execution_plan']['windows'][site]:
            raise ValueError('Input window differs from locked protocol')
        with np.load(inputs/(site+'.npz'), allow_pickle=False) as a:
            temp = float(a['obs_TS_F_MDS_1'][0])+273.15
            theta = float(a['obs_SWC_F_MDS_1'][0])/100.
        if not np.isfinite(temp) or not np.isfinite(theta) or not 0. <= theta <= .45:
            raise ValueError('Missing/invalid initial soil observation; do not fill silently')
        grass = site == 'CN-Cng'; evergreen = site == 'CN-Qia'
        lai = 2. if grass else 3.; maximum = 3. if grass else 5.
        height = geometry['canopy_height_m']; rough = protocol['roughness_assumption']
        z0 = height*rough['momentum_roughness_over_canopy_height']
        plant_p = PlantParameters(allocation=(.5,0.,.5) if grass else (.3,.4,.3),
            turnover_days=(1095. if evergreen else 365., 30.*365., 365.), stem_live_n_fraction=0. if grass else PlantParameters().stem_live_n_fraction)
        carbon = np.array([lai/plant_p.specific_leaf_area, 0. if grass else 8000., 150. if grass else 300.])
        soil_c = np.array([50.,100.,100.,0.,300.,3000.,3000.])
        seasonal = None if evergreen else dict(parameters=asdict(PhenologyParameters(lai_maximum=maximum)),
            initial=asdict(PhenologyState(leaf_fraction=1., phase=PhenologyPhase.ACTIVE, armed=False)))
        common = dict(site=site, scope=protocol['execution_plan']['scope'], **{k:metadata[k] for k in ('latitude','longitude','utc_offset_hours')},
            protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
            column=dict(soil=asdict(SoilParams(name='nominal-loam-height-sensitivity-only')),
                dz=[.1,.3,.6,1.], tsoil=[temp]*4, smc=[theta]*4, ts=temp, lai=lai, fveg=1.,
                z0m=z0, root_frac=[.4,.3,.2,.1], canw=0., swe=0., albedo=.2, emiss=.97,
                options=asdict(replace(clm_options(),OptSurfaceDrag=1,OptRainSnowPartition=5,OptRunoffSurface=6,OptRunoffSubsurface=6))),
            phenology=seasonal, leaf_parameters=asdict(LeafParameters()), leaf_dimension_m=.04,
            biogeochemistry=dict(plant_parameters=asdict(plant_p), decomposition_parameters=asdict(DecompositionParameters(sand_fraction=.4)),
                maximum_lai=maximum, saturated_potential_mpa=-.003,
                plant_state=dict(carbon=carbon.tolist(),nitrogen=(carbon/np.array(plant_p.target_cn)).tolist(),reserve_carbon=100.,reserve_nitrogen=4.),
                soil_state=dict(carbon=soil_c.tolist(),nitrogen=(soil_c/np.array([37.5,37.5,37.5,263.1578947368,8.,11.,11.])).tolist(),mineral_nitrogen=2.)),
            nitrogen_deposition_gn_m2_s=0., aerosol_deposition_kg_m2_s=[0.]*5,
            assumptions=[
                'Nominal common loam hydraulics/thermal coefficients and 2 m depth; not measured site-specific soil.',
                'Uniform initial shallow TS/SWC, zero SWE/canopy water; a cold start rather than equilibrium spinup.',
                'Explicit nominal LAI, C/N pools, SLA, Farquhar/Ball-Berry, root profile, albedo and nutrient availability; no fit to fluxes.',
                'Summer active deciduous initialization for grass/mixed forest; evergreen needleleaf has no deciduous GDD. One aggregate C3 PFT per site, not resolved mixed-forest fractions.',
                'Zero aerosol/N deposition for this controlled short sensitivity only; final site runs need climatological inputs.',
                'All heights have identical initial state and parameters; no score-dependent height, window or case selection.'])
        for tq, wind in itertools.product(geometry['temperature_humidity_candidates_m'], geometry['wind_candidates_m']):
            case = f'{site}-tq{tq:g}-w{wind:g}'
            config = dict(common, case=case, baseline=(tq==geometry['baseline']['temperature_height'] and wind==geometry['baseline']['wind_height']),
                most_geometry=dict(temperature_height=tq,humidity_height=tq,wind_height=wind,
                    displacement=height*rough['displacement_over_canopy_height'],momentum_roughness=z0,
                    heat_roughness=z0*rough['heat_water_roughness_over_momentum'],water_roughness=z0*rough['heat_water_roughness_over_momentum']))
            build_state(config)
            path = output/(case+'.json')
            if path.exists(): raise ValueError('Case configuration exists; keep frozen configurations immutable')
            atomic_json(path, config)
            records.append(dict(case=case,site=site,baseline=config['baseline'],configuration=path.name,
                configuration_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    atomic_json(output/'cases.json',dict(protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),cases=records))
    return records


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('protocol','inputs','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();print(json.dumps({'cases':len(prepare(a.protocol,a.inputs,a.output))}))
