"""Actual Cng cold-day surface that exhausted the float32 secant loop."""
from pathlib import Path
import json
import numpy as np
from noahmp_clm.column import SoilParams
from noahmp_clm.physics.snowpack import Snowpack
from noahmp_clm.physics.soil_enthalpy import SoilEnthalpyState
from noahmp_clm.physics.canopy_interception import CanopyCondensate
from noahmp_clm.physics.canopy_energy import CanopyExchange
from noahmp_clm.physics.canopy_column import surface_step
from noahmp_clm.physics.photosynthesis import LeafParameters
from noahmp_clm.physics.vic import VICParameters
from noahmp_clm.validation import compare_series
from clm_soil_oracle import fixture_suction,reference_liquid

FIXTURES=Path(__file__).with_name('fixtures')

def saved_case():
    with np.load(FIXTURES/'cng_surface_secant_rounding.npz',allow_pickle=False) as z:
        arrays=tuple(fixture_suction(z[f'a{k}'],.355) if k==2 else z[f'a{k}'] for k in range(9))
        return arrays,z['exchange_extension'],z['vic']

def reference_case():
    return json.loads((FIXTURES/'cng_surface_secant_reference.json').read_text())

def numpy_reference(arrays,extension,vic):
    c,g,p,o,a,u,f,v,condensate=(x[0].astype(float) for x in arrays);n=int(o[0])
    snow=Snowpack(c[:n,0],c[:n,1],c[:n,2],c[:n,3],c[:n,4:])
    soil=SoilEnthalpyState(g[:,0],g[:,1],g[:,2],SoilParams(smcmax=p[0],bexp=p[1],smcref=p[2],
        satdk=p[3],quartz=p[4],smcwlt=p[5],saturated_suction_m=p[6]),int(o[1]))
    ext=extension[0].astype(float)
    exchange=CanopyExchange(*u[:11],atmosphere_water_velocity=ext[0],potential_temperature=ext[1])
    return surface_step(snow,soil,v[0],a,u[11],u[12],exchange,
        canopy_condensate=CanopyCondensate(condensate[:,0],condensate[:,1]),root_weights=v[3:],
        vegetated_fraction=u[13],transpiration_limit=u[16],ground_emissivity=u[18],ground_albedo=u[19:21],
        diffuse_fraction=u[21],cosine_zenith=u[22],nitrogen_factor=f[:,0],water_stress=f[:,1],
        leaf_parameters=LeafParameters(),infiltration_parameters=VICParameters(*vic[0,2:].astype(float)))

def test_saved_cng_numpy_surface_closes():
    arrays,extension,vic=saved_case();snow,soil,water,d=numpy_reference(arrays,extension,vic)
    ref=reference_case()
    # Old outputs used a different constitutive law. Retain them as history;
    # this is a new-law challenge with unchanged H and explicit suction .355m.
    assert np.all(np.isfinite([d['h'],d['le'],d['surface_temperature']]))
    np.testing.assert_allclose(soil.liquid/(1000.*soil.dz),
        reference_liquid(soil.water/(1000.*soil.dz),soil.temperature,soil.params,soil.freezing_option),rtol=1e-10,atol=1e-12)
    assert abs(d['water_residual_kgm2'])<1e-10
    assert abs(d['energy_residual_jm2'])<1e-5
    print('Cng NumPy saved surface',d['surface_temperature'],d['h'],d['le'],
          d['water_residual_kgm2'],d['energy_residual_jm2'],
          'constitutive-change H/LE',d['h']-ref['h'],d['le']-ref['le'],flush=True)
