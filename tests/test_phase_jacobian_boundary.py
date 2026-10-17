"""Actual G614 phase operands, independently solved across the melting branch."""
from pathlib import Path
import hashlib,json
import numpy as np
from noahmp_clm.physics.canopy_energy import CanopyExchange
from noahmp_clm.physics.canopy_interception import CanopyCondensate
from noahmp_clm.physics.soil_uptake import SoilWaterSupply
from noahmp_clm.physics.snowpack import Snowpack
from noahmp_clm.physics.photosynthesis import LeafParameters
from noahmp_clm.physics.canopy_phase_energy import solve_canopy_phase_energy
from test_canopy_phase_energy import independent_reference,cases

def operands():
    path=Path(__file__).with_name('fixtures')/'qia_canopy_phase_jacobian.npz'
    meta=json.loads(path.with_suffix('.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['fixture_sha256']
    with np.load(path,allow_pickle=False) as z:v={k:a.astype(float) for k,a in z.items()}
    u=v['u'];b=v['boundary'];soil=v['soil'];c=v['condensate'];n=int(v['snow_count']);snow=v['snow'];f=v['factors']
    exchange=CanopyExchange(*u[:11],atmosphere_water_velocity=v['extension'][0],potential_temperature=v['extension'][1])
    kw=dict(soil_supply=SoilWaterSupply(soil[1:5],soil[5:9],soil[9]),root_enthalpy=b[5:9],vegetated_fraction=u[13],
        transpiration_limit=u[16],ground_evaporation_limit=u[17],ground_emissivity=u[18],ground_albedo=u[19:21],
        diffuse_fraction=u[21],cosine_zenith=u[22],nitrogen_factor=f[:,0],water_stress=f[:,1],
        fixed_ground_temperature=b[1],ground_is_ice=bool(b[2]),ground_shares_soil=bool(b[3]),
        snowpack=Snowpack(snow[:n,0],snow[:n,1],snow[:n,2],snow[:n,3],snow[:n,4:]) if n else None,leaf_parameters=LeafParameters(*v['leaf_parameters']))
    return v,(v['leaf'],u[11],u[12],exchange,CanopyCondensate(c[:,0],c[:,1]),b[4]),kw

def test_actual_phase_boundary_against_independent_network_and_parcels():
    v,args,kw=operands();got=solve_canopy_phase_energy(*args,**kw);ref=independent_reference(*args,**kw)
    maximum_temperature=float(abs(np.r_[got['leaf_temperature'],got['ground_temperature']]-ref['temperature']).max())
    assert maximum_temperature<1e-4
    for key in ['h','le']:np.testing.assert_allclose(got[key],ref[key],rtol=.1,atol=.001)
    np.testing.assert_allclose(got['condensate'].mass,ref['mass'],rtol=.1,atol=1e-7)
    np.testing.assert_allclose(got['condensate'].enthalpy,ref['enthalpy'],rtol=.1,atol=.1)
    assert got['condensate'].ice[0]>0. and got['condensate'].liquid[0]>0.
    assert abs(got['solver_residual']).max()<=5e-5
    assert abs(got['layer_residual']).max()<=5e-5*1.01
    np.testing.assert_allclose(got['condensate'].mass+args[-1]*got['wet_evaporation'],args[-2].mass,rtol=0.,atol=1e-14)
    print('G614 NumPy independent phase',dict(maximum_temperature_k=maximum_temperature,h_difference=got['h']-ref['h'],
        le_difference=got['le']-ref['le'],maximum_solver_residual_wm2=float(abs(got['solver_residual']).max()),
        canopy_water_residual_kgm2=got['canopy_water_residual'],iterations=got['iterations']),flush=True)

def test_independent_network_with_nonunit_nitrogen_and_water_limits():
    _,area,sw,lw,exchange,c,kw=cases()[0]
    kw=dict(kw,nitrogen_factor=.7,water_stress=.4)
    got=solve_canopy_phase_energy(area,sw,lw,exchange,c,1800.,**kw)
    ref=independent_reference(area,sw,lw,exchange,c,1800.,**kw)
    difference=float(abs(np.r_[got['leaf_temperature'],got['ground_temperature']]-ref['temperature']).max())
    assert difference<1e-4
    for key in ['h','le','gpp']:np.testing.assert_allclose(got[key],ref[key],rtol=.1,atol=.001)
    assert abs(got['solver_residual']).max()<=5e-5 and abs(got['layer_residual']).max()<=5e-5*1.01
    print('Independent nonunit C-N/water canopy',dict(temperature_difference_k=difference,gpp_difference=got['gpp']-ref['gpp']),flush=True)
