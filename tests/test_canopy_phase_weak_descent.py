"""Actual cold-year phase edge; the independent NumPy root defines physics."""
from dataclasses import replace
from pathlib import Path
import numpy as np
from noahmp_clm.column import SoilParams
from noahmp_clm.physics.canopy_column import surface_step
from noahmp_clm.physics.canopy_energy import CanopyExchange
from noahmp_clm.physics.canopy_interception import CanopyCondensate
from noahmp_clm.physics.canopy_optics import CanopyOptics
from noahmp_clm.physics.photosynthesis import LeafParameters
from noahmp_clm.physics.snowpack import Snowpack,TF
from noahmp_clm.physics.soil_enthalpy import SoilEnthalpyState
from noahmp_clm.physics.vic import VICParameters


def load_case():
    with np.load(Path(__file__).parent/'fixtures/canopy_phase_weak_descent.npz',allow_pickle=False) as z:raw=dict(z)
    arrays=tuple(raw['input'+str(k)] for k in range(9))
    controls={k[8:]:v for k,v in raw.items() if k.startswith('control/')}
    expected={k[10:]:v for k,v in raw.items() if k.startswith('reference/')}
    return arrays,controls,expected


def reference(arrays,controls):
    c,g,p,o,a,u,f,v,old=[x[0].astype(float) for x in arrays];n=int(o[0])
    pack=Snowpack(c[:n,0],c[:n,1],c[:n,2],c[:n,3],c[:n,4:])
    props=SoilParams(smcmax=p[0],bexp=p[1],smcref=p[2],satdk=p[3],quartz=p[4],smcwlt=p[5],saturated_suction_m=p[6])
    soil=SoilEnthalpyState(g[:,0],g[:,1],g[:,2],props,int(o[1]));exchange=CanopyExchange(*u[:11])
    if 'exchange_extension' in controls:
        extension=controls['exchange_extension'][0]
        exchange=replace(exchange,atmosphere_water_velocity=float(extension[0]),potential_temperature=float(extension[1]))
    parameters=controls['parameters'];parameters=parameters[0] if parameters.ndim==3 else parameters
    leaf=LeafParameters.from_layer_rows(parameters) if parameters.ndim==2 else LeafParameters(*parameters)
    optics=CanopyOptics(*(tuple(row) for row in controls['canopy_optics'][0].reshape(4,2)))
    infiltration=None
    if 'vic' in controls and controls['vic'][0,0]>0.:infiltration=VICParameters(*controls['vic'][0,2:].astype(float))
    pack,soil,_,d=surface_step(pack,soil,float(v[0]),a,float(u[11]),float(u[12]),exchange,
        canopy_condensate=CanopyCondensate(old[:,0],old[:,1]),root_weights=v[3:],infiltration_parameters=infiltration,
        stem_area=controls['stem_area'][0].astype(float),vegetated_fraction=float(u[13]),transpiration_limit=float(u[16]),
        ground_emissivity=float(u[18]),ground_albedo=u[19:21],diffuse_fraction=float(u[21]),cosine_zenith=float(u[22]),
        nitrogen_factor=f[:,0],water_stress=f[:,1],leaf_parameters=leaf,canopy_optics=optics)
    cells=np.zeros((5,9));cells[:,3]=54.5
    if len(pack.mass):cells[:len(pack.mass)]=np.column_stack((pack.mass,pack.enthalpy,pack.thickness,pack.radius_um,pack.impurities))
    d.update(snow_cells=cells,snow_counts=np.asarray(len(pack.mass)),soil_cells=np.column_stack((soil.water,soil.enthalpy,soil.dz)),
        soil_temperature=soil.temperature,condensate=np.column_stack((d['canopy_condensate'].mass,d['canopy_condensate'].enthalpy)))
    return d


def test_actual_phase_edge_has_conservative_numpy_root():
    arrays,controls,_=load_case();got=reference(arrays,controls)
    assert got['surface_root_converged'] and np.max(abs(got['solver_residual']))<=1e-7
    assert abs(got['water_residual_kgm2'])<2e-9 and abs(got['energy_residual_jm2'])<.02
    assert 0.<got['phase_coordinates'][-1]<334000./4180.
    assert got['leaf_temperature'][-1]==TF
    print('actual phase edge NumPy',got['iterations'],got['restarts'],got['phase_coordinates'][-1],
        float(np.max(abs(got['solver_residual']))),got['water_residual_kgm2'],got['energy_residual_jm2'],flush=True)
