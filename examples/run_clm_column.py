"""Complete CLM public-driver example; explicit illustrative pools, not spinup.

Run CPU integration on an EHPC PBS compute node. Native device batch runners
remain in benchmarks/run_fluxnet_native.py and noahmp_clm.gpu.
"""
from dataclasses import replace
import numpy as np
from noahmp_clm import CLMColumnState,ColumnState,Forcing,run_column
from noahmp_clm.physics.biogeochemistry import BGCState
from noahmp_clm.physics.plant_cn import PlantState
from noahmp_clm.physics.decomposition import DecompositionState
from noahmp_clm.physics.canopy_most import MOSTGeometry


def example():
    # These declared illustrative pools are not fitted site initial states.
    leaf_stem_root=np.array([120.,600.,150.]);soil_c=np.array([50.,100.,100.,1000.,500.,5000.,10000.])
    plant=PlantState(leaf_stem_root,leaf_stem_root/np.array([25.,100.,42.]),30.,.05)
    bgc=BGCState(plant,DecompositionState(soil_c,soil_c/25.,1.))
    state=CLMColumnState.from_column(ColumnState(smc=np.full(4,.3))).with_layered_canopy().with_biogeochemistry(bgc)
    state=replace(state,options=replace(state.options,OptRainSnowPartition=5,OptRunoffSurface=6,OptRunoffSubsurface=6))
    forcing=[Forcing(1800.,285.,.006,3.,101325.,sw,320.,0.) for sw in [350.,400.,350.]]
    inputs=[dict(co2_ppm=420.,cosine_zenith=.5,diffuse_fraction=.5,nitrogen_deposition_gn_m2=1e-5) for _ in forcing]
    diagnostics,final=run_column(state,forcing,step_options=inputs,
        most_geometry=MOSTGeometry(wind_height=10.,temperature_height=2.,humidity_height=2.))
    return state,forcing,diagnostics,final


if __name__=='__main__':
    initial,forcing,diagnostics,final=example()
    print('Illustrative 3 x 1800 s, cold start; not annual/spinup/scientific acceptance')
    print('ET mm:',sum(d['et_mm'] for d in diagnostics))
    print('GPP gC/m2:',sum(d['gpp_gc_m2'] for d in diagnostics))
    print('NEE gC/m2:',sum(d['nee_gc_m2'] for d in diagnostics))
    print('Final LAI:',final.lai,'topsoil K:',final.tsoil[0])
