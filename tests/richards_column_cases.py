"""Declarative initial cases; physical reference advancement is EHPC-only."""
from dataclasses import replace
import numpy as np

KINDS=('legacy','aggregate','five_layers','layered','snow','phase2','most','bgc','structure')


def pack(state,kind,dtype):
    if kind=='structure':
        from noahmp_clm.gpu.clm_structure_column import pack_clm_structure_columns
        return pack_clm_structure_columns([state],kind='bgc',dtype=dtype)
    if kind=='bgc':
        from noahmp_clm.gpu.clm_bgc_column import pack_clm_bgc_columns
        return pack_clm_bgc_columns([state],dtype=dtype)
    if kind in ('layered','snow','phase2','most'):
        from noahmp_clm.gpu.clm_layered_column import pack_clm_layered_columns
        return pack_clm_layered_columns([state],dtype=dtype)
    from noahmp_clm.gpu.clm_enthalpy_column import pack_clm_enthalpy_columns
    return pack_clm_enthalpy_columns([state],dtype=dtype)


def initial(kind,dtype):
    from noahmp_clm.column import ColumnState,Forcing
    from noahmp_clm.clm_state import CLMColumnState,clm_options
    from noahmp_clm.physics.snowpack import Snowpack
    from noahmp_clm.physics.canopy_most import MOSTGeometry
    from noahmp_clm.physics.vegetation_structure import CanopyStructureParameters,StemAreaScheme
    old=ColumnState(ts=295.,tsoil=np.array([290.,289.,288.,287.]),smc=np.array([.35,.24,.28,.25]),canw=.0003)
    f=Forcing(1800.,295.,.008,3.,101325.,700.,350.,0.);snow=Snowpack.empty()
    if kind in ('five_layers','snow','phase2'):
        old.ts=265.;old.tsoil[:]=270.;f=Forcing(1800.,268.,.001,3.,101325.,300.,250.,0.)
        snow=Snowpack.from_ice([10.,20.,30.,50.,80.],265.,300.) if kind=='five_layers' else Snowpack.from_ice([10.,20.],265.,300.)
    options=replace(clm_options(),OptSoilSupercoolWater=2) if kind=='phase2' else clm_options()
    state=CLMColumnState.from_column(old,snow=snow,options=options)
    if kind in ('layered','snow','phase2','most'):state=state.with_layered_canopy()
    if kind in ('bgc','structure'):
        from test_clm_bgc_column import case
        state,f,_=case(None,285.,False)
    if kind=='structure':state=state.with_vegetation_structure(CanopyStructureParameters(StemAreaScheme.CN_NONCROP,11.,5.5),sai=.8)
    geometry=MOSTGeometry() if kind=='most' else None
    if geometry is not None:state=replace(state,options=replace(state.options,OptSurfaceDrag=1))
    arrays=pack(state,kind,dtype)
    f2=replace(f,swdown=f.swdown+10.)
    forcing=np.asarray([[list(vars(one).values())] for one in [f,f2]],dtype)
    deposition=np.asarray([[[1.,2.,3.,4.,5.]],[[2.,4.,6.,8.,10.]]],dtype)*dtype(1e-13)
    return arrays,forcing,deposition,geometry


def decode(arrays,kind):
    from test_gpu_clm_enthalpy_column import state_from_row
    from noahmp_clm.physics.canopy_interception import CanopyCondensate
    row=[v[0] for v in arrays];state=state_from_row(row[:7])
    if len(row)>7:
        cc=np.asarray(row[7],float);state=replace(state,canopy_condensate=CanopyCondensate(cc[:,0],cc[:,1]))
    if kind in ('bgc','structure'):
        from test_gpu_plant_cn import parameters
        from noahmp_clm.physics.plant_cn import PlantState
        from noahmp_clm.physics.decomposition import DecompositionState,DecompositionParameters
        from noahmp_clm.physics.biogeochemistry import BGCState,BGCParameters
        ps,flags,pp,plant,plant_p,soil,soil_p,bp=row[8:16]
        assert int(flags[2])==4
        plant=np.asarray(plant,float);soil=np.asarray(soil,float);sp=np.asarray(soil_p,float)
        pools=BGCState(PlantState(plant[:3],plant[3:6],plant[6],plant[7]),DecompositionState(soil[:7],soil[7:14],soil[14]))
        params=BGCParameters(parameters(plant_p),DecompositionParameters(tuple(sp[:7]),tuple(sp[7:10]),*sp[10:]),*map(float,bp))
        state=state.with_biogeochemistry(pools,params)
    if kind=='structure':
        from noahmp_clm.physics.vegetation_structure import CanopyStructureParameters
        ss,pp,oo=row[16:19]
        state=state.with_vegetation_structure(CanopyStructureParameters(int(oo[0]),*map(float,pp),bool(oo[1])),sai=float(ss[0]))
    return state
