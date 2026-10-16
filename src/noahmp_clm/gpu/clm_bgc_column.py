"""Native complete CLM column with prognostic plant and soil carbon/nitrogen.

Sixteen arrays retain the eight water/heat arrays, seasonal state/flags/
parameters, plant state/parameters, soil C/N state/parameters and two BGC
site parameters. Seasonal flags append OptDynamicVeg (4 or 5). Host work is
validation and dispatch only; no NumPy physical tendency feeds either device.
"""
from dataclasses import astuple,replace
import numpy as np
from .clm_layered_column import pack_clm_layered_columns,validate_layered_column
from .plant_cn import pack_plant,validate_plant
from .decomposition import pack_decomposition,validate_decomposition
from .phenology import validate_phenology
from ..physics.phenology import PhenologyState,PhenologyParameters,PhenologyEnvironment
from ..physics.plant_cn import leaf_nitrogen_factor


def pack_clm_bgc_columns(states,**kwargs):
    states=list(states);dtype=kwargs.get('dtype',np.float64)
    if any(s.bgc_state is None for s in states):raise ValueError('Every BGC column needs explicit plant/soil C/N stores')
    factor=kwargs.pop('nitrogen_factor',1.)
    if np.ndim(factor)!=0 or factor!=1.:raise ValueError('BGC derives nitrogen_factor from leaf C/N')
    base=pack_clm_layered_columns([replace(s,bgc_state=None,bgc_parameters=None,phenology_state=None,
        phenology_parameters=None,options=replace(s.options,OptDynamicVeg=4),lai=s.lai) for s in states],
        nitrogen_factor=np.asarray([leaf_nitrogen_factor(s.bgc_state.plant,s.bgc_parameters.plant) for s in states]),**kwargs)
    seasonal=[s.phenology_state or PhenologyState() for s in states]
    ps=np.asarray([[s.gdd,s.leaf_fraction,s.remaining_seconds] for s in seasonal],dtype).reshape(-1,3)
    flags=np.asarray([[v.phase,v.armed,s.options.OptDynamicVeg] for s,v in zip(states,seasonal)],np.int32).reshape(-1,3)
    pp=np.asarray([astuple(s.phenology_parameters or PhenologyParameters()) for s in states],dtype).reshape(-1,6)
    plants=[pack_plant(s.bgc_state.plant,s.bgc_parameters.plant,dtype=dtype) for s in states]
    soils=[pack_decomposition(s.bgc_state.soil,285.,-.01,1800.,parameters=s.bgc_parameters.decomposition,dtype=dtype) for s in states]
    plant=np.asarray([a[0] for a in plants],dtype).reshape(-1,8);plant_p=np.asarray([a[1] for a in plants],dtype).reshape(-1,21)
    soil=np.asarray([a[0] for a in soils],dtype).reshape(-1,15);soil_p=np.asarray([a[2] for a in soils],dtype).reshape(-1,16)
    bp=np.asarray([[s.bgc_parameters.maximum_lai,s.bgc_parameters.saturated_potential_mpa] for s in states],dtype).reshape(-1,2)
    return (*base,ps,flags,pp,plant,plant_p,soil,soil_p,bp)


def pack_bgc_environment(environments,*,dtype=np.float64):
    """Use None for evergreen columns, explicit calendar metadata for option 5."""
    values=list(environments)
    if any(e is not None and not isinstance(e,PhenologyEnvironment) for e in values):raise ValueError('Expected PhenologyEnvironment or None')
    return (np.asarray([[e.daylength_seconds,e.latitude_degrees] if e is not None else [0.,0.] for e in values],dtype).reshape(-1,2),
        np.asarray([[e.increasing_daylength,e.winter_solstice] if e is not None else [0,0] for e in values],np.int32).reshape(-1,2))


def validate_bgc_column(arrays,forcing,environment,calendar,nitrogen,oxygen,xp):
    if len(arrays)!=16:raise ValueError('BGC columns require sixteen state arrays')
    n=len(arrays[0]);dtype=arrays[0].dtype;ps,flags,pp,plant,plant_p,soil,soil_p,bp=arrays[8:]
    validate_layered_column(arrays[:8],forcing,xp.zeros((n,5),dtype),xp)
    if flags.shape!=(n,3) or flags.dtype.kind not in 'iu' or bool(xp.any((flags[:,2]!=4)&(flags[:,2]!=5))):
        raise ValueError('BGC seasonal flags require integer phase/armed/OptDynamicVeg')
    if environment.shape!=(n,2) or environment.dtype!=dtype or bp.shape!=(n,2) or bp.dtype!=dtype:
        raise ValueError('Invalid BGC environment or site parameters')
    if (not bool(xp.all(xp.isfinite(environment))) or not bool(xp.all(xp.isfinite(bp)))
        or bool(xp.any((bp[:,0]<0.)|(bp[:,1]>=0.)))):raise ValueError('Invalid BGC site parameters')
    seasonal=flags[:,2]==5
    if bool(xp.any(seasonal&((abs(environment[:,1])<=19.5)|(abs(environment[:,1])>90.)))):
        raise ValueError('Seasonal BGC needs explicit extratropical latitude')
    if bool(xp.any((~seasonal)&xp.any(environment!=0.,axis=1))):raise ValueError('Evergreen BGC does not use a seasonal environment')
    u=xp.stack((xp.full(n,273.15,dtype),environment[:,0],forcing[:,0]),axis=1)
    validate_phenology(ps,flags[:,:2],u,calendar,pp,xp);validate_plant(plant,plant_p,xp)
    du=xp.concatenate((xp.stack((xp.full(n,273.15,dtype),xp.full(n,-.01,dtype),forcing[:,0],oxygen),axis=1),
        xp.zeros((n,14),dtype),nitrogen[:,None],xp.zeros((n,1),dtype)),axis=1)
    validate_decomposition(soil,du,soil_p,xp)
    for v in (nitrogen,oxygen):
        if v.shape!=(n,) or v.dtype!=dtype or not bool(xp.all(xp.isfinite(v))):raise ValueError('Invalid BGC deposition/oxygen array')
    if bool(xp.any((nitrogen<0.)|(oxygen<0.)|(oxygen>1.))):raise ValueError('Invalid BGC deposition or oxygen factor')
    site=arrays[4];eps=xp.finfo(dtype).eps
    lai=plant[:,0]*plant_p[:,19]/xp.maximum(site[:,1],xp.finfo(dtype).tiny)
    nf=xp.where(plant[:,0]>0.,xp.minimum(1.,plant[:,3]*plant_p[:,3]/xp.maximum(plant[:,0],xp.finfo(dtype).tiny)),0.)
    if bool(xp.any(abs(site[:,0]-lai)>8.*eps*xp.maximum(lai,1.))) or bool(xp.any(abs(site[:,11]-nf)>8.*eps)):
        raise ValueError('BGC LAI/nitrogen aliases disagree with plant C/N')
    if bool(xp.any((site[:,1]==0.)&xp.any(plant[:,:3]>0.,axis=1))):raise ValueError('Bare ground cannot carry plant tissue')
    if bool(xp.any(abs(arrays[6].sum(1)-1.)>8.*eps)):raise ValueError('BGC requires normalized root fractions')


def _sum_rows(a,xp):
    # The same ordered device additions in both APIs; no parallel reduction
    # changes the C/N budget subtraction or root temperature weighting.
    result=xp.zeros(len(a),a.dtype)
    for k in range(a.shape[1]):result=result+a[:,k]
    return result


def _set(a,k,value,backend):
    if backend=='jax':return a.at[:,k].set(value)
    a=a.copy();a[:,k]=value;return a


def _ratio(numerator,denominator,backend):
    if backend=='jax':return numerator/denominator
    from .cuda.clm_bgc_column import _division
    return _division()(numerator,denominator)


def _step(arrays,forcing,environment,calendar,nitrogen,oxygen,backend,**kwargs):
    if backend=='jax':
        import jax.numpy as xp
        from .enthalpy import run_soil_thermodynamics_jax as inverse
        from .phenology import run_phenology_jax as phenology
        from .plant_cn import run_plant_turnover_jax as turnover,plan_plant_allocation_jax as plan,apply_plant_allocation_jax as apply
        from .decomposition import run_decomposition_jax as decompose
        from .clm_layered_column import step_clm_layered_columns_jax as column
        arrays=tuple(map(xp.asarray,arrays));forcing=xp.asarray(forcing)
    else:
        import cupy as xp
        from .cuda.enthalpy import run_soil_thermodynamics_cuda as inverse
        from .cuda.phenology import run_phenology_cuda as phenology
        from .cuda.plant_cn import run_plant_turnover_cuda as turnover,plan_plant_allocation_cuda as plan,apply_plant_allocation_cuda as apply
        from .cuda.decomposition import run_decomposition_cuda as decompose
        from .cuda.clm_layered_column import step_clm_layered_columns_cuda as column
        if any(xp.asarray(arrays[k]).dtype.kind not in 'iu' for k in (3,9)):raise ValueError('BGC and CLM flags must be integers')
        arrays=tuple(xp.asarray(a,xp.int32 if k in (3,9) else xp.float32) for k,a in enumerate(arrays));forcing=xp.asarray(forcing,xp.float32)
    n=len(arrays[0]);dtype=arrays[0].dtype
    if environment is None:
        if bool(xp.any(arrays[9][:,2]==5)):raise ValueError('Seasonal BGC requires an explicit environment')
        environment=xp.zeros((n,2),dtype)
    if calendar is None:
        if bool(xp.any(arrays[9][:,2]==5)):raise ValueError('Seasonal BGC requires an explicit calendar')
        calendar=xp.zeros((n,2),xp.int32)
    environment=xp.asarray(environment,dtype);calendar=xp.asarray(calendar)
    nitrogen=xp.broadcast_to(xp.asarray(nitrogen,dtype),(n,));oxygen=xp.broadcast_to(xp.asarray(oxygen,dtype),(n,))
    validate_bgc_column(arrays,forcing,environment,calendar,nitrogen,oxygen,xp)
    c,g,p,o,site,storage,roots,condensate,ps,flags,pp,plant,plant_p,soil,soil_p,bp=arrays
    seasonal=flags[:,2]==5
    temperature=inverse(xp.concatenate((g[:,2,:],p[:,:2]),axis=1),o[:,1],inverse=True)['temperature_k']
    ph=phenology(ps,flags[:,:2],xp.stack((temperature,environment[:,0],forcing[:,0]),axis=1),calendar,pp)
    next_ps=xp.where(seasonal[:,None],ph['state'],ps)
    next_flags=xp.concatenate((xp.where(seasonal[:,None],ph['flags'],flags[:,:2]),flags[:,2:3]),axis=1)
    after=next_ps[:,1];before=ps[:,1]
    retention=xp.where(seasonal,xp.where(before>0.,xp.minimum(1.,_ratio(after,
        xp.maximum(before,xp.finfo(dtype).tiny),backend)),xp.where(after>0.,1.,0.)),1.)
    target=xp.where(seasonal,ph['lai'],bp[:,0]);growing=((~seasonal)|((next_flags[:,0]==1)|(next_flags[:,0]==2)))&(site[:,1]>0.)
    td=turnover(plant,xp.stack((forcing[:,0],retention),axis=1),plant_p)
    live=td['state'];lai=_ratio(live[:,0]*plant_p[:,19],xp.maximum(site[:,1],xp.finfo(dtype).tiny),backend)
    nf=xp.where(live[:,0]>0.,xp.minimum(1.,_ratio(live[:,3]*plant_p[:,3],
        xp.maximum(live[:,0],xp.finfo(dtype).tiny),backend)),0.)
    used_site=_set(_set(site,0,lai,backend),11,nf,backend)
    out=column(c,g,p,o,used_site,storage,roots,condensate,forcing,**kwargs)
    # Invert the actual post-physical soil enthalpy on device for both root
    # maintenance and topsoil decomposition, including frozen liquid water.
    ng=out['soil_cells'];si=xp.concatenate((ng,xp.broadcast_to(p[:,None,:2],(n,4,2))),axis=2).reshape(-1,5)
    therm=inverse(si,xp.repeat(o[:,1],4),inverse=True)
    temperatures=therm['temperature_k'].reshape(n,4);liquid=therm['liquid_kgm2'].reshape(n,4)
    logq=xp.log(plant_p[:,14]);qweights=xp.exp(_ratio(logq[:,None]*(temperatures-293.15),10.,backend))*roots
    qfactor=_sum_rows(qweights,xp)
    root_t=xp.where(logq==0.,_sum_rows(roots*temperatures,xp),293.15+_ratio(10.*xp.log(qfactor),xp.where(logq==0.,1.,logq),backend))
    saturation=xp.clip(_ratio(liquid[:,0],((1000.*ng[:,0,2])*p[:,0]),backend),1e-4,1.)
    potential=bp[:,1]*xp.exp(-p[:,1]*xp.log(saturation))
    climate=xp.stack((out['diagnostics']['gpp_gcm2'],out['diagnostics']['leaf_respiration_umol_m2s']*12.0107e-6*forcing[:,0],
        forcing[:,1],root_t,forcing[:,0],_ratio(target*site[:,1],plant_p[:,19],backend),growing.astype(dtype)),axis=1)
    planned=plan(live,climate,plant_p)
    du=xp.concatenate((xp.stack((temperatures[:,0],potential,forcing[:,0],oxygen),axis=1),
        td['litter_carbon_gc_m2'],td['litter_nitrogen_gn_m2'],nitrogen[:,None],planned[:,23:24]),axis=1)
    sd=decompose(soil,du,soil_p);ad=apply(planned,sd['plant_uptake_gn_m2'][:,None],plant_p)
    live=ad['state'];earth=sd['state'];gpp=climate[:,0];ra=ad['autotrophic_respiration_gc_m2'];rh=sd['heterotrophic_respiration_gc_m2']
    respiration=ra+rh;nee=respiration-gpp;mass_factor=12.0107e-6*forcing[:,0]
    exchange=dict(gpp_gc_m2=gpp,npp_gc_m2=gpp-ra,nee_gc_m2=nee,autotrophic_respiration_gc_m2=ra,
        heterotrophic_respiration_gc_m2=rh,ecosystem_respiration_gc_m2=respiration,
        soil_respiration_gc_m2=ad['root_respiration_gc_m2']+rh,gpp_umol_m2s=_ratio(gpp,mass_factor,backend),nee_umol_m2s=_ratio(nee,mass_factor,backend))
    oldc=(_sum_rows(plant[:,:3],xp)+plant[:,6])+_sum_rows(soil[:,:7],xp)
    oldn=((_sum_rows(plant[:,3:6],xp)+plant[:,7])+_sum_rows(soil[:,7:14],xp))+soil[:,14]
    newc=(_sum_rows(live[:,:3],xp)+live[:,6])+_sum_rows(earth[:,:7],xp)
    newn=((_sum_rows(live[:,3:6],xp)+live[:,7])+_sum_rows(earth[:,7:14],xp))+earth[:,14]
    cr=(newc-oldc)+nee;nr=(newn-oldn)-nitrogen;eps=xp.finfo(dtype).eps
    cb=128.*eps*xp.maximum(((newc+oldc)+gpp)+respiration,1.);nb=128.*eps*xp.maximum((newn+oldn)+nitrogen,1.)
    if bool(xp.any((abs(cr)>cb)|(abs(nr)>nb))) or not all(bool(xp.all(xp.isfinite(v))) for v in (*exchange.values(),cr,nr,cb,nb)):
        raise RuntimeError('Native whole-column C/N budget exceeds numerical bound')
    final_lai=_ratio(live[:,0]*plant_p[:,19],xp.maximum(site[:,1],xp.finfo(dtype).tiny),backend)
    final_site=_set(_set(site,0,final_lai,backend),11,ad['nitrogen_factor'],backend)
    # GPP in the physical ledger remains the actual surface solve output.
    out['diagnostics'].update(exchange)
    out.update(site=final_site,plant_state=live,soil_cn_state=earth,phenology_state=next_ps,phenology_flags=next_flags,
        phenology_parameters=pp,plant_parameters=plant_p,soil_cn_parameters=soil_p,bgc_parameters=bp)
    out['biogeochemistry']=dict(turnover=td,allocation=ad,decomposition=sd,
        environment=dict(soil_temperature=temperatures[:,0],water_potential_mpa=potential,root_temperature=root_t,root_q10_factor=qfactor),
        lai_before=site[:,0],lai_used=lai,lai_after=final_lai,target_lai=target,nitrogen_factor_used=nf,
        nitrogen_factor_after=ad['nitrogen_factor'],carbon_residual_gc_m2=cr,nitrogen_residual_gn_m2=nr,
        carbon_bound_gc_m2=cb,nitrogen_bound_gn_m2=nb,nitrogen_deposition_gn_m2=nitrogen)
    return out


def step_clm_bgc_columns_jax(c,g,p,o,site,storage,roots,condensate,ps,flags,pp,plant,plant_p,soil,soil_p,bp,
                            forcing,*,environment=None,calendar=None,nitrogen_deposition_gn_m2=0.,oxygen_factor=1.,**kwargs):
    return _step((c,g,p,o,site,storage,roots,condensate,ps,flags,pp,plant,plant_p,soil,soil_p,bp),forcing,
        environment,calendar,nitrogen_deposition_gn_m2,oxygen_factor,'jax',**kwargs)


def next_bgc_arrays(previous,result):
    """Carry all conserved device states; retain immutable parameter arrays."""
    return (result['snow_cells'],result['soil_cells'],previous[2],result['options'],result['site'],result['storage'],previous[6],
        result['condensate'],result['phenology_state'],result['phenology_flags'],previous[10],result['plant_state'],previous[12],
        result['soil_cn_state'],previous[14],previous[15])
