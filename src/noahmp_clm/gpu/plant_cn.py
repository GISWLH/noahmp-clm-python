"""Native plant C/N turnover, demand planning and funded allocation.

Three stages permit actual competition with the soil decomposition kernel.
State: C[3], N[3], reserve C/N. Parameters: four triples, then nine scalars.
Plan: state[8], GPP, dt, available C, growth[3], N deficit[3], paid/unmet
maintenance[3+3], mineral-N demand. All amounts are gC/gN per m2 per step.
"""
from functools import lru_cache
import numpy as np
from ..physics.plant_cn import PlantParameters
from .canopy_energy import _sum

BUDGETS=('carbon_residual_gc_m2','nitrogen_residual_gn_m2','carbon_bound_gc_m2','nitrogen_bound_gn_m2','valid')
ALLOCATION_DIAGNOSTICS=('gpp_gc_m2','npp_gc_m2','autotrophic_respiration_gc_m2','soil_n_demand_gn_m2',
    'soil_n_uptake_gn_m2','nitrogen_factor','root_respiration_gc_m2')


def pack_plant(state,parameters=PlantParameters(),*,dtype=np.float64):
    p=parameters
    return (np.asarray([*state.carbon,*state.nitrogen,state.reserve_carbon,state.reserve_nitrogen],dtype),
        np.asarray([*p.allocation,*p.target_cn,*p.turnover_days,*p.litter_fractions,p.growth_respiration_ratio,
            p.maintenance_base,p.maintenance_q10,p.stem_live_n_fraction,p.maximum_cn_multiplier,
            p.new_carbon_storage_fraction,p.reserve_release_days,p.specific_leaf_area,p.litter_min_cn_multiplier],dtype))


def _arrays(shapes,values,xp):
    dtype=values[0].dtype;n=len(values[0])
    if dtype not in (xp.float32,xp.float64) or any(v.dtype!=dtype or v.shape!=(n,k) for v,k in zip(values,shapes)):
        raise ValueError('Invalid plant native shape or precision')
    if any(not bool(xp.all(xp.isfinite(v))) for v in values):raise ValueError('Nonfinite plant native input')


def validate_plant(state,parameters,xp):
    _arrays((8,21),(state,parameters),xp);s=state;p=parameters;eps=xp.finfo(s.dtype).eps
    bad=(xp.any(s<0.,axis=1)|xp.any((s[:,:3]==0.)&(s[:,3:6]>0.),axis=1)|xp.any(p[:,:12]<0.,axis=1)|
        xp.any(p[:,3:9]<=0.,axis=1)|(abs(p[:,:3].sum(1)-1.)>8.*eps)|(abs(p[:,9:12].sum(1)-1.)>8.*eps)|
        (p[:,12]<0.)|(p[:,12]>1.)|(p[:,13]<0.)|(p[:,14]<1.)|(p[:,14]>5.)|(p[:,15]<0.)|(p[:,15]>1.)|
        (p[:,16]<1.)|(p[:,17]<0.)|(p[:,17]>1.)|(p[:,18]<=0.)|(p[:,19]<=0.)|(p[:,20]<1.))
    if bool(xp.any(bad)):raise ValueError('Invalid native plant state or physiological parameters')


def validate_turnover(state,inputs,parameters,xp):
    validate_plant(state,parameters,xp);_arrays((8,2),(state,inputs),xp)
    if bool(xp.any((inputs[:,0]<=0.)|(inputs[:,0]>86400.)|(inputs[:,1]<0.)|(inputs[:,1]>1.))):
        raise ValueError('Invalid native turnover timestep or retention')


def validate_plan(state,inputs,parameters,xp):
    validate_plant(state,parameters,xp);_arrays((8,7),(state,inputs),xp);u=inputs
    if bool(xp.any((u[:,0]<0.)|(u[:,1]<0.)|(u[:,2]<150.)|(u[:,2]>350.)|(u[:,3]<150.)|(u[:,3]>350.)|
        (u[:,4]<=0.)|(u[:,4]>86400.)|(u[:,5]<0.)|((u[:,6]!=0.)&(u[:,6]!=1.)))):
        raise ValueError('Invalid native plant allocation climate or carbon supply')


def validate_allocation(plan,supply,parameters,xp):
    _arrays((24,1,21),(plan,supply,parameters),xp);validate_plant(plan[:,:8],parameters,xp)
    if bool(xp.any(plan[:,8:]<0.)) or bool(xp.any((plan[:,9]<=0.)|(plan[:,9]>86400.))):
        raise ValueError('Invalid native plant allocation plan')
    if bool(xp.any((supply[:,0]<0.)|(supply[:,0]>plan[:,23]+32.*xp.finfo(plan.dtype).eps*xp.maximum(plan[:,23],1.)))):
        raise ValueError('Native mineral N award exceeds plant demand')


def _budget(old,new,ci,co,ni,no):
    import jax.numpy as j
    from jax import lax
    s=lax.optimization_barrier
    oc=s(_sum(old[:3])+old[6]);nc=s(_sum(new[:3])+new[6])
    on=s(_sum(old[3:6])+old[7]);nn=s(_sum(new[3:6])+new[7])
    cr=s(s(s(nc-oc)-ci)+co);nr=s(s(s(nn-on)-ni)+no);eps=j.finfo(old.dtype).eps
    cb=s(64.*eps*j.maximum(s(s(s(oc+nc)+ci)+co),1.));nb=s(64.*eps*j.maximum(s(s(s(on+nn)+ni)+no),1.))
    valid=j.all(new>=0.)&j.all(j.isfinite(new))&(abs(cr)<=cb)&(abs(nr)<=nb)
    return j.asarray([cr,nr,cb,nb,valid.astype(old.dtype)],old.dtype)


def turnover_kernel_jax(old,u,p):
    import jax.numpy as j
    from jax import lax
    s=lax.optimization_barrier
    loss=-j.expm1(s(-u[0]/s(p[6:9]*86400.)))
    loss=loss.at[0].set(s(1.-s(s(1.-loss[0])*u[1])))
    fc=s(old[:3]*loss);fn=s(old[3:6]*loss)
    recovered=j.maximum(0.,s(fn[0]-s(fc[0]/s(p[20]*p[3]))))
    lc=j.zeros(7,old.dtype).at[:3].set(s(s(fc[0]+fc[2])*p[9:12])).at[3].set(fc[1])
    ln=j.zeros(7,old.dtype).at[:3].set(s(s(s(fn[0]-recovered)+fn[2])*p[9:12])).at[3].set(fn[1])
    new=j.concatenate((s(old[:3]-fc),s(old[3:6]-fn),j.asarray([old[6],s(old[7]+recovered)],old.dtype)))
    return j.concatenate((new,lc,ln,fc,fn,j.asarray([recovered],old.dtype),_budget(old,new,0.,_sum(lc),0.,_sum(ln))))


def plan_kernel_jax(old,u,p):
    import jax.numpy as j
    from jax import lax
    s=lax.optimization_barrier;row=lambda *v:j.asarray(v,old.dtype)
    qair=j.exp(s(j.log(p[14])*s(s(u[2]-293.15)/10.)))
    qroot=j.exp(s(j.log(p[14])*s(s(u[3]-293.15)/10.)))
    demand=row(u[1],s(s(s(s(old[4]*p[15])*p[13])*u[4])*qair),s(s(s(old[5]*p[13])*u[4])*qroot))
    cash=s(old[6]+u[0]);total=_sum(demand)
    paid=s(demand*j.where(total>0.,j.minimum(1.,cash/j.maximum(total,j.finfo(old.dtype).tiny)),1.))
    available=j.maximum(0.,s(cash-total));current=j.maximum(0.,s(u[0]-total))
    stored=j.maximum(0.,s(available-current));release=-j.expm1(s(-u[4]/s(p[18]*86400.)))
    budget=j.where(u[6]>0.,s(s(s(1.-p[17])*current)+s(stored*release)),0.)
    potential=s(p[:3]*s(budget/s(1.+p[12])))
    potential=potential.at[0].set(j.minimum(potential[0],j.maximum(0.,s(u[5]-old[0]))))
    deficit=j.maximum(s(s(old[:3]/p[3:6])-old[3:6]),0.)
    nd=j.maximum(0.,s(s(_sum(deficit)+_sum(s(potential/p[3:6])))-old[7]))
    return j.concatenate((old,row(u[0],u[4],available),potential,deficit,paid,s(demand-paid),row(nd)))


def allocation_kernel_jax(plan,award,p):
    import jax.numpy as j
    from jax import lax
    s=lax.optimization_barrier;old=plan[:8];row=lambda *v:j.asarray(v,old.dtype);tiny=j.finfo(old.dtype).tiny
    supply=s(old[7]+award[0]);deficit=_sum(plan[14:17])
    repair=s(plan[14:17]*j.where(deficit>0.,j.minimum(1.,supply/j.maximum(deficit,tiny)),0.))
    available=j.maximum(0.,s(supply-deficit));target=s(plan[11:14]/p[3:6]);minimum=s(_sum(target)/p[16])
    growth=s(plan[11:14]*j.where(minimum>0.,j.minimum(1.,available/j.maximum(minimum,tiny)),1.))
    target=s(growth/p[3:6]);total=_sum(target)
    ng=s(target*j.where(total>0.,j.minimum(1.,available/j.maximum(total,tiny)),0.))
    rn=j.maximum(0.,s(available-total));gr=s(p[12]*growth)
    rc=j.maximum(0.,s(plan[10]-s(_sum(growth)+_sum(gr))))
    new=j.concatenate((s(old[:3]+growth),s(s(old[3:6]+repair)+ng),row(rc,rn)))
    ra=s(_sum(plan[17:20])+_sum(gr));nf=j.where(new[0]>0.,j.minimum(1.,s(new[3]*p[3])/j.maximum(new[0],tiny)),0.)
    diag=row(plan[8],s(plan[8]-ra),ra,plan[23],award[0],nf,s(plan[19]+gr[2]))
    return j.concatenate((new,growth,ng,repair,plan[17:20],gr,plan[20:23],diag,_budget(old,new,plan[8],ra,award[0],0.)))


def _unpack(output,xp,stage):
    out=dict(state=output[:,:8],carbon=output[:,:3],nitrogen=output[:,3:6],reserve_carbon=output[:,6],reserve_nitrogen=output[:,7])
    if stage=='turnover':
        out.update(litter_carbon_gc_m2=output[:,8:15],litter_nitrogen_gn_m2=output[:,15:22],
            tissue_carbon_loss_gc_m2=output[:,22:25],tissue_nitrogen_loss_gn_m2=output[:,25:28],retranslocated_nitrogen_gn_m2=output[:,28])
        out.update(zip(BUDGETS,output[:,29:].T))
    else:
        for index,key in enumerate(('carbon_growth_gc_m2','nitrogen_growth_gn_m2','nitrogen_repair_gn_m2',
            'maintenance_respiration_gc_m2','growth_respiration_gc_m2','unmet_maintenance_gc_m2')):
            out[key]=output[:,8+index*3:11+index*3]
        out.update(zip(ALLOCATION_DIAGNOSTICS,output[:,26:33].T));out.update(zip(BUDGETS,output[:,33:].T))
    if not bool(xp.all(xp.isfinite(output))) or not bool(xp.all(out['valid']>0.)):
        raise RuntimeError('Native plant C/N domain or numerical budget failed')
    return out


@lru_cache(maxsize=3)
def _compiled(stage):
    import jax
    return jax.jit(jax.vmap({'turnover':turnover_kernel_jax,'plan':plan_kernel_jax,'allocation':allocation_kernel_jax}[stage]))


def run_plant_turnover_jax(state,inputs,parameters):
    import jax.numpy as j
    arrays=tuple(map(j.asarray,(state,inputs,parameters)));validate_turnover(*arrays,j)
    return _unpack(_compiled('turnover')(*arrays),j,'turnover')


def plan_plant_allocation_jax(state,inputs,parameters):
    import jax.numpy as j
    arrays=tuple(map(j.asarray,(state,inputs,parameters)));validate_plan(*arrays,j)
    out=_compiled('plan')(*arrays)
    if not bool(j.all(j.isfinite(out))) or bool(j.any(out<0.)):raise RuntimeError('Invalid native plant demand plan')
    return out


def apply_plant_allocation_jax(plan,supply,parameters):
    import jax.numpy as j
    arrays=tuple(map(j.asarray,(plan,supply,parameters)));validate_allocation(*arrays,j)
    return _unpack(_compiled('allocation')(*arrays),j,'allocation')
