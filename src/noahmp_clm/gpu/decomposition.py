"""Native C/N decomposition with explicit extensive transfer ledgers."""
from functools import lru_cache
import numpy as np
from ..physics.decomposition import DecompositionParameters,DONOR,RECEIVER
from .canopy_energy import _sum

DIAGNOSTICS=('heterotrophic_respiration_gc_m2','heterotrophic_respiration_umol_m2s',
    'temperature_factor','moisture_factor','rate_multiplier','potential_immobilization_gn_m2',
    'immobilization_gn_m2','gross_mineralization_gn_m2','plant_uptake_gn_m2','nitrogen_limitation',
    'carbon_residual_gc_m2','nitrogen_residual_gn_m2','carbon_bound_gc_m2','nitrogen_bound_gn_m2','valid')


def pack_decomposition(state,temperature,potential,dt,*,carbon_input=None,nitrogen_input=None,
    mineral_n_input=0.,plant_n_demand=0.,oxygen_factor=1.,parameters=DecompositionParameters(),dtype=np.float64):
    p=parameters;ci=np.zeros(7) if carbon_input is None else carbon_input;ni=np.zeros(7) if nitrogen_input is None else nitrogen_input
    return (np.asarray([*state.carbon,*state.nitrogen,state.mineral_nitrogen],dtype),
        np.asarray([temperature,potential,dt,oxygen_factor,*ci,*ni,mineral_n_input,plant_n_demand],dtype),
        np.asarray([*p.turnover_years,*p.soil_cn,p.sand_fraction,p.wood_cellulose_fraction,p.q10,
            p.reference_temperature,p.dry_potential_mpa,p.wet_potential_mpa],dtype))


def validate_decomposition(s,u,p,xp):
    n=len(s)
    if s.shape!=(n,15) or u.shape!=(n,20) or p.shape!=(n,16) or s.dtype not in (xp.float32,xp.float64) or u.dtype!=s.dtype or p.dtype!=s.dtype:
        raise ValueError('Invalid decomposition array shapes or precision')
    if not all(bool(xp.all(xp.isfinite(v))) for v in (s,u,p)):raise ValueError('Nonfinite decomposition input')
    bad=(xp.any(s<0.,axis=1)|xp.any(u[:,4:]<0.,axis=1)|xp.any(p[:,:10]<=0.,axis=1)|
        (u[:,0]<150.)|(u[:,0]>350.)|(u[:,1]>0.)|(u[:,2]<=0.)|(u[:,2]>86400.)|(u[:,3]<0.)|(u[:,3]>1.)|
        (p[:,10]<0.)|(p[:,10]>1.)|(p[:,11]<0.)|(p[:,11]>1.)|(p[:,12]<1.)|(p[:,12]>5.)|
        (p[:,13]<250.)|(p[:,13]>330.)|(p[:,14]>=p[:,15])|(p[:,15]>=0.))
    c=s[:,:7]+u[:,4:11];n_organic=s[:,7:14]+u[:,11:18]
    if bool(xp.any(bad)) or bool(xp.any(((s[:,:7]==0.)&(s[:,7:14]>0.))|((c==0.)&(n_organic>0.)))):
        raise ValueError('Invalid decomposition state, climate or parameters')


def decomposition_kernel_jax(old,u,p):
    import jax.numpy as j
    from jax import lax
    s=lax.optimization_barrier;row=lambda *v:j.asarray(v,old.dtype)
    c=s(old[:7]+u[4:11]);n=s(old[7:14]+u[11:18]);mineral=s(old[14]+u[18])
    tf=j.exp(s(j.log(p[12])*s(s(u[0]-p[13])/10.)))
    frozen=s(s(-334000.*j.maximum(s(273.15-u[0]),0.))/s(1000.*u[0]));psi=j.minimum(u[1],frozen)
    ratio=p[14]/j.minimum(psi,p[15])
    wf=j.where(psi<=p[14],0.,j.where(psi>=p[15],1.,j.log(ratio)/j.log(p[14]/p[15])))
    modifier=s(s(tf*wf)*u[3]);fraction=-j.expm1(s(s(-modifier*u[2])/s(p[:7]*31536000.)))
    rf=s(.17+s(.68*p[10]));share=.004/s(1.-rf)
    path=row(1.,1.,1.,p[11],s(1.-p[11]),s(1.-share),share,.42/.45,.03/.45,1.)
    resp=row(.55,.5,.5,0.,0.,rf,rf,.55,.55,.55)
    donor=j.asarray(DONOR);receiver=j.asarray(RECEIVER)
    co=s(s(c[donor]*fraction[donor])*path);no=s(s(n[donor]*fraction[donor])*path)
    ci=s(co*s(1.-resp));ni=j.where(receiver>=4,ci/p[j.clip(receiver-4,0,2)+7],no)
    demand=s(ni-no);potential=_sum(j.maximum(demand,0.));total=s(potential+u[19])
    limitation=j.where(total>0.,j.minimum(1.,mineral/j.maximum(total,j.finfo(old.dtype).tiny)),1.)
    scale=j.where(demand>0.,limitation,1.);co=s(co*scale);ci=s(ci*scale);no=s(no*scale);ni=s(ni*scale)
    dn=s(ni-no);immob=_sum(j.maximum(dn,0.));nmin=_sum(j.maximum(-dn,0.));uptake=s(u[19]*limitation)
    nc,nn=c,n
    for k,(i,q) in enumerate(zip(DONOR,RECEIVER)):
        nc=nc.at[i].set(s(nc[i]-co[k]));nc=nc.at[q].set(s(nc[q]+ci[k]))
        nn=nn.at[i].set(s(nn[i]-no[k]));nn=nn.at[q].set(s(nn[q]+ni[k]))
    eps=j.finfo(old.dtype).eps
    valid=j.all(nc>=-64.*eps*j.maximum(_sum(c),1.))&j.all(nn>=-64.*eps*j.maximum(_sum(n),1.))
    nc=j.maximum(nc,0.);nn=j.maximum(nn,0.);nm=s(j.maximum(0.,s(s(mineral-immob)-uptake))+nmin)
    rh=_sum(s(co-ci));cr=s(s(s(_sum(nc)-_sum(old[:7]))-_sum(u[4:11]))+rh)
    nr=s(s(s(s(s(s(_sum(nn)+nm)-_sum(old[7:14]))-old[14])-_sum(u[11:18]))-u[18])+uptake)
    cb=s(64.*eps*j.maximum(s(s(s(_sum(nc)+_sum(old[:7]))+_sum(u[4:11]))+rh),1.))
    nb=s(64.*eps*j.maximum(s(s(s(s(s(s(_sum(nn)+nm)+_sum(old[7:14]))+old[14])+_sum(u[11:18]))+u[18])+uptake),1.))
    state=j.concatenate((nc,nn,row(nm)))
    diag=row(rh,rh/s(12.0107e-6*u[2]),tf,wf,modifier,potential,immob,nmin,uptake,limitation,cr,nr,cb,nb,valid.astype(old.dtype))
    return j.concatenate((state,co,ci,no,ni,diag))


def unpack_decomposition(output,xp):
    out=dict(state=output[:,:15],carbon=output[:,:7],nitrogen=output[:,7:14],mineral_nitrogen=output[:,14],
        carbon_out_gc_m2=output[:,15:25],carbon_in_gc_m2=output[:,25:35],nitrogen_out_gn_m2=output[:,35:45],nitrogen_in_gn_m2=output[:,45:55])
    out.update(zip(DIAGNOSTICS,output[:,55:].T))
    if (not bool(xp.all(xp.isfinite(output))) or bool(xp.any(output[:,:15]<0.)) or not bool(xp.all(out['valid']>0.))
        or bool(xp.any(abs(out['carbon_residual_gc_m2'])>out['carbon_bound_gc_m2']))
        or bool(xp.any(abs(out['nitrogen_residual_gn_m2'])>out['nitrogen_bound_gn_m2']))):
        raise RuntimeError('Native decomposition violates C/N domain or numerical budget bound')
    return out


@lru_cache(maxsize=1)
def _compiled():
    import jax
    return jax.jit(jax.vmap(decomposition_kernel_jax))


def run_decomposition_jax(state,inputs,parameters):
    import jax.numpy as j
    arrays=tuple(map(j.asarray,(state,inputs,parameters)));validate_decomposition(*arrays,j)
    return unpack_decomposition(_compiled()(*arrays),j)
