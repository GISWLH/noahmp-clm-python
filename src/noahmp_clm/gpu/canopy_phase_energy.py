"""Native coupled layer phase enthalpy and canopy-air energy balance.

The independent NumPy implementation defines the physics. This development
entry has its own device Newton solve and retains every mass/heat parcel.
"""
from functools import lru_cache,partial
import numpy as np
from .canopy_energy import (pack_canopy_case,validate_canopy_batch,unpack_canopy_energy,
    _sum,_dot,_linear_solve,_root_allocation,_positive_scale,_longwave_operator)
from .canopy_boundary import radiation_inputs
from .canopy_snow_radiation import validate_canopy_snow,canopy_snow_kernel_jax
from .snicar import pack_snow_optics,_jax_tables
from .photosynthesis import pack_leaf_parameters

PHASE_LAYERS=('mass','enthalpy','thermal_enthalpy','ice_vapor','liquid_vapor',
    'vapor_enthalpy','vapor_latent_energy','unmet_evaporation','phase_coordinates')
PHASE_SCALARS=('canopy_storage_change_wm2','wet_advected_heat_wm2',
    'transpiration_advected_heat_wm2','ground_nonshortwave_heat','soil_absorbed_shortwave',
    'ground_latent_heat','restarts','max_solver_residual')
PHASE_OUTPUT_SIZE=300


def pack_canopy_phase(leaf,swdown,lwdown,exchange,condensate,dt,*,stem_area=0.,
        fixed_ground_temperature=None,ground_is_ice=False,ground_shares_soil=True,
        root_enthalpy=None,snowpack=None,dtype=np.float64,**kwargs):
    from ..physics.snowpack import Snowpack
    a,u,f,soil,_=pack_canopy_case(leaf,swdown,lwdown,exchange,dtype=dtype,**kwargs)
    if condensate.mass.shape!=(10,):raise ValueError('Ten condensate layers required')
    if kwargs.get('soil_supply') is not None and root_enthalpy is None:
        raise ValueError('Explicit root source enthalpy required')
    rh=np.zeros(4) if root_enthalpy is None else np.asarray(root_enthalpy)
    if rh.shape!=(4,):raise ValueError('Four root enthalpies required')
    if snowpack is not None and fixed_ground_temperature is None:
        raise ValueError('Snow radiation requires fixed surface enthalpy coupling')
    b=np.asarray([fixed_ground_temperature is not None,exchange.air_temperature if fixed_ground_temperature is None else fixed_ground_temperature,
        ground_is_ice,ground_shares_soil,dt,*rh],dtype)
    cells,n,_=pack_snow_optics(Snowpack.empty() if snowpack is None else snowpack,[0.,0.],[0.,0.],dtype=dtype)
    return a,u,f,soil,np.broadcast_to(stem_area,(10,)).astype(dtype),np.column_stack((condensate.mass,condensate.enthalpy)).astype(dtype),b,cells,n


def validate_phase(arrays,xp):
    a,u,f,soil,stem,c,b,snow,n=arrays;size=len(a)
    validate_canopy_batch(a,u,f,soil,xp.broadcast_to(u[:,0,None],(size,11)),xp)
    if stem.shape!=(size,10) or c.shape!=(size,10,2) or b.shape!=(size,9):raise ValueError('Invalid phase canopy shapes')
    if any(x.dtype!=a.dtype for x in (u,f,soil,stem,c,b,snow)) or a.dtype not in (xp.float32,xp.float64):
        raise ValueError('Phase canopy requires matching floating precision')
    if not all(bool(xp.all(xp.isfinite(v))) for v in (stem,c,b)) or bool(xp.any(stem<0.)):
        raise ValueError('Invalid phase canopy values')
    mass,heat=c[:,:,0],c[:,:,1];eps=xp.finfo(a.dtype).eps;rounding=8.*eps*xp.maximum(abs(heat),1.)
    inactive=(u[:,13,None]*(a+stem)==0.)
    bad=(mass<0.)|((mass==0.)&(heat!=0.))|(inactive&(mass>0.))
    bad|=(heat<mass*2100.*(200.-273.15)-rounding)|(heat>mass*(334000.+4180.*(350.-273.15))+rounding)
    flags=b[:,[0,2,3]]
    if bool(xp.any(bad)) or bool(xp.any((flags!=0.)&(flags!=1.))):raise ValueError('Invalid phase state or flags')
    if bool(xp.any((b[:,4]<=0.)|(b[:,1]<200.)|(b[:,1]>350.)|((b[:,2]>0.)&((b[:,0]==0.)|(b[:,1]>xp.asarray(273.15,a.dtype))))
            |((soil[:,0]==0.)&(u[:,16]>0.))|((n>0)&(b[:,0]==0.)))):raise ValueError('Invalid phase boundary or root supply')
    validate_canopy_snow(a+stem,snow,n,radiation_inputs(a+stem,u,xp),xp)


def unpack_phase(out):
    result=unpack_canopy_energy(out[:,:195])
    result.update({k:out[:,195+10*i:205+10*i] for i,k in enumerate(PHASE_LAYERS)})
    result.update({k:out[:,285+i] for i,k in enumerate(PHASE_SCALARS)})
    result['restarts']=result['restarts'].astype('int32');result['snow_absorbed_shortwave']=out[:,293:298]
    result['root_advected_heat_wm2']=out[:,298];result['transpiration_net_heat_wm2']=out[:,299]
    return result


def phase_energy_kernel_jax(leaf,u,f,soil,stem,c,b,snow,count,p,ice,impurity,*,max_iterations=100,tolerance=.005,radiation=None,exchange_extension=None):
    import jax
    import jax.numpy as j
    from jax import lax
    from .photosynthesis import leaf_kernel_jax
    s=lax.optimization_barrier
    water_velocity,heat_temperature=(u[3],u[0]) if exchange_extension is None else exchange_extension
    area=s(leaf+stem);mass=c[:,0];oldh=c[:,1];fv=j.where(_sum(area)>0.,u[13],0.);active=s(fv*area)>0.
    rad=(canopy_snow_kernel_jax(area,snow,count,j.stack((u[11],fv,u[21],u[22],u[19],u[20])),ice,impurity,track_absorption=True)
        if radiation is None else radiation)
    sw=j.concatenate((rad[:10],j.asarray([s(_sum(rad[20:25])+rad[25])])));par=rad[10:20]
    lw=_longwave_operator(area,u[18],fv)
    rho=u[2]/s(s(287.*u[0])*s(1.+s(.61*u[1])));mol=u[2]/s(8.314462618*u[0])
    weights=s(s(rho*1004.)*j.concatenate((s(s(fv*area)*u[5]),j.asarray([s(fv*u[4])]))))
    atm=s(s(s(rho*1004.)*fv)*u[3]);bare=s(s(s(rho*1004.)*s(1.-fv))*u[3]);hden=s(_sum(weights)+atm)
    vc=s(s(s(rho*fv)*area)*u[5]);cap=s(s(.1*fv)*area)
    fw=j.where(cap>0.,j.minimum(1.,j.power(mass/j.where(cap>0.,cap,1.),2./3.)),0.)
    dry=s(s(1.-fw)*(leaf/j.where(area>0.,area,1.)))
    trlimit=j.where(soil[0]>0.,j.minimum(u[16],_sum(j.where(soil[5:9]>0.,soil[1:5],0.))),0.)
    latent_ground=s(2.5e6+j.where(b[2]>0.,334000.,0.));plateau=j.asarray(334000./4180.,u.dtype)
    lower=j.concatenate((j.where(mass>0.,j.full(10,(200.-273.15)*2100./4180.,u.dtype),j.full(10,200.-273.15,u.dtype)),j.asarray([200.,0.],u.dtype)))
    upper=j.concatenate((j.where(mass>0.,plateau+350.-273.15,350.-273.15),j.asarray([350.,190.],u.dtype)))

    def evaluate(z,phase_active):
        x=z[:10];q=s(z[11]*.001)
        tleaf=j.where(phase_active,s(273.15+s(s(s(j.minimum(x,0.)*4180.)/2100.)+j.maximum(s(x-plateau),0.))),s(273.15+x))
        t=j.concatenate((tleaf,z[10:11]));thermal=s(s(mass*4180.)*x)
        tc=s(t-273.15);es=s(611.2*j.exp(s(s(17.67*tc)/s(tc+243.5))))
        qliq=s(.622*es)/s(u[2]-s(.378*es))
        esi=j.exp(s(s(s(9.550426-5723.265/t)+s(3.53068*j.log(t)))-s(.00728332*t)))
        qice=s(.622*esi)/s(u[2]-s(.378*esi))
        dry_fraction=j.where(phase_active,j.clip(s(x*4180.)/334000.,0.,1.),j.asarray(tleaf>273.15,u.dtype))
        frac=j.where(mass>0.,j.clip(thermal/334000.,0.,mass)/j.where(mass>0.,mass,1.),dry_fraction)
        qs=s(s(frac*qliq[:10])+s(s(1.-frac)*qice[:10]))
        rh=s(q*u[2])/s(.622+s(.378*q))/es[:10]
        leaf_input=j.stack((par,tleaf,j.full(10,u[8]),rh,j.full(10,s(u[5]*mol)),s(f[:,0]*j.where(trlimit>0.,f[:,1],0.)),f[:,1]),axis=1)
        gas=jax.vmap(leaf_kernel_jax,in_axes=(0,None))(leaf_input,p)
        amount=s(b[4]*s(s(vc*j.where(qs>q,fw,1.))*s(qs-q)))
        loss=j.minimum(j.maximum(amount,0.),mass);deposit=j.maximum(-amount,0.);remaining=s(mass-loss)
        retained=s((remaining/j.where(mass>0.,mass,1.))*thermal)
        added=s(deposit*s(s(s(1.-frac)*s(2100.*s(tleaf-273.15)))+s(frac*s(334000.+s(4180.*s(tleaf-273.15))))))
        newm=s(remaining+deposit);newh=s(retained+added);adv=s(s(thermal-retained)-added)
        vapor=s(loss-deposit);liquid=s(vapor*frac);iv=s(vapor-liquid);lewet=s(s(2.5e6*vapor)+s(334000.*iv))
        wet=vapor/b[4]
        tr=s(s(s(s(vc*dry)*gas[:,9])/s(s(u[5]*mol)+gas[:,9]))*j.maximum(s(qliq[:10]-q),0.))
        total=_sum(tr);tr=s(tr*j.where(total>0.,j.minimum(1.,trlimit/j.maximum(total,1e-30)),1.))
        uptake=j.where(soil[0]>0.,_root_allocation(_sum(tr),soil),j.zeros(4,u.dtype))
        limit=j.where((soil[0]>0.)&(b[3]>0.),j.minimum(u[17],j.maximum(0.,s(soil[9]-uptake[0]))),u[17])
        gqs=j.where(b[2]>0.,qice[10],qliq[10])
        ge=s(rho*j.stack((s(fv*s(gqs-q)),s(s(1.-fv)*s(gqs-u[1])))))/s(1./j.stack((u[4],water_velocity))+u[9])
        ge=s(j.minimum(ge,0.)+s(j.maximum(ge,0.)*_positive_scale(ge,limit)))
        offset=j.where(hden>0.,_dot(weights,s(t-heat_temperature))/j.maximum(hden,1e-30),0.)
        sensible=s(weights*s(s(t-heat_temperature)-offset));sensible=sensible.at[10].add(s(bare*s(t[10]-heat_temperature)))
        emission=j.concatenate((s(5.670374419e-8*s(s(t*t)*s(t*t))),u[12:13]))
        rn=s(sw+jax.vmap(lambda row:_dot(row,emission))(lw))
        latent=j.concatenate((s(lewet/b[4]+s(2.5e6*tr)),j.asarray([s(latent_ground*_sum(ge))])))
        take=_sum(uptake);root=j.where(take>0.,_dot(uptake,b[5:9])/j.maximum(take,1e-30),0.)
        trheat=s(tr*s(s(334000.+s(4180.*s(tleaf-273.15)))-root))
        storage=s(newh-oldh)/b[4];thermal_delta=s(thermal-oldh)/b[4]
        raw=s(s(rn-sensible)-latent);gh=j.where(b[0]>0.,raw[10],s(s(u[7]*s(t[10]-u[6]))+u[10]))
        physical=raw.at[:10].add(-s(s(storage+adv/b[4])+trheat)).at[10].add(-gh)
        residual=raw.at[:10].add(-s(thermal_delta+trheat)).at[10].add(-gh)
        residual=residual.at[:10].set(j.where(active,residual[:10],s(u[0]-tleaf)))
        residual=residual.at[10].set(j.where(b[0]>0.,s(b[1]-t[10]),residual[10]))
        airwater=s(s(s(_sum(wet)+_sum(tr))+ge[0])-s(s(s(rho*fv)*water_velocity)*s(q-u[1])))
        allres=j.concatenate((residual,j.asarray([j.where(fv>0.,s(airwater*2.5e6),s(s(q-u[1])*1e5))])))
        out=j.zeros(PHASE_OUTPUT_SIZE,u.dtype).at[:10].set(tleaf).at[10:15].set(j.stack((t[10],s(heat_temperature+offset),q,s(fv*_dot(leaf,j.where(active,gas[:,0],0.))),s(fv*_dot(leaf,j.where(active,gas[:,5],0.))))))
        out=out.at[15:26].set(sensible).at[26:30].set(j.stack((_sum(sensible),_sum(latent),gh,_sum(rn))))
        out=out.at[30:40].set(wet).at[40:50].set(tr).at[50].set(_sum(ge)).at[51:55].set(uptake).at[55:66].set(physical).at[66].set(_sum(physical))
        out=out.at[69].set(_sum(s(s(newm-mass)+vapor))).at[71].set(s(s(_sum(sensible)-s(atm*offset))-s(bare*s(t[10]-heat_temperature)))).at[72].set(airwater)
        out=out.at[73].set(rad[26]).at[74].set(s(u[12]-_sum(s(rn-sw)))).at[75:195].set(gas.ravel())
        fields=j.stack((newm,newh,thermal,iv,liquid,adv,lewet,j.maximum(s(amount-loss),0.),x))
        out=out.at[195:285].set(fields.ravel()).at[285:293].set(j.stack((_sum(storage),_sum(adv)/b[4],_dot(tr,s(334000.+s(4180.*s(tleaf-273.15)))),s(gh-sw[10]),rad[25],latent_ground,j.asarray(0.,u.dtype),j.max(abs(allres))))).at[293:298].set(rad[20:25])
        out=out.at[298].set(_dot(uptake,b[5:9])).at[299].set(_sum(trheat))
        return allres,out

    old=j.concatenate((j.where(mass>0.,oldh/j.where(mass>0.,s(mass*4180.),1.),s(u[0]-273.15)),j.stack((j.where(b[0]>0.,b[1],u[0]),s(u[1]*1000.)))))
    # XLA can combine Ci/Cw across a multiply/divide and change the cold
    # initial enthalpy by one ulp. Preserve both native float32 operations.
    ambient=old.at[:10].set(j.where(mass>0.,j.where(u[0]<273.15,s(s(s(u[0]-273.15)*2100.)/4180.),s(plateau+s(u[0]-273.15))),s(u[0]-273.15)))
    ambient=j.where(u[0]==j.asarray(273.15,u.dtype),old,ambient)
    r0,_=evaluate(old,mass>0.);ra,_=evaluate(ambient,mass>0.);choose=_dot(ra,ra)<_dot(r0,r0)
    initial=j.where(choose,ambient,old);alternative=j.where(choose,old,ambient)
    available=j.any(initial!=alternative)
    step_size=j.asarray(.02 if u.dtype==j.float32 else 1e-6,u.dtype);qstep=j.asarray(.002 if u.dtype==j.float32 else 1e-7,u.dtype)
    def body(state):
        iteration,x,restarts,avail,failed,phase_active=state
        r,out=evaluate(x,phase_active)
        fresh=(~phase_active)&active&(abs(x[:10])<=.05)&(out[30:40]<0.)
        phase_active=phase_active|fresh
        x=x.at[:10].set(j.where(fresh,j.where(x[:10]<0.,s(s(x[:10]*2100.)/4180.),s(plateau+x[:10])),x[:10]))
        lo_bound=lower.at[:10].set(j.where(phase_active,j.full(10,(200.-273.15)*2100./4180.,u.dtype),lower[:10]))
        hi_bound=upper.at[:10].set(j.where(phase_active,s(s(plateau+350.)-273.15),upper[:10]))
        converted=alternative.at[:10].set(j.where(phase_active&(mass==0.),
            j.where(alternative[:10]<0.,s(s(alternative[:10]*2100.)/4180.),s(plateau+alternative[:10])),alternative[:10]))
        r,out=evaluate(x,phase_active)
        depleted=phase_active&(out[195:205]==0.)&(x[:10]>0.)&(x[:10]<plateau)&(abs(r[:10])>tolerance)
        x=x.at[:10].set(j.where(depleted,j.where(r[:10]>0.,plateau,0.),x[:10]));r,_=evaluate(x,phase_active)
        def column(k):
            step=j.where(k==11,qstep,step_size)
            lo=x.at[k].set(j.maximum(lo_bound[k],s(x[k]-step)));hi=x.at[k].set(j.minimum(hi_bound[k],s(x[k]+step)))
            a,_=evaluate(lo,phase_active);d,_=evaluate(hi,phase_active);return s(d-a)/s(hi[k]-lo[k])
        jac=jax.vmap(column)(j.arange(12)).T
        delta=_linear_solve(jac,-r);scale=j.maximum(1.,j.max(abs(delta)/j.concatenate((j.full(10,20.,u.dtype),j.asarray([5.,5.],u.dtype)))))
        delta=delta/scale;norm=_dot(r,r)
        def backtrack(k,carry):
            chosen,accepted,chosen_norm=carry
            # exp2 is lowered through an approximate exponential by XLA;
            # these exact binary half-steps must not depend on that lowering.
            fraction=j.asarray([2.**(-i) for i in range(24)],u.dtype)[k]
            trial=j.clip(s(x+s(delta*fraction)),lo_bound,hi_bound);rr,_=evaluate(trial,phase_active)
            trial_norm=_dot(rr,rr);good=(~accepted)&(trial_norm<norm)
            return j.where(good,trial,chosen),accepted|good,j.where(good,trial_norm,chosen_norm)
        # The first strict descent is final. Match NumPy's early break and
        # avoid evaluating the remaining fractions after it has been chosen.
        def backtrack_step(state):
            k,carry=state
            return k+1,backtrack(k,carry)
        _,(trial,accepted,chosen_norm)=lax.while_loop(
            lambda state:(state[0]<24)&(~state[1][1]),backtrack_step,
            (j.asarray(0,j.int32),(x,j.asarray(False),norm)))
        # Try the first exact phase edge for existing and newly deposited
        # parcels, retaining old stores and strict complete-residual descent.
        edges=j.stack((j.zeros(10,u.dtype),j.full(10,plateau,u.dtype)),axis=1)
        direction=delta[:10]
        fractions=s(edges-x[:10,None])/j.where(direction[:,None]!=0.,direction[:,None],1.)
        eligible=phase_active[:,None]&(direction[:,None]!=0.)&(fractions>0.)&(fractions<=1.)
        scores=j.where(eligible,fractions,j.asarray(2.,u.dtype)).reshape(20)
        selected=j.argmin(scores);alpha=scores[selected]
        def phase_edge_trial(_):
            candidate=j.clip(s(x+s(delta*alpha)),lo_bound,hi_bound)
            candidate=candidate.at[selected//2].set(j.where(selected%2==0,j.asarray(0.,u.dtype),plateau))
            rr,_=evaluate(candidate,phase_active);event_norm=_dot(rr,rr)
            good=event_norm<chosen_norm
            return j.where(good,candidate,trial),accepted|good,j.where(good,event_norm,chosen_norm)
        trial,accepted,chosen_norm=lax.cond(alpha<=1.,phase_edge_trial,lambda _:(trial,accepted,chosen_norm),operand=None)
        slow=(~accepted)|(chosen_norm>s(.999*norm))
        def humidity_restart(_):
            lo=trial.at[11].set(lo_bound[11]);hi=trial.at[11].set(hi_bound[11])
            rl,_=evaluate(lo,phase_active);rh,_=evaluate(hi,phase_active);bracket=(rl[11]>=0.)&(rh[11]<=0.)
            def bisect(state):
                k,low,high=state;mid=s(.5*s(low+high));rm,_=evaluate(mid,phase_active)
                return k+1,j.where(rm[11]>0.,mid,low),j.where(rm[11]>0.,high,mid)
            # A dynamic loop avoids 32 extra gas-exchange evaluations in
            # batches where every Newton step made sufficient progress.
            _,lo,hi=lax.while_loop(lambda state:(state[0]<32)&slow&bracket,bisect,(j.asarray(0,j.int32),lo,hi))
            candidate=s(.5*s(lo+hi));rr,_=evaluate(candidate,phase_active)
            good=bracket&(_dot(rr,rr)<chosen_norm)
            return j.where(good,candidate,trial),good
        trial,rescued=lax.cond(slow,humidity_restart,lambda _:(trial,j.asarray(False)),operand=None)
        accepted=accepted|rescued
        def interval_restart(_):
            # An original centred difference can straddle0 or LF/Cw. Keep
            # the usual trajectory unless both original recovery paths fail.
            def local_column(k):
                step=j.where(k==11,qstep,step_size)
                lo=x.at[k].set(j.maximum(lo_bound[k],s(x[k]-step)));hi=x.at[k].set(j.minimum(hi_bound[k],s(x[k]+step)))
                phase=(k<10)&phase_active[j.minimum(k,9)]
                cold=phase&(x[k]<0.);melt=phase&(x[k]>0.)&(x[k]<plateau);warm=phase&(x[k]>plateau)
                lo=lo.at[k].set(j.maximum(lo[k],j.where(melt,0.,j.where(warm,plateau,lo[k]))))
                hi=hi.at[k].set(j.minimum(hi[k],j.where(cold,0.,j.where(melt,plateau,hi[k]))))
                a,_=evaluate(lo,phase_active);d,_=evaluate(hi,phase_active)
                return s(d-a)/s(hi[k]-lo[k])
            local_jac=jax.vmap(local_column)(j.arange(12)).T
            delta=_linear_solve(local_jac,-r)
            scale=j.maximum(1.,j.max(abs(delta)/j.concatenate((j.full(10,20.,u.dtype),j.asarray([5.,5.],u.dtype)))))
            delta=delta/scale
            def local_backtrack(k,carry):
                chosen,good=carry;fraction=j.asarray([2.**(-i) for i in range(24)],u.dtype)[k]
                candidate=j.clip(s(x+s(delta*fraction)),lo_bound,hi_bound);rr,_=evaluate(candidate,phase_active)
                better=(~good)&(_dot(rr,rr)<norm)
                return j.where(better,candidate,chosen),good|better
            def local_step(state):
                k,carry=state
                return k+1,local_backtrack(k,carry)
            _,result=lax.while_loop(lambda state:(state[0]<24)&(~state[1][1]),
                local_step,(j.asarray(0,j.int32),(trial,j.asarray(False))))
            return result
        trial,phase_rescued=lax.cond(~accepted,interval_restart,lambda _:(trial,j.asarray(False)),operand=None)
        accepted=accepted|phase_rescued
        wide_rescued=j.asarray(False)
        if u.dtype==j.float64:
            # Match NumPy's final-stall recovery only after the original
            # fine-stencil, humidity, phase and alternate-seed paths fail.
            # Keep the float32 JAX/CUDA algorithm and arithmetic unchanged.
            def wider_restart(_):
                def attempt(clipped,carry):
                    chosen,good=carry
                    def wide_column(k):
                        step=j.where(k==11,j.asarray(1e-5,u.dtype),j.asarray(1e-4,u.dtype))
                        lo=x.at[k].set(j.maximum(lo_bound[k],s(x[k]-step)))
                        hi=x.at[k].set(j.minimum(hi_bound[k],s(x[k]+step)))
                        phase=(k<10)&phase_active[j.minimum(k,9)]
                        cold=phase&(x[k]<0.);melt=phase&(x[k]>0.)&(x[k]<plateau);warm=phase&(x[k]>plateau)
                        lo=lo.at[k].set(j.where(clipped,j.maximum(lo[k],j.where(melt,0.,j.where(warm,plateau,lo[k]))),lo[k]))
                        hi=hi.at[k].set(j.where(clipped,j.minimum(hi[k],j.where(cold,0.,j.where(melt,plateau,hi[k]))),hi[k]))
                        a,_=evaluate(lo,phase_active);d,_=evaluate(hi,phase_active)
                        return s(d-a)/s(hi[k]-lo[k])
                    wide_jac=jax.vmap(wide_column)(j.arange(12)).T
                    direction=_linear_solve(wide_jac,-r)
                    direction=direction/j.maximum(1.,j.max(abs(direction)/j.concatenate((j.full(10,20.,u.dtype),j.asarray([5.,5.],u.dtype)))))
                    def wide_step(state):
                        k,current,done=state
                        fraction=j.asarray([2.**(-i) for i in range(24)],u.dtype)[k]
                        candidate=j.clip(s(x+s(direction*fraction)),lo_bound,hi_bound)
                        rr,_=evaluate(candidate,phase_active);better=_dot(rr,rr)<norm
                        return k+1,j.where(better,candidate,current),better
                    _,chosen,good=lax.while_loop(lambda state:(state[0]<24)&(~state[2]),wide_step,(j.asarray(0,j.int32),chosen,good))
                    return chosen,good
                first=attempt(j.asarray(False),(trial,j.asarray(False)))
                return lax.cond(first[1],lambda _:first,lambda _:attempt(j.asarray(True),first),operand=None)
            trial,wide_rescued=lax.cond((~accepted)&(~avail),wider_restart,lambda _:(trial,j.asarray(False)),operand=None)
            accepted=accepted|wide_rescued
        retry=(~accepted)&avail
        return iteration+1,j.where(retry,converted,trial),restarts+(retry|rescued|phase_rescued|wide_rescued).astype(j.int32),avail&(~retry),~(accepted|retry),phase_active
    def condition(state):
        it,x,_,_,failed,phase_active=state;r,_=evaluate(x,phase_active)
        return (it<max_iterations)&(~failed)&(j.max(abs(r))>tolerance)
    it,x,restarts,_,failed,phase_active=lax.while_loop(condition,body,(j.asarray(0,j.int32),initial,j.asarray(0,j.int32),available,j.asarray(False),mass>0.))
    r,out=evaluate(x,phase_active);good=(~failed)&(j.max(abs(r))<=tolerance)&(j.max(abs(out[55:66]))<=tolerance*1.05)
    return out.at[67].set(it+1).at[68].set(good.astype(u.dtype)).at[291].set(restarts)


@lru_cache(maxsize=8)
def _compiled(max_iterations,tolerance):
    import jax
    return jax.jit(jax.vmap(partial(phase_energy_kernel_jax,max_iterations=max_iterations,tolerance=tolerance),in_axes=(*([0]*9),None,None,None)))


def run_canopy_phase_jax(*arrays,parameters=None,max_iterations=100,tolerance=None,raise_on_failure=True):
    import jax.numpy as j
    arrays=tuple(map(j.asarray,arrays));validate_phase(arrays,j);dtype=arrays[0].dtype
    tol=(.005 if dtype==j.float32 else 5e-5) if tolerance is None else tolerance
    if not isinstance(max_iterations,int) or max_iterations<1 or not np.isfinite(tol) or tol<=0.:raise ValueError('Invalid phase solver controls')
    out=_compiled(max_iterations,float(tol))(*arrays,j.asarray(pack_leaf_parameters(parameters,np.dtype(dtype))),*_jax_tables(str(dtype)))
    if not bool(j.all(j.isfinite(out))):raise RuntimeError('Nonfinite phase canopy output')
    result=unpack_phase(out)
    if raise_on_failure and not bool(j.all(result['converged'])):raise RuntimeError('Phase canopy did not converge; inspect with raise_on_failure=False')
    return result
