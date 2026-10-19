"""Device development reference for closed-column Richards liquid flow.

Newton and water-flux arithmetic run on the selected device. The host only
dispatches adaptive steps and disconnected hydraulic blocks, as in MOST.
This component is not connected to the production column or heat transport.
"""
from functools import lru_cache
import math


def _properties(u,p):
    import jax.numpy as j
    from jax import lax
    s=lax.optimization_barrier
    dz,por,k,psi,b,ref=p.T;maximum=s(s(1000.*dz)*por)
    safe=j.where(maximum>0.,maximum,1.);saturated=u>maximum
    water=j.minimum(u,maximum);dw=j.where(saturated,0.,1.)
    relative=water/safe
    limited=j.maximum(relative,s(.01*ref)/j.where(por>0.,por,1.))
    raw=s(psi*j.power(limited,-b));suction=j.minimum(raw,1e5)
    floor=(relative<limited)|(raw>=1e5);entry=j.minimum(psi,1e5)
    head=j.where(saturated,s(-entry+s(s(s(u-maximum)*b)*entry)/safe),-suction)
    dh=j.where(saturated,s(b*entry)/safe,j.where(floor,0.,s(b*raw)/j.where(water>0.,water,1.)))
    return water,dw,head,dh


def _system(u,old,p,dt):
    import jax.numpy as j
    from jax import lax
    s=lax.optimization_barrier
    dt=s(j.asarray(dt,u.dtype))
    water,dw,head,dh=_properties(u,p);dz,por,k,psi,b,ref=p.T
    def midpoint(total,thickness):
        next_=s(total+thickness)
        return next_,s(next_-s(.5*thickness))
    _,z=lax.scan(midpoint,j.asarray(0.,u.dtype),dz);distance=j.diff(z)
    theta=water/s(1000.*dz);dtheta=dw/s(1000.*dz)
    reference=s(ref[:-1]+ref[1:]);relative=s(theta[:-1]+theta[1:])/reference
    power=s(s(b[:-1]+b[1:])+3.);material=s(j.sqrt(k[:-1])*j.sqrt(k[1:]))
    face=s(material*j.power(relative,power))
    dk=s(s(material*power)*j.power(relative,s(power-1.)))/reference
    gradient=s(1.-j.diff(head)/distance);q=s(face*gradient)
    left=s(s(s(dk*dtheta[:-1])*gradient)+s(face*dh[:-1])/distance)
    right=s(s(s(dk*dtheta[1:])*gradient)-s(face*dh[1:])/distance)
    residual=s(s(water-old)+s(s(1000.*dt)*j.diff(j.concatenate((j.zeros(1,u.dtype),q,j.zeros(1,u.dtype))))))
    matrix=j.diag(dw)
    for index in range(len(u)-1):
        a=s(s(1000.*dt)*left[index]);b_=s(s(1000.*dt)*right[index])
        matrix=matrix.at[index,index].set(s(matrix[index,index]+a))
        matrix=matrix.at[index,index+1].set(b_)
        matrix=matrix.at[index+1,index].set(-a)
        matrix=matrix.at[index+1,index+1].set(s(matrix[index+1,index+1]-b_))
    return residual,q,matrix


def _gaussian(a,r):
    """Small partial-pivot solve, with the same order as the CUDA kernel."""
    import jax.numpy as j
    from jax import lax
    s=lax.optimization_barrier;n=len(r)
    def eliminate(k,carry):
        a,r,ok=carry
        pivot=j.argmax(j.where(j.arange(n)>=k,j.abs(a[:,k]),-1.))
        row=a[k];value=r[k]
        a=a.at[k].set(a[pivot]).at[pivot].set(row);r=r.at[k].set(r[pivot]).at[pivot].set(value)
        good=j.abs(a[k,k])>1e-30;den=j.where(good,a[k,k],1.)
        def row_step(i,state):
            a,r=state;factor=a[i,k]/den
            a=a.at[i].set(s(a[i]-s(factor*a[k])));r=r.at[i].set(s(r[i]-s(factor*r[k])))
            return a,r
        a,r=lax.fori_loop(k+1,n,row_step,(a,r))
        return a,r,ok&good
    a,r,ok=lax.fori_loop(0,n,eliminate,(a,r,j.asarray(True)))
    def back(k,x):
        i=n-1-k
        value=lax.fori_loop(i+1,n,lambda z,v:s(v-s(a[i,z]*x[z])),r[i])
        return x.at[i].set(value/j.where(j.abs(a[i,i])>1e-30,a[i,i],1.))
    x=lax.fori_loop(0,n,back,j.zeros_like(r))
    return x,ok&j.all(j.isfinite(x))


def implicit_step_jax(old,guess,p,dt):
    import jax.numpy as j
    from jax import lax
    s=lax.optimization_barrier
    dt=s(j.asarray(dt,old.dtype))
    capacity=s(s(1000.*p[:,0])*p[:,1])
    tolerance=j.maximum(j.asarray(1e-10,old.dtype),s(s(8.*j.finfo(old.dtype).eps)*j.maximum(1.,j.max(j.abs(old)))))
    def advance(carry):
        count,u,ok,done=carry;residual,q,a=_system(u,old,p,dt)
        norm=j.max(j.abs(residual));finished=norm<=tolerance
        direction,linear_ok=_gaussian(a,-residual)
        def trial(c):
            k,scale,chosen,found=c;candidate=s(u+s(scale*direction))
            allowed=j.all(j.isfinite(candidate))&j.all(candidate>=0.)
            allowed=allowed&~(j.all(candidate>=capacity)&(j.sum(old)<s(j.sum(capacity)-1e-8)))
            value,_,_=_system(candidate,old,p,dt)
            accept=allowed&linear_ok&(j.max(j.abs(value))<norm)
            return k+1,s(.5*scale),j.where(accept,candidate,chosen),found|accept
        _,_,chosen,found=lax.while_loop(lambda c:(c[0]<35)&~c[3],trial,
            (j.asarray(0),j.asarray(1.,old.dtype),u,j.asarray(False)))
        return count+1,j.where(finished,u,chosen),ok&(finished|found),done|finished
    count,u,ok,done=lax.while_loop(lambda c:(c[0]<45)&c[2]&~c[3],advance,
        (j.asarray(0),guess,j.asarray(True),j.asarray(False)))
    residual,q,_=_system(u,old,p,dt);transfer=s(s(1000.*dt)*q)
    water=s(s(old+j.concatenate((j.zeros(1,old.dtype),transfer)))-j.concatenate((transfer,j.zeros(1,old.dtype))))
    rounding=s(s(8.*j.finfo(old.dtype).eps)*j.maximum(capacity,1.))
    good=ok&done&j.all(water>=0.)&j.all(water<=capacity+rounding)&j.all(j.isfinite(water))
    return j.concatenate((water,u,transfer,j.stack((j.max(j.abs(residual)),count.astype(old.dtype),good.astype(old.dtype)))))


@lru_cache(maxsize=1)
def _compiled_step():
    import jax
    return jax.jit(implicit_step_jax)


def _set(a,where,value,xp):
    if xp.__name__.startswith('jax'):return a.at[where].set(value)
    a[where]=value
    return a


def _sum(a,xp):
    value=xp.asarray(0.,a.dtype)
    for k in range(len(a)):value=value+a[k]
    return value


def _head(u,p,xp):
    dz,por,k,psi,b,ref=p.T;maximum=1000.*dz*por;defined=por>0.
    # CuPy divide rounds the quotient; XLA float32 uses reciprocal/multiply.
    # Match the two device operations in host-dispatched coefficients too.
    relative=xp.minimum(u,maximum)*xp.reciprocal(xp.where(defined,maximum,1.))
    limited=xp.maximum(relative,(.01*ref)*xp.reciprocal(xp.where(defined,por,1.)))
    raw=psi*xp.power(limited,-b);entry=xp.minimum(psi,1e5)
    head=xp.where(u>maximum,-entry+((u-maximum)*b*entry)*xp.reciprocal(xp.where(defined,maximum,1.)),-xp.minimum(raw,1e5))
    return xp.where(defined,head,0.),defined&((relative<limited)|(raw>=1e5))


def _one(old,p,duration,xp,stepper,absolute,relative,budget):
    n=len(old);zero=xp.zeros(n-1,old.dtype);head,floor=_head(old,p,xp);defined=p[:,1]>0.
    result=dict(water_kg_m2=old.copy(),pressure_head_m=head,pressure_defined=defined,
        pressure_floor_active=floor,interface_transfer_kg_m2=zero.copy(),forward_transfer_kg_m2=zero.copy(),
        reverse_transfer_kg_m2=zero.copy(),accepted_substeps=xp.asarray(0,xp.int32),rejected_substeps=xp.asarray(0,xp.int32),
        maximum_nonlinear_residual_kg_m2=xp.asarray(0.,old.dtype),maximum_temporal_error_kg_m2=xp.asarray(0.,old.dtype))
    connected=(p[:,2]>0.)&defined
    if not bool(xp.all(connected)):
        begin=0
        while begin<n:
            if not bool(connected[begin]):begin+=1;continue
            end=begin+1
            while end<n and bool(connected[end]):end+=1
            part=_one(old[begin:end],p[begin:end],duration,xp,stepper,absolute,relative,budget)
            for key in ('water_kg_m2','pressure_head_m','pressure_floor_active'):result[key]=_set(result[key],slice(begin,end),part[key],xp)
            for key in ('interface_transfer_kg_m2','forward_transfer_kg_m2','reverse_transfer_kg_m2'):result[key]=_set(result[key],slice(begin,end-1),part[key],xp)
            for key in ('accepted_substeps','rejected_substeps'):result[key]=result[key]+part[key]
            for key in ('maximum_nonlinear_residual_kg_m2','maximum_temporal_error_kg_m2'):result[key]=xp.maximum(result[key],part[key])
            begin=end
    elif bool(xp.all(old==1000.*p[:,0]*p[:,1])):
        z=xp.cumsum(p[:,0])-.5*p[:,0]
        result['pressure_head_m']=z+xp.max(-p[:,3]-z);result['pressure_floor_active']=xp.zeros(n,bool)
    else:
        current=old.copy();coordinate=old.copy();elapsed=0.;step=min(duration,3600.);accepted=rejected=0
        total=zero.copy();compensation=zero.copy();forward=zero.copy();reverse=zero.copy()
        while elapsed<duration:
            if accepted+rejected>=budget:raise RuntimeError('Device Richards adaptive budget exhausted')
            step=min(step,duration-elapsed)
            coarse=stepper(current,coordinate,p,step)
            first=stepper(current,coordinate,p,.5*step)
            second=stepper(first[:n],first[n:2*n],p,.5*step)
            good=bool((coarse[-1]>0.)&(first[-1]>0.)&(second[-1]>0.))
            if not good:
                rejected+=1;step*=.5
                if step<1e-7:raise RuntimeError('Device Richards nonlinear substep underflow')
                continue
            fine=second[:n];error=xp.abs(fine-coarse[:n]);tendency=xp.maximum(xp.abs(fine-current),xp.abs(coarse[:n]-current))
            ratio=float(xp.max(error*xp.reciprocal(absolute+relative*tendency)))
            if ratio>1.:
                rejected+=1;step*=max(.2,.8/math.sqrt(ratio));continue
            for part in (first[2*n:3*n-1],second[2*n:3*n-1]):
                value=part-compensation;updated=total+value;compensation=(updated-total)-value;total=updated
                forward=forward+xp.maximum(part,0.);reverse=reverse+xp.maximum(-part,0.)
            current=fine;coordinate=second[n:2*n];elapsed+=step;accepted+=1
            result['maximum_nonlinear_residual_kg_m2']=xp.maximum(result['maximum_nonlinear_residual_kg_m2'],xp.maximum(first[-3],second[-3]))
            result['maximum_temporal_error_kg_m2']=xp.maximum(result['maximum_temporal_error_kg_m2'],xp.max(error))
            step*=min(2.,max(.5,.8/math.sqrt(max(ratio,1e-12))))
        head,floor=_head(coordinate,p,xp)
        result.update(water_kg_m2=old+xp.concatenate((xp.zeros(1,old.dtype),total))-xp.concatenate((total,xp.zeros(1,old.dtype))),
            pressure_head_m=head,pressure_floor_active=floor,interface_transfer_kg_m2=total,
            forward_transfer_kg_m2=forward,reverse_transfer_kg_m2=reverse,
            accepted_substeps=xp.asarray(accepted,xp.int32),rejected_substeps=xp.asarray(rejected,xp.int32))
    water=result['water_kg_m2'];mass=_sum(water,xp)-_sum(old,xp)
    bound=32.*xp.finfo(old.dtype).eps*xp.maximum(_sum(xp.abs(water),xp)+_sum(xp.abs(old),xp),1.)
    if not bool(xp.abs(mass)<=bound):raise RuntimeError('Device Richards extensive water budget failed')
    if bool(xp.any(water<0.)):raise RuntimeError('Device Richards generated negative liquid storage')
    result.update(mass_residual_kg_m2=mass,mass_bound_kg_m2=bound)
    return result


def _run(water,hydraulics,dt,xp,stepper,*,absolute_tolerance_kg_m2=.002,relative_tendency_tolerance=.01,max_substeps=4096):
    water=xp.asarray(water);p=xp.asarray(hydraulics);dt=xp.asarray(dt)
    if water.ndim!=2 or not 1<=water.shape[1]<=10 or p.shape!=(*water.shape,6) or dt.shape!=(len(water),):
        raise ValueError('Richards requires water(B,L), hydraulics(B,L,6), dt(B), with 1--10 layers')
    if water.dtype not in (xp.float32,xp.float64) or p.dtype!=water.dtype or dt.dtype!=water.dtype:
        raise ValueError('Richards requires matching float32/float64')
    if not all(bool(xp.all(xp.isfinite(v))) for v in (water,p,dt)):raise ValueError('Nonfinite Richards input')
    dz,por,k,psi,b,ref=[p[:,:,i] for i in range(6)]
    valid_ref=((ref>0.)&(ref<1.))|((ref==1.)&(por==0.))
    if (bool(xp.any((dz<=0.)|(por<0.)|(por>=1.)|(k<0.)|(psi<=0.)|(b<=0.)|~valid_ref|
            ((por==0.)&(k!=0.))|(water<0.)|(water>1000.*dz*por))) or bool(xp.any(dt<=0.))):
        raise ValueError('Invalid Richards water, pores or hydraulic coefficients')
    if (not math.isfinite(absolute_tolerance_kg_m2) or absolute_tolerance_kg_m2<=0.
            or not math.isfinite(relative_tendency_tolerance) or not 0.<relative_tendency_tolerance<=.1
            or type(max_substeps) is not int or max_substeps<1):raise ValueError('Invalid Richards controls')
    if not len(water):raise ValueError('Richards requires at least one column')
    columns=[_one(water[i],p[i],float(dt[i]),xp,stepper,absolute_tolerance_kg_m2,relative_tendency_tolerance,max_substeps) for i in range(len(water))]
    return {key:xp.stack([c[key] for c in columns]) for key in columns[0]}


def run_richards_jax(water,hydraulics,dt,**controls):
    import jax.numpy as j
    def stepper(old,guess,p,duration):
        # The kernel ABI must receive the physical precision. A Python
        # scalar is a weak float64 with x64 enabled; conversion inside a
        # fused loop can be reassociated with the 1000*dt coefficient.
        return _compiled_step()(old,guess,p,j.asarray(duration,old.dtype))
    return _run(water,hydraulics,dt,j,stepper,**controls)
