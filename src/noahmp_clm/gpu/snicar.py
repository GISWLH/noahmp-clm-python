"""Native JAX five-band snow optics and conservative layer fluxes.

UCAR tables are transferred as data. Interpolation, impurity mixing, delta
scaling, thin-slab exponential and adding/doubling all run on the device.
"""
from functools import lru_cache
import numpy as np


def optical_tables(dtype=np.float64):
    """Pack immutable source tables, without evaluating any column physics."""
    from ..physics.snicar import snow_tables
    source=snow_tables()
    radius=source['optics_radius_um']
    if not np.array_equal(radius,np.arange(30.,1501.)):
        raise ValueError('Snow optics requires the documented 30--1500 um, 1 um grid')
    keys=('ext_cff_mss','ss_alb','asm_prm')
    ice=np.array([[source[f'optics_{key}_ice_{mode}'] for key in keys]
                  for mode in ('drc','dfs')],dtype=dtype)
    impurities=np.array([[source[f'optics_{key}_{species}'] for key in keys]
        for species in ('bcphil','dust01','dust02','dust03','dust04')],dtype=dtype)
    return ice,impurities


def pack_snow_optics(pack,direct_flux,diffuse_flux,*,ground_albedo=(.15,.3),cosine_zenith=.5,dtype=np.float64):
    """Return five cells [mass, radius, BC, dust1..4], layer count and boundary.

    Boundary: visible/NIR direct, visible/NIR diffuse, two ground albedos, mu.
    Inactive cells have zero mass/tracers and an arbitrary valid radius.
    """
    cells=np.zeros((5,7),dtype=dtype);cells[:,1]=54.5
    n=len(pack.mass)
    if n>5: raise ValueError('Snow optics supports at most five layers')
    cells[:n,0]=pack.mass;cells[:n,1]=pack.radius_um;cells[:n,2:]=pack.impurities
    boundary=np.asarray([*direct_flux,*diffuse_flux,*ground_albedo,cosine_zenith],dtype)
    return cells,np.asarray(n,np.int32),boundary


def validate_snow_optics(cells,counts,boundary,xp):
    if cells.ndim!=3 or cells.shape[1:]!=(5,7) or counts.shape!=(len(cells),) or boundary.shape!=(len(cells),7):
        raise ValueError('Snow optics shapes must be (n,5,7), (n,), (n,7)')
    if cells.dtype not in (xp.float32,xp.float64) or boundary.dtype!=cells.dtype or counts.dtype.kind not in 'iu':
        raise ValueError('Snow optics requires matching floats and integer counts')
    if not all(bool(xp.all(xp.isfinite(x))) for x in (cells,boundary)):
        raise ValueError('Nonfinite snow optical input')
    active=xp.arange(5)[None,:]<counts[:,None]
    if bool(xp.any((counts<0)|(counts>5))) or bool(xp.any(cells<0)) or bool(xp.any(
        (active&((cells[:,:,0]<=0)|(cells[:,:,1]<=0)))|(~active&(xp.sum(cells[:,:,2:],axis=2)!=0))|(~active&(cells[:,:,0]!=0)))):
        raise ValueError('Invalid snow layer mass, tracer or padding')
    if bool(xp.any((boundary[:,:4]<0))) or bool(xp.any((boundary[:,4:6]<0)|(boundary[:,4:6]>1))) or bool(xp.any((boundary[:,6]<=0)|(boundary[:,6]>1))):
        raise ValueError('Invalid snow radiation boundary')


def slab_coefficients_jax(tau,omega,g,mu,eddington):
    """One passive scattering slab; no host or NumPy physics is called."""
    import jax.numpy as j
    from jax import lax
    s=lax.optimization_barrier
    divisor=s(1.-s(s(omega*g)*g));tau=s(tau*divisor)
    loss=s(1.-omega)/divisor;w=s(1.-loss);g=g/s(1.+g)
    a=j.where(eddington,s(7.-s(w*s(4.+s(3.*g))))/4.,s(2.-s(w*s(1.+g))))
    b=j.where(eddington,s(s(w*s(4.-s(3.*g)))-1.)/4.,s(w*s(1.-g)))
    beta=j.where(eddington,s(2.-s(s(3.*g)*mu))/4.,s(1.-s(g*mu))/2.)
    matrix=j.array([[-a,b,s(w*s(1.-beta))/mu],[-b,a,-s(w*beta)/mu],[0.,0.,-1./mu]],dtype=tau.dtype)
    norm=j.max(s(s(j.abs(matrix[:,0])+j.abs(matrix[:,1]))+j.abs(matrix[:,2])))
    magnitude=s(tau*norm)/.1
    fraction,exponent=j.frexp(j.maximum(magnitude,1.))
    count=exponent-j.where(fraction==.5,1,0)
    width=j.ldexp(tau,-count);small=s(matrix*width)
    def multiply(left,right):
        first=s(left[:,0,None]*right[None,0,:]);second=s(left[:,1,None]*right[None,1,:])
        third=s(left[:,2,None]*right[None,2,:])
        return s(s(first+second)+third)
    # Integrate the absorption equation alongside the three fluxes:
    # A' = 2(1-w)(D+U) + (1-w)B/mu. This retains weak absorption without
    # subtracting two O(1) scattering coefficients at the initial thin slab.
    # Reuse the already rounded direct-beam matrix entry. A separately
    # compiled (loss/mu)*width can choose a different reciprocal ordering.
    source=j.stack((s(s(2.*loss)*width),s(s(2.*loss)*width),-s(loss*small[2,2])))
    def taylor(order,carry):
        term,value,abs_term,integral=carry
        term=s(multiply(term,small)/order.astype(tau.dtype))
        prod=s(s(s(abs_term[0]*small[0])+s(abs_term[1]*small[1]))+s(abs_term[2]*small[2]))
        abs_term=j.where(order==1,source,s(prod/order.astype(tau.dtype)))
        return term,s(value+term),abs_term,s(integral+abs_term)
    identity=j.eye(3,dtype=tau.dtype)
    _,transfer,_,integral=lax.fori_loop(1,13,taylor,(identity,identity,j.zeros(3,tau.dtype),j.zeros(3,tau.dtype)))
    r=-transfer[1,0]/transfer[1,1];t=1./transfer[1,1]
    up=-transfer[1,2]/transfer[1,1];down=s(transfer[0,2]+s(transfer[0,1]*up))
    direct=j.exp(-width/mu)
    ad=s(integral[0]+s(integral[1]*r));ab=s(integral[2]+s(integral[1]*up))
    t=s(s(1.-r)-ad);down=s(s(s(1.-direct)-up)-ab)
    def double(_,values):
        r,t,u,d,beam,ad,ab=values;den=s(1.-s(r*r));tt=s(t*t)
        nr=s(r+s(tt*r)/den)
        nu=s(u+s(t*s(s(r*d)+s(u*beam)))/den)
        nb=s(beam*beam)
        middle=s(d+s(s(r*u)*beam))/den
        nad=s(ad+s(ad*s(s(1.+r)*t))/den)
        nab=s(s(ab*s(1.+beam))+s(ad*s(s(s(1.+r)*middle)+s(u*beam))))
        return nr,s(s(1.-nr)-nad),nu,s(s(s(1.-nb)-nu)-nab),nb,nad,nab
    result=lax.fori_loop(0,count,double,(r,t,up,down,direct,ad,ab))
    return j.stack(result[:5])


def spectrum_jax(optics,boundary,eddington):
    """Five cells by (tau, omega, g); beam/diffuse/albedo/mu boundary.

    Return absorbed[5], beam/down/up[6 each], reflected/ground/residual.
    Zero-tau padding is transparent; padded interface fluxes repeat the last
    physical interface, so the fixed bottom index is valid for empty snow.
    """
    import jax
    import jax.numpy as j
    from jax import lax
    s=lax.optimization_barrier
    beam0,diffuse0,albedo,mu=boundary
    r,t,u,d,bt=jax.vmap(lambda v:slab_coefficients_jax(*v,mu,eddington))(optics).T
    beam=j.zeros(6,optics.dtype).at[0].set(beam0)
    beam=lax.fori_loop(0,5,lambda i,x:x.at[i+1].set(s(x[i]*bt[i])),beam)
    su,sd=s(u*beam[:-1]),s(d*beam[:-1])
    re=j.zeros_like(beam).at[5].set(albedo);so=j.zeros_like(beam).at[5].set(s(albedo*beam[5]))
    def eliminate(idx,carry):
        re,so=carry;i=4-idx;den=s(1.-s(r[i]*re[i+1]))
        nr=s(r[i]+s(s(t[i]*t[i])*re[i+1])/den)
        ns=s(su[i]+s(t[i]*s(so[i+1]+s(re[i+1]*sd[i])))/den)
        return re.at[i].set(nr),so.at[i].set(ns)
    re,so=lax.fori_loop(0,5,eliminate,(re,so))
    down=j.zeros_like(beam).at[0].set(diffuse0)
    def forward(i,x):
        value=s(s(s(t[i]*x[i])+s(r[i]*so[i+1]))+sd[i])/s(1.-s(r[i]*re[i+1]))
        return x.at[i+1].set(value)
    down=lax.fori_loop(0,5,forward,down);up=s(s(re*down)+so)
    absorbed=s(s(s(s(s(beam[:-1]+down[:-1])+up[1:])-beam[1:])-down[1:])-up[:-1])
    absorbed=j.where(optics[:,0]==0.,0.,absorbed)
    ground=s(s(1.-albedo)*s(beam[5]+down[5]))
    total=lax.fori_loop(0,5,lambda i,x:s(x+absorbed[i]),j.zeros_like(ground))
    residual=s(s(s(s(beam0+diffuse0)-up[0])-total)-ground)
    return j.concatenate((absorbed,beam,down,up,j.stack((up[0],ground,residual))))


def snow_shortwave_kernel_jax(cells,count,boundary,ice_table,impurity_table):
    import jax
    import jax.numpy as j
    from jax import lax
    s=lax.optimization_barrier
    active=j.arange(5)<count;mass=cells[:,0];radius=j.clip(cells[:,1],30.,1500.)
    lower=j.minimum(j.floor(radius-30.).astype(j.int32),1469);weight=s(radius-s(lower.astype(cells.dtype)+30.))
    def band(mode,band):
        table=ice_table[mode,:,band,:]
        lookup=s(table[:,lower]+s(weight[None,:]*s(table[:,lower+1]-table[:,lower])))
        tau=s(mass*lookup[0]);scatter=s(tau*lookup[1]);moment=s(scatter*lookup[2])
        def mix(species,values):
            tau,scatter,moment=values;ext,alb,g=impurity_table[species,:,band]
            impurity=s(cells[:,species+2]*ext);scattering=s(impurity*alb)
            return s(tau+impurity),s(scatter+scattering),s(moment+s(scattering*g))
        tau,scatter,moment=lax.fori_loop(0,5,mix,(tau,scatter,moment))
        optical=j.stack((tau,j.where(tau>0.,scatter/j.maximum(tau,1e-30),0.),j.where(scatter>0.,moment/j.maximum(scatter,1e-30),0.)),axis=1)
        weights=j.array([[1.,.494,.181,.121,.204],[1.,.586,.202,.109,.103]],cells.dtype)
        flux=s(boundary[2*mode+j.where(band==0,0,1)]*weights[mode,band])
        bc=j.stack((j.where(mode==0,flux,0.),j.where(mode==1,flux,0.),boundary[4+j.where(band==0,0,1)],boundary[6]))
        return spectrum_jax(optical,bc,band==0),optical.T
    conditions,optics=jax.vmap(lambda mode:jax.vmap(lambda b:band(mode,b))(j.arange(5)))(j.arange(2))
    return assemble_snow_output_jax(conditions,optics,active&(cells[:,1]!=radius),boundary)


def assemble_snow_output_jax(conditions,optics,clipped,boundary):
    """Collect mode/band solutions without changing their flux budgets."""
    import jax
    import jax.numpy as j
    from jax import lax
    s=lax.optimization_barrier
    spectral=s(conditions[0,:,:5]+conditions[1,:,:5]).T
    ground=s(conditions[0,:,24]+conditions[1,:,24]);reflected=s(conditions[0,:,23]+conditions[1,:,23])
    def sum5(v):return lax.fori_loop(0,5,lambda i,x:s(x+v[i]),j.zeros_like(v[0]))
    absorbed=jax.vmap(sum5)(spectral);tg=sum5(ground);tr=sum5(reflected)
    incoming=s(s(s(boundary[0]+boundary[1])+boundary[2])+boundary[3])
    albedo=j.where(incoming>0.,tr/j.maximum(incoming,1e-30),0.)
    residual=s(s(s(incoming-tr)-sum5(absorbed))-tg)
    return j.concatenate((absorbed,spectral.ravel(),ground,reflected,
        j.stack((tg,tr,albedo,residual,(incoming>0.).astype(boundary.dtype))),clipped.astype(boundary.dtype),conditions.ravel(),optics.ravel()))


def unpack_snow_shortwave(out):
    return dict(absorbed=out[:,:5],spectral_absorbed=out[:,5:30].reshape(-1,5,5),
        spectral_ground_absorbed=out[:,30:35],spectral_reflected=out[:,35:40],
        ground_absorbed=out[:,40],reflected=out[:,41],albedo=out[:,42],residual=out[:,43],
        albedo_defined=out[:,44]>0,radius_clipped=out[:,45:50]>0,
        conditions=out[:,50:310].reshape(-1,2,5,26),optics=out[:,310:].reshape(-1,2,5,3,5))


@lru_cache(maxsize=1)
def _compiled():
    import jax
    return jax.jit(jax.vmap(snow_shortwave_kernel_jax,in_axes=(0,0,0,None,None)))


@lru_cache(maxsize=2)
def _jax_tables(dtype):
    import jax.numpy as j
    return tuple(j.asarray(x) for x in optical_tables(np.dtype(dtype)))


def run_snow_shortwave_jax(cells,counts,boundary):
    import jax.numpy as j
    c,n,b=map(j.asarray,(cells,counts,boundary));validate_snow_optics(c,n,b,j)
    result=_compiled()(c,n,b,*_jax_tables(str(c.dtype)))
    if not bool(j.all(j.isfinite(result))):raise RuntimeError('Nonfinite snow optical solve')
    return unpack_snow_shortwave(result)
