"""Five-band, delta-scaled SNICAR approximation using public UCAR Mie tables.

CLM5 technical note 2.3.62--68 defines optical mixing and delta scaling. Visible
uses Eddington and NIR hemispheric-mean closure (Toon et al., 1989). The original
solver here uses thin-slab matrix exponentials followed by adding/doubling;
it does not copy CTSM code or perturb the sun angle near analytic singularities.
BC is represented as one hydrophilic class; four dust classes are explicit.
"""
from functools import lru_cache
from importlib.resources import files
import numpy as np

from .canopy_radiation import _solve_layers

NIR_DIRECT=np.array([.494,.181,.121,.204])
NIR_DIFFUSE=np.array([.586,.202,.109,.103])


@lru_cache(maxsize=1)
def snow_tables():
    with files('noahmp_clm').joinpath('data/snow_tables.npz').open('rb') as stream:
        with np.load(stream,allow_pickle=False) as data:
            values={k:data[k].copy() for k in data.files}
    for v in values.values(): v.setflags(write=False)
    return values


def bulk_optics(pack, *, diffuse=False):
    """Return unscaled tau, omega, g with shape (layers,5).

    Liquid droplets use the surrounding ice optical properties, a stated
    approximation valid for small liquid contents, not ponded water. Radii
    are constrained to the lookup range; the caller gets a clipping flag.
    """
    table=snow_tables();mode='dfs' if diffuse else 'drc'
    radii=table['optics_radius_um'];r=np.clip(pack.radius_um,radii[0],radii[-1])
    props=[]
    for key in ('ext_cff_mss','ss_alb','asm_prm'):
        props.append(np.stack([np.interp(r,radii,row)
            for row in table[f'optics_{key}_ice_{mode}']],axis=1))
    extinction,omega,g=props
    tau=pack.mass[:,None]*extinction
    scatter=tau*omega;moment=scatter*g
    for j,species in enumerate(('bcphil','dust01','dust02','dust03','dust04')):
        impurity=pack.impurities[:,j,None]*table[f'optics_ext_cff_mss_{species}']
        scattering=impurity*table[f'optics_ss_alb_{species}']
        tau+=impurity;scatter+=scattering
        moment+=scattering*table[f'optics_asm_prm_{species}']
    return dict(tau=tau,omega=np.divide(scatter,tau,out=np.zeros_like(tau),where=tau>0),
        g=np.divide(moment,scatter,out=np.zeros_like(tau),where=scatter>0),
        radius_clipped=pack.radius_um!=r)


def delta_scale(tau,omega,g):
    divisor=1.-omega*g*g
    return tau*divisor,omega*(1.-g*g)/divisor,g/(1.+g)


def layer_coefficients(tau,omega,g,cosine_zenith,*,eddington=False):
    """Stable slab scattering coefficients, including direct-beam sources.

    The local exponential has norm <= 0.1 and a twelve-term Taylor expansion.
    Doubling acts on passive scattering coefficients, never the exponentially
    growing transfer matrix of an optically thick snow layer.
    """
    tau,w,g=np.broadcast_arrays(np.asarray(tau,float),omega,g)
    if (tau.ndim!=1 or not all(np.all(np.isfinite(v)) for v in (tau,w,g))
            or np.any(tau<0) or np.any((w<0)|(w>1)) or np.any((g<0)|(g>=1))
            or not 0<cosine_zenith<=1):
        raise ValueError('Invalid snow optical properties or sun angle')
    mu=cosine_zenith
    tau,w,g=delta_scale(tau,w,g)
    if eddington:
        a=(7.-w*(4.+3.*g))/4.;b=(w*(4.-3.*g)-1.)/4.
        beta=(2.-3.*g*mu)/4.
    else:
        a=2.-w*(1.+g);b=w*(1.-g);beta=(1.-g*mu)/2.
    # A single thin transfer matrix is safe even when the characteristic
    # diffuse eigenvalue equals direct-beam attenuation (the Toon resonance).
    result=np.empty((5,len(tau)))
    for j in range(len(tau)):
        matrix=np.array([[-a[j],b[j],w[j]*(1.-beta[j])/mu],
            [-b[j],a[j],-w[j]*beta[j]/mu],[0.,0.,-1./mu]])
        magnitude=tau[j]*np.max(np.sum(np.abs(matrix),axis=1))
        count=max(0,int(np.ceil(np.log2(max(magnitude/.1,1.)))))
        width=tau[j]/2.**count
        term=np.eye(3);transfer=term.copy()
        for order in range(1,13):
            term=(term@(matrix*width))/order;transfer+=term
        r=-transfer[1,0]/transfer[1,1];t=1./transfer[1,1]
        up=-transfer[1,2]/transfer[1,1]
        down=transfer[0,2]+transfer[0,1]*up
        direct=np.exp(-width/mu)
        for _ in range(count):
            den=1.-r*r
            r,t,up,down,direct=(r+t*t*r/den,t*t/den,
                up+t*(r*down+up*direct)/den,
                down*direct+t*(down+r*up*direct)/den,direct*direct)
            if w[j]==1.:
                t=1.-r;down=1.-direct-up
        result[:,j]=r,t,up,down,direct
    return tuple(result)


def spectral_fluxes(tau,omega,g,direct_flux,diffuse_flux,ground_albedo,
                    cosine_zenith,*,eddington=False):
    """One-band interface flux solution; coefficients may use either closure."""
    if not (0<=ground_albedo<=1 and direct_flux>=0 and diffuse_flux>=0
            and np.all(np.isfinite([direct_flux,diffuse_flux]))):
        raise ValueError('Invalid snow radiation boundary')
    r,t,up,down,beam_t=layer_coefficients(tau,omega,g,cosine_zenith,eddington=eddington)
    direct=np.r_[direct_flux,direct_flux*np.cumprod(beam_t)]
    df,uf=_solve_layers(r,t,up*direct[:-1],down*direct[:-1],diffuse_flux,
        ground_albedo,ground_albedo*direct[-1])
    absorbed=direct[:-1]+df[:-1]+uf[1:]-direct[1:]-df[1:]-uf[:-1]
    ground=(1.-ground_albedo)*(direct[-1]+df[-1])
    return dict(absorbed=absorbed,ground_absorbed=float(ground),reflected=float(uf[0]),
        direct=direct,diffuse_down=df,diffuse_up=uf,
        residual=float(direct_flux+diffuse_flux-uf[0]-absorbed.sum()-ground))


def snow_shortwave(pack,direct_flux,diffuse_flux,*,ground_albedo=(.15,.3),
                   cosine_zenith=.5):
    """Five-band reflected and absorbed W/m2 for visible/NIR boundary fluxes.

    ``direct_flux`` and ``diffuse_flux`` each contain [visible, NIR] W/m2.
    Direct and diffuse use their own UCAR optics and fixed within-NIR weights.
    This permits canopy-to-snow spectral flux coupling without guessing a new
    visible fraction below the canopy. An empty pack returns bare-ground flux.
    """
    direct_flux=np.asarray(direct_flux,float);diffuse_flux=np.asarray(diffuse_flux,float)
    albedo=np.asarray(ground_albedo,float)
    if (direct_flux.shape!=(2,) or diffuse_flux.shape!=(2,) or albedo.shape!=(2,)
            or not all(np.all(np.isfinite(x)) for x in (direct_flux,diffuse_flux,albedo))
            or np.any(direct_flux<0) or np.any(diffuse_flux<0) or np.any((albedo<0)|(albedo>1))
            or not 0<cosine_zenith<=1):
        raise ValueError('Snow radiation requires valid two-band direct/diffuse fluxes')
    absorbed=np.zeros((pack.mass.size,5));ground=np.zeros(5);reflection=np.zeros(5)
    conditions=[]
    for diffuse,flux,weights in ((False,direct_flux,NIR_DIRECT),(True,diffuse_flux,NIR_DIFFUSE)):
        optics=bulk_optics(pack,diffuse=diffuse)
        spectral=np.r_[flux[0],flux[1]*weights]
        runs=[]
        for band in range(5):
            run=spectral_fluxes(optics['tau'][:,band],optics['omega'][:,band],optics['g'][:,band],
                0. if diffuse else spectral[band],spectral[band] if diffuse else 0.,
                albedo[0 if band==0 else 1],cosine_zenith,eddington=band==0)
            absorbed[:,band]+=run['absorbed'];ground[band]+=run['ground_absorbed']
            reflection[band]+=run['reflected'];runs.append(run)
        conditions.append(runs)
    incident=float(direct_flux.sum()+diffuse_flux.sum())
    return dict(absorbed=absorbed.sum(axis=1),spectral_absorbed=absorbed,
        ground_absorbed=float(ground.sum()),spectral_ground_absorbed=ground,
        reflected=float(reflection.sum()),spectral_reflected=reflection,
        albedo=float(reflection.sum()/incident) if incident>0 else None,
        residual=float(incident-reflection.sum()-absorbed.sum()-ground.sum()),
        conditions=conditions)
