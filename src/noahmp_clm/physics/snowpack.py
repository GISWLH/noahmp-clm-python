"""Five-layer snow mass/enthalpy state and conservative vertical remapping.

Original implementation based on CLM5 technical note 2.6 and 2.8. Enthalpy is
relative to ice at freezing: H = (mi*ci+mw*cw)*(T-Tf)+Lf*mw, J/m2. Layer masses
are kg/m2, thickness is m; impurities are kg/m2 (BC plus four dust classes).
This state is independent of the released three-layer snow approximation.
"""
from dataclasses import dataclass
import numpy as np

TF=273.15
CI=2100.
CW=4180.
LF=334000.
RHO_ICE=917.
RHO_WATER=1000.
MAX_LAYERS=5


def equilibrium(mass,enthalpy):
    """Diagnose equilibrium temperature, ice and water from extensive state.

    The melting interval has exactly constant temperature; no artificial heat
    capacity spike or temperature truncation is used. Empty cells have T=Tf.
    """
    m,h=np.broadcast_arrays(np.asarray(mass,float),np.asarray(enthalpy,float))
    liquid=np.clip(h/LF,0.,m)
    positive=np.maximum(m,1e-300)
    offset=np.where(h<0.,h/(CI*positive),np.where(h>m*LF,(h-m*LF)/(CW*positive),0.))
    return TF+np.where(m>0.,offset,0.),m-liquid,liquid


def target_thickness(depth):
    """CLM5 2.6 depth-based grid, limited to the requested five layers.

    A positive pack thinner than 1 cm remains one explicit layer to preserve
    its heat and mass; it is never discarded into an untracked thin-snow store.
    """
    if not np.isfinite(depth) or depth<0: raise ValueError('Invalid snow depth')
    if depth==0.: return np.empty(0)
    if depth<=.03: return np.array([depth])
    if depth<=.04: return np.full(2,depth/2.)
    if depth<=.07: return np.array([.02,depth-.02])
    if depth<=.12: return np.array([.02,(depth-.02)/2.,(depth-.02)/2.])
    if depth<=.18: return np.array([.02,.05,depth-.07])
    if depth<=.29: return np.array([.02,.05,(depth-.07)/2.,(depth-.07)/2.])
    if depth<=.41: return np.array([.02,.05,.11,depth-.18])
    if depth<=.64: return np.array([.02,.05,.11,(depth-.18)/2.,(depth-.18)/2.])
    return np.array([.02,.05,.11,.23,depth-.41])


@dataclass(frozen=True)
class Snowpack:
    mass: np.ndarray
    enthalpy: np.ndarray
    thickness: np.ndarray
    radius_um: np.ndarray | None = None
    impurities: np.ndarray | None = None

    def __post_init__(self):
        m=np.array(self.mass,float,copy=True)
        h=np.array(self.enthalpy,float,copy=True)
        z=np.array(self.thickness,float,copy=True)
        n=m.size
        r=np.full(n,54.5) if self.radius_um is None else np.array(self.radius_um,float,copy=True)
        a=np.zeros((n,5)) if self.impurities is None else np.array(self.impurities,float,copy=True)
        if m.ndim!=1 or n>MAX_LAYERS or h.shape!=(n,) or z.shape!=(n,) or r.shape!=(n,) or a.shape!=(n,5):
            raise ValueError('Snowpack requires zero to five matching layers and five impurity species')
        if (not all(np.all(np.isfinite(x)) for x in (m,h,z,r,a)) or np.any(m<=0.)
                or np.any(z<=0.) or np.any(r<=0.) or np.any(a<0.)):
            raise ValueError('Invalid snow mass, thickness, enthalpy, radius or impurity mass')
        for name,value in (('mass',m),('enthalpy',h),('thickness',z),('radius_um',r),('impurities',a)):
            value.setflags(write=False)
            object.__setattr__(self,name,value)

    @classmethod
    def empty(cls): return cls(np.empty(0),np.empty(0),np.empty(0))

    @classmethod
    def from_ice(cls,mass,temperature,density=300.,radius_um=54.5):
        m,t,rho,r=np.broadcast_arrays(np.atleast_1d(mass).astype(float),temperature,density,radius_um)
        if np.any(t>TF) or np.any(t<150.) or np.any(rho<30.) or np.any(rho>RHO_ICE):
            raise ValueError('Invalid initial ice temperature or density')
        return cls(m,m*CI*(t-TF),m/rho,r)

    @property
    def temperature(self): return equilibrium(self.mass,self.enthalpy)[0]

    @property
    def ice(self): return equilibrium(self.mass,self.enthalpy)[1]

    @property
    def liquid(self): return equilibrium(self.mass,self.enthalpy)[2]

    @property
    def density(self): return self.mass/self.thickness

    @property
    def swe(self): return float(self.mass.sum()/RHO_WATER)

    def with_enthalpy(self,h):
        return Snowpack(self.mass,h,self.thickness,self.radius_um,self.impurities)


def remap_layers(mass,enthalpy,thickness,radius_um,impurities,new_thickness):
    """Integrate piecewise constant extensive contents over overlapping cells.

    Input may contain more than five layers after a snowfall insertion; output
    is a validated five-layer state. Mass, enthalpy, volume and impurities are
    conserved. Radius mixing conserves the mass-weighted inverse-radius proxy
    for optical surface area, not an arithmetic mean of density or radius.
    """
    m,h,z,r=[np.asarray(v,float) for v in (mass,enthalpy,thickness,radius_um)]
    a,nz=np.asarray(impurities,float),np.asarray(new_thickness,float)
    if (m.ndim!=1 or any(v.shape!=m.shape for v in (h,z,r)) or a.shape!=(m.size,5)
            or nz.ndim!=1 or nz.size>5 or np.any(m<=0.) or np.any(z<=0.) or np.any(nz<=0.)
            or not all(np.all(np.isfinite(v)) for v in (m,h,z,r,a,nz)) or np.any(r<=0.) or np.any(a<0.)
            or not np.isclose(z.sum(),nz.sum(),rtol=1e-12,atol=1e-14)):
        raise ValueError('Invalid snow remapping geometry or contents')
    if not m.size: return Snowpack.empty()
    old_edges=np.r_[0.,np.cumsum(z)]
    new_edges=np.r_[0.,np.cumsum(nz)]
    overlap=np.maximum(0.,np.minimum(new_edges[1:,None],old_edges[None,1:])-
        np.maximum(new_edges[:-1,None],old_edges[None,:-1]))
    weights=overlap/z[None,:]
    nm=weights@m
    nr=nm/(weights@(m/r))
    return Snowpack(nm,weights@h,nz,nr,weights@a)


def regrid(pack):
    return remap_layers(pack.mass,pack.enthalpy,pack.thickness,pack.radius_um,pack.impurities,
        target_thickness(float(pack.thickness.sum())))


def add_snowfall(pack,amount_kgm2,temperature,density=100.,radius_um=54.5,impurities=None):
    """Insert cold new snow and conservatively remap without mixing the whole pack."""
    if not np.isfinite(amount_kgm2) or amount_kgm2<0: raise ValueError('Invalid snowfall mass')
    if amount_kgm2==0: return pack
    fresh=Snowpack.from_ice([amount_kgm2],temperature,density,radius_um)
    tracer=np.zeros((1,5)) if impurities is None else np.asarray(impurities,float).reshape(1,5)
    m=np.r_[fresh.mass,pack.mass];h=np.r_[fresh.enthalpy,pack.enthalpy]
    z=np.r_[fresh.thickness,pack.thickness];r=np.r_[fresh.radius_um,pack.radius_um]
    a=np.concatenate((tracer,pack.impurities))
    return remap_layers(m,h,z,r,a,target_thickness(float(z.sum())))


def add_rain(pack,amount_kgm2,temperature):
    """Mix rain into the first layer, retaining its liquid enthalpy.

    A caller with an empty pack must route rain directly to soil instead.
    Cold rain carries supercooled liquid enthalpy and can freeze on mixing.
    """
    if not pack.mass.size: raise ValueError('Rain on an empty pack belongs to soil hydrology')
    if not np.isfinite(amount_kgm2) or amount_kgm2<0 or not 200.<=temperature<=350.:
        raise ValueError('Invalid rain mass or temperature')
    m,h=pack.mass.copy(),pack.enthalpy.copy()
    m[0]+=amount_kgm2;h[0]+=amount_kgm2*(LF+CW*(temperature-TF))
    return Snowpack(m,h,pack.thickness,pack.radius_um,pack.impurities)


def drain_liquid(pack,*,retention=.033,impermeable_porosity=.05,scavenging=(.2,.02,.02,.01,.01),
                 return_ledger=False):
    """Top-down liquid retention/drainage, carrying water enthalpy and impurities.

    CLM-style irreducible saturation and ice-pore blockage. Downward transfer
    is limited by the receiving cell's pore volume INCLUDING refreezing. Water
    can perch above a blocked cell. Overflow that cannot fit within pore space
    leaves as local runoff, not as fictitious flow through impermeable ice.
    A fully melted cell has no snow skeleton; its untransferred water runs off.
    Returns (new_pack, water_out_kgm2, enthalpy_out_jm2, impurity_out_kgm2).
    The four-value return combines bottom drainage and local runoff. A column
    driver must use ``return_ledger=True`` to get (new_pack, ledger) with both
    paths separated, before sending drainage to soil and runoff out of column.
    Enthalpy of warm meltwater includes all post-melt excess energy. This is a
    bucket representation of perched overflow, not lateral snow hydrodynamics.
    """
    if not 0<=retention<1 or not 0<=impermeable_porosity<1:
        raise ValueError('Invalid snow retention or porosity threshold')
    scav=np.asarray(scavenging,float)
    if scav.shape!=(5,) or not np.all(np.isfinite(scav)) or np.any((scav<0)|(scav>1)):
        raise ValueError('Invalid impurity scavenging')
    m,h,z,a=pack.mass.copy(),pack.enthalpy.copy(),pack.thickness.copy(),pack.impurities.copy()
    water,heat,tracer=0.,0.,np.zeros(5)
    runoff,runoff_heat,runoff_tracer=0.,0.,np.zeros(5)
    for j in range(m.size):
        m[j]+=water;h[j]+=heat;a[j]+=tracer
        t,ice,liq=equilibrium(m[j],h[j])
        # Freezing expansion can enlarge a saturated layer; ice cannot have
        # a volume smaller than its actual mass / ice density.
        z[j]=max(z[j],float(ice)/RHO_ICE)
        pore=max(0.,1.-float(ice)/(RHO_ICE*z[j]))
        cap=retention*pore*z[j]*RHO_WATER
        potential=(float(liq) if ice==0. else max(0.,float(liq)-cap) if pore>=impermeable_porosity else 0.)
        specific=LF+CW*(float(t)-TF)
        water=potential
        if j+1<m.size and potential>0.:
            _,next_ice,_=equilibrium(m[j+1],h[j+1])
            if 1.-next_ice/(RHO_ICE*z[j+1])<impermeable_porosity:
                water=0.
            else:
                # Analytic equilibrium at each trial includes latent freezing
                # in the receiver. This avoids accepting 1000 kg/m3 water
                # volume and subsequently overfilling it with 917 kg/m3 ice.
                def volume(incoming):
                    _,ni,nl=equilibrium(m[j+1]+incoming,h[j+1]+incoming*specific)
                    return float(ni/RHO_ICE+nl/RHO_WATER)
                if volume(potential)>z[j+1]:
                    low,high=0.,potential
                    for _ in range(48):
                        mid=(low+high)*.5
                        if volume(mid)<=z[j+1]: low=mid
                        else: high=mid
                    water=low
        overflow=(float(liq)-water if ice==0. else
            max(0.,float(liq)-water-pore*z[j]*RHO_WATER))
        total=water+overflow
        heat=water*specific
        fraction=np.minimum(1.,scav*total/m[j])
        exported=a[j].copy() if total==m[j] else a[j]*fraction
        tracer=exported*(water/total) if total>0. else np.zeros(5)
        runoff+=overflow;runoff_heat+=overflow*specific
        runoff_tracer+=exported-tracer
        m[j]-=total;h[j]-=total*specific;a[j]-=exported
        if m[j]==0.: h[j]=0.
    keep=m>0.
    new=(regrid(Snowpack(m[keep],h[keep],z[keep],pack.radius_um[keep],a[keep]))
        if np.any(keep) else Snowpack.empty())
    if return_ledger:
        return new,dict(drainage_kgm2=float(water),drainage_enthalpy_jm2=float(heat),
            drainage_impurities_kgm2=tracer,runoff_kgm2=float(runoff),
            runoff_enthalpy_jm2=float(runoff_heat),runoff_impurities_kgm2=runoff_tracer)
    return new,float(water+runoff),float(heat+runoff_heat),tracer+runoff_tracer


def remove_sublimation(pack,amount_kgm2):
    """Remove top-down ice mass and its sensible enthalpy; retain aerosol mass.

    Surface latent heat (Ls, including fusion) is handled by the energy solver;
    this advective energy is a separate term in the total enthalpy budget.
    A caller must cap demand to ice supply before computing its latent flux.
    """
    if not np.isfinite(amount_kgm2) or amount_kgm2<0 or amount_kgm2>pack.ice.sum()+1e-12:
        raise ValueError('Sublimation exceeds snow ice supply')
    m,h,z,a=pack.mass.copy(),pack.enthalpy.copy(),pack.thickness.copy(),pack.impurities.copy()
    remaining=amount_kgm2;export=0.;residue=np.zeros(5)
    for j in range(m.size):
        t,ice,_=equilibrium(m[j],h[j])
        loss=min(remaining,float(ice));energy=loss*CI*(float(t)-TF)
        z[j]*=(m[j]-loss)/m[j]
        m[j]-=loss;h[j]-=energy;remaining-=loss;export+=energy
        a[j]+=residue;residue=np.zeros(5)
        if m[j]==0.: h[j]=0.;residue=a[j].copy();a[j]=0.
    keep=m>0.
    new=(regrid(Snowpack(m[keep],h[keep],z[keep],pack.radius_um[keep],a[keep]))
        if np.any(keep) else Snowpack.empty())
    return new,float(export),residue
