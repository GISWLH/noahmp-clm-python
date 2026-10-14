"""Extensive CLM development state, separate from the released 70-float ABI."""
from dataclasses import dataclass,field,replace
from copy import deepcopy
import numpy as np
from .column import ColumnState
from .options import PhysicsOptions
from .physics.snowpack import Snowpack,TF,LF,CW
from .physics.soil_enthalpy import SoilEnthalpyState


def clm_options():
    return PhysicsOptions(OptSnowAlbedo=3,OptSoilTemperatureBottom=1)


@dataclass(frozen=True)
class CLMColumnState:
    """Independent snow, soil and intercepted-liquid mass/enthalpy stores.

    No lossy conversion to the old three-snow-layer GPU ABI is performed.
    The optical, site and option fields apply to this single independent PFT.
    """
    ground: SoilEnthalpyState
    snow: Snowpack=field(default_factory=Snowpack.empty)
    canopy_water_kgm2: float=0.
    canopy_enthalpy_jm2: float=0.
    lai: float=3.
    fveg: float=.8
    z0m: float=.05
    albedo: float=.2
    emiss: float=.97
    surface_temperature: float=285.
    root_frac: np.ndarray | None=None
    options: PhysicsOptions=field(default_factory=clm_options)
    terrain_index: float=0.
    groundwater_depth: float=2.

    def __post_init__(self):
        values=[self.canopy_water_kgm2,self.canopy_enthalpy_jm2,self.lai,self.fveg,self.z0m,
            self.albedo,self.emiss,self.surface_temperature,self.terrain_index,self.groundwater_depth]
        if (not np.all(np.isfinite(values)) or self.canopy_water_kgm2<0. or self.lai<0.
                or not 0<=self.fveg<=1 or self.z0m<=0 or not 0<=self.albedo<=1
                or not 0<self.emiss<=1 or not 200<=self.surface_temperature<=350
                or (self.canopy_water_kgm2==0 and self.canopy_enthalpy_jm2!=0)):
            raise ValueError('Invalid extensive CLM state/site parameter')
        if self.ground.freezing_option!=self.options.OptSoilSupercoolWater:
            raise ValueError('Soil enthalpy phase curve and CLM options disagree')
        roots=self.root_frac
        if roots is None:
            roots=np.exp(-3.*(np.cumsum(self.ground.dz)-.5*self.ground.dz))
        roots=np.array(roots,float,copy=True)
        if roots.shape!=self.ground.water.shape or not np.all(np.isfinite(roots)) or np.any(roots<0):
            raise ValueError('Invalid CLM root fractions')
        if roots.sum()>0: roots/=roots.sum()
        roots.setflags(write=False);object.__setattr__(self,'root_frac',roots)

    @classmethod
    def from_column(cls,column,*,snow=None,options=None,canopy_temperature=None):
        """Initialize from released state; select SNICAR/insulated CLM profile.

        Legacy snow is interpreted as ice; a warm or liquid pack requires an
        explicit Snowpack. Canopy temperature defaults to legacy surface T.
        Soil/site parameters are copied, never shared between initial columns.
        """
        c=deepcopy(column)
        options=replace(c.options,OptSnowAlbedo=3,OptSoilTemperatureBottom=1) if options is None else options
        c.options=options
        if snow is None:
            active=c.snow_swe_layers>0
            snow=(Snowpack.from_ice(1000.*c.snow_swe_layers[active],c.snow_t_layers[active],c.snow_density_layers[active])
                if np.any(active) else Snowpack.empty())
        t=c.ts if canopy_temperature is None else float(canopy_temperature)
        return cls(SoilEnthalpyState.from_column(c),snow,1000.*c.canw,1000.*c.canw*(LF+CW*(t-TF)),
            c.lai,c.fveg,c.z0m,c.albedo,c.emiss,c.ts,c.root_frac,options,c.terrain_index,c.groundwater_depth)

    @property
    def swe(self): return self.snow.swe
    @property
    def canw(self): return self.canopy_water_kgm2/1000.
    @property
    def tsoil(self): return self.ground.temperature
    @property
    def smc(self): return self.ground.water/(1000.*self.ground.dz)
    @property
    def water_kgm2(self): return float(self.ground.water.sum()+self.snow.mass.sum()+self.canopy_water_kgm2)
    @property
    def enthalpy_jm2(self): return float(self.ground.enthalpy.sum()+self.snow.enthalpy.sum()+self.canopy_enthalpy_jm2)

    def parameter_view(self):
        """Read-only-use adapter for legacy stress/drag/runoff parameterizations.

        This temporary ColumnState never evolves snow; the actual pack stays
        in Snowpack, including all five layers and its liquid/impurity stores.
        """
        return ColumnState(dz=self.ground.dz.copy(),tsoil=self.tsoil,smc=self.smc,
            soil=deepcopy(self.ground.params),soil_ice=self.ground.ice/(1000.*self.ground.dz),
            ts=self.surface_temperature,swe=self.swe,canw=self.canw,lai=self.lai,fveg=self.fveg,
            z0m=self.z0m,albedo=self.albedo,emiss=self.emiss,root_frac=self.root_frac.copy(),
            options=self.options,terrain_index=self.terrain_index,groundwater_depth=self.groundwater_depth)
