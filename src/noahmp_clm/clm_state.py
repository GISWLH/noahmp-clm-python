"""Extensive CLM development state, separate from the released 70-float ABI."""
from dataclasses import dataclass,field,replace
from copy import deepcopy
import numpy as np
from .column import ColumnState
from .options import PhysicsOptions
from .physics.snowpack import Snowpack,TF,LF,CW
from .physics.soil_enthalpy import SoilEnthalpyState
from .physics.canopy_interception import CanopyCondensate
from .physics.phenology import PhenologyState,PhenologyParameters
from .physics.biogeochemistry import BGCState,BGCParameters
from .physics.crop import CropState,CropParameters,CropPhase
from .physics.irrigation import IrrigationState,IrrigationParameters
from .physics.wetland import WetlandState,WetlandParameters
from .physics.tile_drainage import TileDrainageParameters
from .physics.vegetation_structure import CanopyStructureParameters,StemAreaScheme
from .physics.canopy_capacity import TemperatureMemory,CanopyCapacityParameters


def clm_options():
    return PhysicsOptions(OptSnowAlbedo=3,OptSoilTemperatureBottom=1)


@dataclass(frozen=True)
class CLMColumnState:
    """Independent snow, soil and optional layered canopy mass/enthalpy stores.

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
    canopy_condensate: CanopyCondensate | None=None
    phenology_state: PhenologyState | None=None
    phenology_parameters: PhenologyParameters | None=None
    bgc_state: BGCState | None=None
    bgc_parameters: BGCParameters | None=None
    crop_state: CropState | None=None
    crop_parameters: CropParameters | None=None
    irrigation_state: IrrigationState | None=None
    irrigation_parameters: IrrigationParameters | None=None
    wetland_state: WetlandState | None=None
    wetland_parameters: WetlandParameters | None=None
    tile_drainage_parameters: TileDrainageParameters | None=None
    sai: float=0.
    canopy_structure_parameters: CanopyStructureParameters | None=None
    canopy_capacity_state: TemperatureMemory | None=None
    canopy_capacity_parameters: CanopyCapacityParameters | None=None

    def __post_init__(self):
        if self.canopy_capacity_state is not None or self.canopy_capacity_parameters is not None:
            if (not isinstance(self.canopy_capacity_state,TemperatureMemory)
                    or not isinstance(self.canopy_capacity_parameters,CanopyCapacityParameters)
                    or self.canopy_condensate is None):
                raise ValueError('Canopy capacity requires traits, retained temperature memory and a layered canopy')
        structure=self.canopy_structure_parameters
        if not np.isfinite(self.sai) or self.sai<0.:
            raise ValueError('Invalid total stem area index')
        if structure is not None:
            if not isinstance(structure,CanopyStructureParameters) or self.canopy_condensate is None:
                raise ValueError('Explicit canopy structure requires parameters and a layered canopy')
            if structure.scheme==StemAreaScheme.ANNUAL_C3_CROP and self.crop_state is None:
                raise ValueError('Annual crop structure requires the actual crop state')
            if self.crop_state is not None and structure.scheme in (StemAreaScheme.CN_NONCROP,StemAreaScheme.CN_GENERIC_CROP):
                raise ValueError('Annual crop uses annual-crop or prescribed structure')
        elif self.sai!=0.:
            raise ValueError('Nonzero SAI requires explicit canopy geometry parameters')
        if self.options.OptDynamicVeg==5:
            if (not isinstance(self.phenology_state,PhenologyState)
                or not isinstance(self.phenology_parameters,PhenologyParameters)
                or self.canopy_condensate is None or len(self.ground.water)<3):
                raise ValueError('Dynamic vegetation requires phenology state/parameters, layered canopy and third soil layer')
            p=self.phenology_parameters
            object.__setattr__(self,'lai',p.lai_minimum+(p.lai_maximum-p.lai_minimum)*self.phenology_state.leaf_fraction)
        elif self.phenology_state is not None or self.phenology_parameters is not None:
            raise ValueError('Phenology state/parameters require OptDynamicVeg=5')
        if self.bgc_state is not None or self.bgc_parameters is not None:
            if (not isinstance(self.bgc_state,BGCState) or not isinstance(self.bgc_parameters,BGCParameters)
                or self.canopy_condensate is None):
                raise ValueError('BGC requires C/N state, parameters and a layered canopy')
            from .physics.plant_cn import plant_lai
            object.__setattr__(self,'lai',plant_lai(self.bgc_state.plant,self.fveg,self.bgc_parameters.plant))
        if self.options.OptCropModel==1:
            if (not isinstance(self.crop_state,CropState) or not isinstance(self.crop_parameters,CropParameters)
                or self.bgc_state is None or self.options.OptDynamicVeg!=4 or not self.fveg>0.):
                raise ValueError('Crop requires crop state/parameters, BGC, positive vegetation cover and OptDynamicVeg=4')
            plant=self.bgc_state.plant
            if self.crop_state.phase in (CropPhase.FALLOW,CropPhase.SOWN) and (plant.total_carbon>0. or plant.total_nitrogen>0.):
                raise ValueError('Fallow/unemerged crop cannot carry live or reserve plant biomass')
        elif self.crop_state is not None or self.crop_parameters is not None:
            raise ValueError('Crop state/parameters require OptCropModel=1')
        if self.options.OptIrrigation!=0:
            if (self.options.OptIrrigation!=1 or self.options.OptIrrigationMethod!=3
                or not isinstance(self.irrigation_state,IrrigationState) or not isinstance(self.irrigation_parameters,IrrigationParameters)):
                raise ValueError('Irrigation requires option 1, surface method 3 and explicit state/parameters')
            if self.irrigation_parameters.target_theta>self.ground.params.smcmax:
                raise ValueError('Irrigation target exceeds soil porosity')
        elif self.irrigation_state is not None or self.irrigation_parameters is not None or self.options.OptIrrigationMethod!=0:
            raise ValueError('Irrigation state, parameters and method require OptIrrigation=1')
        if self.options.OptWetlandModel!=0:
            if (self.options.OptWetlandModel!=1 or not isinstance(self.wetland_state,WetlandState)
                or not isinstance(self.wetland_parameters,WetlandParameters) or self.bgc_state is None):
                raise ValueError('Wetland option 1 requires a methane store, parameters and BGC')
            if self.wetland_parameters.active_depth_m>float(self.ground.dz.sum()):
                raise ValueError('Wetland active depth exceeds the soil column')
            if self.wetland_parameters.donor_q10!=self.bgc_parameters.decomposition.q10:
                raise ValueError('Wetland donor Q10 must match actual decomposition')
        elif self.wetland_state is not None or self.wetland_parameters is not None:
            raise ValueError('Wetland state/parameters require OptWetlandModel=1')
        if self.options.OptTileDrainage!=0:
            if self.options.OptTileDrainage!=1 or not isinstance(self.tile_drainage_parameters,TileDrainageParameters):
                raise ValueError('Tile drainage requires option 1 and explicit parameters')
            if self.tile_drainage_parameters.field_capacity>self.ground.params.smcmax:
                raise ValueError('Tile field capacity exceeds soil porosity')
            if self.tile_drainage_parameters.tube_depth_m>float(self.ground.dz.sum()):
                raise ValueError('Tile depth exceeds the soil column')
        elif self.tile_drainage_parameters is not None:raise ValueError('Tile parameters require OptTileDrainage=1')
        # When present, layer M/H are authoritative. Scalar fields remain
        # read-only aggregate aliases for diagnostics and parameter adapters.
        if self.canopy_condensate is not None:
            if not isinstance(self.canopy_condensate,CanopyCondensate):raise ValueError('Invalid layered canopy state')
            object.__setattr__(self,'canopy_water_kgm2',float(self.canopy_condensate.mass.sum()))
            object.__setattr__(self,'canopy_enthalpy_jm2',float(self.canopy_condensate.enthalpy.sum()))
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
        if self.bgc_state is not None and roots.sum()==0.:
            raise ValueError('BGC requires a nonempty root distribution')
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

    def with_phenology(self,parameters,initial=None):
        """Enable simplified GDD leaf display with fixed maximum vegetation cover.

        Layer M/H is retained when LAI changes. The next interception stage
        routes any excess liquid/ice and its heat to the ground. Plant carbon
        pools are separate from this leaf-availability controller.
        """
        return replace(self,options=replace(self.options,OptDynamicVeg=5),
            phenology_state=PhenologyState() if initial is None else initial,phenology_parameters=parameters)

    def with_biogeochemistry(self,state,parameters=None):
        """Attach explicit C/N pools; actual LAI becomes a leaf-carbon alias.

        Retain all water/enthalpy. Seasonal option 5 supplies a leaf target,
        not biomass; any capacity excess is routed by the next physical step.
        """
        return replace(self,bgc_state=state,bgc_parameters=BGCParameters() if parameters is None else parameters)

    def with_layered_canopy(self,layers=10):
        """Convert shared mass/enthalpy to equal leaf-area layers, retaining heat.

        Existing extensive M/H determines phase; a liquid store initialized
        below Tf is consequently reinterpreted at phase equilibrium. Calling
        this on an already layered canopy never redistributes its state.
        """
        if not isinstance(layers,int) or layers<1:raise ValueError('Positive canopy layer count required')
        if self.canopy_condensate is not None:
            if len(self.canopy_condensate.mass)!=layers:raise ValueError('Explicit conservative canopy remapping required')
            return self
        return replace(self,canopy_condensate=CanopyCondensate(np.full(layers,self.canopy_water_kgm2/layers),np.full(layers,self.canopy_enthalpy_jm2/layers)))

    def with_canopy_capacity(self,parameters,memory):
        """Attach explicit C3 traits and the complete restartable T10 history."""
        return replace(self,canopy_capacity_parameters=parameters,canopy_capacity_state=memory)

    def with_vegetation_structure(self,parameters,*,sai=None):
        """Enable explicit total/exposed leaf+stem geometry, retaining all M/H.

        Prescribed SAI requires its actual initial input. Dynamic schemes
        initialize at their reference floor/annual-crop ratio unless supplied.
        Height and vegetation form are explicit parameters, not inferred from
        flux scores. The legacy state and its70-float ABI are unchanged.
        """
        if not isinstance(parameters,CanopyStructureParameters):raise TypeError('Expected canopy structure parameters')
        if sai is None:
            if parameters.scheme==StemAreaScheme.PRESCRIBED:raise ValueError('Prescribed structure requires explicit initial SAI')
            if parameters.scheme==StemAreaScheme.CN_NONCROP:sai=.5
            elif parameters.scheme==StemAreaScheme.CN_GENERIC_CROP:sai=.05
            else:sai=.2*self.lai
        layered=self.with_layered_canopy() if self.canopy_condensate is None else self
        return replace(layered,sai=sai,canopy_structure_parameters=parameters)

    def with_crop(self,parameters=None,initial=None):
        """Attach an annual crop to explicit BGC pools, retaining canopy M/H.

        Existing seasonal phenology is incompatible. Fallow starts require
        empty plant pools; attaching a growing crop requires its actual phase.
        """
        return replace(self,options=replace(self.options,OptCropModel=1),
            crop_state=CropState() if initial is None else initial,
            crop_parameters=CropParameters() if parameters is None else parameters)

    def with_irrigation(self,parameters=None,initial=None):
        """Enable moisture-triggered surface application from explicit supply."""
        if parameters is None:
            parameters=IrrigationParameters(target_theta=self.ground.params.smcref,wilting_theta=self.ground.params.smcwlt)
        return replace(self,options=replace(self.options,OptIrrigation=1,OptIrrigationMethod=3),
            irrigation_state=IrrigationState() if initial is None else initial,irrigation_parameters=parameters)

    def with_wetland(self,parameters=None,initial=None):
        """Add a conserved methane-C pool to the actual BGC column."""
        if self.bgc_state is None:raise ValueError('Wetland requires explicit BGC carbon donors')
        if parameters is None:parameters=WetlandParameters(donor_q10=self.bgc_parameters.decomposition.q10)
        return replace(self,options=replace(self.options,OptWetlandModel=1),
            wetland_state=WetlandState() if initial is None else initial,wetland_parameters=parameters)

    def with_tile_drainage(self,parameters=None):
        """Enable the simple actual-soil outlet at an explicitly modeled depth."""
        if parameters is None:parameters=TileDrainageParameters(field_capacity=self.ground.params.smcref)
        return replace(self,options=replace(self.options,OptTileDrainage=1),tile_drainage_parameters=parameters)

    @property
    def carbon_gc_m2(self):
        if self.bgc_state is None:raise ValueError('Total carbon requires BGC')
        return (self.bgc_state.total_carbon+(self.crop_state.seed_carbon if self.crop_state is not None else 0.)
            +(self.wetland_state.methane_gc_m2 if self.wetland_state is not None else 0.))

    @property
    def nitrogen_gn_m2(self):
        if self.bgc_state is None:raise ValueError('Total nitrogen requires BGC')
        return self.bgc_state.total_nitrogen+(self.crop_state.seed_nitrogen if self.crop_state is not None else 0.)
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
