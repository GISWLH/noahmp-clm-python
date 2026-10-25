! Independent test caller, containing no implementation of official physics.
! All diagnostic/drainage operations below call the unchanged pinned libclm.a.
program drainage_caller
  use iso_fortran_env, only: input_unit, output_unit
  use shr_kind_mod, only: r8 => shr_kind_r8
  use ESMF, only: ESMF_Initialize, ESMF_Finalize, ESMF_SUCCESS
  use spmdMod, only: spmd_init, MPI_COMM_WORLD
  use clm_time_manager, only: set_timemgr_init, timemgr_init, advance_timestep, get_nstep
  use clm_varpar, only: nlevsoi, nlevgrnd, nlevsno
  use clm_varctl, only: use_vichydro
  use ColumnType, only: col
  use LandunitType, only: lun
  use decompMod, only: bounds_type
  use SoilHydrologyType, only: soilhydrology_type
  use SoilStateType, only: soilstate_type
  use WaterStateType, only: waterstate_type
  use WaterfluxType, only: waterflux_type
  use TemperatureType, only: temperature_type
  use SoilHydrologyMod, only: SoilHydReadNML, ThetaBasedWaterTable, PerchedWaterTable, &
       PerchedLateralFlow, LateralFlowPowerLaw
  implicit none
  integer, parameter :: n=4
  integer :: ierr, unit, j, dt, nsteps, istep, ipc_step
  integer :: hydro_filter(1)=[1], urban_filter(1)=[1]
  real(r8) :: slope, baseflow, shared_water(n), shared_enthalpy(n)
  character(len=1024) :: filename
  type(bounds_type) :: bounds
  type(soilhydrology_type) :: h
  type(soilstate_type) :: s
  type(waterstate_type) :: w
  type(waterflux_type) :: f
  type(temperature_type) :: t

  call get_command_argument(1, filename)
  open(newunit=unit, file=trim(filename), status='old', action='read')
  read(unit,*) dt, slope, baseflow, nsteps
  if (nsteps /= 960 .or. dt /= 90 .or. baseflow /= 0._r8) stop 4
  call MPI_Init(ierr)
  if (ierr /= 0) stop 2
  call spmd_init(MPI_COMM_WORLD, 1)
  call ESMF_Initialize(rc=ierr)
  if (ierr /= ESMF_SUCCESS) stop 3
  call set_timemgr_init(calendar_in='NO_LEAP', start_ymd_in=20010101, start_tod_in=0, &
       ref_ymd_in=20010101, ref_tod_in=0, stop_ymd_in=20020101, stop_tod_in=0, &
       perpetual_run_in=.false., nelapse_in=nsteps+1, dtime_in=dt)
  call timemgr_init()
  nlevsoi=n; nlevgrnd=n; nlevsno=0; use_vichydro=.false.
  bounds%begg=1; bounds%endg=1; bounds%begl=1; bounds%endl=1
  bounds%begc=1; bounds%endc=1; bounds%begp=1; bounds%endp=1
  bounds%begCohort=1; bounds%endCohort=1; bounds%level=0; bounds%clump_index=1
  allocate(col%z(1,n), col%zi(1,0:n), col%dz(1,n), col%nbedrock(1), col%topo_slope(1))
  allocate(col%snl(1), col%landunit(1), lun%urbpoi(1))
  col%snl=0; col%landunit=1; lun%urbpoi=.false.; col%nbedrock=n
  col%topo_slope=slope
  allocate(w%h2osoi_liq_col(1,n), w%h2osoi_ice_col(1,n), w%h2osoi_vol_col(1,n))
  allocate(w%h2osno_col(1), w%h2osfc_col(1)); w%h2osno_col=0; w%h2osfc_col=0
  allocate(t%t_soisno_col(1,n))
  allocate(s%watsat_col(1,n), s%bsw_col(1,n), s%sucsat_col(1,n), s%hksat_col(1,n))
  allocate(s%eff_porosity_col(1,n), s%hk_l_col(1,n))
  do j=1,n
     read(unit,*) col%dz(1,j), s%watsat_col(1,j), s%bsw_col(1,j), s%sucsat_col(1,j), &
          s%hksat_col(1,j), w%h2osoi_liq_col(1,j), w%h2osoi_ice_col(1,j), t%t_soisno_col(1,j)
  enddo
  close(unit)
  col%zi(1,0)=0
  do j=1,n
     col%zi(1,j)=col%zi(1,j-1)+col%dz(1,j)
     col%z(1,j)=0.5_r8*(col%zi(1,j-1)+col%zi(1,j))
  enddo
  s%eff_porosity_col=s%watsat_col-w%h2osoi_ice_col/(917._r8*col%dz)
  s%hk_l_col=0
  w%h2osoi_vol_col=0
  allocate(h%zwt_col(1), h%zwt_perched_col(1), h%frost_table_col(1), h%icefrac_col(1,n))
  allocate(h%depth_col(1,n), h%c_param_col(1), h%dsmax_col(1), h%max_moist_col(1,n))
  allocate(h%moist_col(1,n), h%ds_col(1), h%Wsvic_col(1), h%hkdepth_col(1))
  allocate(h%wa_col(1), h%ice_col(1,n), h%qcharge_col(1))
  h%origflag=0; h%h2osfcflag=1; h%depth_col=0; h%c_param_col=0; h%dsmax_col=0
  h%max_moist_col=0; h%moist_col=0; h%ds_col=0; h%Wsvic_col=0; h%hkdepth_col=1
  h%wa_col=0; h%ice_col=0; h%qcharge_col=0; h%icefrac_col=0
  h%zwt_col=0; h%zwt_perched_col=0; h%frost_table_col=0
  allocate(f%qflx_drain_col(1), f%qflx_drain_perched_col(1), f%qflx_rsub_sat_col(1))
  allocate(f%qflx_snwcp_liq_col(1), f%qflx_ice_runoff_xs_col(1), f%qflx_dew_grnd_col(1))
  allocate(f%qflx_dew_snow_col(1), f%qflx_sub_snow_col(1), f%qflx_qrgwl_col(1))
  f%qflx_drain_col=0; f%qflx_drain_perched_col=0; f%qflx_rsub_sat_col=0
  f%qflx_snwcp_liq_col=0; f%qflx_ice_runoff_xs_col=0; f%qflx_dew_grnd_col=0
  f%qflx_dew_snow_col=0; f%qflx_sub_snow_col=0; f%qflx_qrgwl_col=0
  open(newunit=unit, file='hydrology.nml', status='replace')
  write(unit,*) '&soilhydrology_inparm'
  write(unit,*) ' baseflow_scalar=', baseflow
  write(unit,*) '/'
  close(unit)
  call SoilHydReadNML('hydrology.nml')
  call ThetaBasedWaterTable(bounds, 1, hydro_filter, 0, urban_filter, h, s, w, f)
  call PerchedWaterTable(bounds, 1, hydro_filter, 0, urban_filter, h, s, t, w, f)
  write(*,'(a)') 'PB_IPC_READY'
  flush(output_unit)
  do istep=1,nsteps
  read(input_unit,*,iostat=ierr) ipc_step
  if (ierr /= 0 .or. ipc_step /= istep) stop 11
  do j=1,n
     read(input_unit,*,iostat=ierr) shared_water(j), shared_enthalpy(j), &
          t%t_soisno_col(1,j), w%h2osoi_liq_col(1,j), w%h2osoi_ice_col(1,j)
     if (ierr /= 0) stop 12
  enddo
  ! Only boundary state import; no thermodynamic or hydrologic implementation.
  s%eff_porosity_col=s%watsat_col-w%h2osoi_ice_col/(917._r8*col%dz)
  call ThetaBasedWaterTable(bounds, 1, hydro_filter, 0, urban_filter, h, s, w, f)
  write(*,'(a,4(1x,es25.17))') 'PB_OFFICIAL_SHARED_WATER', shared_water
  write(*,'(a,4(1x,es25.17))') 'PB_OFFICIAL_SHARED_ENTHALPY', shared_enthalpy
  write(*,'(a,1(1x,es25.17))') 'PB_OFFICIAL_INITIAL_SNOW', w%h2osno_col(1)
  write(*,'(a,2(1x,i8))') 'PB_OFFICIAL_STEP', istep, get_nstep()
  write(*,'(a,3(1x,es25.17))') 'PB_OFFICIAL_DIAG', h%zwt_col(1), h%zwt_perched_col(1), h%frost_table_col(1)
  write(*,'(a,4(1x,es25.17))') 'PB_OFFICIAL_INITIAL_LIQ', w%h2osoi_liq_col(1,:)
  write(*,'(a,4(1x,es25.17))') 'PB_OFFICIAL_INITIAL_ICE', w%h2osoi_ice_col(1,:)
  write(*,'(a,3(1x,es25.17))') 'PB_OFFICIAL_INITIAL_OTHER', w%h2osfc_col(1), h%wa_col(1), h%qcharge_col(1)
  call PerchedLateralFlow(bounds, 1, hydro_filter, 0, urban_filter, h, s, w, f)
  write(*,'(a,2(1x,es25.17))') 'PB_OFFICIAL_PERCHED', f%qflx_drain_perched_col(1), h%zwt_perched_col(1)
  write(*,'(a,4(1x,es25.17))') 'PB_OFFICIAL_AFTER_PERCH_LIQ', w%h2osoi_liq_col(1,:)
  write(*,'(a,4(1x,es25.17))') 'PB_OFFICIAL_AFTER_PERCH_ICE', w%h2osoi_ice_col(1,:)
  write(*,'(a,3(1x,es25.17))') 'PB_OFFICIAL_AFTER_PERCH_OTHER', w%h2osfc_col(1), h%wa_col(1), h%qcharge_col(1)
  call LateralFlowPowerLaw(bounds, 1, hydro_filter, 0, urban_filter, h, s, w, f)
  write(*,'(a,2(1x,es25.17))') 'PB_OFFICIAL_MAIN', f%qflx_drain_col(1), h%zwt_col(1)
  write(*,'(a,4(1x,es25.17))') 'PB_OFFICIAL_FINAL_LIQ', w%h2osoi_liq_col(1,:)
  write(*,'(a,4(1x,es25.17))') 'PB_OFFICIAL_FINAL_ICE', w%h2osoi_ice_col(1,:)
  write(*,'(a,6(1x,es25.17))') 'PB_OFFICIAL_OTHER', w%h2osfc_col(1), h%wa_col(1), h%qcharge_col(1), &
       f%qflx_rsub_sat_col(1), f%qflx_ice_runoff_xs_col(1), f%qflx_qrgwl_col(1)
  write(*,'(a,4(1x,es25.17))') 'PB_OFFICIAL_TEMPERATURE', t%t_soisno_col(1,:)
  write(*,'(a,1(1x,es25.17))') 'PB_OFFICIAL_FINAL_SNOW', w%h2osno_col(1)
  write(*,'(a,4(1x,es25.17))') 'PB_OFFICIAL_AUX_FLUXES', f%qflx_snwcp_liq_col(1), &
       f%qflx_dew_grnd_col(1), f%qflx_dew_snow_col(1), f%qflx_sub_snow_col(1)
  call advance_timestep()
  write(*,'(a,2(1x,i8))') 'PB_OFFICIAL_ENDSTEP', istep, get_nstep()
  flush(output_unit)
  enddo
  ! Receive/echo the final external phase state before diagnostic-only readback.
  read(input_unit,*,iostat=ierr) ipc_step
  if (ierr /= 0 .or. ipc_step /= nsteps+1) stop 13
  do j=1,n
     read(input_unit,*,iostat=ierr) shared_water(j), shared_enthalpy(j), &
          t%t_soisno_col(1,j), w%h2osoi_liq_col(1,j), w%h2osoi_ice_col(1,j)
     if (ierr /= 0) stop 14
  enddo
  write(*,'(a,4(1x,es25.17))') 'PB_OFFICIAL_FINAL_SHARED_WATER', shared_water
  write(*,'(a,4(1x,es25.17))') 'PB_OFFICIAL_FINAL_SHARED_ENTHALPY', shared_enthalpy
  write(*,'(a,4(1x,es25.17))') 'PB_OFFICIAL_FINAL_SHARED_TEMPERATURE', t%t_soisno_col(1,:)
  write(*,'(a,4(1x,es25.17))') 'PB_OFFICIAL_FINAL_SHARED_LIQ', w%h2osoi_liq_col(1,:)
  write(*,'(a,4(1x,es25.17))') 'PB_OFFICIAL_FINAL_SHARED_ICE', w%h2osoi_ice_col(1,:)
  s%eff_porosity_col=s%watsat_col-w%h2osoi_ice_col/(917._r8*col%dz)
  call ThetaBasedWaterTable(bounds, 1, hydro_filter, 0, urban_filter, h, s, w, f)
  call PerchedWaterTable(bounds, 1, hydro_filter, 0, urban_filter, h, s, t, w, f)
  write(*,'(a,3(1x,es25.17))') 'PB_OFFICIAL_FINAL_REDIAG', h%zwt_col(1), h%zwt_perched_col(1), h%frost_table_col(1)
  call ESMF_Finalize(rc=ierr)
  call MPI_Finalize(ierr)
end program drainage_caller
