module effective_ripple

    use iso_fortran_env, only: real64
    implicit none
    
    ! Grid parameters
    integer :: nr, nz, nphi
    real(8) :: rmin, rmax, zmin, zmax, phimin, phimax
    real(8), allocatable :: r_grid(:), z_grid(:), phi_grid(:)
    
    ! Number of coil groups
    integer :: nextcur
    
    ! 5D arrays: Hermite coefficients for each coil group (nextcur, 0:7, nr, nz, nphi)
    real(8), allocatable :: fherm_br_arr(:,:,:,:,:)
    real(8), allocatable :: fherm_bz_arr(:,:,:,:,:)
    real(8), allocatable :: fherm_bp_arr(:,:,:,:,:)
    
    ! Interpolation method flags
    integer :: ilinx, iliny, ilinz
    
    ! Tracing parameters (shared by all scans)
    integer :: nturn, nphi_trace
    integer :: npoints
    integer :: trace_verbose = 0
    ! Physical constants
    real(8), parameter :: PI = 3.141592653589793d0
    
    ! =====================================================================
    ! Thread-private variables - recalculated for each extcur scan
    ! =====================================================================
    ! 4D arrays: total field Hermite coefficients for current extcur (0:7, nr, nz, nphi)
    real(8), allocatable :: fherm_br_sum(:,:,:,:)
    real(8), allocatable :: fherm_bz_sum(:,:,:,:)
    real(8), allocatable :: fherm_bp_sum(:,:,:,:)
    
    ! Current scan input/output (kept public)
    ! real(8), allocatable :: extcur_current(:)
    ! initial_rz_current, initial_gradpsi_current removed
    real(8), allocatable :: fieldline_gradpsi_data_current(:,:)
    real(8) :: Bboundary_current
    real(8) :: epsilon_eff_current
    ! integer :: trace_error_code

contains

    !============================================================================
    ! Subroutine: initialize_field
    ! Purpose: One-time initialization of field data (dB/dI), called once at startup
    ! Input: br_input, bz_input, bp_input - dB/dI data for each coil group
    !        grid parameters
    ! Output: global public variables are set
    !============================================================================
    subroutine initialize_field(br_input, bz_input, bp_input, &
                               rmin_in, rmax_in, nr_in, &
                               zmin_in, zmax_in, nz_in, &
                               phimin_in, phimax_in, nphi_in, nextcur_in)
      implicit none
      integer, parameter :: R8=SELECTED_REAL_KIND(12,100)
      
      ! Input: dB/dI matrices (nextcur, nphi, nz, nr)
      integer, intent(in) :: nr_in, nz_in, nphi_in, nextcur_in
      real(8), intent(in) :: rmin_in, rmax_in, zmin_in, zmax_in, phimin_in, phimax_in
      real(8), intent(in) :: br_input(nextcur_in, nphi_in, nz_in, nr_in)
      real(8), intent(in) :: bz_input(nextcur_in, nphi_in, nz_in, nr_in)
      real(8), intent(in) :: bp_input(nextcur_in, nphi_in, nz_in, nr_in)
      
      ! Local variables
      integer :: i, j, k, ic, ier
      real(8) :: dr, dz, dphi

      ! Save grid to globals
      nr = nr_in
      nz = nz_in
      nphi = nphi_in
      nextcur = nextcur_in
      rmin = rmin_in
      rmax = rmax_in
      zmin = zmin_in
      zmax = zmax_in
      phimin = phimin_in
      phimax = phimax_in

      ! Allocate grid arrays
      if (allocated(r_grid)) deallocate(r_grid)
      if (allocated(z_grid)) deallocate(z_grid)
      if (allocated(phi_grid)) deallocate(phi_grid)
      allocate(r_grid(nr), z_grid(nz), phi_grid(nphi))

      dr = (rmax - rmin) / real(nr - 1, 8)
      dz = (zmax - zmin) / real(nz - 1, 8)
      dphi = (phimax - phimin) / real(nphi - 1, 8)
      do i=1,nr;   r_grid(i)  = rmin  + real(i-1, 8)*dr;  enddo
      do j=1,nz;   z_grid(j)  = zmin  + real(j-1, 8)*dz;  enddo
      do k=1,nphi; phi_grid(k)= phimin+ real(k-1, 8)*dphi;enddo

      ! Allocate 5D Hermite coefficient arrays: one per coil group (nextcur, 0:7, nr, nz, nphi)
      if (allocated(fherm_br_arr)) deallocate(fherm_br_arr)
      if (allocated(fherm_bz_arr)) deallocate(fherm_bz_arr)
      if (allocated(fherm_bp_arr)) deallocate(fherm_bp_arr)
      allocate(fherm_br_arr(nextcur, 0:7, nr, nz, nphi))
      allocate(fherm_bz_arr(nextcur, 0:7, nr, nz, nphi))
      allocate(fherm_bp_arr(nextcur, 0:7, nr, nz, nphi))

      ! Fill and setup Hermite coefficients for each coil group separately
      do ic = 1, nextcur
        ! Fill function values for coil group ic
        do k = 1, nphi
          do j = 1, nz
            do i = 1, nr
              fherm_br_arr(ic, 0, i, j, k) = br_input(ic, k, j, i)
              fherm_bz_arr(ic, 0, i, j, k) = bz_input(ic, k, j, i)
              fherm_bp_arr(ic, 0, i, j, k) = bp_input(ic, k, j, i)
            enddo
          enddo
        enddo

        ! Setup Hermite interpolation coefficients for coil group ic
        call r8akherm3p(r_grid, nr, z_grid, nz, phi_grid, nphi, &
                        fherm_br_arr(ic,:,:,:,:), nr, nz, &
                        ilinx, iliny, ilinz, 0, 0, 0, ier)
        if (ier /= 0) then
          if (trace_verbose /= 0) write(*,'(A,I0,A,I0)') 'Error in r8akherm3p for Br (coil ', ic, '): ier = ', ier
          return
        endif

        call r8akherm3p(r_grid, nr, z_grid, nz, phi_grid, nphi, &
                        fherm_bz_arr(ic,:,:,:,:), nr, nz, &
                        ilinx, iliny, ilinz, 0, 0, 0, ier)
        if (ier /= 0) then
          if (trace_verbose /= 0) write(*,'(A,I0,A,I0)') 'Error in r8akherm3p for Bz (coil ', ic, '): ier = ', ier
          return
        endif

        call r8akherm3p(r_grid, nr, z_grid, nz, phi_grid, nphi, &
                        fherm_bp_arr(ic,:,:,:,:), nr, nz, &
                        ilinx, iliny, ilinz, 0, 0, 0, ier)
        if (ier /= 0) then
          if (trace_verbose /= 0) write(*,'(A,I0,A,I0)') 'Error in r8akherm3p for Bp (coil ', ic, '): ier = ', ier
          return
        endif
      end do

      if (trace_verbose /= 0) write(*,'(A,I0,A)') 'Field initialization completed for ', nextcur, ' coil groups.'
    end subroutine initialize_field

    !============================================================================
    ! Subroutine: set_trace_parameters
    ! Purpose: Set field line tracing parameters (shared by all scans)
    ! Input: nturn_in - number of turns, nphi_in - points per turn
    !============================================================================
    subroutine set_trace_parameters(nturn_in, nphi_in)
      implicit none
      integer, intent(in) :: nturn_in, nphi_in
      
      nturn = nturn_in
      nphi_trace = nphi_in
      npoints = nturn * nphi_trace
      
      ! write(*,'(A,I0,A,I0,A,I0,A)') 'Trace parameters set: nturn=', nturn, &
      !                                ', nphi=', nphi_trace, ', npoints=', npoints
    end subroutine set_trace_parameters

    !============================================================================
    ! Subroutine: set_trace_verbose
    ! Purpose: allow external (e.g. Python) code to enable/disable module writes
    ! Input: flag - 0 to silence writes, non-zero to enable
    !============================================================================
    subroutine set_trace_verbose(flag)
      implicit none
      integer, intent(in) :: flag
      trace_verbose = flag
    end subroutine set_trace_verbose

    function get_trace_verbose() result(flag)
      implicit none
      integer :: flag
      flag = trace_verbose
    end function get_trace_verbose

    !============================================================================
    ! Subroutine: compute_ripple
    ! Purpose: Main computation interface - given extcur and initial conditions, compute effective ripple
    ! Input: extcur - current combination
    !        initial_rz - initial position [R, Z]
    !        initial_gradpsi - initial grad_psi [r, z, phi]
    !        save_fieldline - flag to save field line data
    ! Output: epsilon_eff - effective ripple
    !         Bboundary - boundary magnetic field strength
    !============================================================================
    subroutine compute_ripple(extcur, initial_rz, initial_gradpsi, &
                         epsilon_eff, Bboundary, fieldline_data, trace_istate)
      implicit none
      
      real(8), intent(in) :: extcur(:)
      real(8), intent(in) :: initial_rz(2)
      real(8), intent(in) :: initial_gradpsi(3)
      real(8), intent(out) :: epsilon_eff
      real(8), intent(out) :: Bboundary
      real(8), intent(out) :: fieldline_data(:, :)
      integer, intent(out) :: trace_istate
      
      real(8), allocatable :: fieldline_local(:,:)
      real(8), allocatable :: geocur(:)

      trace_istate = 0
      
      if (.not. allocated(fherm_br_arr)) then
        write(*,'(A)') 'Error: Field not initialized. Call initialize_field first.'
        epsilon_eff = 0.0d0
        Bboundary = 0.0d0
        trace_istate = -100
        return
      endif
      
      if (npoints <= 0) then
        write(*,'(A)') 'Error: Trace parameters not set. Call set_trace_parameters first.'
        epsilon_eff = 0.0d0
        Bboundary = 0.0d0
        trace_istate = -101
        return
      endif

      ! if (allocated(extcur_current)) deallocate(extcur_current)
      ! allocate(extcur_current(size(extcur)))
      ! extcur_current = extcur
      ! initial_rz_current = initial_rz
      ! initial_gradpsi_current = initial_gradpsi
      
      call sum_bfield_internal(extcur)
      
      allocate(fieldline_local(npoints, 20))
      call trace_gradpsi_internal(fieldline_local, initial_rz, initial_gradpsi, trace_istate)

      if (trace_istate /= 0) then
        epsilon_eff = 0.0d0
        Bboundary = 0.0d0
        deallocate(fieldline_local)
        return
      endif
      
      allocate(geocur(npoints))
      call geodesic_curvature_internal(fieldline_local, geocur, Bboundary)
      
      call effective_ripple_internal(fieldline_local, geocur, epsilon_eff)
      
      Bboundary_current = Bboundary
      epsilon_eff_current = epsilon_eff
      
      if (allocated(fieldline_gradpsi_data_current)) deallocate(fieldline_gradpsi_data_current)
      allocate(fieldline_gradpsi_data_current(npoints, 20))
      fieldline_gradpsi_data_current = fieldline_local


      if (size(fieldline_data, 1) >= npoints .and. size(fieldline_data, 2) >= 20) then
        fieldline_data(1:npoints, 1:20) = fieldline_local
      else
        write(*,'(A)') 'Warning: fieldline_data array too small'
      endif

      deallocate(fieldline_local, geocur)
      
    end subroutine compute_ripple

    !============================================================================
    ! Internal subroutine: sum_bfield_internal
    ! Weighted sum of magnetic field according to current extcur_current
    !============================================================================
    subroutine sum_bfield_internal(extcur)
      implicit none
      real(8), intent(in) :: extcur(:)
      integer :: ic

      if (allocated(fherm_br_sum)) deallocate(fherm_br_sum)
      if (allocated(fherm_bz_sum)) deallocate(fherm_bz_sum)
      if (allocated(fherm_bp_sum)) deallocate(fherm_bp_sum)
      allocate(fherm_br_sum(0:7, nr, nz, nphi))
      allocate(fherm_bz_sum(0:7, nr, nz, nphi))
      allocate(fherm_bp_sum(0:7, nr, nz, nphi))

      fherm_br_sum = 0.0d0
      fherm_bz_sum = 0.0d0
      fherm_bp_sum = 0.0d0

      do ic = 1, nextcur
        fherm_br_sum(:,:,:,:) = fherm_br_sum(:,:,:,:) + extcur(ic) * fherm_br_arr(ic,:,:,:,:)
        fherm_bz_sum(:,:,:,:) = fherm_bz_sum(:,:,:,:) + extcur(ic) * fherm_bz_arr(ic,:,:,:,:)
        fherm_bp_sum(:,:,:,:) = fherm_bp_sum(:,:,:,:) + extcur(ic) * fherm_bp_arr(ic,:,:,:,:)
      end do

    end subroutine sum_bfield_internal

    !============================================================================
    ! Internal subroutine: interpolate_field
    ! Interpolate magnetic field and its derivatives at a given point
    !============================================================================
    subroutine interpolate_field(r, z, phi, br_interp, bz_interp, bp_interp, &
                   br_r, br_z, br_phi, &
                   bz_r, bz_z, bz_phi, &
                   bp_r, bp_z, bp_phi, trace_istate)
      implicit none
      integer, parameter :: R8=SELECTED_REAL_KIND(12,100)
      real(8), intent(in) :: r, z, phi
      real(8), intent(out) :: br_interp, bz_interp, bp_interp
      real(8), intent(out) :: br_r, br_z, br_phi
      real(8), intent(out) :: bz_r, bz_z, bz_phi
      real(8), intent(out) :: bp_r, bp_z, bp_phi
      integer, intent(out) :: trace_istate
      integer :: ict(8), ier 
      real(8) :: fval(8)

      trace_istate = 0
      if (r < rmin .or. r > rmax .or. z < zmin .or. z > zmax) then
        trace_istate = -1000
        if (trace_verbose /= 0) write(*, '(A,E15.6,A,E15.6,A)') &
        'Error: Point out of bounds for interpolation: R=', r, ', Z=', z, ', Phi=', phi
        return
      end if


      ict(1:8) = 0
      ict(1) = 1; ict(2) = 1; ict(3) = 1; ict(4) = 1

      call r8herm3ev(r, z, phi, r_grid, nr, z_grid, nz, phi_grid, nphi, &
                     ilinx, iliny, ilinz, fherm_br_sum, nr, nz, ict, fval, ier)
      if (ier /= 0) then
        trace_istate = -1000 - abs(ier)
        if (trace_verbose /= 0) write(*, '(A,E15.6,A,E15.6,A,E15.6,A,I0)') &
        'Error in R: ', r, ', Z: ', z, ', Phi: ', phi, &
        ' for Br interpolation: ier = ', ier
        ! trace_error_code = -1000 - abs(ier)
        ! if (present(trace_istate)) trace_istate = trace_error_code
        return
      endif
      br_interp = fval(1); br_r = fval(2); br_z = fval(3); br_phi = fval(4)

      call r8herm3ev(r, z, phi, r_grid, nr, z_grid, nz, phi_grid, nphi, &
                     ilinx, iliny, ilinz, fherm_bz_sum, nr, nz, ict, fval, ier)
      if (ier /= 0) then
        trace_istate = -1000 - abs(ier)
        if (trace_verbose /= 0) write(*, '(A,E15.6,A,E15.6,A,E15.6,A,I0)') &
        'Error in R: ', r, ', Z: ', z, ', Phi: ', phi, &
        ' for Bz interpolation: ier = ', ier
        ! trace_error_code = -1000 - abs(ier)
        ! if (present(trace_istate)) trace_istate = trace_error_code
        return
      endif
      bz_interp = fval(1); bz_r = fval(2); bz_z = fval(3); bz_phi = fval(4)

      call r8herm3ev(r, z, phi, r_grid, nr, z_grid, nz, phi_grid, nphi, &
                     ilinx, iliny, ilinz, fherm_bp_sum, nr, nz, ict, fval, ier)
      if (ier /= 0) then
        trace_istate = -1000 - abs(ier)
        if (trace_verbose /= 0) write(*, '(A,E15.6,A,E15.6,A,E15.6,A,I0)') &
        'Error in R: ', r, ', Z: ', z, ', Phi: ', phi, &
        ' for Bp interpolation: ier = ', ier
        ! trace_error_code = -1000 - abs(ier)
        ! if (present(trace_istate)) trace_istate = trace_error_code
        return
      endif
      bp_interp = fval(1); bp_r = fval(2); bp_z = fval(3); bp_phi = fval(4)
    end subroutine interpolate_field

    !============================================================================
    ! Internal subroutine: trace_gradpsi_internal
    ! Trace field lines and compute grad_psi evolution
    !============================================================================
    subroutine trace_gradpsi_internal(fieldline_gradpsi_data, initial_rz, initial_gradpsi, trace_istate)
      implicit none
      real(8), intent(out) :: fieldline_gradpsi_data(:, :)
      real(8), intent(in)  :: initial_rz(2)
      real(8), intent(in)  :: initial_gradpsi(3)
      integer, intent(out) :: trace_istate

      integer :: i
      integer :: neq, itol, itask, iopt, lrw, liw, mf
      integer :: istate_local
      integer :: interp_istate
      real(8) :: rtol, atol, phi, phi_stop
      real(8), allocatable :: v(:), rwork(:)
      integer, allocatable :: iwork(:)
      real(8) :: phi_for_interp

      trace_istate = 0

      ! trace_istate = 0

      if (npoints <= 0) then
         trace_istate = -101
        return
      end if

      neq = 5
      mf = 10
      lrw = 20 + 16*neq
      liw = 20
      atol = 1.0d-12
      rtol = 0.0d0
      itol = 1
      itask = 1
      istate_local = 1
      iopt = 0

      allocate(v(1:neq), rwork(1:lrw), iwork(1:liw))

      v(1) = initial_rz(1)
      v(2) = initial_rz(2)
      v(3) = initial_gradpsi(1)
      v(4) = initial_gradpsi(2)
      v(5) = initial_gradpsi(3)

      phi = 0.0d0

      do i = 1, npoints
        phi_stop = phi + 2.0d0 * PI / real(nphi_trace, 8)

        fieldline_gradpsi_data(i, 1:3) = [ v(1), v(2), phi ]

        phi_for_interp = phi
        call normalize_phi(phi_for_interp)

        call interpolate_field(v(1), v(2), phi_for_interp, &
                              fieldline_gradpsi_data(i, 4), &
                              fieldline_gradpsi_data(i, 5), &
                              fieldline_gradpsi_data(i, 6), &
                              fieldline_gradpsi_data(i, 12), &
                              fieldline_gradpsi_data(i, 13), &
                              fieldline_gradpsi_data(i, 14), &
                              fieldline_gradpsi_data(i, 15), &
                              fieldline_gradpsi_data(i, 16), &
                              fieldline_gradpsi_data(i, 17), &
                              fieldline_gradpsi_data(i, 18), &
                              fieldline_gradpsi_data(i, 19), &
                              fieldline_gradpsi_data(i, 20),trace_istate)
        if (trace_istate /= 0) then
          if (trace_verbose /= 0) write(*, '(A,I0)') 'Warning: Interpolation error during trace at point ', i
          if (i <= npoints) then
            fieldline_gradpsi_data(i:npoints, 1:20) = 0.0d0
          end if
          exit
        end if

        fieldline_gradpsi_data(i, 7) = sqrt(fieldline_gradpsi_data(i, 4)**2 &
                                          + fieldline_gradpsi_data(i, 5)**2 &
                                          + fieldline_gradpsi_data(i, 6)**2)

        fieldline_gradpsi_data(i, 8) = v(3)
        fieldline_gradpsi_data(i, 9) = v(4)
        fieldline_gradpsi_data(i, 10) = v(5)

        fieldline_gradpsi_data(i, 11) = sqrt(v(3)**2 + v(4)**2 + (v(5)/v(1))**2)
        ! fieldline_gradpsi_data columns:
        ! 1:R, 2:Z, 3:phi, 4:Br, 5:Bz, 6:Bp, 7:|B|,
        ! 8:dpsi/dr, 9:dpsi/dz, 10:dpsi/dphi, 11:|grad_psi|,
        ! 12:dBr/dr, 13:dBr/dz, 14:dBr/dphi,15:dBz/dr, 16:dBz/dz, 17:dBz/dphi,18:dBp/dr, 19:dBp/dz, 20:dBp/dphi
        ! (used later for geodesic curvature calculation)



        call dlsode(gradpsi_ode, neq, v, phi, phi_stop, &
                    itol, rtol, atol, itask, &
                    istate_local, iopt, rwork, lrw, &
                    iwork, liw, jacobian_stub_5d, mf)

        if (istate_local < 0) then
          trace_istate = istate_local
          if (trace_verbose /= 0) write(*, '(A, I0)') 'Warning: LSODE solver returned ISTATE = ', istate_local
          if (i <= npoints) then
            fieldline_gradpsi_data(i:npoints, 1:20) = 0.0d0
          end if
          exit
        end if

        phi = phi_stop

      end do

      deallocate(v, rwork, iwork)

    end subroutine trace_gradpsi_internal

    subroutine normalize_phi(phi_inout)
      implicit none
      real(8), intent(inout) :: phi_inout
      real(8) :: phi_range
      
      phi_range = phimax - phimin
      
      do while (phi_inout > phimax)
        phi_inout = phi_inout - phi_range
      end do
      
      do while (phi_inout < phimin)
        phi_inout = phi_inout + phi_range
      end do
    end subroutine normalize_phi

    subroutine gradpsi_ode(neq, t, v, vdot)
      implicit none
      integer, intent(in) :: neq
      real(8), intent(in) :: t
      real(8), intent(in) :: v(neq)
      real(8), intent(out) :: vdot(neq)
      
      real(8) :: r, z, phi, phi_normalized
      real(8) :: br, bz, bp
      real(8) :: br_r, br_z, br_phi
      real(8) :: bz_r, bz_z, bz_phi
      real(8) :: bp_r, bp_z, bp_phi
      real(8) :: P, G, Q
      real(8) :: zero_threshold
      integer :: trace_istate

      r = v(1)
      z = v(2)
      phi = t
      P = v(3)
      G = v(4)
      Q = v(5)
      !P=∂ψ/∂R, G=∂ψ/∂Z, Q=(∂ψ/∂φ)


      phi_normalized = phi
      call normalize_phi(phi_normalized)

      call interpolate_field(r, z, phi_normalized, &
                            br, bz, bp, &
                            br_r, br_z, br_phi, &
                            bz_r, bz_z, bz_phi, &
                            bp_r, bp_z, bp_phi, trace_istate)

      zero_threshold = 1.0d-15
      if (abs(bp) < zero_threshold) then
        vdot(1:5) = 0.0d0
        return
      end if

      vdot(1) = r * br / bp
      vdot(2) = r * bz / bp

      vdot(3) = (-r/bp)*(br_r*P+((1/r)*bp_r-bp/r**2)*Q+bz_r*G)
      vdot(4) = (-r/bp)*(br_z*P+((1/r)*bp_z)*Q+bz_z*G)
      vdot(5) = (-r/bp)*(br_phi*P+((1/r)*bp_phi)*Q+bz_phi*G)

    end subroutine gradpsi_ode

    subroutine jacobian_stub_5d(neq, t, y, ml, mu, pd, nrowpd)
      implicit none
      integer, intent(in) :: neq, ml, mu, nrowpd
      real(8), intent(in) :: t, y(neq)
      real(8), intent(out) :: pd(nrowpd, neq)
      pd = 0.0d0
      return
    end subroutine jacobian_stub_5d

    subroutine geodesic_curvature_internal(fieldline_gradpsi_data, geocur, Bboundary)
      implicit none

      !========================================================
      ! Input / Output
      !========================================================
      real(8), intent(in)  :: fieldline_gradpsi_data(:, :)
      real(8), intent(out) :: geocur(:)
      real(8), intent(out) :: Bboundary

      !========================================================
      ! Local variables
      !========================================================
      integer :: i, npoints

      real(8) :: r, r_inv
      real(8) :: Br, Bphi, Bz
      real(8) :: dBr_dr, dBr_dz, dBr_dphi
      real(8) :: dBphi_dr, dBphi_dz, dBphi_dphi
      real(8) :: dBz_dr, dBz_dz, dBz_dphi

      real(8) :: Bmag
      real(8) :: bR_n, bphi_n, bZ_n

      ! (b·∇)b
      real(8) :: bdb_R, bdb_phi, bdb_Z

      ! d|B|/d(R,phi,Z)
      real(8) :: dBmag_dr, dBmag_dphi, dBmag_dz

      ! derivatives of b = B/|B|
      real(8) :: dbhatR_dr, dbhatR_dphi, dbhatR_dz
      real(8) :: dbhatphi_dr, dbhatphi_dphi, dbhatphi_dz
      real(8) :: dbhatZ_dr, dbhatZ_dphi, dbhatZ_dz

      ! curvature κ
      real(8) :: kappa_R, kappa_phi, kappa_Z

      ! b × κ (with metric factors)
      real(8) :: bxk_R, bxk_phi, bxk_Z

      ! grad psi
      real(8) :: P, Q, G
      real(8) :: gradpsi_R, gradpsi_phi, gradpsi_Z
      real(8) :: gradpsi_mag

      !========================================================
      ! Initialization
      !========================================================
      npoints   = size(fieldline_gradpsi_data, 1)
      geocur    = 0.0d0
      Bboundary = 0.0d0

      !========================================================
      ! Loop along field line
      !========================================================
      do i = 1, npoints

        !------------------------------------
        ! 1. Coordinates
        !------------------------------------
        r = fieldline_gradpsi_data(i, 1)
        if (r < 1.0d-14) then
          geocur(i) = 0.0d0
          cycle
        end if
        r_inv = 1.0d0 / r

        !------------------------------------
        ! 2. Magnetic field
        !------------------------------------
        Br   = fieldline_gradpsi_data(i, 4)
        Bz   = fieldline_gradpsi_data(i, 5)
        Bphi = fieldline_gradpsi_data(i, 6)

        Bmag = fieldline_gradpsi_data(i, 7)
        if (Bmag < 1.0d-15) then
          geocur(i) = 0.0d0
          cycle
        end if
        
        ! Unit vector b = B / |B|
        bR_n   = Br   / Bmag
        bphi_n = Bphi / Bmag
        bZ_n   = Bz   / Bmag

        ! Accumulate for average B
        Bboundary = Bboundary + Bmag

        !------------------------------------
        ! 3. Magnetic field derivatives
        !------------------------------------
        dBr_dr     = fieldline_gradpsi_data(i, 12)
        dBr_dz     = fieldline_gradpsi_data(i, 13)
        dBr_dphi   = fieldline_gradpsi_data(i, 14)

        dBz_dr     = fieldline_gradpsi_data(i, 15)
        dBz_dz     = fieldline_gradpsi_data(i, 16)
        dBz_dphi   = fieldline_gradpsi_data(i, 17)

        dBphi_dr   = fieldline_gradpsi_data(i, 18)
        dBphi_dz   = fieldline_gradpsi_data(i, 19)
        dBphi_dphi = fieldline_gradpsi_data(i, 20)

        !====================================================
        ! 4. Compute (b·∇)b from b = B/|B| (Nemov Eq.17-compatible form)
        !====================================================
        dBmag_dr   = (Br * dBr_dr     + Bphi * dBphi_dr   + Bz * dBz_dr)   / Bmag
        dBmag_dphi = (Br * dBr_dphi   + Bphi * dBphi_dphi + Bz * dBz_dphi) / Bmag
        dBmag_dz   = (Br * dBr_dz     + Bphi * dBphi_dz   + Bz * dBz_dz)   / Bmag

        dbhatR_dr     = (dBr_dr   - bR_n   * dBmag_dr)   / Bmag
        dbhatR_dphi   = (dBr_dphi - bR_n   * dBmag_dphi) / Bmag
        dbhatR_dz     = (dBr_dz   - bR_n   * dBmag_dz)   / Bmag

        dbhatphi_dr   = (dBphi_dr   - bphi_n * dBmag_dr)   / Bmag
        dbhatphi_dphi = (dBphi_dphi - bphi_n * dBmag_dphi) / Bmag
        dbhatphi_dz   = (dBphi_dz   - bphi_n * dBmag_dz)   / Bmag

        dbhatZ_dr     = (dBz_dr   - bZ_n   * dBmag_dr)   / Bmag
        dbhatZ_dphi   = (dBz_dphi - bZ_n   * dBmag_dphi) / Bmag
        dbhatZ_dz     = (dBz_dz   - bZ_n   * dBmag_dz)   / Bmag

        ! b·∇ = bR_n*∂/∂R + (bphi_n/R)*∂/∂phi + bZ_n*∂/∂Z
        bdb_R = bR_n * dbhatR_dr + bphi_n * r_inv * dbhatR_dphi + bZ_n * dbhatR_dz &
          - bphi_n * bphi_n * r_inv

        bdb_phi = bR_n * dbhatphi_dr + bphi_n * r_inv * dbhatphi_dphi + bZ_n * dbhatphi_dz &
            + bR_n * bphi_n * r_inv

        bdb_Z = bR_n * dbhatZ_dr + bphi_n * r_inv * dbhatZ_dphi + bZ_n * dbhatZ_dz

        !====================================================
        ! 6. Curvature κ = (b·∇)b
        !====================================================
        kappa_R   = bdb_R
        kappa_phi = bdb_phi
        kappa_Z   = bdb_Z

        !====================================================
        ! 7. Compute b × κ WITH METRIC FACTORS
        !====================================================
        ! Cylindrical coordinates: h_R=1, h_phi=R, h_Z=1
        ! (A×B)_R = (A_phi*B_Z - A_Z*B_phi)
        ! (A×B)_phi = (A_Z*B_R - A_R*B_Z)
        ! (A×B)_Z = (A_R*B_phi - A_phi*B_R)
        
        bxk_R   = bphi_n * kappa_Z - bZ_n * kappa_phi
        bxk_phi = bZ_n * kappa_R - bR_n * kappa_Z
        bxk_Z   = bR_n * kappa_phi - bphi_n * kappa_R

        !====================================================
        ! 8. Compute ∇ψ components from Nemov variables
        !====================================================
        ! P = ∂ψ/∂R, G = ∂ψ/∂Z, Q = R*(∂ψ/∂φ)
        P = fieldline_gradpsi_data(i, 8)
        G = fieldline_gradpsi_data(i, 9)
        Q = fieldline_gradpsi_data(i, 10)

        ! Physical components of ∇ψ
        gradpsi_R   = P                     ! ∂ψ/∂R
        gradpsi_phi = Q * r_inv    ! (1/R)∂ψ/∂φ = Q/R  
        gradpsi_Z   = G                     ! ∂ψ/∂Z

        gradpsi_mag = fieldline_gradpsi_data(i, 11)
        if (gradpsi_mag < 1.0d-14) then
          geocur(i) = 0.0d0
          cycle
        end if

        !====================================================
        ! 9. Compute geodesic curvature κ_g = (∇ψ/|∇ψ|)·(b×κ)
        !====================================================
        geocur(i) = ( bxk_R   * gradpsi_R   &
                    + bxk_phi * gradpsi_phi &
                    + bxk_Z   * gradpsi_Z ) / gradpsi_mag

      end do

      !========================================================
      ! 10. Average |B| along field line
      !========================================================
      if (npoints > 0) then
        Bboundary = Bboundary / real(npoints, 8)
      else
        Bboundary = 0.0d0
      end if

      ! ! DEBUG: Print geocur statistics
      ! block
      !   real(8) :: geocur_min, geocur_max, geocur_mean, geocur_std
      !   integer :: ii, n_nonzero
      !   geocur_min = geocur(1)
      !   geocur_max = geocur(1)
      !   geocur_mean = 0.0d0
      !   n_nonzero = 0
        
      !   do ii = 1, npoints
      !     if (abs(geocur(ii)) > 1.0d-20) n_nonzero = n_nonzero + 1
      !     geocur_min = min(geocur_min, geocur(ii))
      !     geocur_max = max(geocur_max, geocur(ii))
      !     geocur_mean = geocur_mean + geocur(ii)
      !   end do
      !   geocur_mean = geocur_mean / real(npoints, 8)
        
      !   geocur_std = 0.0d0
      !   do ii = 1, npoints
      !     geocur_std = geocur_std + (geocur(ii) - geocur_mean)**2
      !   end do
      !   geocur_std = sqrt(geocur_std / real(npoints, 8))
        
      !   write(*, '(A)') '========== DEBUG: geodesic_curvature_internal =========='
      !   write(*, '(A, E15.6)') 'geocur min  = ', geocur_min
      !   write(*, '(A, E15.6)') 'geocur max  = ', geocur_max
      !   write(*, '(A, E15.6)') 'geocur mean = ', geocur_mean
      !   write(*, '(A, E15.6)') 'geocur std  = ', geocur_std
      !   write(*, '(A, I0, A, I0)') 'nonzero geocur: ', n_nonzero, ' / ', npoints
      !   write(*, '(A)') '======================================================'
      ! end block

    end subroutine geodesic_curvature_internal

    subroutine effective_ripple_internal(fieldline_gradpsi_data, geocur, epsilon_eff)
      implicit none
      real(8), intent(in) :: fieldline_gradpsi_data(:, :)
      real(8), intent(in) :: geocur(:)
      real(8), intent(out) :: epsilon_eff

      ! Local variables
      integer :: i, j, k, n_b, n_w, npts
      real(8) :: bmax, bmin, b0, dbp, bp,bphi
      real(8) :: b, ds
      real(8), allocatable :: h_i(:), h_j(:, :), i_j(:, :)
      real(8) :: e1, e2, e3
      real(8) :: r, dphi
      real(8) :: grad_psi
      real(8) :: sqrt_term

      npts = size(fieldline_gradpsi_data, 1)

      if (npts < 2) then
        if (trace_verbose /= 0) write(*, '(A)') 'Error: Not enough data points for ripple calculation'
        epsilon_eff = 0.0d0
        return
      end if
      
      
      n_b = 5000
      n_w = 5000
      dphi = 2.0d0 * PI / real(nphi_trace, 8)
      ds = 0.0d0
      allocate(h_i(n_b), h_j(n_b, n_w), i_j(n_b, n_w))
      h_i(:) = 0.0d0
      h_j(:, :) = 0.0d0
      i_j(:, :) = 0.0d0

      bmax = fieldline_gradpsi_data(1, 7)
      bmin = fieldline_gradpsi_data(1, 7)
      do i = 1, npts
        if (fieldline_gradpsi_data(i, 7) > bmax) bmax = fieldline_gradpsi_data(i, 7)
        if (fieldline_gradpsi_data(i, 7) < bmin) bmin = fieldline_gradpsi_data(i, 7)
      end do
      
      if (bmax < 1.0d-15 .or. bmax <= bmin) then
        if (trace_verbose /= 0) write(*, '(A)') 'Error: Invalid magnetic field range'
        epsilon_eff = 0.0d0
        deallocate(h_i, h_j, i_j)
        return
      end if

      b0 = bmax
      dbp = (bmax - bmin) / (real(n_b - 1, 8) * b0)

      do j = 1, n_b
        h_j(j, :) = 0.0d0
        i_j(j, :) = 0.0d0
        h_i(j) = 0.0d0
        k = 1
        bp = bmin / b0 + real(j - 1, 8) * dbp

        do i = 1, npts
          b = fieldline_gradpsi_data(i, 7)
          r = fieldline_gradpsi_data(i, 1)
          bphi = fieldline_gradpsi_data(i, 6)
          grad_psi = fieldline_gradpsi_data(i, 11)
          
          if (b < 1.0d-15 .or. grad_psi < 1.0d-15 .or. abs(bphi) < 1.0d-15) cycle
          
          if (i < npts) then
            ds = r * b / abs(bphi) * dphi
          else
            ds = 0.0d0
          end if

          if (bp > b / b0) then
            sqrt_term = bp - b / b0
            if (sqrt_term > 0.0d0) then
              h_j(j, k) = h_j(j, k) + 1.0d0 / bp * ds / b * sqrt(sqrt_term) &
                        * (4.0d0 * b0 / b - 1.0d0 / bp) * abs(grad_psi) * geocur(i)
            end if
            
            sqrt_term = 1.0d0 - b / (b0 * bp)
            if (sqrt_term > 0.0d0) then
              i_j(j, k) = i_j(j, k) + ds / b * sqrt(sqrt_term)
            end if
            
            if (i < npts) then
              if (bp < fieldline_gradpsi_data(i+1, 7) / b0) then
                if (i_j(j, k) > 1.0d-15) then
                  h_i(j) = h_i(j) + h_j(j, k)**2 / i_j(j, k)
                end if
                k = k + 1
                if (k > n_w) exit
              end if
            else
              if (i_j(j, k) > 1.0d-15) then
                h_i(j) = h_i(j) + h_j(j, k)**2 / i_j(j, k)
              end if
            end if
          end if
        end do
      end do

      e1 = 0.0d0
      e2 = 0.0d0
      e3 = 0.0d0

      do i = 1, n_b
        e1 = e1 + h_i(i) * dbp
      end do
      
      ! ! DEBUG: Check h_i distribution
      ! block
      !   integer :: n_nonzero_h
      !   real(8) :: h_i_min, h_i_max, h_i_sum_check
      !   n_nonzero_h = 0
      !   h_i_min = 1.0d30
      !   h_i_max = 0.0d0
      !   h_i_sum_check = 0.0d0
        
      !   do i = 1, n_b
      !     if (h_i(i) > 1.0d-20) then
      !       n_nonzero_h = n_nonzero_h + 1
      !       h_i_min = min(h_i_min, h_i(i))
      !       h_i_max = max(h_i_max, h_i(i))
      !     end if
      !     h_i_sum_check = h_i_sum_check + abs(h_i(i))
      !   end do
        
      !   write(*, '(A)') '========== DEBUG: b-prime scan results =========='
      !   write(*, '(A, I0, A, I0)') 'nonzero h_i: ', n_nonzero_h, ' / ', n_b
      !   write(*, '(A, E15.6)') 'h_i min (nonzero) = ', h_i_min
      !   write(*, '(A, E15.6)') 'h_i max          = ', h_i_max
      !   write(*, '(A, E15.6)') 'e1 (before dbp scaling) ≈ ', h_i_sum_check / real(n_b, 8)
      !   write(*, '(A)') '================================================'
      ! end block

      do i = 1, npts
        b = fieldline_gradpsi_data(i, 7)
        r = fieldline_gradpsi_data(i, 1)
        grad_psi = fieldline_gradpsi_data(i, 11)
        bphi = fieldline_gradpsi_data(i, 6)
        
        if (b < 1.0d-15 .or. grad_psi < 1.0d-15 .or. abs(bphi) < 1.0d-15) cycle
        
        if (i < npts) then
          ds = r * b / abs(bphi) * dphi
        else
          ds = 0.0d0
        end if
        
        e2 = e2 + ds / b
        e3 = e3 + ds / b * abs(grad_psi)
      end do

      ! ! DEBUG: Print intermediate values
      ! write(*, '(A)') '========== DEBUG: effective_ripple_internal =========='
      ! write(*, '(A, E15.6)') 'e1 (H^2/I integral over b'') = ', e1
      ! write(*, '(A, E15.6)') 'e2 (integral ds/B)          = ', e2
      ! write(*, '(A, E15.6)') 'e3 (integral ds/B |grad_psi|) = ', e3
      ! write(*, '(A, I0, A, I0)') 'n_b samples processed: ', n_b, ', n_w max width: ', n_w
      ! write(*, '(A)') '======================================================'

      if (e3 > 1.0d-15 .and. e2 > 1.0d-15) then
        epsilon_eff = (e1 * e2 / (e3**2)) * (PI * 1d0**2) / (8.0d0 * sqrt(2.0d0))
      else
        epsilon_eff = 0.0d0
      end if

      deallocate(h_i, h_j, i_j)

    end subroutine effective_ripple_internal


!=======================================================================
! 最终修正版 effective_ripple_internal
! 已修复所有声明位置 + 使用平均磁场 B0 + 正确Gauss映射
!=======================================================================
! subroutine effective_ripple_internal(fieldline_data, geocur, epsilon_eff)
!     implicit none
!     real(8), intent(in)  :: fieldline_data(:, :)
!     real(8), intent(in)  :: geocur(:)
!     real(8), intent(out) :: epsilon_eff

!     integer, parameter :: n_gauss = 64
!     real(8), parameter :: EPS = 1.0d-14
!     real(8), parameter :: SAFETY = 1.03d0
!     real(8), parameter :: PI = 3.141592653589793d0

!     integer :: npts, i, j, nseg
!     real(8) :: b0_ref, bp_min, bp_max, e1, e2, e3, factor
!     real(8) :: r, Bphi, dphi, bp, xi, H2_over_I, H_seg, I_seg

!     real(8), allocatable :: B(:), gp(:), kg(:), ds_over_B(:)
!     real(8), allocatable :: cum_dsB(:), cum_dsB_gp(:)
!     real(8), allocatable :: x_g(:), w_g(:)
!     integer, allocatable :: min_idx(:)

!     npts = size(fieldline_data, 1)
!     epsilon_eff = 0.0d0
!     if (npts < 20) return

!     allocate(B(npts), gp(npts), kg(npts), ds_over_B(npts))
!     allocate(cum_dsB(0:npts), cum_dsB_gp(0:npts))

!     B  = fieldline_data(:,7)
!     gp = fieldline_data(:,11)
!     kg = geocur

!     b0_ref = sum(B) / real(npts,8)          ! 平均磁场

!     bp_min = minval(B) / b0_ref
!     bp_max = maxval(B) / b0_ref * SAFETY

!     ! ====================== ds/B 累积 ======================
!     cum_dsB(0) = 0.0d0
!     cum_dsB_gp(0) = 0.0d0

!     do i = 1, npts-1
!         r    = fieldline_data(i,1)
!         Bphi = fieldline_data(i,6)
!         dphi = 2.0d0 * PI / real(nphi_trace, kind=8)

!         ds_over_B(i) = merge(r * dphi / abs(Bphi), 0.0d0, abs(Bphi)>1.0d-14)

!         cum_dsB(i)    = cum_dsB(i-1)    + ds_over_B(i)
!         cum_dsB_gp(i) = cum_dsB_gp(i-1) + ds_over_B(i) * gp(i)
!     end do
!     cum_dsB(npts)    = cum_dsB(npts-1)
!     cum_dsB_gp(npts) = cum_dsB_gp(npts-1)

!     e2 = cum_dsB(npts)
!     e3 = cum_dsB_gp(npts)

!     if (e2 < EPS .or. e3 < EPS) then
!         deallocate(B, gp, kg, ds_over_B, cum_dsB, cum_dsB_gp)
!         return
!     end if

!     call find_local_minima(B, min_idx, nseg)
!     if (nseg < 2) then
!         deallocate(B, gp, kg, ds_over_B, cum_dsB, cum_dsB_gp, min_idx)
!         return
!     end if

!     allocate(x_g(n_gauss), w_g(n_gauss))
!     call gauss_legendre_01(n_gauss, x_g, w_g)

!     e1 = 0.0d0
!     do j = 1, n_gauss
!         xi = 2.0d0 * x_g(j) - 1.0d0
!         bp = 0.5d0*(bp_max + bp_min) + 0.5d0*(bp_max - bp_min)*xi

!         H2_over_I = 0.0d0
!         do i = 1, nseg-1
!             call integrate_bounce_segment(bp, min_idx(i), min_idx(i+1), &
!                  B, gp, kg, ds_over_B, b0_ref, H_seg, I_seg)
!             if (I_seg > EPS) H2_over_I = H2_over_I + H_seg**2 / I_seg
!         end do

!         e1 = e1 + H2_over_I * w_g(j) * 0.5d0 * (bp_max - bp_min)
!     end do

!     factor = (PI * 1.0d0**2) / (8.0d0 * sqrt(2.0d0))
!     epsilon_eff = (e1 * e2 / (e3**2)) * factor * b0_ref

!     deallocate(B, gp, kg, ds_over_B, cum_dsB, cum_dsB_gp, x_g, w_g, min_idx)

! end subroutine effective_ripple_internal


!=======================================================================
! Gauss-Legendre 节点和权重 (区间 [0,1])
!=======================================================================
subroutine gauss_legendre_01(ng, x, w)
    implicit none
    integer, intent(in)  :: ng
    real(8), intent(out) :: x(ng), w(ng)
    real(8), allocatable :: xt(:), wt(:)

    allocate(xt(ng), wt(ng))
    call gauleg(-1.0d0, 1.0d0, xt, wt, ng)
    
    x = 0.5d0 * (xt + 1.0d0)
    w = 0.5d0 * wt
    
    deallocate(xt, wt)
end subroutine gauss_legendre_01

!=======================================================================
! 经典 Gauss-Legendre (区间 [-1,1])
!=======================================================================
subroutine gauleg(x1, x2, x, w, n)
    implicit none
    real(8), intent(in)  :: x1, x2
    integer, intent(in)  :: n
    real(8), intent(out) :: x(n), w(n)

    integer, parameter :: MAXIT = 20
    real(8), parameter :: EPS = 3.0d-14
    integer :: i, j, k, m
    real(8) :: xm, xl, z, z1, p1, p2, p3, pp

    m = (n + 1)/2
    xm = 0.5d0*(x2 + x1)
    xl = 0.5d0*(x2 - x1)

    do i = 1, m
        z = cos(PI * (real(i,8) - 0.25d0) / (real(n,8) + 0.5d0))
        do j = 1, MAXIT
            p1 = 1.0d0; p2 = 0.0d0
            do k = 1, n
                p3 = p2
                p2 = p1
                p1 = ((2.0d0*k - 1.0d0)*z*p2 - (k-1.0d0)*p3) / real(k,8)
            end do
            pp = n * (z*p1 - p2) / (z*z - 1.0d0)
            z1 = z
            z  = z1 - p1/pp
            if (abs(z - z1) < EPS) exit
        end do
        x(i)     = xm - xl*z
        x(n+1-i) = xm + xl*z
        w(i)     = 2.0d0*xl / ((1.0d0 - z*z)*pp*pp)
        w(n+1-i) = w(i)
    end do
end subroutine gauleg

!=======================================================================
! 查找 B 的局部极小值索引
!=======================================================================
subroutine find_local_minima(B, idx, n)
    real(8), intent(in) :: B(:)
    integer, allocatable, intent(out) :: idx(:)
    integer, intent(out) :: n

    integer :: i, m, count
    m = size(B)
    allocate(idx(m+2))

    idx(1) = 1
    count = 1

    do i = 2, m-1
        if (B(i) < B(i-1) .and. B(i) < B(i+1)) then
            count = count + 1
            idx(count) = i
        end if
    end do

    idx(count+1) = m
    n = count + 1
end subroutine find_local_minima

!=======================================================================
! 单弹跳段积分
!=======================================================================
subroutine integrate_bounce_segment(bp, i1, i2, B, gp, kg, ds_over_B, b0_ref, Hout, Iout)
    real(8), intent(in) :: bp, b0_ref
    integer, intent(in) :: i1, i2
    real(8), intent(in) :: B(:), gp(:), kg(:), ds_over_B(:)
    real(8), intent(out) :: Hout, Iout

    integer :: k
    real(8) :: b_loc, sqrtH, sqrtI, termH, termI

    Hout = 0.0d0
    Iout = 0.0d0

    do k = i1, i2-1
        b_loc = B(k) / b0_ref
        if (bp <= b_loc) cycle

        sqrtH = sqrt(max(0.0d0, bp - b_loc))
        sqrtI = sqrt(max(0.0d0, 1.0d0 - b_loc/bp))

        termH = (1.0d0/bp) * ds_over_B(k) * sqrtH * &
                (4.0d0*b0_ref/B(k) - 1.0d0/bp) * gp(k) * kg(k)

        termI = ds_over_B(k) * sqrtI

        Hout = Hout + termH
        Iout = Iout + termI
    end do
end subroutine
    
end module effective_ripple