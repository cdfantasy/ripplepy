
import numpy as np
from scipy.optimize import root
from scipy.integrate import simpson
from importlib import import_module
from .mgrid import MGrid


try:
    # Prefer the f90wrap-generated high-level API.
    _effective_ripple_mod = import_module(".effective_ripple", __package__)
    Effective_Ripple = _effective_ripple_mod.Effective_Ripple
except ImportError as e:
    print(f"Failed to import Effective_Ripple wrapper: {e}")
    Effective_Ripple = None


class MagneticField:
    """Wrap the compiled ripple field state and expose an object-oriented API."""

    def __init__(
        self,
        mgrid,
        extcur,
        full_torus=True,
        phimin=0.0,
        phimax=None,
    ):
        if Effective_Ripple is None:
            raise ImportError("Effective_Ripple was not imported successfully.")

        self.mgrid = mgrid
        self.full_torus = full_torus
        self.phimin = phimin
        self.phimax = phimax if phimax is not None else (2 * np.pi if full_torus else 2 * np.pi / mgrid.nfp)
        self.nr = mgrid.nr
        self.nz = mgrid.nz
        self.nphi = mgrid.nphi
        self.rmin = mgrid.rmin
        self.rmax = mgrid.rmax
        self.zmin = mgrid.zmin
        self.zmax = mgrid.zmax
        self.n_ext_cur = mgrid.n_ext_cur
        self.extcur = np.asarray(extcur, dtype=np.float64) if extcur is not None else 1.0 * np.ones(mgrid.n_ext_cur, dtype=np.float64)

    @classmethod
    def from_mgrid_file(cls, mgrid_filename, extcur, nfp, full_torus=True):
        """Create a MagneticField instance from an mgrid file and initialize the compiled backend."""
        mgrid = MGrid.from_file(mgrid_filename)
        mgrid.expand_to_full_torus(nfp=nfp, full_torus=full_torus)
        field = cls(mgrid, extcur, full_torus=full_torus)
        field.initialize_backend()
        return field

    def initialize_backend(self):
        """Load the grid into the compiled ripple backend."""
        Effective_Ripple.initialize_field(
            self.mgrid.br_arr,
            self.mgrid.bz_arr,
            self.mgrid.bp_arr,
            self.rmin,
            self.rmax,
            self.nr,
            self.zmin,
            self.zmax,
            self.nz,
            self.phimin,
            self.phimax,
            self.nphi,
            self.mgrid.n_ext_cur,
        )
        self.set_extcur(self.extcur)
        return self

    def set_extcur(self, extcur):
        """Update the active external current set used by the backend."""
        self.extcur = np.asarray(extcur, dtype=np.float64)
        Effective_Ripple.sum_bfield_internal(self.extcur)
        return self

    def interpolate(self, r, z, phi):
        """Interpolate the magnetic field and derivatives at one or many points."""
        if Effective_Ripple is None:
            raise ImportError("Effective_Ripple was not imported successfully.")

        Effective_Ripple.sum_bfield_internal(self.extcur)

        if np.isscalar(r):
            result = Effective_Ripple.interpolate_field(float(r), float(z), float(phi))
            return np.array(result, dtype=np.float64)

        r_arr = np.atleast_1d(r).astype(np.float64)
        z_arr = np.atleast_1d(z).astype(np.float64)
        phi_arr = np.atleast_1d(phi).astype(np.float64)

        if not (len(r_arr) == len(z_arr) == len(phi_arr)):
            raise ValueError("r, z, and phi arrays must have the same length")

        results = np.zeros((len(r_arr), 12), dtype=np.float64)
        for index in range(len(r_arr)):
            results[index, :] = Effective_Ripple.interpolate_field(
                r_arr[index], z_arr[index], phi_arr[index]
            )
        return results

    def field_matrix(self, r, z, phi):
        """Alias for interpolate to keep the physical meaning explicit."""
        return self.interpolate(r, z, phi)

    def __repr__(self):
        return (
            "MagneticField(" 
            f"nr={self.nr}, nz={self.nz}, nphi={self.nphi}, "
            f"n_ext_cur={self.n_ext_cur}, full_torus={self.full_torus})"
        )

class FieldLineTracer:
    """Trace field lines using the compiled ripple backend."""

    def __init__(self, magnetic_field, nturn=2, nphi=360):
        if Effective_Ripple is None:
            raise ImportError("Effective_Ripple was not imported successfully.")
        self.magnetic_field = magnetic_field
        self.nturn = int(nturn)
        self.nphi = int(nphi)

    def set_trace_parameters(self, nturn=None, nphi=None):
        if nturn is not None:
            self.nturn = int(nturn)
        if nphi is not None:
            self.nphi = int(nphi)
        print(f"Trace parameters set: nturn={self.nturn}, nphi={self.nphi}")
        return self

    def trace(self, initial_rz, initial_gradpsi=None):
        """Trace a field line and return the raw Fortran output matrix."""
        initial_rz = np.asarray(initial_rz, dtype=np.float64)
        if initial_rz.shape != (2,):
            raise ValueError("initial_rz must contain exactly two values: (R, Z)")

        if initial_gradpsi is None:
            initial_gradpsi = np.zeros(3, dtype=np.float64, order="F")
        else:
            initial_gradpsi = np.asarray(initial_gradpsi, dtype=np.float64, order="F")
            if initial_gradpsi.shape != (3,):
                raise ValueError("initial_gradpsi must contain exactly three values")

        self.magnetic_field.set_extcur(self.magnetic_field.extcur)
        Effective_Ripple.set_trace_parameters(self.nturn, self.nphi)

        fieldline_data = np.zeros((self.nturn * self.nphi, 20), dtype=np.float64, order="F")
        Effective_Ripple.trace_gradpsi_internal(fieldline_data, initial_rz, initial_gradpsi)
        return fieldline_data

class epsilon_eff:
    """Python-side post-processing for Nemov-style ripple diagnostics.

    Integration strategy:
    1) Core smooth parts: Newton-Cotes (Simpson).
    2) Near turning-point endpoints: Gauss-Legendre quadrature.
    3) Endpoint asymptotics: linearized analytic correction.
    """

    def __init__(self, magnetic_field, nturn=2, nphi=360, gl_order=8, nc_points=33, endpoint_fraction=0.12):
        if Effective_Ripple is None:
            raise ImportError("Effective_Ripple was not imported successfully.")
        self.magnetic_field = magnetic_field
        self.nturn = int(nturn)
        self.nphi = int(nphi)
        self.gl_order = int(gl_order)
        self.nc_points = int(max(5, nc_points))
        self.endpoint_fraction = float(endpoint_fraction)
        self._tiny = 1.0e-14

    def set_trace_parameters(self, nturn=None, nphi=None):
        if nturn is not None:
            self.nturn = int(nturn)
        if nphi is not None:
            self.nphi = int(nphi)
        print(f"Trace parameters set: nturn={self.nturn}, nphi={self.nphi}")
        return self

    def trace(self, initial_rz, initial_gradpsi=None):
        initial_rz = np.asarray(initial_rz, dtype=np.float64)
        if initial_rz.shape != (2,):
            raise ValueError("initial_rz must contain exactly two values: (R, Z)")

        if initial_gradpsi is None:
            initial_gradpsi = np.zeros(3, dtype=np.float64, order="F")
        else:
            initial_gradpsi = np.asarray(initial_gradpsi, dtype=np.float64, order="F")
            if initial_gradpsi.shape != (3,):
                raise ValueError("initial_gradpsi must contain exactly three values")

        self.magnetic_field.set_extcur(self.magnetic_field.extcur)
        Effective_Ripple.set_trace_parameters(self.nturn, self.nphi)

        fieldline_data = np.zeros((self.nturn * self.nphi, 20), dtype=np.float64, order="F")
        Effective_Ripple.trace_gradpsi_internal(fieldline_data, initial_rz, initial_gradpsi)
        return fieldline_data

    def _validate_fieldline_data(self, fieldline_data):
        arr = np.asarray(fieldline_data, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] < 20:
            raise ValueError("fieldline_data must be a 2D array with at least 20 columns")
        if arr.shape[0] < 4:
            raise ValueError("fieldline_data must contain at least 4 points")
        return arr

    def compute_initial_gradpsi(self, initial_rz):
        
        r0, z0 = initial_rz
        Br0, Bz0, Bphi0 = self.magnetic_field.interpolate(r0, z0, 0.0)[:3]
        B_mag0 = np.sqrt(Br0**2 + Bz0**2 + Bphi0**2)
        gradpsi_r, gradpsi_z, gradpsi_phi = [1,0,0]
        return np.array([gradpsi_r, gradpsi_z, gradpsi_phi], dtype=np.float64)

    def compute_ripple_python(
        self,
        initial_rz,
        initial_gradpsi=None,
        major_radius=1.0,
        n_b=1200,
        stel_sym=True,
    ):
        """Trace with Fortran and post-process geodesic curvature and epsilon in Python."""
        if initial_gradpsi is None and stel_sym:
            initial_gradpsi = self.compute_initial_gradpsi(initial_rz)

        fieldline_data = self.trace(initial_rz, initial_gradpsi=initial_gradpsi)
        geocur, bboundary = self.compute_geodesic_curvature_literature(fieldline_data)
        epsilon_value = self.compute_effective_ripple_literature(
            fieldline_data,
            geocur,
            major_radius=major_radius,
            n_b=n_b,
        )
        return epsilon_value, bboundary, fieldline_data, geocur

    @staticmethod
    def _split_fieldline_columns(data):
        """Return fieldline columns using Fortran's documented 1-based layout.

        1:R, 2:Z, 3:phi, 4:Br, 5:Bz, 6:Bphi, 7:|B|,
        8:P, 9:G, 10:Q, 11:|grad_psi|,
        12:dBr/dR, 13:dBr/dZ, 14:dBr/dphi,
        15:dBz/dR, 16:dBz/dZ, 17:dBz/dphi,
        18:dBphi/dR, 19:dBphi/dZ, 20:dBphi/dphi.
        """
        return {
            "r": data[:, 0],
            "z": data[:, 1],
            "phi": data[:, 2],
            "br": data[:, 3],
            "bz": data[:, 4],
            "bphi": data[:, 5],
            "bmag": data[:, 6],
            "p": data[:, 7],
            "g": data[:, 8],
            "q": data[:, 9],
            "gradpsi_mag": data[:, 10],
            "dbr_dr": data[:, 11],
            "dbr_dz": data[:, 12],
            "dbr_dphi": data[:, 13],
            "dbz_dr": data[:, 14],
            "dbz_dz": data[:, 15],
            "dbz_dphi": data[:, 16],
            "dbphi_dr": data[:, 17],
            "dbphi_dz": data[:, 18],
            "dbphi_dphi": data[:, 19],
        }

    def _gauss_legendre(self, func, a, b):
        if b <= a:
            return 0.0
        nodes, weights = np.polynomial.legendre.leggauss(self.gl_order)
        x = 0.5 * (b - a) * nodes + 0.5 * (a + b)
        return 0.5 * (b - a) * np.sum(weights * func(x))

    def _composite_newton_cotes(self, func, a, b):
        if b <= a:
            return 0.0
        x = np.linspace(a, b, self.nc_points)
        y = func(x)
        return float(simpson(y, x=x))

    def _build_path_coordinate(self, fieldline_data):
        cols = self._split_fieldline_columns(fieldline_data)
        phi = cols["phi"]
        r = cols["r"]
        bmag = np.maximum(cols["bmag"], self._tiny)
        bphi = cols["bphi"]

        dphi = np.abs(np.diff(phi))
        if dphi.size == 0:
            return np.array([0.0], dtype=np.float64)

        step = np.abs(r[:-1]) * bmag[:-1] * dphi / np.maximum(np.abs(bphi[:-1]), self._tiny)
        step = np.where(np.isfinite(step), step, 0.0)

        s = np.zeros(fieldline_data.shape[0], dtype=np.float64)
        s[1:] = np.cumsum(step)
        if s[-1] <= self._tiny:
            s = np.linspace(0.0, 1.0, fieldline_data.shape[0], dtype=np.float64)
        return s

    def compute_geodesic_curvature_literature(self, fieldline_data):
        """Compute k_g using Eq. (17): k_g = (grad(psi-hat))·(b x kappa)."""
        data = self._validate_fieldline_data(fieldline_data)
        cols = self._split_fieldline_columns(data)

        r = cols["r"]
        br = cols["br"]
        bz = cols["bz"]
        bphi = cols["bphi"]
        bmag = np.maximum(cols["bmag"], self._tiny)

        p = cols["p"]
        g = cols["g"]
        q = cols["q"]
        gradpsi_mag = np.maximum(cols["gradpsi_mag"], self._tiny)

        dbr_dr, dbr_dz, dbr_dphi = cols["dbr_dr"], cols["dbr_dz"], cols["dbr_dphi"]
        dbz_dr, dbz_dz, dbz_dphi = cols["dbz_dr"], cols["dbz_dz"], cols["dbz_dphi"]
        dbphi_dr, dbphi_dz, dbphi_dphi = cols["dbphi_dr"], cols["dbphi_dz"], cols["dbphi_dphi"]

        inv_b = 1.0 / bmag
        r_safe = np.maximum(r, self._tiny)
        r_inv = 1.0 / r_safe

        bR = br * inv_b
        bZ = bz * inv_b
        bPhi = bphi * inv_b

        dBmag_dr = (br * dbr_dr + bz * dbz_dr + bphi * dbphi_dr) * inv_b
        dBmag_dz = (br * dbr_dz + bz * dbz_dz + bphi * dbphi_dz) * inv_b
        dBmag_dphi = (br * dbr_dphi + bz * dbz_dphi + bphi * dbphi_dphi) * inv_b

        inv_b2 = inv_b * inv_b
        dbR_dr = dbr_dr * inv_b - br * dBmag_dr * inv_b2
        dbR_dz = dbr_dz * inv_b - br * dBmag_dz * inv_b2
        dbR_dphi = dbr_dphi * inv_b - br * dBmag_dphi * inv_b2

        dbZ_dr = dbz_dr * inv_b - bz * dBmag_dr * inv_b2
        dbZ_dz = dbz_dz * inv_b - bz * dBmag_dz * inv_b2
        dbZ_dphi = dbz_dphi * inv_b - bz * dBmag_dphi * inv_b2

        dbPhi_dr = dbphi_dr * inv_b - bphi * dBmag_dr * inv_b2
        dbPhi_dz = dbphi_dz * inv_b - bphi * dBmag_dz * inv_b2
        dbPhi_dphi = dbphi_dphi * inv_b - bphi * dBmag_dphi * inv_b2

        bdb_R = bR * dbR_dr + (bPhi * r_inv) * dbR_dphi + bZ * dbR_dz - (bPhi * bPhi) * r_inv
        bdb_Phi = bR * dbPhi_dr + (bPhi * r_inv) * dbPhi_dphi + bZ * dbPhi_dz + (bR * bPhi) * r_inv
        bdb_Z = bR * dbZ_dr + (bPhi * r_inv) * dbZ_dphi + bZ * dbZ_dz

        bxk_R = bPhi * bdb_Z - bZ * bdb_Phi
        bxk_Phi = bZ * bdb_R - bR * bdb_Z
        bxk_Z = bR * bdb_Phi - bPhi * bdb_R

        gradpsi_R = p
        gradpsi_Phi = q * r_inv
        gradpsi_Z = g

        geocur = (bxk_R * gradpsi_R + bxk_Phi * gradpsi_Phi + bxk_Z * gradpsi_Z) / gradpsi_mag
        geocur = np.where(np.isfinite(geocur), geocur, 0.0)

        bboundary = float(np.mean(bmag))
        return geocur.astype(np.float64), bboundary

    def _find_wells(self, s, x, bp):
        wells = []
        inside = bool(x[0] < bp)
        start = s[0] if inside else None

        for i in range(len(x) - 1):
            x0 = x[i]
            x1 = x[i + 1]
            s0 = s[i]
            s1 = s[i + 1]

            if (x0 - bp) * (x1 - bp) < 0.0:
                t = (bp - x0) / (x1 - x0)
                sc = s0 + t * (s1 - s0)
                if inside:
                    if start is not None and sc > start:
                        wells.append((start, sc))
                    inside = False
                    start = None
                else:
                    inside = True
                    start = sc

        if inside and start is not None and s[-1] > start:
            wells.append((start, s[-1]))
        return wells

    def _integrate_turning_interval(self, s, x, weight, bp, left, right, mode):
        if right <= left:
            return 0.0

        total_length = right - left
        delta = min(self.endpoint_fraction * total_length, 0.5 * total_length)

        def interp_x(u):
            return np.interp(u, s, x)

        def interp_w(u):
            return np.interp(u, s, weight)

        def kernel(u):
            x_u = interp_x(u)
            if mode == "h":
                return np.sqrt(np.maximum(bp - x_u, 0.0))
            return np.sqrt(np.maximum(1.0 - x_u / bp, 0.0))

        def full_integrand(u):
            return interp_w(u) * kernel(u)

        def endpoint_cap(is_left):
            if delta <= self._tiny:
                return 0.0
            if is_left:
                a, b = left, left + delta
                edge = left
                probe = min(edge + 1.0e-4 * total_length, right)
                c = max((bp - interp_x(probe)) / max(probe - edge, self._tiny), self._tiny)

                def lin_part(u):
                    du = u - edge
                    if mode == "h":
                        return np.sqrt(np.maximum(c * du, 0.0))
                    return np.sqrt(np.maximum((c / bp) * du, 0.0))
            else:
                a, b = right - delta, right
                edge = right
                probe = max(edge - 1.0e-4 * total_length, left)
                c = max((bp - interp_x(probe)) / max(edge - probe, self._tiny), self._tiny)

                def lin_part(u):
                    du = edge - u
                    if mode == "h":
                        return np.sqrt(np.maximum(c * du, 0.0))
                    return np.sqrt(np.maximum((c / bp) * du, 0.0))

            w0 = float(interp_w(edge))
            lin_integrand = lambda u: w0 * lin_part(u)

            i_num = self._gauss_legendre(full_integrand, a, b)
            i_lin_num = self._gauss_legendre(lin_integrand, a, b)

            if mode == "h":
                i_lin_exact = w0 * (2.0 / 3.0) * np.sqrt(c) * (delta ** 1.5)
            else:
                i_lin_exact = w0 * (2.0 / 3.0) * np.sqrt(c / bp) * (delta ** 1.5)

            return i_num + (i_lin_exact - i_lin_num)

        left_cap = endpoint_cap(is_left=True)
        right_cap = endpoint_cap(is_left=False)

        core_a = left + delta
        core_b = right - delta
        core = self._composite_newton_cotes(full_integrand, core_a, core_b) if core_b > core_a else 0.0
        return float(left_cap + core + right_cap)

    def _integrate_bp_hybrid(self, bp, h):
        bp = np.asarray(bp, dtype=np.float64)
        h = np.asarray(h, dtype=np.float64)
        if bp.size < 5:
            return float(np.trapz(h, bp))

        a = bp[0]
        b = bp[-1]
        if b <= a:
            return 0.0

        span = b - a
        delta = min(0.08 * span, 0.5 * span)

        def interp_h(x):
            return np.interp(x, bp, h)

        left_gauss = self._gauss_legendre(interp_h, a, a + delta)
        right_gauss = self._gauss_legendre(interp_h, b - delta, b)

        nfit = min(8, bp.size)

        xL = bp[:nfit] - a
        yL = np.maximum(h[:nfit], 0.0)
        AL = np.column_stack([np.sqrt(np.maximum(xL, 0.0)), xL])
        cL, _, _, _ = np.linalg.lstsq(AL, yL, rcond=None)
        left_analytic = (2.0 / 3.0) * cL[0] * (delta ** 1.5) + 0.5 * cL[1] * (delta ** 2)

        xR = b - bp[-nfit:]
        yR = np.maximum(h[-nfit:], 0.0)
        AR = np.column_stack([np.sqrt(np.maximum(xR, 0.0)), xR])
        cR, _, _, _ = np.linalg.lstsq(AR, yR, rcond=None)
        right_analytic = (2.0 / 3.0) * cR[0] * (delta ** 1.5) + 0.5 * cR[1] * (delta ** 2)

        left_corr = left_analytic - left_gauss
        right_corr = right_analytic - right_gauss

        mid_mask = (bp >= a + delta) & (bp <= b - delta)
        if np.count_nonzero(mid_mask) >= 3:
            mid_val = simpson(h[mid_mask], x=bp[mid_mask])
        else:
            mid_val = np.trapz(h, bp)

        return float(mid_val + left_gauss + right_gauss + left_corr + right_corr)

    def compute_effective_ripple_literature(
        self,
        fieldline_data,
        geocur,
        major_radius=1.0,
        n_b=1200,
    ):
        """Compute epsilon_eff with Eq. (29), H with Eq. (30), I with Eq. (31)."""
        data = self._validate_fieldline_data(fieldline_data)
        cols = self._split_fieldline_columns(data)
        geocur = np.asarray(geocur, dtype=np.float64)
        if geocur.shape[0] != data.shape[0]:
            raise ValueError("geocur length must match fieldline_data rows")

        b = np.maximum(cols["bmag"], self._tiny)
        gradpsi = np.maximum(np.abs(cols["gradpsi_mag"]), self._tiny)

        b0 = float(np.max(b))
        if b0 <= self._tiny:
            return 0.0

        x = b / b0
        bp_min = max(float(np.min(x)) + 1.0e-8, 1.0e-8)
        bp_max = 1.0 - 1.0e-8
        if bp_max <= bp_min:
            return 0.0

        s = self._build_path_coordinate(data)

        inv_b = 1.0 / b
        # Eq. (30) integrand base: (dl/B) * |grad psi| * k_g * (...) * sqrt(...)
        weight_h_base = inv_b * np.abs(gradpsi) * geocur
        # Eq. (31) integrand base: (dl/B) * sqrt(...)
        weight_i_base = inv_b

        bp_grid = np.linspace(bp_min, bp_max, int(max(100, n_b)), dtype=np.float64)
        h_i = np.zeros_like(bp_grid)

        for j, bp in enumerate(bp_grid):
            wells = self._find_wells(s, x, bp)
            if not wells:
                continue

            # Eq. (30): prefactor in front of sqrt term.
            pref = (1.0 / bp) * (4.0 / np.maximum(x, self._tiny) - 1.0 / bp)
            weight_h = weight_h_base * pref
            total_h = 0.0

            for left, right in wells:
                if right - left <= self._tiny:
                    continue

                h_j = self._integrate_turning_interval(
                    s,
                    x,
                    weight_h,
                    bp,
                    left,
                    right,
                    mode="h",
                )
                i_j = self._integrate_turning_interval(
                    s,
                    x,
                    weight_i_base,
                    bp,
                    left,
                    right,
                    mode="i",
                )

                if i_j > self._tiny:
                    total_h += (h_j * h_j) / i_j

            h_i[j] = total_h

        e1 = self._integrate_bp_hybrid(bp_grid, h_i)
        e2 = float(simpson(inv_b, x=s))
        e3 = float(simpson(inv_b * gradpsi, x=s))

        if e2 <= self._tiny or e3 <= self._tiny:
            return 0.0

        # Eq. (29)
        epsilon_local = (e1 * e2 / (e3 * e3)) * (np.pi / (8.0 * np.sqrt(2.0)))
        return float((major_radius ** 2) * epsilon_local)


class AxisFinder:
    """Find the magnetic axis by repeatedly calling FieldLineTracer.trace."""

    def __init__(self, tracer):
        self.tracer = tracer

    def find_axis(self, initial_rz, xtol=1e-10, max_iter=200, timeout=10.0):
        from scipy.optimize import root

        initial_rz = np.asarray(initial_rz, dtype=np.float64)
        if initial_rz.shape != (2,):
            raise ValueError("initial_rz must contain exactly two values: (R, Z)")

        print("\nSearching for magnetic axis...")
        print(f"  Initial guess: R={initial_rz[0]:.6f}, Z={initial_rz[1]:.6f}")
        print(f"  Timeout: {timeout:.1f} s")

        final_fieldline_data = None

        def axis_residual(current_rz):
            nonlocal final_fieldline_data
            try:
                fieldline_data = self.tracer.trace(current_rz, initial_gradpsi=np.zeros(3, dtype=np.float64, order="F"))
                final_fieldline_data = fieldline_data.copy()
                final_index = self.tracer.nphi
                return np.array(
                    [
                        fieldline_data[final_index, 0] - current_rz[0],
                        fieldline_data[final_index, 1] - current_rz[1],
                    ],
                    dtype=np.float64,
                )
            except Exception as exc:
                print(
                    f"  Warning: fieldline tracing failed at R={current_rz[0]:.6f}, "
                    f"Z={current_rz[1]:.6f}: {exc}"
                )
                return np.array([1e10, 1e10], dtype=np.float64)

        def run_solver():
            return root(
                axis_residual,
                initial_rz,
                method="hybr",
                tol=xtol,
                options={
                    "maxfev": max_iter * (len(initial_rz) + 1),
                    "factor": 100,
                },
            )

        if timeout is None or timeout <= 0:
            result = run_solver()
        else:
            import signal

            class AxisTimeoutError(Exception):
                pass

            def timeout_handler(signum, frame):
                raise AxisTimeoutError("Magnetic axis search timed out")

            if not hasattr(signal, "SIGALRM"):
                print("Warning: signal-based timeout is not available on this platform; running without timeout.")
                result = run_solver()
            else:
                previous_handler = signal.getsignal(signal.SIGALRM)
                previous_alarm = signal.alarm(0)
                try:
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(int(np.ceil(timeout)))
                    result = run_solver()
                except AxisTimeoutError:
                    print(f"  ✗ Timeout: magnetic axis search exceeded {timeout:.1f} s")
                    return np.array([np.nan, np.nan]), np.nan, None, True
                except Exception as exc:
                    print(f"  ✗ Error during axis search: {exc}")
                    return np.array([np.nan, np.nan]), np.nan, None, True
                finally:
                    signal.alarm(previous_alarm)
                    signal.signal(signal.SIGALRM, previous_handler)

        distance = np.linalg.norm(result.fun)

        if final_fieldline_data is not None:
            major_radius = np.mean(
                np.sqrt(final_fieldline_data[:self.tracer.nphi, 0] ** 2 + final_fieldline_data[:self.tracer.nphi, 1] ** 2)
            )
        else:
            major_radius = np.nan

        print("  Optimization completed:")
        print(f"    Axis position: R={result.x[0]:.10f}, Z={result.x[1]:.10f}")
        print(f"    Major radius R0: {major_radius:.10f}")
        print(f"    Distance error: {distance:.2e}")
        print(f"    Converged: {result.success}")

        trace_error_flag = (final_fieldline_data is None) or (not result.success)
        return result.x, major_radius, final_fieldline_data, trace_error_flag

