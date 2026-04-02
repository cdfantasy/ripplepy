
import numpy as np
from scipy.optimize import root
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
        self.extcur = np.asarray(extcur, dtype=np.float64)
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

