
import numpy as np
from scipy.optimize import root
from scipy.integrate import simpson
from importlib import import_module
from .mgrid import MGrid

try:
    _effective_ripple_mod = import_module(".effective_ripple", __package__)
    Effective_Ripple = _effective_ripple_mod.Effective_Ripple
except ImportError as e:
    print(f"Failed to import Effective_Ripple wrapper: {e}")
    Effective_Ripple = None


_CURRENT_N_EXT_CUR = None
_CURRENT_NTURN = 200
_CURRENT_NPHI = 360


def set_trace_verbose(flag, verbose=True):
    if Effective_Ripple is None:
        raise ImportError("Effective_Ripple was not imported successfully.")
    val = 1 if bool(flag) else 0
    Effective_Ripple.set_trace_verbose(int(val))
    if verbose:
        try:
            cur = int(Effective_Ripple.get_trace_verbose())
        except Exception:
            cur = val
        print(f"✓ trace_verbose set to {cur}")


def get_trace_verbose():
    if Effective_Ripple is None:
        raise ImportError("Effective_Ripple was not imported successfully.")
    return int(Effective_Ripple.get_trace_verbose())


def initialize_mgrid_field(mgrid_filename, nfp, full_torus=True):
    if Effective_Ripple is None:
        raise ImportError("Effective_Ripple was not imported successfully.")
    mgrid = MGrid.from_file(mgrid_filename)
    print(f"✓ Loaded mgrid from '{mgrid_filename}' with shape (nr={mgrid.nr}, nz={mgrid.nz}, nphi={mgrid.nphi})")
    mgrid.expand_to_full_torus(nfp=nfp, full_torus=full_torus)
    phimin = 0.0
    phimax = 2 * np.pi if full_torus else 2 * np.pi / nfp
    Effective_Ripple.initialize_field(
        mgrid.br_arr, mgrid.bz_arr, mgrid.bp_arr,
        mgrid.rmin, mgrid.rmax, mgrid.nr,
        mgrid.zmin, mgrid.zmax, mgrid.nz,
        phimin, phimax, mgrid.nphi, mgrid.n_ext_cur,
    )
    global _CURRENT_N_EXT_CUR
    _CURRENT_N_EXT_CUR = int(mgrid.n_ext_cur)
    return mgrid


def set_extcur(extcur):
    if Effective_Ripple is None:
        raise ImportError("Effective_Ripple was not imported successfully.")
    if extcur is None:
        if _CURRENT_N_EXT_CUR is None:
            raise ValueError("Field not initialized.")
        extcur = np.ones(_CURRENT_N_EXT_CUR, dtype=np.float64)
    extcur_array = np.asarray(extcur, dtype=np.float64)
    Effective_Ripple.sum_bfield_internal(extcur_array)
    return extcur_array


def get_bfield_matrix(extcur, r, z, phi):
    set_extcur(extcur)
    if np.isscalar(r):
        result = Effective_Ripple.interpolate_field(float(r), float(z), float(phi))
        return np.array(result, dtype=np.float64)
    r_arr = np.atleast_1d(r).astype(np.float64)
    z_arr = np.atleast_1d(z).astype(np.float64)
    phi_arr = np.atleast_1d(phi).astype(np.float64)
    if not (len(r_arr) == len(z_arr) == len(phi_arr)):
        raise ValueError("r, z, phi arrays must have same length")
    results = np.zeros((len(r_arr), 13), dtype=np.float64)
    for i in range(len(r_arr)):
        results[i, :] = Effective_Ripple.interpolate_field(r_arr[i], z_arr[i], phi_arr[i])
    return results


def set_trace_parameters(nturn, nphi, npart=5000, verbose=True):
    """Set tracing parameters (nturn, nphi, optional npart) in Fortran backend."""
    if Effective_Ripple is None:
        raise ImportError("Effective_Ripple was not imported successfully.")
    global _CURRENT_NTURN, _CURRENT_NPHI
    _CURRENT_NTURN = int(nturn)
    _CURRENT_NPHI = int(nphi)
    if npart is not None:
        Effective_Ripple.set_trace_parameters(int(nturn), int(nphi), int(npart))
    else:
        Effective_Ripple.set_trace_parameters(int(nturn), int(nphi))
    if verbose:
        part_str = f", npart={npart}" if npart is not None else ""
        print(f"✓ Trace parameters set: nturn={nturn}, nphi={nphi}{part_str}")


def trace_fieldline(initial_rz=None, initial_gradpsi=None, nturn=400, nphi=360, extcur=None, verbose=False):
    if Effective_Ripple is None:
        raise ImportError("Effective_Ripple was not imported successfully.")
    initial_rz = np.asarray(initial_rz, dtype=np.float64)
    if initial_rz.shape != (2,):
        raise ValueError("initial_rz must be shape (2,)")
    if initial_gradpsi is None:
        initial_gradpsi = np.zeros(3, dtype=np.float64, order="F")
    else:
        initial_gradpsi = np.asarray(initial_gradpsi, dtype=np.float64, order="F")
        if initial_gradpsi.shape != (3,):
            raise ValueError("initial_gradpsi must be shape (3,)")
    if extcur is not None:
        set_extcur(extcur)
    set_trace_parameters(nturn, nphi, verbose=False)
    npoints = int(nturn) * int(nphi)
    fieldline_data = np.zeros((npoints, 20), dtype=np.float64, order="F")
    initial_rz_f = np.asfortranarray(np.asarray(initial_rz, dtype=np.float64))
    initial_gradpsi_f = np.asfortranarray(np.asarray(initial_gradpsi, dtype=np.float64))
    trace_istate = Effective_Ripple.trace_gradpsi_internal(fieldline_data, initial_rz_f, initial_gradpsi_f)
    if trace_istate != 0 and verbose:
        print(f"✗ Field line tracing failed with istate={trace_istate}")
    return fieldline_data, trace_istate


def compute_epstot(initial_rz, initial_gradpsi=None, return_fieldline=False, verbose=True):
    """Compute ε_eff^(3/2).  nturn/nphi/npart from prior set_trace_parameters()."""
    if Effective_Ripple is None:
        raise ImportError("Effective_Ripple was not imported successfully.")
    nturn = _CURRENT_NTURN
    nphi = _CURRENT_NPHI
    initial_rz_array = np.asarray(initial_rz, dtype=np.float64)
    if initial_gradpsi is None:
        initial_gradpsi_array = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        initial_gradpsi_array = np.asarray(initial_gradpsi, dtype=np.float64)
    npoints_val = nturn * nphi
    fieldline_data = np.zeros((npoints_val, 20), dtype=np.float64, order='F')
    initial_rz_f = np.asfortranarray(initial_rz_array)
    initial_gradpsi_f = np.asfortranarray(initial_gradpsi_array)
    result = Effective_Ripple.compute_ripple(
        initial_rz_f, initial_gradpsi_f, fieldline_data,
    )
    epsilon_eff, Bboundary_val, R0, trace_istate = result
    if trace_istate != 0:
        if verbose:
            print(f"✗ Field line tracing failed with istate={trace_istate}")
        if return_fieldline:
            return None, 0.0, np.zeros((0, 20)), trace_istate
        return None, 0.0, trace_istate
    Bboundary = np.array([Bboundary_val], dtype=np.float64)
    if verbose:
        print(f"✓ ε_eff^(3/2) = {epsilon_eff:.6e}  (R0={R0:.4f})")
    if return_fieldline:
        return epsilon_eff, Bboundary, fieldline_data, trace_istate
    else:
        return epsilon_eff, Bboundary, trace_istate


def calculate_plasma_params(fieldline_data, axis_data, nturn, nphi, Rm):
    R = fieldline_data[:, 0].reshape((nturn, nphi)).T
    Z = fieldline_data[:, 1].reshape((nturn, nphi)).T
    Phi = fieldline_data[:, 2].reshape((nturn, nphi)).T
    R_axis = axis_data[:nphi, 0].reshape(-1, 1)
    Z_axis = axis_data[:nphi, 1].reshape(-1, 1)
    thetas = np.arctan2(Z - Z_axis, R - R_axis)
    R_sorted = np.zeros_like(R)
    Z_sorted = np.zeros_like(Z)
    for i in range(nphi):
        idx = np.argsort(thetas[i, :])
        R_sorted[i, :] = R[i, idx]
        Z_sorted[i, :] = Z[i, idx]
    R_next = np.roll(R_sorted, -1, axis=1)
    Z_next = np.roll(Z_sorted, -1, axis=1)
    vol_sum = np.sum((R_next - R_sorted) * (R_next + R_sorted) * (Z_next + Z_sorted))
    volume = abs(vol_sum) * (np.pi / nphi)
    am = np.sqrt(volume / (2 * np.pi**2 * Rm))
    d_theta = np.unwrap(thetas[:, 0])
    d_phi = np.unwrap(Phi[:, 0])
    iota = (d_theta[-1] - d_theta[0]) / (d_phi[-1] - d_phi[0]) if len(d_phi) > 1 else 0.0
    return volume, am, abs(iota)


def plot_fieldline_3d(fieldline_data, color_by_b=True, colorscale="Viridis", line_width=4, title=None):
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError("plotly required") from exc
    data = np.asarray(fieldline_data, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] < 7:
        raise ValueError("fieldline_data must be 2D with >=7 columns")
    r_lines, z_lines, phi_lines, b_line = data[:, 0], data[:, 1], data[:, 2], data[:, 6]
    x_lines = r_lines * np.cos(phi_lines)
    y_lines = r_lines * np.sin(phi_lines)
    kw = {"width": line_width}
    if color_by_b:
        kw.update({"color": b_line, "colorscale": colorscale, "colorbar": {"title": "|B| (T)"}})
    fig = go.Figure(data=[go.Scatter3d(x=x_lines, y=y_lines, z=z_lines, mode="lines", line=kw)])
    fig.update_layout(scene={"xaxis_title": "X", "yaxis_title": "Y", "zaxis_title": "Z"},
                      title=title or "Fieldline", scene_aspectmode="data")
    return fig


def compute_initial_gradpsi_nemov(extcur, R0, Z0, phi0=0.0, verbose=True):
    try:
        b = get_bfield_matrix(extcur, R0, Z0, phi0)
        Br0, Bz0, Bphi0 = b[0], b[1], b[2]
    except Exception as exc:
        if verbose:
            print(f"Error: {exc}")
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    N_R, N_phi, N_Z = -Bphi0, Br0, 0.0
    nrm = np.sqrt(N_R**2 + N_phi**2 + N_Z**2)
    if nrm < 1e-15:
        n_R, n_phi, n_Z = 1.0, 0.0, 0.0
    else:
        n_R, n_phi, n_Z = N_R / nrm, N_phi / nrm, N_Z / nrm
    return np.array([n_R, n_Z, R0**2 * n_phi], dtype=np.float64)


def find_axis(initial_rz, xtol=1e-10, max_iter=200, delta_r=0.01, verbose=False):
    from scipy.optimize import root
    if verbose:
        print(f"\nSearching for magnetic axis...")
    def axis_residual(candidate_rz):
        candidate_rz = np.array(candidate_rz, dtype=np.float64, order='F')
        fld, ist = trace_fieldline(initial_rz=candidate_rz, initial_gradpsi=None, nturn=2, nphi=360, verbose=False)
        if ist != 0:
            return np.array([1e10, 1e10])
        return np.array([fld[360, 0] - candidate_rz[0], fld[360, 1] - candidate_rz[1]])
    trial_points = [
        np.array(initial_rz, dtype=np.float64),
        np.array([initial_rz[0] + delta_r, initial_rz[1]], dtype=np.float64),
        np.array([initial_rz[0] - delta_r, initial_rz[1]], dtype=np.float64),
    ]
    trial_results = sorted(
        [(np.linalg.norm(axis_residual(p)), p) for p in trial_points],
        key=lambda x: x[0],
    )
    start_rz = trial_results[0][1]
    result = root(axis_residual, start_rz, method='hybr', tol=xtol, options={'maxfev': max_iter, 'factor': 100})
    if result.success:
        fld, ist = trace_fieldline(initial_rz=result.x, nturn=2, nphi=360, verbose=False)
    else:
        ist = -1
        fld = None
    if fld is not None and ist == 0:
        R0 = np.mean(np.sqrt(fld[:360, 0]**2 + fld[:360, 1]**2))
        return result.x, R0, fld, True
    else:
        return None, None, None, False
