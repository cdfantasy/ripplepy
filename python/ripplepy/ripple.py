
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


def fieldline_smoothness_flag(fieldline_data, nturn, nphi,
                               drift_r_frac=0.02, drift_z_abs=0.02,
                               max_step=0.10):
    """Fast smoothness / confinement check for a traced field line.

    This is intentionally a cheap O(npoints) numpy pass — orders of
    magnitude faster than `trace_gradpsi_internal` itself — and is meant to
    be used as a first-pass oracle for island mapping.  It does not attempt a
    full Poincare analysis; it only rejects lines that are clearly not
    confined to a smooth surface:

      * non-finite or zero-filled data
      * an unphysically large single-step jump in (R, Z)
      * monotonic/random drift of the per-turn mean R
      * per-turn mean Z drifting off the Z=0 symmetry plane

    Returns True when the line looks like a regular confined field line.
    Thresholds are deliberately loose; ambiguous lines pass this check and
    can be re-tested with a higher-level criterion later.
    """
    if fieldline_data is None or fieldline_data.size == 0:
        return False
    R = np.asarray(fieldline_data[:, 0], dtype=np.float64)
    Z = np.asarray(fieldline_data[:, 1], dtype=np.float64)
    n = int(nturn) * int(nphi)
    if R.size < n or Z.size < n:
        return False
    if not (np.isfinite(R).all() and np.isfinite(Z).all()):
        return False
    if np.all(R == 0.0) and np.all(Z == 0.0):
        return False

    # 1) Single-step jump: a field line is integrated with small steps, so a
    #    jump of several cm between consecutive points indicates escape or
    #    an interpolation discontinuity.
    if R.size > 1:
        dR = np.abs(np.diff(R))
        dZ = np.abs(np.diff(Z))
        if float(dR.max()) > float(max_step) or float(dZ.max()) > float(max_step):
            return False

    # 2) Per-turn mean R drift: on a smooth flux surface the turn-averaged
    #    R is stationary; a stochastic/escaping line wanders.
    n_turns = R.size // int(nphi)
    if n_turns >= 3:
        turn_R = R[:n_turns * int(nphi)].reshape(n_turns, int(nphi))
        turn_Z = Z[:n_turns * int(nphi)].reshape(n_turns, int(nphi))
        mean_R = turn_R.mean(axis=1)
        mean_Z = turn_Z.mean(axis=1)
        r0 = float(np.mean(mean_R))
        drift_r = float(np.max(mean_R) - np.min(mean_R))
        if drift_r > max(drift_r_frac * max(r0, 1e-12), 0.005):
            return False
        if float(np.max(np.abs(mean_Z))) > float(drift_z_abs):
            return False

    return True


def fieldline_smoothness_poincare(fieldline_data, nturn, nphi,
                                   axis_rz=None, n_harmonics=4,
                                   residual_rms_frac_tol=0.05,
                                   max_angular_gap=1.0,
                                   min_points=16):
    """Judge smoothness from the phi=0 Poincare section of a traced line.

    A regular magnetic surface gives a set of (R, Z) Poincare points that lie
    on a smooth closed curve; a stochastic/escaping line gives an annular
    cloud, and a low-order rational surface gives only a few distinct points.

    The check therefore:
      1. takes every nphi-th point (phi = 0 section);
      2. forms polar coordinates (theta, r) around the magnetic axis (or the
         Poincare centroid when axis_rz is not supplied);
      3. fits r(theta) with a low-order Fourier series (k=0..n_harmonics);
      4. rejects when the normalised fit residual is too large, or when the
         points leave a large angular gap (island chain / too few crossings).

    This is O(n_turn) and effectively free compared with the field-line
    integration itself.

    Returns
    -------
    smooth : bool
    metrics : dict
        {n_points, residual_rms_frac, max_gap_rad}
    """
    npoints = int(nturn) * int(nphi)
    if fieldline_data is None or fieldline_data.ndim < 2:
        return False, {"n_points": 0}
    if fieldline_data.shape[0] < npoints:
        return False, {"n_points": 0}

    idx = np.arange(0, npoints, int(nphi), dtype=int)
    R = np.asarray(fieldline_data[idx, 0], dtype=np.float64)
    Z = np.asarray(fieldline_data[idx, 1], dtype=np.float64)

    metrics = {"n_points": int(R.size)}
    if R.size < int(min_points):
        return False, metrics
    if not (np.isfinite(R).all() and np.isfinite(Z).all()):
        return False, metrics
    if np.all(R == 0.0) and np.all(Z == 0.0):
        return False, metrics

    if axis_rz is not None:
        R0 = float(axis_rz[0])
        Z0 = float(axis_rz[1])
    else:
        R0 = float(np.median(R))
        Z0 = float(np.median(Z))

    theta = np.arctan2(Z - Z0, R - R0)
    r = np.hypot(R - R0, Z - Z0)

    order = np.argsort(theta)
    theta_s = theta[order]
    r_s = r[order]

    # FFT low-pass fit of r(theta): interpolate the (sorted but non-uniform)
    # Poincare points onto a uniform grid, then keep harmonics 0..n_harmonics.
    # This is O(n log n) and avoids LAPACK; for n_turn ~ 200 it is negligible.
    n_uni = 2 ** int(np.ceil(np.log2(max(64, len(theta_s)))))
    theta_grid = np.linspace(-np.pi, np.pi, n_uni, endpoint=False)
    r_grid = np.interp(theta_grid, theta_s, r_s, period=2.0 * np.pi)

    spec = np.fft.rfft(r_grid)
    n_keep = int(n_harmonics)
    spec_filtered = spec.copy()
    if n_keep + 1 < len(spec_filtered):
        spec_filtered[n_keep + 1:] = 0.0
    r_fit = np.fft.irfft(spec_filtered, n=n_uni)

    residual = r_grid - r_fit
    rms_frac = float(np.sqrt(np.mean(residual**2))
                     / max(float(np.mean(r_grid)), 1e-12))
    metrics["residual_rms_frac"] = rms_frac

    # Angular coverage, including the wrap-around gap.
    gaps = np.diff(theta_s)
    max_gap = 0.0 if gaps.size == 0 else float(np.max(gaps))
    wrap_gap = float(2.0 * np.pi - (theta_s[-1] - theta_s[0]))
    max_gap = max(max_gap, wrap_gap)
    metrics["max_gap_rad"] = max_gap

    smooth = (rms_frac <= float(residual_rms_frac_tol)
              and max_gap <= float(max_angular_gap))
    return bool(smooth), metrics


def trace_fieldline(initial_rz=None, initial_gradpsi=None, nturn=400, nphi=360, extcur=None, verbose=False,
                    check_smoothness=False,
                    smoothness_axis_rz=None,
                    smoothness_n_harmonics=4,
                    smoothness_residual_rms_frac_tol=0.05,
                    smoothness_max_angular_gap=1.0,
                    smoothness_min_points=16):
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
    if check_smoothness and trace_istate == 0:
        smooth, _ = fieldline_smoothness_poincare(
            fieldline_data, nturn, nphi,
            axis_rz=smoothness_axis_rz,
            n_harmonics=smoothness_n_harmonics,
            residual_rms_frac_tol=smoothness_residual_rms_frac_tol,
            max_angular_gap=smoothness_max_angular_gap,
            min_points=smoothness_min_points)
        if not smooth:
            trace_istate = -2001   # traced, but not a smooth confined field line
            if verbose:
                print("✗ Field line tracing succeeded but Poincare smoothness check failed")
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


def find_axis(initial_rz, xtol=1e-10, max_iter=200, delta_r=0.01, nphi=360,
              verbose=False):
    """Find the magnetic axis by solving the one-turn return-map fixed point.

    Robustness improvements over a bare `root` call:
      * 5 trial points (nominal, +-delta_r in R, +-delta_r in Z) are ranked
        by their one-turn return residual before starting the solver;
      * if the best trial fails, the next-best trial is attempted (up to 3);
      * the final axis is re-traced and accepted only if it actually closes
        on itself to within a tolerance consistent with `xtol`.
    """
    from scipy.optimize import root
    nphi = int(nphi)
    n_one_turn = nphi   # index of the point after exactly one toroidal turn
    if verbose:
        print("\nSearching for magnetic axis...")

    def axis_residual(candidate_rz):
        candidate_rz = np.array(candidate_rz, dtype=np.float64, order='F')
        fld, ist = trace_fieldline(
            initial_rz=candidate_rz, initial_gradpsi=None,
            nturn=2, nphi=nphi, verbose=False)
        if ist != 0:
            return np.array([1e10, 1e10])
        return np.array([fld[n_one_turn, 0] - candidate_rz[0],
                         fld[n_one_turn, 1] - candidate_rz[1]])

    init = np.asarray(initial_rz, dtype=np.float64)
    trial_points = [
        init,
        init + np.array([delta_r, 0.0]),
        init - np.array([delta_r, 0.0]),
        init + np.array([0.0, delta_r]),
        init - np.array([0.0, delta_r]),
    ]
    # De-duplicate (paranoia if delta_r == 0)
    unique_trials = []
    for p in trial_points:
        if not any(np.allclose(p, q) for q in unique_trials):
            unique_trials.append(p)

    trial_results = sorted(
        [(np.linalg.norm(axis_residual(p)), p) for p in unique_trials],
        key=lambda x: x[0],
    )

    # A good axis returns to its starting point after one turn.  Be slightly
    # more tolerant than the solver's own `xtol`; find_axis is used with
    # xtol between 1e-5 and 1e-10, and the trace interpolation itself has a
    # finite accuracy floor.
    accept_tol = max(1e-6, 10.0 * max(float(xtol), 1e-12))

    for _, start_rz in trial_results[:3]:
        result = root(axis_residual, start_rz, method='hybr', tol=xtol,
                      options={'maxfev': max_iter, 'factor': 100})
        if not result.success:
            continue
        fld, ist = trace_fieldline(
            initial_rz=result.x, nturn=2, nphi=nphi, verbose=False)
        if fld is None or ist != 0:
            continue
        final_res = float(np.linalg.norm(
            [fld[n_one_turn, 0] - result.x[0],
             fld[n_one_turn, 1] - result.x[1]]))
        if final_res <= accept_tol:
            R0 = float(np.mean(
                np.sqrt(fld[:n_one_turn, 0]**2 + fld[:n_one_turn, 1]**2)))
            if verbose:
                print(f"  Axis found: R={result.x[0]:.6f}, Z={result.x[1]:.6f}, "
                      f"one-turn residual={final_res:.2e}")
            return result.x, R0, fld, True

    return None, None, None, False


def find_axis_any(rmin, rmax, rstep, z0=0.0, xtol=1e-6,
                  max_iter=100, delta_r=0.01, axis_z_tol=1e-6,
                  nphi=360, verbose=False):
    """Scan R at fixed Z and return as soon as one valid axis is found.

    This is the early-exit version of `find_axis_multi_guess`, used when the
    caller does not care whether several axes exist.  In a box where most
    samples are axis-feasible it cuts the average axis-scan cost by roughly
    (rmax-rmin)/rstep, because the scan stops after the first good R guess.
    """
    axes = []
    r = float(rmin)
    while r <= float(rmax) + 1e-12:
        guess = np.array([r, float(z0)], dtype=np.float64)
        try:
            axis_rz, _, _, ok = find_axis(
                guess, xtol=xtol, max_iter=max_iter,
                delta_r=delta_r, nphi=nphi, verbose=False)
        except Exception:
            axis_rz, ok = None, False
        if ok and abs(axis_rz[1]) <= axis_z_tol:
            axes.append((float(axis_rz[0]), float(axis_rz[1])))
            break
        r += float(rstep)
    return axes


def find_axis_multi_guess(rmin, rmax, rstep, z0=0.0, xtol=1e-6,
                          max_iter=100, delta_r=0.01, axis_z_tol=1e-6,
                          nphi=360, verbose=False):
    """Scan R at fixed Z and return every distinct magnetic axis found.

    Each R guess is passed to the (already robust) `find_axis`; this wrapper
    implements the multi-guess strategy for island mapping: a coil-current
    vector is axis-feasible if ANY R in the scan reaches an axis with
    |Z_axis| <= axis_z_tol.

    Returns
    -------
    axes : list[tuple[float, float]]
        Distinct (R, Z) axes found.  The caller can mark a sample as
        multi-axis when several axes are sufficiently separated.
    """
    axes = []
    r = float(rmin)
    while r <= float(rmax) + 1e-12:
        guess = np.array([r, float(z0)], dtype=np.float64)
        try:
            axis_rz, _, _, ok = find_axis(
                guess, xtol=xtol, max_iter=max_iter,
                delta_r=delta_r, nphi=nphi, verbose=False)
        except Exception:
            axis_rz, ok = None, False
        if ok and abs(axis_rz[1]) <= axis_z_tol:
            # Store distinct axes only (avoid duplicates from neighbouring
            # R guesses converging to the same fixed point).
            if not any(np.hypot(axis_rz[0] - a[0], axis_rz[1] - a[1]) < 1e-4
                       for a in axes):
                axes.append((float(axis_rz[0]), float(axis_rz[1])))
        r += float(rstep)
    return axes
