"""Diagnose axis-feasibility flips between old and new find_axis_any params.

Replicates the smoke pre-survey oracle (32 Sobol samples on the engineering
box, seed=48) and compares:
  OLD path: find_axis_any(max_iter=100, no fail-fast, nphi=180) + a second
            full find_axis(nphi=360, xtol=1e-5) refinement, |Z| re-check.
  NEW path: find_axis_any(max_iter=100, fail_residual_tol=RMAX-RMIN, nphi=180),
            axis used directly.

For each flipped sample it prints, per R grid point, whether the OLD solver
succeeds, and attributes the NEW failure to fail-fast (trial residual >
RMAX-RMIN) vs maxfev.  For old-axis samples lost by the new path it also runs
the L2/L3 short+full traces to see whether a real full-feasible point is being
missed.

Run:  PYTHONPATH=python python tests/diag_axis_flip.py
"""
import numpy as np
from pathlib import Path

from ripplepy import initialize_mgrid_field, set_extcur
from ripplepy.ripple import fieldline_smoothness_poincare, find_axis, find_axis_any
from ripplepy.islands import sample_bounds

BASE = Path(__file__).resolve().parent.parent
ENGINEERING_BOUNDS = np.array([
    [ 50000.0,  50000.0],
    [     0.0,  10000.0],
    [     0.0,  10000.0],
    [-220000.0, -40000.0],
    [-100000.0, -10000.0],
])
RMIN, RMAX, RSTEP = 1.00, 1.35, 0.05
Z_TOL = 1e-6
SHORT_NTURN, SHORT_NPHI = 5, 36
FULL_NTURN, FULL_NPHI = 20, 72
DELT_R = 0.06


def _grid_rs():
    rs, r = [], float(RMIN)
    while r <= float(RMAX) + 1e-12:
        rs.append(r)
        r += float(RSTEP)
    return rs


def axis_old(point):
    set_extcur(point)
    axes = find_axis_any(RMIN, RMAX, RSTEP, z0=0.0, xtol=1e-6, max_iter=100,
                         delta_r=0.01, axis_z_tol=Z_TOL, nphi=180)
    if not axes:
        return None, "L1-no-axis"
    for (Rc, Zc) in axes:
        guess = np.array([Rc, Zc], dtype=np.float64)
        try:
            axis_rz, _, _, ok = find_axis(guess, xtol=1e-5, max_iter=100,
                                          delta_r=0.01, nphi=360)
        except Exception:
            axis_rz, ok = None, False
        if ok and abs(axis_rz[1]) <= Z_TOL:
            return axis_rz, "ok"
    return None, "refine-failed"


def axis_new(point):
    set_extcur(point)
    axes = find_axis_any(RMIN, RMAX, RSTEP, z0=0.0, xtol=1e-6, max_iter=100,
                         delta_r=0.01, axis_z_tol=Z_TOL, nphi=180,
                         scan_center=None, fail_residual_tol=RMAX - RMIN)
    if not axes:
        return None, "L1-no-axis"
    return np.asarray(axes[0]), "ok"


def _l3_status(axis_rz):
    """L2 short trace + L3 full trace + smoothness, mirroring _map_point."""
    start = np.array([axis_rz[0] + DELT_R, axis_rz[1]], dtype=np.float64,
                     order="F")
    _, short_ist = _trace(start, SHORT_NTURN, SHORT_NPHI)
    if short_ist != 0:
        return f"short_ist={short_ist}"
    fld, full_ist = _trace(start, FULL_NTURN, FULL_NPHI)
    if full_ist != 0:
        return f"full_ist={full_ist}"
    smooth, _ = fieldline_smoothness_poincare(
        fld, FULL_NTURN, FULL_NPHI, axis_rz=axis_rz,
        n_harmonics=4, residual_rms_frac_tol=0.05,
        max_angular_gap=1.0, min_points=8)
    return "FULL-FEASIBLE" if smooth else "not-smooth"


def _trace(start, nturn, nphi):
    from ripplepy import trace_fieldline
    return trace_fieldline(initial_rz=start, nturn=nturn, nphi=nphi,
                           verbose=False)


def main():
    initialize_mgrid_field(str(BASE / "tests/test_file/mgrid_h1_design.nc"),
                           nfp=3, full_torus=False)
    samples = sample_bounds(ENGINEERING_BOUNDS, 32, seed=48)
    grid_rs = _grid_rs()

    n_old = n_new = 0
    flips = 0
    for k, pt in enumerate(samples):
        if k % 4 == 0 or k == len(samples) - 1:
            print(f"  sample {k + 1}/{len(samples)} ...", flush=True)
        a_old, s_old = axis_old(pt)
        a_new, s_new = axis_new(pt)
        old_ok, new_ok = a_old is not None, a_new is not None
        n_old += old_ok
        n_new += new_ok
        if old_ok == new_ok:
            continue
        flips += 1
        print(f"\nsample {k}: OLD={s_old} "
              f"({'R=%.5f Z=%.2e' % (a_old[0], a_old[1]) if old_ok else '--'})"
              f" | NEW={s_new} "
              f"({'R=%.5f Z=%.2e' % (a_new[0], a_new[1]) if new_ok else '--'})")
        if old_ok and not new_ok:
            print(f"  L2/L3 with OLD axis: {_l3_status(a_old)}")
        set_extcur(pt)
        for r in grid_rs:
            guess = np.array([r, 0.0], dtype=np.float64)
            _, _, _, ok_old = find_axis(guess, xtol=1e-6, max_iter=100,
                                        delta_r=0.01, nphi=180)
            _, _, _, ok_fast = find_axis(guess, xtol=1e-6, max_iter=100,
                                         delta_r=0.01, nphi=180,
                                         fail_residual_tol=RMAX - RMIN)
            _, _, _, ok_new = find_axis(guess, xtol=1e-6, max_iter=100,
                                        delta_r=0.01, nphi=180,
                                        fail_residual_tol=RMAX - RMIN)
            cause = "same" if ok_old == ok_new else (
                "FAIL-FAST" if not ok_fast else "maxfev")
            print(f"  R={r:.2f}: old(hybr100)={'Y' if ok_old else 'n'} "
                  f"fast(hybr100)={'Y' if ok_fast else 'n'} "
                  f"new(hybr40)={'Y' if ok_new else 'n'}  -> {cause}")

    print(f"\naxis_feasible: OLD={n_old}/32  NEW={n_new}/32  "
          f"(flipped samples: {flips})")


if __name__ == "__main__":
    main()
