#!/usr/bin/env python3
"""Analyse the eps "altitude" map stored by the island mapping.

Reads a mapping HDF5 (samples, full_feasible, eps), fits a quadratic
eps(u) over the full-feasible samples in the island's normalised free-coil
space, finds the constrained minimum (unconstrained quadratic minimum,
projected onto the island's empirical ellipsoid when outside), verifies the
candidates with a full-resolution evaluation, and reports whether the optimum
presses the sampled-region boundary (=> extend the mapping in that direction).

Usage:
    python tests/h1_eps_analyse.py [path_to_h5] [--npart 5000]
"""

import argparse
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
MGRID_PATH = BASE / "tests" / "test_file" / "mgrid_h1_design.nc"
NFP = 3
FULL_TORUS = False
RIDGE = 1e-4


def _normalise(samples: np.ndarray, bounds: np.ndarray, free: np.ndarray
               ) -> np.ndarray:
    lo = bounds[free, 0]
    span = bounds[free, 1] - bounds[free, 0]
    span[span <= 1e-12] = 1.0
    return (samples[:, free] - lo) / span


def _to_currents(u: np.ndarray, bounds: np.ndarray, free: np.ndarray
                 ) -> np.ndarray:
    x = bounds[:, 0].copy()
    x[free] = bounds[free, 0] + u * (bounds[free, 1] - bounds[free, 0])
    return x


def fit_quadratic(u: np.ndarray, eps: np.ndarray):
    """Fit eps ~= a + g'u + 1/2 u'Hu on normalised free coords u (n x d).

    Returns (a, g, H, r2).
    """
    n, d = u.shape
    cols = [np.ones(n)] + [u[:, i] for i in range(d)]
    k = 1 + d
    for i in range(d):
        for j in range(i, d):
            cols.append(u[:, i] * u[:, j])
    X = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(X, eps, rcond=None)
    pred = X @ coef
    ss_res = float(np.sum((eps - pred) ** 2))
    ss_tot = float(np.sum((eps - eps.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    g = coef[1:1 + d]
    H = np.zeros((d, d))
    k = 1 + d
    for i in range(d):
        for j in range(i, d):
            c = float(coef[k]); k += 1
            if i == j:
                H[i, j] = 2.0 * c
            else:
                H[i, j] = H[j, i] = c
    # Ridge proportional to H's own scale keeps the fit stable.
    H = H + RIDGE * np.eye(d) * max(1.0, np.trace(H) / d)
    return float(coef[0]), g, H, r2


def constrained_min(g: np.ndarray, H: np.ndarray, mu: np.ndarray,
                    Sigma_inv: np.ndarray, r2: float):
    """min 1/2 u'Hu + g'u  s.t.  (u-mu)' Sigma^-1 (u-mu) <= r2.

    H must be positive-definite.  Returns (u_min, lam, on_boundary):
    lam == 0 and on_boundary False when the unconstrained min is interior.
    """
    def u_of(lam):
        K = H + lam * Sigma_inv
        rhs = lam * (Sigma_inv @ mu) - g
        return np.linalg.solve(K, rhs)

    def phi(lam):
        u = u_of(lam)
        return float((u - mu) @ Sigma_inv @ (u - mu)) - r2

    u_free = np.linalg.solve(H, -g)
    if phi(0.0) <= 0:
        return u_free, 0.0, False
    lam = 1e-6
    while phi(lam) > 0 and lam < 1e12:
        lam *= 2.0
    lo, hi = 0.0, lam
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if phi(mid) > 0:
            lo = mid
        else:
            hi = mid
    lam_c = 0.5 * (lo + hi)
    return u_of(lam_c), lam_c, True


def evaluate_point(extcur: np.ndarray, delt_r: float, initial_rz: np.ndarray,
                   nturn: int, nphi: int, npart: int):
    """Full DE-style evaluation of one candidate: axis + trace + eps."""
    from ripplepy import (compute_epstot, compute_initial_gradpsi_nemov,
                          find_axis, set_extcur, set_trace_parameters)
    set_extcur(extcur)
    axis_rz, _, _, ok = find_axis(initial_rz, xtol=1e-5, max_iter=100,
                                  delta_r=0.01, verbose=False)
    if not ok:
        return None, None, "axis not found"
    start = np.array([axis_rz[0] + delt_r, axis_rz[1]], dtype=np.float64,
                     order="F")
    gradpsi = compute_initial_gradpsi_nemov(extcur, start[0], start[1],
                                            verbose=False)
    set_trace_parameters(nturn, nphi, npart=npart, verbose=False)
    res = compute_epstot(start, initial_gradpsi=gradpsi,
                         return_fieldline=False, verbose=False)
    eps = res[0]
    if eps is None or np.isnan(eps):
        return axis_rz, None, f"eps failed (istate={res[2]})"
    return axis_rz, float(eps), "ok"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("h5", nargs="?",
                    default=str(BASE / "tests" / "h1_islands"
                                / "islands_dr0.12.h5"))
    ap.add_argument("--npart", type=int, default=5000)
    ap.add_argument("--nturn", type=int, default=400)
    ap.add_argument("--nphi", type=int, default=720)
    args = ap.parse_args()

    from ripplepy import initialize_mgrid_field
    from ripplepy.islands import load_island_mapping_h5

    res = load_island_mapping_h5(Path(args.h5))
    samples = np.asarray(res["samples"], dtype=np.float64)
    bounds = np.asarray(res["bounds"], dtype=np.float64)
    delt_r = float(res["delt_r"])
    full = res["full_feasible"].astype(bool)
    eps = np.asarray(res.get("eps", np.full(len(samples), np.nan)),
                     dtype=np.float64)
    valid = full & np.isfinite(eps)
    free = np.flatnonzero(bounds[:, 1] - bounds[:, 0] > 1e-12)

    print(f"HDF5: {args.h5} (delt_r={delt_r}, full_feasible={full.sum()}, "
          f"eps-valid={valid.sum()})")
    if int(valid.sum()) < 10:
        print("Too few eps-valid samples (need >= 10). Re-run the mapping "
              "with COMPUTE_EPS=True.")
        return

    u = _normalise(samples, bounds, free)[valid]
    e = eps[valid]
    mu = u.mean(axis=0)
    cov = np.cov(u, rowvar=False) + 1e-9 * np.eye(len(free))
    Sigma_inv = np.linalg.inv(cov)
    diff = u - mu
    md = np.sqrt(np.sum((diff @ Sigma_inv) * diff, axis=1))
    r_max = float(md.max())

    # L1: best sampled point (the mapping is itself a global search).
    k0 = int(np.argmin(e))
    x_best_sample = samples[valid][k0]
    print(f"\n[L1] best sampled eps = {e[k0]:.6f} at currents "
          f"{np.round(x_best_sample, 1).tolist()}")

    # L2: quadratic fit + constrained minimum.
    a, g, H, r2 = fit_quadratic(u, e)
    print(f"\n[L2] quadratic fit R^2 = {r2:.3f}")
    eigH = np.linalg.eigvalsh(H)
    print(f"     H eigenvalues = {np.round(eigH, 4).tolist()} "
          f"({'convex' if eigH.min() > 0 else 'NOT convex'})")
    if eigH.min() <= 0:
        print("     H not positive-definite -> constrained min skipped; "
              "rely on L1 best-of-samples.")
        candidates = [("L1 best sample", x_best_sample)]
    else:
        u_free = np.linalg.solve(H, -g)
        x_free = _to_currents(u_free, bounds, free)
        u_c, lam_c, on_boundary = constrained_min(g, H, mu, Sigma_inv,
                                                  r_max ** 2)
        x_c = _to_currents(u_c, bounds, free)
        print(f"     unconstrained min at currents "
              f"{np.round(x_free, 1).tolist()}")
        print(f"     constrained min at currents {np.round(x_c, 1).tolist()} "
              f"({'on boundary' if on_boundary else 'interior'})")
        candidates = [("L2 constrained min", x_c), ("L1 best sample",
                                                    x_best_sample)]
        if on_boundary:
            d = u_free - mu
            nd = np.linalg.norm(d)
            if nd > 0:
                d = d / nd
                delta = d * (bounds[free, 1] - bounds[free, 0])  # A per coil
                order = np.argsort(-np.abs(delta))
                print("     -> minimum presses the sampled-region boundary; "
                      "extend the mapping along:")
                for j in order:
                    c = int(free[j])
                    sign = "up" if delta[j] >= 0 else "down"
                    print(f"        coil {c}: {sign} by ~{abs(delta[j]):.0f} A")

    # Verify candidates with a full-resolution evaluation.
    initialize_mgrid_field(str(MGRID_PATH), nfp=NFP, full_torus=FULL_TORUS)
    axis_r = float(res["axis_used_RZ"][valid][k0, 0])
    if not np.isfinite(axis_r):
        axis_r = 1.27
    print(f"\n[verify] nturn={args.nturn}, nphi={args.nphi}, "
          f"npart={args.npart} (warm axis R={axis_r:.4f})")
    for label, x in candidates:
        ax, eps_v, status = evaluate_point(x, delt_r, np.array([axis_r, 0.0]),
                                           args.nturn, args.nphi, args.npart)
        if status == "ok":
            print(f"  {label:18s}: eps = {eps_v:.6f}  (axis R={ax[0]:.4f})")
        else:
            print(f"  {label:18s}: {status}")


if __name__ == "__main__":
    main()
