#!/usr/bin/env python3
"""Per-island analysis of the eps "altitude" map stored by the island mapping.

For EACH island in the HDF5:
  L1  best-of-samples (min eps among the island's full-feasible members)
  L2  island-local quadratic fit of eps(u) + constrained minimum on the
      island's own covariance ellipsoid (skipped when H is not convex)
  verify  every candidate with a full-resolution evaluation
          (find_axis + trace + eps at npart=5000)
  plasma  calculate_plasma_params of the verified best (volume, minor
          radius, iota)
  plot    Poincare section (phi=0) of the island's best verified surface

A summary bar chart of each island's best verified eps is written alongside.

Usage:
    python tests/h1_eps_analyse.py [path_to_h5] [--npart 5000]
                                    [--outdir DIR]
"""

import argparse
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
MGRID_PATH = BASE / "tests" / "test_file" / "mgrid_h1_design.nc"
NFP = 3
FULL_TORUS = False
RIDGE = 1e-4
MIN_EPS_VALID = 10   # below this, skip the quadratic fit, rely on L1


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
    """Fit eps ~= a + g'u + 1/2 u'Hu on normalised free coords u (n x d)."""
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
    H = H + RIDGE * np.eye(d) * max(1.0, np.trace(H) / d)
    return float(coef[0]), g, H, r2


def constrained_min(g: np.ndarray, H: np.ndarray, mu: np.ndarray,
                    Sigma_inv: np.ndarray, r2: float):
    """min 1/2 u'Hu + g'u  s.t.  (u-mu)' Sigma^-1 (u-mu) <= r2 (H PD)."""
    def u_of(lam):
        return np.linalg.solve(H + lam * Sigma_inv,
                               lam * (Sigma_inv @ mu) - g)

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
    """Full DE-style evaluation: axis + trace + eps + plasma parameters.

    Returns (axis_rz, eps, status, plasma, fieldline), with
    plasma = (volume, minor_radius, iota) or None.
    """
    from ripplepy import (calculate_plasma_params, compute_epstot,
                          compute_initial_gradpsi_nemov, find_axis,
                          set_extcur, set_trace_parameters)
    set_extcur(extcur)
    axis_rz, R0, axis_fld, ok = find_axis(initial_rz, xtol=1e-5, max_iter=100,
                                          delta_r=0.01, nphi=nphi,
                                          verbose=False)
    if not ok:
        return None, None, "axis not found", None, None
    start = np.array([axis_rz[0] + delt_r, axis_rz[1]], dtype=np.float64,
                     order="F")
    gradpsi = compute_initial_gradpsi_nemov(extcur, start[0], start[1],
                                            verbose=False)
    set_trace_parameters(nturn, nphi, npart=npart, verbose=False)
    res = compute_epstot(start, initial_gradpsi=gradpsi,
                         return_fieldline=True, verbose=False)
    eps = res[0]
    if eps is None or np.isnan(eps):
        return axis_rz, None, f"eps failed (istate={res[3]})", None, None
    plasma = None
    try:
        vol, minor_r, iota = calculate_plasma_params(
            res[2], axis_fld, nturn, nphi, float(R0))
        plasma = (float(vol), float(minor_r), float(iota))
    except Exception:
        plasma = None
    return axis_rz, float(eps), "ok", plasma, res[2]


def _plot_poincare(fieldline, axis_rz, nphi, title, outpath):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"    (matplotlib unavailable; skip plot {outpath})")
        return
    idx = np.arange(0, fieldline.shape[0], nphi)   # phi=0 section per turn
    R, Z = fieldline[:, 0], fieldline[:, 1]
    plt.figure(figsize=(5, 5))
    plt.scatter(R[idx], Z[idx], s=3, alpha=0.8, label="phi=0 section")
    plt.scatter([axis_rz[0]], [axis_rz[1]], c="red", marker="x", s=50,
                label="axis")
    plt.axis("equal")
    plt.xlabel("R (m)")
    plt.ylabel("Z (m)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"    plot -> {outpath}")


def _plot_summary(labels, best_eps, outpath):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"    (matplotlib unavailable; skip plot {outpath})")
        return
    plt.figure(figsize=(max(4, 1.6 * len(labels)), 4))
    plt.bar(range(len(labels)), best_eps, color="steelblue")
    plt.xticks(range(len(labels)), labels, rotation=15)
    plt.ylabel("best eps_eff^(3/2) (verified)")
    plt.title("Best verified eps per island")
    for i, v in enumerate(best_eps):
        plt.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"    plot -> {outpath}")


def analyze_island(isl: dict, samples: np.ndarray, bounds: np.ndarray,
                   delt_r: float, eps: np.ndarray, full: np.ndarray,
                   free: np.ndarray, axis_used_RZ: np.ndarray,
                   args) -> dict | None:
    """Analyse one island: L1/L2, verify, plasma params, Poincare plot."""
    name = f"island {isl['island_id']}"
    inds = np.asarray(isl["sample_indices"], dtype=int)
    valid_isl = inds[np.isfinite(eps[inds]) & full[inds]]
    n_mem = int(isl.get("n_points", len(inds)))
    print(f"\n[{name}] members={n_mem}, eps-valid={len(valid_isl)}")
    if len(valid_isl) == 0:
        print("  (no eps-valid members)")
        return None

    u = _normalise(samples, bounds, free)[valid_isl]
    e = eps[valid_isl]

    # L1: best sampled point in this island.
    k0 = int(np.argmin(e))
    x_best = samples[valid_isl][k0]
    print(f"  [L1] best sampled eps = {e[k0]:.6f} at currents "
          f"{np.round(x_best, 1).tolist()}")
    candidates = [("L1 best sample", x_best)]

    # L2: island-local quadratic fit + constrained minimum on the island's
    # own covariance ellipsoid (skipped when H is not positive-definite).
    on_boundary = False
    if len(valid_isl) >= MIN_EPS_VALID:
        a, g, H, r2 = fit_quadratic(u, e)
        eigH = np.linalg.eigvalsh(H)
        print(f"  [L2] quadratic fit R^2 = {r2:.3f}; H eig = "
              f"{np.round(eigH, 3).tolist()} "
              f"({'convex' if eigH.min() > 0 else 'NOT convex'})")
        if eigH.min() > 0:
            mu = np.asarray(isl.get("center_free"), dtype=np.float64)
            cov = np.asarray(isl.get("cov_free"), dtype=np.float64)
            if mu.shape != (len(free),) or cov.shape != (len(free), len(free)):
                mu = u.mean(axis=0)
                cov = np.cov(u, rowvar=False)
            cov = cov + 1e-9 * np.eye(len(free))
            Sigma_inv = np.linalg.inv(cov)
            diff = u - mu
            md = np.sqrt(np.sum((diff @ Sigma_inv) * diff, axis=1))
            r_max = float(md.max())
            u_c, lam_c, on_boundary = constrained_min(g, H, mu, Sigma_inv,
                                                      r_max ** 2)
            x_c = _to_currents(u_c, bounds, free)
            print(f"  [L2] constrained min at currents "
                  f"{np.round(x_c, 1).tolist()} "
                  f"({'on boundary' if on_boundary else 'interior'})")
            candidates.append(("L2 constrained min", x_c))
            if on_boundary:
                d = u_c - mu
                nd = np.linalg.norm(d)
                if nd > 0:
                    delta = (d / nd) * (bounds[free, 1] - bounds[free, 0])
                    order = np.argsort(-np.abs(delta))
                    print("       -> presses the sampled-region boundary; "
                          "extend the mapping along:")
                    for j in order:
                        c = int(free[j])
                        sign = "up" if delta[j] >= 0 else "down"
                        print(f"          coil {c}: {sign} by "
                              f"~{abs(delta[j]):.0f} A")

    # Warm-start axis: the L1 best member's recorded axis R (fallback: the
    # island's mean axis R).
    axis_r = float(axis_used_RZ[valid_isl][k0, 0])
    if not np.isfinite(axis_r):
        axis_r = float(isl.get("mean_axis_R", 1.27))

    print(f"  [verify] nturn={args.nturn}, nphi={args.nphi}, "
          f"npart={args.npart}")
    best = None   # (eps, label, axis_rz, plasma, fieldline)
    for label, x in candidates:
        ax, eps_v, status, plasma, fld = evaluate_point(
            x, delt_r, np.array([axis_r, 0.0]), args.nturn, args.nphi,
            args.npart)
        if status == "ok":
            vol, mr, iota = plasma if plasma else (np.nan,) * 3
            print(f"    {label:18s}: eps={eps_v:.6f}  axis R={ax[0]:.4f}  "
                  f"minor_r={mr:.4f}  iota={iota:.3f}  vol={vol:.6f}")
            if best is None or eps_v < best[0]:
                best = (eps_v, label, ax, plasma, fld)
        else:
            print(f"    {label:18s}: {status}")

    if best is not None:
        out = args.outdir / f"eps_poincare_isl{isl['island_id']}_dr{delt_r:g}.png"
        _plot_poincare(best[4], best[2], args.nphi,
                       f"{name} best ({best[1]}), eps={best[0]:.4f}", out)

    return {"island_id": int(isl["island_id"]), "name": name,
            "best_eps": best[0] if best else np.nan,
            "n_members": n_mem}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("h5", nargs="?",
                    default=str(BASE / "tests" / "h1_islands"
                                / "islands_dr0.12.h5"))
    ap.add_argument("--npart", type=int, default=5000)
    ap.add_argument("--nturn", type=int, default=400)
    ap.add_argument("--nphi", type=int, default=360)
    ap.add_argument("--outdir", type=str, default=None,
                    help="plot output dir (default: next to the HDF5)")
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
    axis_used_RZ = np.asarray(res["axis_used_RZ"], dtype=np.float64)
    free = np.flatnonzero(bounds[:, 1] - bounds[:, 0] > 1e-12)
    args.outdir = Path(args.outdir) if args.outdir else Path(args.h5).parent
    args.outdir.mkdir(parents=True, exist_ok=True)

    print(f"HDF5: {args.h5} (delt_r={delt_r}, full_feasible={full.sum()}, "
          f"eps-valid={int(np.isfinite(eps).sum())})")

    islands = res.get("islands", [])
    if not islands:
        # No clustered islands (e.g. smoke): treat the whole feasible set as
        # one pseudo-island so the analysis still runs.
        islands = [{"island_id": 0, "n_points": int(full.sum()),
                    "sample_indices": np.flatnonzero(full)}]

    initialize_mgrid_field(str(MGRID_PATH), nfp=NFP, full_torus=FULL_TORUS)

    labels, best_eps = [], []
    for isl in islands:
        info = analyze_island(isl, samples, bounds, delt_r, eps, full, free,
                              axis_used_RZ, args)
        if info is not None:
            labels.append(info["name"])
            best_eps.append(info["best_eps"])

    if labels:
        _plot_summary(labels, best_eps,
                      args.outdir / f"eps_islands_dr{delt_r:g}.png")
        i_best = int(np.nanargmin(best_eps))
        print(f"\n[summary] best island: {labels[i_best]} "
              f"(verified eps={best_eps[i_best]:.6f})")


if __name__ == "__main__":
    main()
