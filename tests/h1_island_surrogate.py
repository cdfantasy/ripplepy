#!/usr/bin/env python3
"""Stage 4: cluster the eps field and fit one quadratic surrogate per island.

Reads the one-pass scan (``scan_all.h5``), takes for every configuration the
surface closest to a = A0 (so all configurations are compared at the same
radius), keeps the trustworthy ones (eps > 0 and a clean grad-psi drift),
clusters them in the normalised current box, and fits

    f(eps)  ~  a + b.v + v^T C v          v = B^T (u - centre) / sd

per island, with f = ln or the identity: **both are fitted and the one with the
smaller cross-validation error wins**.  The CV is always reported as the
*relative* error in eps, so the two transforms are directly comparable.  (eps
spans a factor of ~20 over the sampled region, so the log normally wins by a
wide margin; if it does not, the region is already nearly linear and the linear
fit is the more honest model.)  The fit is
done in the PCA basis of the island's own point cloud, restricted to the
directions that actually have spread -- the islands are thin slabs (31:1 measured
earlier), so coefficients along a thin direction would otherwise be
unconstrained noise.

Everything printed is a *diagnostic*: R^2, k-fold cross-validation error, the
condition number of the design matrix, and the spread of each PCA direction.
Only islands whose CV error is small enough should be used for prediction.

Outputs
  surrogate.h5   one group per island: centre, basis, sd, coefficients, metrics
  (plus the candidate list in the stdout report)

  python -u tests/h1_island_surrogate.py
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

BASE = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
_SCAN_CANDIDATES = [BASE / "tests" / "h1_full_scan" / "scan_all.h5",
                    BASE / "tests" / "h1_radius_scan" / "scan_all.h5"]
SCAN_H5 = next((p for p in _SCAN_CANDIDATES if p.exists()), _SCAN_CANDIDATES[0])
OUT_H5 = SCAN_H5.parent / "surrogate.h5"

A0 = 0.10               # reference minor radius of the config-level eps

BOUNDS = np.array([
    [50000.0, 50000.0],     # coil 0: TF, fixed
    [800.0, 5500.0],        # coil 1
    [0.0, 5000.0],          # coil 2
    [-100000.0, 0.0],       # coil 3
    [-100000.0, 0.0],       # coil 4
])

QC_MAX = 1.5            # grad-psi drift above this = corrupted eps, drop it
MIN_POINTS = 80         # an island needs at least this many points to be fitted
MAX_DIMS = 4            # PCA directions used (4 = no reduction)
SD_FRACTION = 0.05      # a direction is usable if sd > SD_FRACTION * sd_max
N_FOLDS = 5
TRUST_SIGMA = 2.5       # prediction only inside a Mahalanobis radius
DBSCAN_MIN_SAMPLES = 8
DBSCAN_EPS_SWEEP = (0.04, 0.06, 0.08, 0.10, 0.15)
N_CANDIDATES = 5
FIT_TRANSFORMS = ("log", "identity")   # both are tried; the better CV wins


# ---------------------------------------------------------------------------
def load_reference(path=SCAN_H5, a0=A0):
    """Interpolate every surface quantity to exactly a = a0.

    Picking the nearest face is NOT good enough: the radial grid is coarse
    (dr = 0.02 plus one bisection), so the nearest face to a0 sits anywhere in
    [0.082, 0.116], and eps varies steeply with radius (d ln eps / da ~ 19/m,
    i.e. a 0.01 m offset is a 20% change).  That jitter would contaminate the
    model.  Since the walk stores every surface, each quantity is interpolated
    along the column instead, which needs the column to bracket a0.
    """
    with h5py.File(path, "r") as f:
        S = f["samples"][:]
        keys = ("a", "eps", "gradpsi_drift", "dr", "iota", "volume",
                "elongation", "status")
        prof = {k: f["profile"][k][:] for k in keys}
    ok = prof["status"] == 0
    n = len(S)
    ref = {k: np.full(n, np.nan) for k in
           ("eps", "a", "dr", "iota", "volume", "elongation",
            "gradpsi_drift")}
    for i in range(n):
        m = ok[i] & np.isfinite(prof["a"][i]) & (prof["a"][i] > 0)
        if m.sum() < 2:
            continue
        am = prof["a"][i][m]
        em = prof["eps"][i][m]
        if not (np.isfinite(em).all() and em.min() > 0):
            continue
        o = np.argsort(am)
        am, em = am[o], em[o]
        if not (am[0] < a0 < am[-1] or np.any(np.abs(am - a0) < 1e-6)):
            continue
        idx = np.flatnonzero(m)[o]
        ref["eps"][i] = float(np.exp(np.interp(a0, am, np.log(em))))
        ref["a"][i] = a0
        for k in ("dr", "iota", "volume", "elongation"):
            v = prof[k][i][idx]
            if np.isfinite(v).all():
                ref[k][i] = float(np.interp(a0, am, v))
        j = int(np.argmin(np.abs(am - a0)))     # drift is a QC ratio
        ref["gradpsi_drift"][i] = prof["gradpsi_drift"][i][idx[j]]
    return S, ref


# ---------------------------------------------------------------------------
def dbscan_lite(X, eps, min_samples):
    """Plain DBSCAN on a KD-tree (no sklearn dependency)."""
    n = len(X)
    labels = np.full(n, -1, dtype=int)
    tree = cKDTree(X)
    neigh = tree.query_ball_point(X, eps)
    core = np.array([len(nb) >= min_samples for nb in neigh])
    cid = 0
    for i in range(n):
        if labels[i] != -1 or not core[i]:
            continue
        stack, labels[i] = [i], cid
        while stack:
            j = stack.pop()
            for k in neigh[j]:
                if labels[k] == -1:
                    labels[k] = cid
                    if core[k]:
                        stack.append(k)
        cid += 1
    return labels


def quad_terms(n_dims):
    """Index list of the quadratic design matrix columns."""
    terms = [("1", ())]
    for i in range(n_dims):
        terms.append((f"v{i}", (i,)))
    for i in range(n_dims):
        for j in range(i, n_dims):
            terms.append((f"v{i}v{j}", (i, j)))
    return terms


def design(v, terms):
    cols = []
    for _, idx in terms:
        if not idx:
            cols.append(np.ones(len(v)))
        else:
            cols.append(np.prod(v[:, idx], axis=1))
    return np.column_stack(cols)


def to_fit(e, transform):
    """Map eps to the space the quadratic is fitted in."""
    return np.log(e) if transform == "log" else np.asarray(e, dtype=float)


def to_eps(yhat, transform):
    """Map a fit-space prediction back to eps (relative error is what matters)."""
    out = np.exp(yhat) if transform == "log" else np.asarray(yhat, dtype=float)
    return np.maximum(out, 1e-30)


def fit_quadratic(v, y, terms):
    X = design(v, terms)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef, X


def cross_validate(v, e, terms, transform, n_folds, rng):
    """k-fold CV reported as the RELATIVE error in eps for every transform."""
    y = to_fit(e, transform)
    n = len(y)
    idx = rng.permutation(n)
    folds = np.array_split(idx, min(n_folds, n))
    err = np.full(n, np.nan)
    for f in folds:
        tr = np.setdiff1d(idx, f)
        coef, _ = fit_quadratic(v[tr], y[tr], terms)
        pred = to_eps(design(v[f], terms) @ coef, transform)
        err[f] = np.abs(pred - e[f]) / np.maximum(e[f], 1e-30)
    return err


# ---------------------------------------------------------------------------
def fit_island(rows, U, E, rng, tag):
    """Local PCA -> quadratic fit in the well-spread subspace, for every
    transform in FIT_TRANSFORMS; the one with the smaller CV wins."""
    u = U[rows]
    e = E[rows]
    c = u.mean(axis=0)
    d = u - c
    _, _, vt = np.linalg.svd(d, full_matrices=False)
    proj = d @ vt.T                      # (n, 4) PCA scores
    sd = proj.std(axis=0)
    keep = sd > SD_FRACTION * sd[0]
    keep = np.flatnonzero(keep)[:MAX_DIMS]
    if len(keep) == 0:
        return None
    v = proj[:, keep] / sd[keep]
    terms = quad_terms(len(keep))

    results = {}
    for tr in FIT_TRANSFORMS:
        y = to_fit(e, tr)
        coef, X = fit_quadratic(v, y, terms)
        pred = X @ coef
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        cv = cross_validate(v, e, terms, tr, N_FOLDS, rng)
        results[tr] = {"coef": coef, "y": y, "r2": 1.0 - ss_res / max(ss_tot, 1e-30),
                       "cv": cv, "cond": float(np.linalg.cond(X))}
    chosen = min(results, key=lambda t: np.nanmedian(results[t]["cv"]))
    r = results[chosen]
    coef = r["coef"]

    # Hessian of the chosen fit, in whitened coordinates
    n = len(keep)
    Hm = np.zeros((n, n))
    for name, idx in terms:
        if len(idx) == 2:
            k = terms.index((name, idx))
            if idx[0] == idx[1]:
                Hm[idx[0], idx[0]] = 2.0 * coef[k]
            else:
                Hm[idx[0], idx[1]] = coef[k]
                Hm[idx[1], idx[0]] = coef[k]
    eig = np.linalg.eigvalsh(Hm)

    print(f"  [{tag}] n={len(rows)}  dims kept={len(keep)}/{len(sd)}")
    for tr in FIT_TRANSFORMS:
        q = results[tr]
        mark = "  <-- chosen" if tr == chosen else ""
        print(f"        {tr:8s}: R2={q['r2']:.3f}  "
              f"CV med={np.nanmedian(q['cv']) * 100:5.1f}%  "
              f"p90={np.nanpercentile(q['cv'], 90) * 100:5.1f}%  "
              f"cond={q['cond']:.1e}{mark}")
    print("        PCA sd (normalised box): " +
          "  ".join(f"{x:.4f}" for x in sd) +
          ("   <- dropped: " + ", ".join(str(i) for i in range(len(sd))
                                         if i not in keep) if len(keep) < len(sd) else ""))
    print("        Hessian eigenvalues (" + chosen + "): " +
          "  ".join(f"{x:+.3f}" for x in eig) +
          ("   [all negative: interior maximum]" if np.all(eig < 0) else ""))
    return {
        "centre": c, "basis": vt[keep], "sd": sd[keep], "coef": coef,
        "terms": terms, "n": len(rows), "r2": r["r2"],
        "cv_med": float(np.nanmedian(r["cv"])),
        "cv_p90": float(np.nanpercentile(r["cv"], 90)), "cond": r["cond"],
        "eig": eig, "keep": keep, "sd_all": sd, "rows": rows, "v": v,
        "y": r["y"], "e": e, "transform": chosen,
    }


def propose(fit, bf, span, n_cand, tag):
    """Constrained minimum of the quadratic inside the data-supported region.

    Three guards, all of which were violated by the first version:
      * the search box is the observed extent of each whitened axis, not a fixed
        +/-TRUST_SIGMA -- the earlier one produced a candidate 5 sigma out,
        where a saddle-shaped model keeps decreasing for no physical reason;
      * the candidate is clipped to the coil box, and its prediction is
        re-evaluated at the clipped point (the whitened -> current map is not
        box preserving);
      * candidates that leave the data cloud or predict well below the best
        observed value are flagged as extrapolations instead of being reported
        as improvements.
    """
    v, y, terms, coef = fit["v"], fit["y"], fit["terms"], fit["coef"]
    d_m = np.linalg.norm(v, axis=1)
    r_lim = float(np.percentile(d_m, 98))
    lo, hi = v.min(axis=0), v.max(axis=0)
    e_obs = fit["e"]

    def f(x):
        arr = np.asarray(design(x.reshape(1, -1), terms) @ coef).ravel()
        return float(arr[0])

    starts = v[np.argsort(y)[:5]]
    cands = []
    for s0 in starts:
        res = minimize(f, s0, method="L-BFGS-B", bounds=list(zip(lo, hi)))
        if res.success:
            cands.append(res.x)
    if not cands:
        return []
    # dedupe by proximity (several starts converge to the same point)
    uniq = []
    for x in cands:
        if all(np.linalg.norm(x - u) > 1e-3 for u in uniq):
            uniq.append(x)
    cands = np.array(uniq)

    out = []
    for x in cands:
        u = fit["centre"] + (x * fit["sd"]) @ fit["basis"]
        u_clip = np.clip(u, 0.0, 1.0)
        clipped = bool(np.max(np.abs(u - u_clip)) > 1e-9)
        vv = ((u_clip - fit["centre"]) @ fit["basis"].T) / fit["sd"]
        eps_pred = float(to_eps(np.array([f(vv)]), fit["transform"])[0])
        dist = np.linalg.norm(v - vv, axis=1)
        j = int(np.argmin(dist))
        out.append({"u": u_clip, "v": vv,
                    "currents": bf[:, 0] + u_clip * span,
                    "eps_pred": eps_pred, "mahalanobis": float(np.linalg.norm(vv)),
                    "clipped": clipped, "nn_dist": float(dist[j]),
                    "nn_eps": float(e_obs[j])})
    out.sort(key=lambda d: d["eps_pred"])

    print(f"  [{tag}] data Mahalanobis radius: med {np.median(d_m):.2f} "
          f"98% {r_lim:.2f}   best observed eps {e_obs.min():.4f}")
    for c in out[:n_cand]:
        flags = []
        if c["mahalanobis"] > r_lim:
            flags.append("OUTSIDE DATA")
        if c["eps_pred"] < 0.8 * float(e_obs.min()):
            flags.append("EXTRAPOLATED")
        if c["clipped"]:
            flags.append("clipped to box")
        print(f"    pred eps={c['eps_pred']:.4f}  d_M={c['mahalanobis']:.2f}  "
              f"nearest obs eps={c['nn_eps']:.4f} at d={c['nn_dist']:.2f}"
              + ("   <-- " + ", ".join(flags) if flags else ""))
        print(f"      extcur=[{', '.join(f'{x:.1f}' for x in c['currents'])}]")
    return out[:n_cand]


def local_fit(rows, U, E, rng, k=200):
    """Refit the quadratic on the k nearest neighbours of the best point."""
    i_best = rows[int(np.argmin(E[rows]))]
    d = np.linalg.norm(U[rows] - U[i_best], axis=1)
    near = rows[np.argsort(d)[:min(k, len(rows))]]
    return fit_island(near, U, E, rng, f"local k={len(near)} around best")


# ---------------------------------------------------------------------------
def main():
    S, ref = load_reference()
    e = ref["eps"]
    qc = ref["gradpsi_drift"]
    have = np.isfinite(e) & (e > 0)
    good = have & np.isfinite(qc) & (qc <= QC_MAX)
    S, e = S[good], e[good]
    print(f"scan input: {SCAN_H5}\n  {int(have.sum())} configurations reach"
          f" a ~ {A0} m; {int(good.sum())} clean (eps>0, drift <= {QC_MAX})")
    if good.sum() < MIN_POINTS:
        print("  too few clean configurations to fit anything")
        return

    span = BOUNDS[:, 1] - BOUNDS[:, 0]
    free = span > 0
    bf = BOUNDS[free]
    U = (S[:, free] - bf[:, 0]) / span[free]

    # --- clustering: how many islands, and is the answer stable in eps? -----
    nn = cKDTree(U).query(U, k=2)[0][:, 1]
    print(f"\nnormalised nearest-neighbour distance: med {np.median(nn):.4f} "
          f"p95 {np.percentile(nn, 95):.4f} max {nn.max():.4f}")
    best_labels, best_eps, best_score = None, None, -1
    print("\nDBSCAN eps sweep (min_samples=%d):" % DBSCAN_MIN_SAMPLES)
    for eps in DBSCAN_EPS_SWEEP:
        lab = dbscan_lite(U, eps, DBSCAN_MIN_SAMPLES)
        sizes = sorted([int((lab == k).sum()) for k in np.unique(lab) if k >= 0],
                       reverse=True)
        noise = int((lab == -1).sum())
        print(f"  eps={eps:.2f}: {len(sizes)} clusters, sizes {sizes[:6]}, "
              f"noise {noise} ({100 * noise / len(lab):.0f}%)")
        score = (sizes[0] if sizes else 0) - noise * 0.5
        if score > best_score:
            best_score, best_labels, best_eps = score, lab, eps
    labels = best_labels
    print(f"\nusing eps={best_eps:.2f}")

    # --- per-island fit ----------------------------------------------------
    rng = np.random.default_rng(0)
    fits = []
    print("\nper-island quadratic (whitened PCA coords; both transforms tried):")
    for k in np.unique(labels):
        if k < 0:
            continue
        rows = np.flatnonzero(labels == k)
        if len(rows) < MIN_POINTS:
            print(f"  [island {k}] n={len(rows)} < {MIN_POINTS}, skipped")
            continue
        f = fit_island(rows, U, e, rng, f"island {k}")
        if f is not None:
            f["label"] = int(k)
            fits.append(f)

    # --- candidates --------------------------------------------------------
    print("\nconstrained minima of the quadratic (data-supported region):")
    span_free = span[free]
    for f in fits:
        cands = propose(f, bf, span_free, N_CANDIDATES,
                        f"island {f['label']} global (transform={f['transform']}, "
                        f"CV {f['cv_med'] * 100:.1f}%)")
        lf = local_fit(f["rows"], U, e, rng)
        if lf is not None:
            lc = propose(lf, bf, span_free, N_CANDIDATES,
                         f"island {f['label']} local (transform={lf['transform']}, "
                         f"CV {lf['cv_med'] * 100:.1f}%)")
            cands = (cands or []) + (lc or [])
            f["local"] = lf
        f["candidates"] = cands

    # --- save --------------------------------------------------------------
    out = Path(OUT_H5)
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w") as h:
        h.attrs["a0"] = A0
        h.attrs["dbscan_eps"] = best_eps
        h.attrs["n_clean"] = int(good.sum())
        for f in fits:
            g = h.create_group(f"island_{f['label']}")
            g.create_dataset("centre", data=f["centre"])
            g.create_dataset("basis", data=f["basis"])
            g.create_dataset("sd", data=f["sd"])
            g.create_dataset("coef", data=f["coef"])
            g.create_dataset("sd_all", data=f["sd_all"])
            g.create_dataset("rows", data=f["rows"])
            g.attrs["n"] = f["n"]
            g.attrs["r2"] = f["r2"]
            g.attrs["cv_med"] = f["cv_med"]
            g.attrs["cv_p90"] = f["cv_p90"]
            g.attrs["cond"] = f["cond"]
            g.attrs["eig"] = json.dumps([float(x) for x in f["eig"]])
            g.attrs["terms"] = json.dumps([t[0] for t in f["terms"]])
            g.attrs["transform"] = f["transform"]
            g.create_dataset("obs_eps", data=f["e"])
            g.create_dataset("fit_y", data=f["y"])
            if f.get("candidates"):
                g.create_dataset("cand_currents",
                                 data=np.array([c["currents"]
                                                for c in f["candidates"]]))
                g.create_dataset("cand_eps",
                                 data=np.array([c["eps_pred"]
                                               for c in f["candidates"]]))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
