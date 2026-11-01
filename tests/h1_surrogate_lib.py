#!/usr/bin/env python3
"""Build the multi-radius quadratic surrogate library for the H1 scan.

Every slice is fitted independently: the surfaces of each configuration are
interpolated along the radial column to exactly ``a = a0`` (log-eps
interpolation), the model is a quadratic in the PCA-whitened normalised coil
currents, and the accuracy is the k-fold cross-validated *relative* error of
``eps_eff^(3/2)``.

The resulting ``surrogate_lib.h5`` contains, per slice

  * the whitening (centre, PCA basis, per-direction scale),
  * the quadratic coefficients and derived diagnostics (R^2, CV error,
    condition number, Hessian eigenvalues),
  * the support sample (normalised currents, eps, Mahalanobis radius),

together with the current-box definition, so that ``predict_eps`` /
``mahalanobis`` below can be used without re-reading the scan.

Usage
-----
  python -u tests/h1_surrogate_lib.py              # build + report + write h5
  python -u tests/h1_surrogate_lib.py --check      # rebuild, compare to the
                                                   # reference numbers only
  python -u tests/h1_surrogate_lib.py --query 50000,2000,10,-5000,-33000,0.10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

BASE = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
SCAN_H5 = BASE / "tests" / "h1_full_scan" / "scan_all.h5"
OUT_H5 = BASE / "tests" / "h1_full_scan" / "surrogate_lib.h5"

A0_LIST = (0.05, 0.06, 0.08, 0.10, 0.12)
MARGIN = 0.02          # D1 = A: LCFS >= a0 + MARGIN for the analysis population

BOUNDS = np.array([
    [50000.0, 50000.0],     # coil 0: TF, fixed
    [800.0, 5500.0],        # coil 1
    [0.0, 5000.0],          # coil 2
    [-100000.0, 0.0],       # coil 3
    [-100000.0, 0.0],       # coil 4
])
FREE = np.array([1, 2, 3, 4])          # coils that are actually varied

QC_MAX = 1.5            # grad-psi drift above this = corrupted eps, drop it
N_FOLDS = 5
MAX_DIMS = 4            # PCA directions used in the fit
SD_FRACTION = 0.05      # a direction is usable if sd > SD_FRACTION * sd_max
SEED = 0

# Reference numbers from the hand-off note (used as a regression check only).
_EXPECTED = {
    0.05: {"n": 6884, "cv_med": 0.023, "cv_p90": 0.059, "r2": 0.987},
    0.06: {"n": 5744, "cv_med": 0.022, "cv_p90": 0.058, "r2": 0.982},
    0.08: {"n": 3961, "cv_med": 0.025, "cv_p90": 0.062, "r2": 0.967},
    0.10: {"n": 2489, "cv_med": 0.030, "cv_p90": 0.110, "r2": 0.916},
    0.12: {"n": 1257, "cv_med": 0.047, "cv_p90": 0.144, "r2": 0.849},
}


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_reference(path: Path = SCAN_H5, a0: float = 0.10):
    """Interpolate every surface quantity to exactly a = a0.

    The nearest face is not good enough: the radial grid is coarse (dr = 0.02
    plus one bisection) and the nearest face to a0 sits anywhere in
    [0.082, 0.116], while eps varies steeply with radius.  Each quantity is
    therefore interpolated along the column (log eps, linear for the rest), and
    only columns that bracket a0 are used.
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
    with h5py.File(path, "r") as f:
        ref["lcfs_a"] = f["scalars/lcfs_a"][:]
    for i in range(n):
        m = ok[i] & np.isfinite(prof["a"][i]) & (prof["a"][i] > 0)
        if m.sum() < 2:
            continue
        idx = np.flatnonzero(m)
        am = prof["a"][i][m]
        em = prof["eps"][i][m]
        if not (np.isfinite(em).all() and em.min() > 0):
            continue
        o = np.argsort(am)
        am, em = am[o], em[o]
        if not (am[0] < a0 < am[-1] or np.any(np.abs(am - a0) < 1e-6)):
            continue
        idx = idx[o]
        ref["eps"][i] = float(np.exp(np.interp(a0, am, np.log(em))))
        ref["a"][i] = a0
        for k in ("dr", "iota", "volume", "elongation"):
            v = prof[k][i][idx]
            if np.isfinite(v).all():
                ref[k][i] = float(np.interp(a0, am, v))
        j = int(np.argmin(np.abs(am - a0)))     # drift is a QC ratio
        ref["gradpsi_drift"][i] = prof["gradpsi_drift"][i][idx[j]]
    return S, ref


def normalised_currents(S: np.ndarray) -> np.ndarray:
    span = BOUNDS[FREE, 1] - BOUNDS[FREE, 0]
    return (S[:, FREE] - BOUNDS[FREE, 0]) / span


# ---------------------------------------------------------------------------
# quadratic model
# ---------------------------------------------------------------------------
def quad_terms(n_dims: int):
    terms = [("1", ())]
    for i in range(n_dims):
        terms.append((f"v{i}", (i,)))
    for i in range(n_dims):
        for j in range(i, n_dims):
            terms.append((f"v{i}v{j}", (i, j)))
    return terms


def design(v: np.ndarray, terms) -> np.ndarray:
    cols = []
    for _, idx in terms:
        if not idx:
            cols.append(np.ones(len(v)))
        else:
            cols.append(np.prod(v[:, idx], axis=1))
    return np.column_stack(cols)


def fit_quadratic(v, y, terms):
    X = design(v, terms)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef, X


def cross_validate(v, e, terms, n_folds: int, rng) -> np.ndarray:
    """k-fold CV, reported as the relative error in eps for every point."""
    y = np.log(e)
    n = len(y)
    idx = rng.permutation(n)
    folds = np.array_split(idx, min(n_folds, n))
    err = np.full(n, np.nan)
    for f in folds:
        tr = np.setdiff1d(idx, f)
        coef, _ = fit_quadratic(v[tr], y[tr], terms)
        pred = np.exp(design(v[f], terms) @ coef)
        err[f] = np.abs(pred - e[f]) / np.maximum(e[f], 1e-30)
    return err


def hessian_from_coef(coef, terms, n_dims: int) -> np.ndarray:
    H = np.zeros((n_dims, n_dims))
    names = [t[0] for t in terms]
    for name, idx in terms:
        if len(idx) == 2:
            k = names.index(name)
            if idx[0] == idx[1]:
                H[idx[0], idx[0]] = 2.0 * coef[k]
            else:
                H[idx[0], idx[1]] = coef[k]
                H[idx[1], idx[0]] = coef[k]
    return H


def fit_slice(S: np.ndarray, ref: dict, a0: float, rng) -> dict:
    """Fit one radius slice; returns everything needed for prediction."""
    e = ref["eps"]
    qc = ref["gradpsi_drift"]
    good = (np.isfinite(e) & (e > 0) & np.isfinite(qc) & (qc <= QC_MAX)
            & (ref["lcfs_a"] >= a0 + MARGIN))
    U = normalised_currents(S)[good]
    E = e[good]

    centre = U.mean(axis=0)
    d = U - centre
    _, _, vt = np.linalg.svd(d, full_matrices=False)
    proj = d @ vt.T
    sd_all = proj.std(axis=0)
    keep = np.flatnonzero(sd_all > SD_FRACTION * sd_all[0])[:MAX_DIMS]
    if len(keep) == 0:
        raise RuntimeError(f"no usable PCA direction at a0={a0}")
    sd = sd_all[keep]
    v = proj[:, keep] / sd
    terms = quad_terms(len(keep))

    coef, X = fit_quadratic(v, np.log(E), terms)
    pred = X @ coef
    ss_res = np.sum((np.log(E) - pred) ** 2)
    ss_tot = np.sum((np.log(E) - np.log(E).mean()) ** 2)
    cv = cross_validate(v, E, terms, N_FOLDS, rng)
    H = hessian_from_coef(coef, terms, len(keep))

    return {
        "a0": float(a0),
        "n": int(good.sum()),
        "centre": centre,
        "basis": vt[keep],
        "sd": sd,
        "sd_all": sd_all,
        "keep": keep,
        "coef": coef,
        "terms": terms,
        "r2": float(1.0 - ss_res / max(ss_tot, 1e-30)),
        "cv_med": float(np.nanmedian(cv)),
        "cv_p90": float(np.nanpercentile(cv, 90)),
        "cond": float(np.linalg.cond(X)),
        "hessian": H,
        "eig": np.linalg.eigvalsh(H),
        "U": U,
        "eps": E,
        "dm": np.linalg.norm(v, axis=1),   # Mahalanobis radius of members
    }


# ---------------------------------------------------------------------------
# build / I/O / query
# ---------------------------------------------------------------------------
def build(scan_h5: Path = SCAN_H5, out_h5: Path = OUT_H5) -> dict:
    rng = np.random.default_rng(SEED)
    slices = []
    for a0 in A0_LIST:
        S, ref = load_reference(scan_h5, a0)
        sl = fit_slice(S, ref, a0, rng)
        slices.append(sl)
        exp = _EXPECTED.get(a0) if MARGIN == 0.0 else None
        if exp:
            print(f"  a0={a0:.2f}  n={sl['n']:5d}  "
                  f"CV med={sl['cv_med'] * 100:4.1f}%  "
                  f"p90={sl['cv_p90'] * 100:4.1f}%  R2={sl['r2']:.3f}  "
                  f"eig={np.array2string(sl['eig'], precision=3)}")

    out_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_h5, "w") as f:
        f.attrs["bounds"] = BOUNDS
        f.attrs["margin"] = MARGIN
        f.attrs["free"] = FREE
        f.attrs["qc_max"] = QC_MAX
        f.attrs["n_folds"] = N_FOLDS
        f.attrs["seed"] = SEED
        f.attrs["transform"] = "log"
        for i, sl in enumerate(slices):
            g = f.create_group(f"slice_{i}")
            g.attrs["a0"] = sl["a0"]
            g.attrs["n"] = sl["n"]
            g.attrs["r2"] = sl["r2"]
            g.attrs["cv_med"] = sl["cv_med"]
            g.attrs["cv_p90"] = sl["cv_p90"]
            g.attrs["cond"] = sl["cond"]
            g.attrs["sd_all"] = sl["sd_all"]
            g.attrs["keep"] = sl["keep"]
            g.attrs["eig"] = sl["eig"]
            g.create_dataset("centre", data=sl["centre"])
            g.create_dataset("basis", data=sl["basis"])
            g.create_dataset("sd", data=sl["sd"])
            g.create_dataset("coef", data=sl["coef"])
            g.create_dataset("hessian", data=sl["hessian"])
            g.create_dataset("U", data=sl["U"])
            g.create_dataset("eps", data=sl["eps"])
            g.create_dataset("dm", data=sl["dm"])
    print(f"  saved -> {out_h5}")
    return {"slices": slices, "bounds": BOUNDS, "free": FREE}


def load_library(path: Path = OUT_H5) -> dict:
    out = {"slices": []}
    with h5py.File(path, "r") as f:
        out["bounds"] = f.attrs["bounds"]
        out["free"] = f.attrs["free"]
        for name in f:
            g = f[name]
            out["slices"].append({
                "a0": float(g.attrs["a0"]),
                "n": int(g.attrs["n"]),
                "r2": float(g.attrs["r2"]),
                "cv_med": float(g.attrs["cv_med"]),
                "cv_p90": float(g.attrs["cv_p90"]),
                "cond": float(g.attrs["cond"]),
                "sd": g["sd"][:],
                "eig": g.attrs["eig"],
                "centre": g["centre"][:],
                "basis": g["basis"][:],
                "coef": g["coef"][:],
                "U": g["U"][:],
                "eps": g["eps"][:],
                "dm": g["dm"][:],
            })
    out["slices"].sort(key=lambda d: d["a0"])
    return out


def _free_currents(currents) -> np.ndarray:
    """Accept a free (4,) current vector or a full (5,) extcur vector."""
    x = np.asarray(currents, dtype=np.float64)
    if x.shape[-1] == 5:
        x = x[..., FREE]
    if x.shape[-1] != 4:
        raise ValueError(f"expected 4 free or 5 total currents, got shape {x.shape}")
    return x


def slice_value(lib: dict, sl: dict, currents) -> float:
    span = lib["bounds"][FREE, 1] - lib["bounds"][FREE, 0]
    x = _free_currents(currents)
    u = (x - lib["bounds"][FREE, 0]) / span
    v = (sl["basis"] @ (u - sl["centre"])) / sl["sd"]
    return float(design(v.reshape(1, -1), quad_terms(len(sl["sd"])))[0]
                 @ sl["coef"])


def predict_eps(lib: dict, currents, a: float) -> float:
    """Predict eps_eff^(3/2) at minor radius ``a`` (log-linear in a)."""
    a0s = [sl["a0"] for sl in lib["slices"]]
    if a < a0s[0] - 1e-9 or a > a0s[-1] + 1e-9:
        raise ValueError(f"a={a} outside the fitted range [{a0s[0]}, {a0s[-1]}]")
    j = int(np.argmin(np.abs(np.asarray(a0s) - a)))
    if abs(a0s[j] - a) < 1e-9:
        return float(np.exp(slice_value(lib, lib["slices"][j], currents)))
    hi = j if a0s[j] > a else j + 1
    lo = hi - 1
    y_lo = slice_value(lib, lib["slices"][lo], currents)
    y_hi = slice_value(lib, lib["slices"][hi], currents)
    w = (a - a0s[lo]) / (a0s[hi] - a0s[lo])
    return float(np.exp((1 - w) * y_lo + w * y_hi))


def mahalanobis(lib: dict, currents, a: float) -> float:
    """Mahalanobis radius of a point w.r.t. the nearest slice's support."""
    a0s = [sl["a0"] for sl in lib["slices"]]
    j = int(np.argmin(np.abs(np.asarray(a0s) - a)))
    sl = lib["slices"][j]
    span = lib["bounds"][FREE, 1] - lib["bounds"][FREE, 0]
    u = (_free_currents(currents) - lib["bounds"][FREE, 0]) / span
    v = (sl["basis"] @ (u - sl["centre"])) / sl["sd"]
    return float(np.linalg.norm(v))


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="build and compare against the reference numbers")
    ap.add_argument("--query", default=None,
                    help="comma-separated extcur (5 values) [, a]")
    args = ap.parse_args()

    if args.query:
        lib = load_library()
        vals = [float(v) for v in args.query.split(",")]
        extcur, a = (vals[:-1], vals[-1]) if len(vals) == 6 else (vals, 0.10)
        print(f"extcur={extcur}  a={a}")
        print(f"  eps_eff^(3/2) = {predict_eps(lib, extcur, a):.6f}")
        print(f"  Mahalanobis d_M = {mahalanobis(lib, extcur, a):.3f}")
        return

    print("building multi-radius quadratic surrogate library")
    lib = build()
    if args.check:
        ok = True
        for sl in lib["slices"]:
            exp = _EXPECTED.get(round(sl["a0"], 2))
            if exp is None:
                continue
            good = (sl["n"] == exp["n"]
                    and abs(sl["cv_med"] - exp["cv_med"]) <= 0.005
                    and abs(sl["cv_p90"] - exp["cv_p90"]) <= 0.006
                    and abs(sl["r2"] - exp["r2"]) <= 0.005)
            ok &= good
            print(f"  a0={sl['a0']:.2f}: {'OK' if good else 'MISMATCH'} "
                  f"(n {sl['n']}/{exp['n']}, "
                  f"CV {sl['cv_med'] * 100:.1f}/{exp['cv_med'] * 100:.1f}%, "
                  f"p90 {sl['cv_p90'] * 100:.1f}/{exp['cv_p90'] * 100:.1f}%, "
                  f"R2 {sl['r2']:.3f}/{exp['r2']:.3f})")
        print("  regression check:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
