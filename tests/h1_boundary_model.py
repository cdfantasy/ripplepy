#!/usr/bin/env python3
"""Data-driven, exportable model of the feasibility boundary (D2).

Target: predict whether a current vector belongs to the analysis population
F = {x : the certified column brackets a0 and LCFS >= a0 + margin} (D1 = A,
margin = 0.02 m).  The model is a logistic classifier on smooth features of the
normalised currents; its zero level set is the boundary and its probability is
reported with uncertainty (calibration / conservative threshold).

Candidate feature maps:
  * polynomial degree 1 / 2 / 3 (standardised currents);
  * Gaussian RBF features with centres drawn from the training folds.

Metrics: AUC, average precision (baseline = positive rate), Brier score,
TPR at FPR=5/10%, precision at TPR=90%, and the *boundary-band* AUC (positives
with LCFS in [threshold, threshold+0.02], negatives with LCFS in
[threshold-0.02, threshold)).  The last number is the honest one for "where is
the boundary"; the global AUC is dominated by easy, far-from-boundary pairs.

Writes ``tests/h1_full_scan/boundary_model.json`` and ``boundary_model.h5``
(the chosen model's coefficients / centres, exportable and usable without the
oracle).

  python -u tests/h1_boundary_model.py
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.stats import rankdata

BASE = Path(__file__).resolve().parent.parent
SCAN_H5 = BASE / "tests" / "h1_full_scan" / "scan_all.h5"
OUT_JSON = BASE / "tests" / "h1_full_scan" / "boundary_model.json"
OUT_H5 = BASE / "tests" / "h1_full_scan" / "boundary_model.h5"

A0 = 0.10
MARGIN = 0.02
THRESH = A0 + MARGIN
BAND = 0.02
N_FOLDS = 5
SEED = 0
LAM = 1e-3

BOUNDS = np.array([
    [50000.0, 50000.0], [800.0, 5500.0], [0.0, 5000.0],
    [-100000.0, 0.0], [-100000.0, 0.0],
])
FREE = np.array([1, 2, 3, 4])


def auc(y, p):
    r = rankdata(p)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def average_precision(y, p):
    o = np.argsort(-p)
    yy = y[o]
    tp = np.cumsum(yy)
    prec = tp / np.arange(1, len(yy) + 1)
    rec = tp / max(1, yy.sum())
    ap = 0.0
    prev = 0.0
    for k in range(len(yy)):
        ap += (rec[k] - prev) * prec[k]
        prev = rec[k]
    return float(ap)


def brier(y, p):
    return float(np.mean((p - y) ** 2))


def _poly(U, deg, mu=None, sd=None):
    Z = U if mu is None else (U - mu) / sd
    if mu is None:
        mu, sd = Z.mean(0), Z.std(0)
        Z = (U - mu) / sd
    cols = [np.ones(len(Z))] + [Z[:, i] for i in range(Z.shape[1])]
    if deg >= 2:
        for i in range(Z.shape[1]):
            for j in range(i, Z.shape[1]):
                cols.append(Z[:, i] * Z[:, j])
    if deg >= 3:
        for i in range(Z.shape[1]):
            for j in range(i, Z.shape[1]):
                for k in range(j, Z.shape[1]):
                    cols.append(Z[:, i] * Z[:, j] * Z[:, k])
    return np.column_stack(cols), (mu, sd)


def _rbf(U, centres, sigma):
    d2 = ((U[:, None, :] - centres[None, :, :]) ** 2).sum(-1)
    return np.exp(-d2 / (2.0 * sigma ** 2))


def _fit_logit(X, y, lam=LAM, maxiter=3000):
    def fg(w):
        z = X @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))
        f = np.mean(np.logaddexp(0, z) - y * z) + lam * w @ w
        g = X.T @ (p - y) / len(y) + 2.0 * lam * w
        return f, g
    r = minimize(fg, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B",
                 options={"maxiter": maxiter})
    return r.x


def _predict(X, w):
    return 1.0 / (1.0 + np.exp(-np.clip(X @ w, -60, 60)))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def _logit(p):
    q = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(q / (1 - q))


def _platt(p, y):
    """Platt scaling of cross-validated scores (A, B)."""
    z = _logit(p)

    def fg(ab):
        a, b = ab
        q = _sigmoid(a * z + b)
        f = np.mean(np.logaddexp(0, a * z + b) - y * (a * z + b))
        g = [np.mean((q - y) * z), np.mean(q - y)]
        return f, np.array(g)
    r = minimize(fg, [1.0, 0.0], jac=True, method="L-BFGS-B")
    return float(r.x[0]), float(r.x[1])


def _choose_thresholds(pc, y):
    out = {}
    neg = np.sort(pc[y == 0])
    for target in (0.05, 0.10):
        thr = float(np.quantile(neg, 1 - target))
        pred = pc >= thr
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        out[f"fpr{int(target*100)}"] = {
            "threshold": thr,
            "tpr": tp / max(1, int(y.sum())),
            "fpr": fp / max(1, int((1 - y).sum())),
            "precision": tp / max(1, tp + fp),
            "n_predicted": int(pred.sum()),
        }
    best = (0.0, 0.0, 0.0)
    for thr in np.quantile(pc, np.linspace(0.5, 0.999, 200)):
        pred = pc >= thr
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, int(y.sum()))
        f1 = 2 * prec * rec / max(1e-12, prec + rec)
        if f1 > best[0]:
            best = (f1, float(thr), prec)
    out["max_f1"] = {"threshold": best[1], "f1": best[0], "precision": best[2]}
    return out


def _build(kind: str, U_train, U_all, rng):
    if kind.startswith("poly"):
        deg = int(kind[-1])
        mu, sd = U_train.mean(0), U_train.std(0) + 1e-12
        return _poly(U_all, deg, mu, sd)[0]
    if kind.startswith("rbf"):
        n_c = int(kind.split("_")[1])
        idx = rng.choice(len(U_train), size=min(n_c, len(U_train)), replace=False)
        centres = U_train[idx]
        tree = cKDTree(U_train)
        d, _ = tree.query(U_train[rng.choice(len(U_train),
                                           size=min(2000, len(U_train)),
                                           replace=False)], k=2)
        sigma = 0.20  # 4-D centre spacing for ~300 centres is ~0.2
        # (median nearest-neighbour distance, ~0.05, makes the kernels far too
        # narrow; the earlier RBF AUC of 0.60 was this bug)
        return _rbf(U_all, centres, sigma), centres, sigma
    raise ValueError(kind)


def cv_scores(kind, U, y):
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(y))
    folds = np.array_split(idx, N_FOLDS)
    p = np.zeros(len(y))
    for f in folds:
        tr = np.setdiff1d(idx, f)
        out = _build(kind, U[tr], U, rng)
        X = out[0] if isinstance(out, tuple) else out
        w = _fit_logit(X[tr], y[tr])
        p[f] = _predict(X[f], w)
    return p


def evaluate(y, p, lcfs):
    pos_band = (y == 1) & (lcfs < THRESH + BAND)
    neg_band = (y == 0) & (lcfs >= THRESH - BAND)
    band = pos_band | neg_band
    band_auc = auc(y[band], p[band]) if band.sum() > 10 else np.nan
    # working points
    order = np.argsort(-p)
    tp = np.cumsum(y[order])
    fp = np.cumsum(1 - y[order])
    tpr = tp / max(1, y.sum())
    fpr = fp / max(1, (1 - y).sum())
    def tpr_at(fpr_t):
        k = int(np.searchsorted(fpr, fpr_t))
        return float(tpr[min(k, len(tpr) - 1)])
    prec90 = float(tp[np.searchsorted(tpr, 0.9)] / (np.searchsorted(tpr, 0.9) + 1)) \
        if (tpr >= 0.9).any() else np.nan
    return {
        "auc": float(auc(y, p)),
        "ap": average_precision(y, p),
        "brier": brier(y, p),
        "tpr_at_fpr5": tpr_at(0.05),
        "tpr_at_fpr10": tpr_at(0.10),
        "precision_at_tpr90": prec90,
        "boundary_band_auc": float(band_auc),
        "n_band": int(band.sum()),
        "pos_rate": float(y.mean()),
    }


def main():
    with h5py.File(SCAN_H5, "r") as f:
        S = f["samples"][:]
        lcfs = f["scalars/lcfs_a"][:]
    span = BOUNDS[FREE, 1] - BOUNDS[FREE, 0]
    U = (S[:, FREE] - BOUNDS[FREE, 0]) / span
    y = (lcfs >= THRESH).astype(np.int64)
    print(f"boundary model: N={len(U)}  positives (LCFS>={THRESH:.2f} m): "
          f"{int(y.sum())} ({y.mean() * 100:.2f}%)")

    kinds = ["poly1", "poly2", "poly3", "rbf_300"]
    results, oof = {}, {}
    for kind in kinds:
        p = cv_scores(kind, U, y)
        oof[kind] = p
        A, B = _platt(p, y)
        pc = _sigmoid(A * _logit(p) + B)
        m = evaluate(y, pc, lcfs)
        m["platt_A"], m["platt_B"] = A, B
        results[kind] = m
        print(f"  {kind:8s} AUC={m['auc']:.3f}  AP={m['ap']:.3f}  "
              f"Brier={m['brier']:.4f}  TPR@FPR5={m['tpr_at_fpr5']:.2f}  "
              f"TPR@FPR10={m['tpr_at_fpr10']:.2f}  "
              f"P@TPR90={m['precision_at_tpr90']:.2f}  "
              f"bandAUC={m['boundary_band_auc']:.3f} (n={m['n_band']})")

    # export the best smooth model (highest AP; report all metrics)
    best = max(results, key=lambda k: results[k]["ap"])
    print(f"  chosen exportable model: {best}")
    rng = np.random.default_rng(SEED)
    out = _build(best, U, U, rng)
    X = out[0] if isinstance(out, tuple) else out
    w = _fit_logit(X, y)
    A, B = _platt(oof[best], y)
    pc = _sigmoid(A * _logit(oof[best]) + B)
    thresholds = _choose_thresholds(pc, y)
    print(f"  Platt: A={A:.3f} B={B:.3f}")
    for k, v in thresholds.items():
        print(f"    threshold {k}: p={v['threshold']:.4f}  "
              + (f"TPR={v['tpr']:.2f} FPR={v['fpr']:.2f} "
                 f"precision={v['precision']:.2f}" if k != "max_f1"
                 else f"F1={v['f1']:.2f} precision={v['precision']:.2f}"))
    with h5py.File(OUT_H5, "w") as f:
        f.attrs["kind"] = best
        f.attrs["a0"] = A0
        f.attrs["margin"] = MARGIN
        f.attrs["threshold"] = THRESH
        f.create_dataset("coef", data=w)
        f.attrs["platt_A"], f.attrs["platt_B"] = A, B
        for k, v in thresholds.items():
            f.attrs[f"thr_{k}"] = v["threshold"]
        if best.startswith("rbf"):
            f.create_dataset("centres", data=out[1])
            f.attrs["sigma"] = out[2]
        else:
            f.create_dataset("mu", data=U.mean(0))
            f.create_dataset("sd", data=U.std(0))
    OUT_JSON.write_text(json.dumps(
        {"threshold": THRESH, "margin": MARGIN, "chosen": best,
         "platt_A": A, "platt_B": B, "operating_points": thresholds,
         "metrics": results}, indent=2) + "\n")
    print(f"  saved -> {OUT_JSON} and {OUT_H5}")


if __name__ == "__main__":
    main()
