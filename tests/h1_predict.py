#!/usr/bin/env python3
"""Query the H1 feasibility / eps estimator for one or many current sets.

Feasibility definition: a certified surface reaches a = 0.10 m with
LCFS >= 0.12 m (2 cm margin) under the scan gates.

Usage:
    python -u tests/h1_predict.py "50000,1638,10,-5368,-33476"
    python -u tests/h1_predict.py "1638,10,-5368,-33476"          # TF=50 kA
    python -u tests/h1_predict.py --mode balanced "50000,..."
    python -u tests/h1_predict.py --batch currents.csv --out result.csv
    python -u tests/h1_predict.py --demo

Modes:
    screening   : candidate if d <= 0.10  (recall ~96%, precision ~25%)
    balanced    : candidate if d <= 0.08  (recall ~88%, precision ~34%)
    conservative: candidate if d <= 0.06  (recall ~59%, precision ~48%)
In all modes d > 0.12 -> likely infeasible (0.05% feasible in the 32k test).

The output is an estimate, not a certificate.  For final design decisions run
the full-tier measurement on the candidate (about 20 s).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "tests"))
from h1_surrogate_lib import (BOUNDS, FREE, load_library, mahalanobis,  # noqa: E402
                              predict_eps, normalised_currents)
import h1_boundary_model as bm  # noqa: E402

DATA = BASE / "tests" / "h1_full_scan"
A0 = 0.10
D_REJ = 0.12
MODES = {"screening": 0.10, "balanced": 0.08, "conservative": 0.06}
P_FPR5 = 0.201


class Predictor:
    def __init__(self, scan_h5: Path | None = None,
                 boundary_h5: Path | None = None,
                 lib_h5: Path | None = None):
        scan_h5 = scan_h5 or DATA / "scan_all.h5"
        boundary_h5 = boundary_h5 or DATA / "boundary_model.h5"
        lib_h5 = lib_h5 or DATA / "surrogate_lib.h5"
        with h5py.File(scan_h5, "r") as f:
            Str = f["samples"][:]
            ltr = f["scalars/lcfs_a"][:]
        self.U = normalised_currents(Str)
        y = ltr >= A0 + 0.02
        self.tree_feas = cKDTree(self.U[y])
        self.tree_infeas = cKDTree(self.U[~y])
        self.tree_all = cKDTree(self.U)
        with h5py.File(boundary_h5, "r") as f:
            self.coef = f["coef"][:]
            self.A = float(f.attrs["platt_A"])
            self.B = float(f.attrs["platt_B"])
        self.mu, self.sd = self.U.mean(0), self.U.std(0) + 1e-12
        self.lib = load_library(lib_h5)
        hyb = DATA / "hybrid_boundary.h5"
        self.hyb = None
        if hyb.exists():
            with h5py.File(hyb, "r") as f:
                self.hyb = {"w": f["w"][:], "mu": f["mu"][:], "sd": f["sd"][:],
                            "A": float(f.attrs["platt_A"]),
                            "B": float(f.attrs["platt_B"]),
                            "thr5": float(f.attrs["thr_fpr5"])}

    def check_box(self, cur):
        if cur.shape != (5,):
            return "need 5 currents (TF,I1,I2,I3,I4) in A"
        for k in range(5):
            lo, hi = BOUNDS[k]
            if not (lo - 1e-6 <= cur[k] <= hi + 1e-6):
                return (f"out of scanned box: coil {k} = {cur[k]:.1f} A, "
                        f"allowed [{lo:.1f}, {hi:.1f}] A")
        return None

    def predict(self, cur, mode="screening"):
        d_acc = MODES[mode]
        err = self.check_box(cur)
        if err:
            return {"state": "out_of_box", "mode": mode, "note": err}
        u = (cur[FREE] - BOUNDS[FREE, 0]) / (BOUNDS[FREE, 1] - BOUNDS[FREE, 0])
        X, _ = bm._poly(u.reshape(1, -1), 3, self.mu, self.sd)
        p12 = float(bm._sigmoid(self.A * (X @ self.coef)[0] + self.B))
        d1 = float(self.tree_feas.query(u.reshape(1, -1), k=1)[0][0])
        d5 = float(self.tree_feas.query(u.reshape(1, -1), k=5)[0][0][-1])
        dm = float(mahalanobis(self.lib, cur, A0))
        eps = float(predict_eps(self.lib, cur, A0))
        p_hyb = None
        if self.hyb is not None:
            di = float(self.tree_infeas.query(u.reshape(1, -1), k=1)[0][0])
            nf = self.tree_feas.query_ball_point(u.reshape(1, -1), 0.10,
                                                 return_length=True)[0]
            na = self.tree_all.query_ball_point(u.reshape(1, -1), 0.10,
                                                return_length=True)[0]
            frac = float(nf) / (float(na) + 1e-9)
            z = float(np.log(max(p12, 1e-9) / max(1 - p12, 1e-9)))
            F = np.array([z, d1, d5, di, frac, dm, di / (d1 + 1e-9)])
            Fs = (F - self.hyb["mu"]) / self.hyb["sd"]
            raw = float(Fs @ self.hyb["w"])
            p_hyb = float(1.0 / (1.0 + np.exp(-np.clip(
                self.hyb["A"] * raw + self.hyb["B"], -60, 60))))
        if d1 <= d_acc:
            state = "candidate_feasible"
            conf = ("high" if mode == "conservative" else
                    "medium" if mode == "balanced" else "screening")
        elif d1 > D_REJ:
            state = "likely_infeasible"
            conf = "rejected"
        else:
            state = "unknown_near_boundary"
            conf = "unknown"
        out = {"state": state, "mode": mode, "confidence": conf,
               "p12": p12, "d1": d1, "d5": d5, "dM": dm,
               "eps_pred": eps, "eps_error": self._band(dm, d1, state)}
        if p_hyb is not None:
            out["p_hybrid"] = p_hyb
            out["p_hybrid_thr_fpr5"] = self.hyb["thr5"]
        return out

    @staticmethod
    def _band(dm, d1, state):
        if state == "likely_infeasible":
            return "eps undefined if no surface; prediction for reference only"
        if dm <= 2.6 and d1 <= 0.08:
            return "typical 1.4%, p90 4.3% (walk); 1.3%/4.35% (complete full tier)"
        if dm <= 2.6:
            return "typical 1.7%, p90 8.3%; near-support edge, prefer measurement"
        if dm <= 3.2:
            return "typical 1.5%, p90 11%; tail extrapolation"
        if dm <= 4.0:
            return "typical 4.6%, p90 17%; tail"
        return "typical 8%, p90 26%+; full-tier measurement required"


def _parse(s):
    vals = [float(v) for v in s.replace(",", " ").split()]
    if len(vals) == 4:
        vals = [50000.0] + vals
    return np.array(vals, float)


def demo():
    pred = Predictor()
    cases = {
        "measured best (feasible, tail)": [50000, 1638, 10, -5368, -33476],
        "feasible interior (LCFS=0.155)": [50000, 4407.1, 2464.4, -82653.6, -62341.7],
        "just below threshold (LCFS=0.102)": [50000, 3431.0, 2144.9, -10922.8, -66632.1],
        "clearly infeasible (LCFS=0.018)": [50000, 2034.5, 2324.4, -32801.5, -74462.8],
    }
    for name, c in cases.items():
        print(f"\n{name}: {c}")
        print(pred.predict(np.array(c, float), "screening"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("currents", nargs="?", help="4 or 5 currents in A")
    ap.add_argument("--mode", choices=list(MODES), default="screening")
    ap.add_argument("--batch", help="CSV file with 4 or 5 current columns (no header)")
    ap.add_argument("--out", help="output CSV for --batch")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    if args.demo:
        demo()
        return
    pred = Predictor()
    if args.batch:
        rows = []
        with open(args.batch) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                cur = _parse(line)
                r = pred.predict(cur, args.mode)
                r["I_TF_A"], r["I1_A"], r["I2_A"], r["I3_A"], r["I4_A"] = cur
                rows.append(r)
        keys = ["state", "mode", "confidence", "p12", "d1", "d5", "dM",
                "eps_pred", "eps_error", "I_TF_A", "I1_A", "I2_A", "I3_A", "I4_A"]
        out = args.out or "predictions.csv"
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in keys})
        print(f"wrote {len(rows)} rows -> {out}")
        return
    if not args.currents:
        ap.error("give 4 or 5 currents in A, or use --demo/--batch")
    print(pred.predict(_parse(args.currents), args.mode))


if __name__ == "__main__":
    main()
