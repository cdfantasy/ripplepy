#!/usr/bin/env python3
"""Stage 5: local densification around the best measured configurations.

Why this instead of "let the quadratic extrapolate":
stage 4 showed that the fitted quadratic, pushed outside the cloud of measured
points, predicts improvements that are not merely inaccurate but *infeasible* --
all four of its candidates had no closed flux surface at the reference radius.
The model only knows eps, it has no notion of feasibility, and its Hessian along
the escape direction is nearly flat (|lambda| ~ 0.01), so a long extrapolation
is driven by a linear term the data never constrained.

So the model is used only where it is validated: inside a Mahalanobis radius of
~2, where its cross-validated error is 1.3 %.  Here it is used for nothing more
than choosing *where* to sample: around each of the best observed points the
local covariance defines a small ellipsoid, and new currents are drawn from it at
a fraction of the local scale.  Every candidate goes through the full oracle
(walk + eps on the reference surface), so nothing is trusted without measurement.

  python -u tests/h1_local_search.py
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from ripplepy.radial_scan import ScanConfig, load_scan_h5, scan_box

BASE = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
MGRID_PATH = BASE / "tests" / "test_file" / "mgrid_h1_design.nc"
NFP = 3
EPS_H5 = BASE / "tests" / "h1_radial_scan" / "eps_ref.h5"
OUT_H5 = BASE / "tests" / "h1_radial_scan" / "local_search.h5"

BOUNDS = np.array([
    [50000.0, 50000.0],     # coil 0: TF, fixed
    [0.0, 10000.0],         # coil 1
    [0.0, 10000.0],         # coil 2
    [-100000.0, 0.0],       # coil 3
    [-100000.0, 0.0],       # coil 4
])

A0 = 0.10           # reference minor radius (must match eps_ref.h5)
QC_MAX = 1.5        # grad-psi drift cut used in stage 3/4
N_SEEDS = 8         # best observed points used as centres
PER_SEED = 24       # candidates drawn around each seed
K_LOCAL = 40        # neighbours defining the local covariance
SHRINK = 0.35       # draw at 0.35 of the local (1 sigma) scale
MIN_SEP = 0.25      # reject candidates closer than this (x local NN) to an
                    # existing sample -- they would add no new information
DRY_RUN = False     # True: only generate and report, do not run the oracle
SEED_RNG = 0
PROCESSES = 64      # 0 = serial
DR_MAX = 0.20
NPHI = 360
PROFILE_NTURN = 100
PROFILE_NPART = 200
FULL_NTURN = 400
FULL_NPART = 2000
RESIDUAL_TOL = 0.05
GAP_NORM_TOL = 15.0


# ---------------------------------------------------------------------------
def main():
    with h5py.File(EPS_H5, "r") as f:
        S = f["samples"][:]
        e = f["ref"]["eps"][:]
        qc = f["ref"]["gradpsi_drift"][:]
        a0_file = float(f.attrs["a0"])
    assert abs(a0_file - A0) < 1e-9, f"eps_ref was built with a0={a0_file}"

    good = np.isfinite(e) & (e > 0) & np.isfinite(qc) & (qc <= QC_MAX)
    S, e = S[good], e[good]
    order = np.argsort(e)
    print(f"{len(S)} clean columns, best eps {e[order[0]]:.6f}")

    span = BOUNDS[:, 1] - BOUNDS[:, 0]
    free = span > 0
    U = (S[:, free] - BOUNDS[free, 0]) / span[free]

    from scipy.spatial import cKDTree
    tree = cKDTree(U)
    nn = tree.query(U, k=2)[0][:, 1]
    print(f"local spacing: median NN {np.median(nn):.4f}")

    rng = np.random.default_rng(SEED_RNG)
    seeds = order[:N_SEEDS]
    cands = []
    for i in seeds:
        d, idx = tree.query(U[i], k=K_LOCAL)
        loc = U[idx]
        cov = np.cov(loc.T) + np.eye(U.shape[1]) * 1e-10
        step = SHRINK * np.sqrt(np.diag(cov))
        for _ in range(PER_SEED):
            u = U[i] + rng.normal(0.0, step)
            u = np.clip(u, 0.0, 1.0)
            cands.append(u)
    cands = np.array(cands)
    # drop candidates that sit on top of an existing sample (no new information)
    # and duplicates among themselves
    keep = []
    lim = MIN_SEP * float(np.median(nn))
    for u in cands:
        if tree.query(u, k=1)[0] < lim:
            continue
        if keep and min(np.linalg.norm(u - k) for k in keep) < 0.01:
            continue
        keep.append(u)
    cands = np.array(keep) if keep else np.zeros((0, U.shape[1]))
    cur = np.tile(BOUNDS[:, 0], (len(cands), 1))
    cur[:, free] = BOUNDS[free, 0] + cands * span[free]
    print(f"generated {len(cands)} candidates around {len(seeds)} seeds "
          f"(shrunk to {SHRINK} of the local scale; separation limit {lim:.4f})")
    if DRY_RUN or len(cands) == 0:
        print("dry run, stopping before the oracle")
        for c in cands[:5]:
            print("  u=", np.round(c, 4))
        return

    cfg = ScanConfig(
        mgrid_path=str(MGRID_PATH), nfp=NFP, dr_max=DR_MAX, nphi=NPHI,
        profile_nturn=PROFILE_NTURN, profile_npart=PROFILE_NPART,
        full_nturn=FULL_NTURN, full_npart=FULL_NPART,
        residual_tol=RESIDUAL_TOL, gap_norm_tol=GAP_NORM_TOL)
    flat = scan_box(cfg, BOUNDS, samples=cur, processes=PROCESSES,
                    output_h5=Path(OUT_H5), progress_interval=32,
                    eps_at_a=A0)

    new = flat["scalars"]["ref_eps"]
    ok = np.isfinite(new)
    if ok.any():
        j = int(np.argmin(np.where(ok, new, np.inf)))
        print(f"\nnew best eps {new[j]:.6f} at "
              f"lcfs_dr={flat['scalars']['lcfs_dr'][j]:.2f}, "
              f"V={flat['scalars']['lcfs_volume'][j]:.3f}")
        print("  extcur=[%s]" % ", ".join(f"{x:.1f}" for x in cur[j]))
        print(f"  improvement over previous best: "
              f"{100 * (new[j] / e[order[0]] - 1):+.1f}%")
    else:
        print("\nno candidate produced an eps value")


if __name__ == "__main__":
    main()
