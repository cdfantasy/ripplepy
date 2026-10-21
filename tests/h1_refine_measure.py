#!/usr/bin/env python3
"""Re-measure the best configurations of the one-pass scan at the full tier.

The one-pass scan measures eps on every surface at the cheap walk tier
(nturn=100, npart=500).  That is fine for ranking and for mapping eps(a), but
the absolute values are a few percent off the full tier (nturn=400, npart=2000)
used for published numbers.  This script therefore takes the best N
configurations and re-measures two surfaces at the full tier:

    * the reference surface  a = A0
    * the outermost surface  dr = lcfs_dr

It also prints the ratio full/walk eps at the reference surface, which is the
calibration factor between the two tiers.

  python -u tests/h1_refine_measure.py
"""

from __future__ import annotations

import time
from pathlib import Path

import h5py
import numpy as np

from ripplepy import initialize_mgrid_field, set_extcur
from ripplepy.radial_scan import (ScanConfig, TIER_FULL, _axis_ref,
                                  load_scan_h5, measure_surface)

BASE = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
SCAN_H5 = BASE / "tests" / "h1_full_scan" / "scan_all.h5"
OUT_H5 = BASE / "tests" / "h1_full_scan" / "eps_refine.h5"

A0 = 0.10          # reference minor radius [m]
N_BEST = 500       # how many configurations to re-measure
FULL_NTURN = 400
FULL_NPART = 2000
NPHI = 360
QC_MAX = 1.5       # grad-psi drift above this is flagged, not filtered
PROCESSES = 64

_R = {}


def _init(cfg):
    _R["cfg"] = cfg


def _dr_at_a(dr, a, a_target):
    """Interpolate the dr where the column reaches minor radius `a_target`."""
    m = np.isfinite(dr) & np.isfinite(a) & (a > 0)
    if m.sum() < 2:
        return None
    x, y = dr[m], a[m]
    o = np.argsort(y)
    y, x = y[o], x[o]
    if a_target < y[0] or a_target > y[-1]:
        return None
    return float(np.interp(a_target, y, x))


def _measure_one(args):
    i, extcur, axis_rz, dr_a0, dr_lcfs = args
    cfg = _R["cfg"]
    extcur = np.asarray(extcur, dtype=np.float64)
    set_extcur(extcur)
    ax = _axis_ref(np.asarray(axis_rz, dtype=np.float64), cfg.nphi)
    if ax is None:
        return i, None
    out = {"dr_lcfs": dr_lcfs}
    for tag, dr in (("a0", dr_a0), ("lcfs", dr_lcfs)):
        if dr is None or not np.isfinite(dr):
            continue
        row = measure_surface(extcur, ax, float(dr), FULL_NTURN, NPHI,
                              FULL_NPART, TIER_FULL, cfg, compute_eps=True)
        out[f"dr_{tag}"] = float(dr)
        out[f"a_{tag}"] = row["a"]
        out[f"eps_{tag}"] = row["eps"]
        out[f"drift_{tag}"] = row["gradpsi_drift"]
        out[f"volume_{tag}"] = row["volume"]
        out[f"iota_{tag}"] = row["iota"]
        out[f"status_{tag}"] = row["status"]
    return i, out


def main():
    cfg = ScanConfig(mgrid_path=str(BASE / "tests" / "test_file"
                                    / "mgrid_h1_design.nc"),
                     nfp=3, nphi=NPHI)
    initialize_mgrid_field(cfg.mgrid_path, cfg.nfp)

    d = load_scan_h5(SCAN_H5)
    sc, prof, S = d["scalars"], d["profile"], d["samples"]
    dr, a, eps_walk = prof["dr"], prof["a"], prof["eps"]
    ok = np.isfinite(eps_walk) & (eps_walk > 0)
    print(f"scan: {len(S):,} configurations, {int(ok.sum()):,} walks with a"
          f" screening eps")

    # screening value at the reference surface: the face closest to a = A0
    idx_a0 = np.full(len(S), -1)
    eps_a0_walk = np.full(len(S), np.nan)
    for i in range(len(S)):
        m = np.isfinite(a[i]) & ok[i]
        if not m.any():
            continue
        j = int(np.argmin(np.abs(a[i][m] - A0)))
        idx_a0[i] = np.flatnonzero(m)[j]
        eps_a0_walk[i] = eps_walk[i][m][j]
    good = np.isfinite(eps_a0_walk)
    print(f"  {int(good.sum()):,} with a screening eps at a ~ {A0} m")

    order = np.argsort(np.where(good, eps_a0_walk, np.inf))
    sel = order[:min(N_BEST, int(good.sum()))]
    print(f"  re-measuring the best {len(sel)} at the full tier"
          f" (nturn={FULL_NTURN}, npart={FULL_NPART})", flush=True)

    tasks = []
    for i in sel:
        j = idx_a0[i]
        dr_a0 = float(dr[i][j]) if j >= 0 else None
        lc = sc["lcfs_dr"][i]
        tasks.append((int(i), S[i], np.array([sc["axis_R"][i],
                                              sc["axis_Z"][i]]),
                      dr_a0, float(lc) if np.isfinite(lc) else None))

    t0 = time.perf_counter()
    res: list = [None] * len(tasks)
    if PROCESSES > 0:
        import multiprocessing
        with multiprocessing.Pool(PROCESSES, initializer=_init,
                                  initargs=(cfg,)) as pool:
            for k, (pos, out) in enumerate(
                    pool.imap_unordered(_measure_one, tasks, chunksize=2),
                    start=1):
                res[pos] = out
                if k % 50 == 0:
                    print(f"    {k}/{len(tasks)}  {time.perf_counter() - t0:.0f}s",
                          flush=True)
    else:
        _init(cfg)
        for pos, task in enumerate(tasks):
            _, out = _measure_one(task)
            res[pos] = out
    print(f"  measured in {time.perf_counter() - t0:.0f}s", flush=True)

    # ---- collect and save ------------------------------------------------
    rows = [(t[0], r) for t, r in zip(tasks, res) if r is not None]
    print(f"  {len(rows)}/{len(tasks)} succeeded")
    eps_a0 = np.array([r.get("eps_a0", np.nan) for _, r in rows])
    eps_lc = np.array([r.get("eps_lcfs", np.nan) for _, r in rows])
    ratio = eps_a0 / eps_a0_walk[[i for i, _ in rows]]
    m = np.isfinite(ratio) & (eps_a0 > 0)
    if m.any():
        print(f"  full/walk eps at a={A0}: median {np.median(ratio[m]):.3f},"
              f" p10 {np.percentile(ratio[m], 10):.3f},"
              f" p90 {np.percentile(ratio[m], 90):.3f}")
    print(f"  eps at the LCFS: min {np.nanmin(eps_lc):.4f}"
          f"  median {np.nanmedian(eps_lc):.4f}")
    k = int(np.nanargmin(np.where(np.isfinite(eps_a0), eps_a0, np.inf)))
    print(f"  best full-tier eps at a={A0}: {eps_a0[k]:.6f}"
          f"  (walk-tier value {eps_a0_walk[rows[k][0]]:.6f})")

    OUT_H5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(OUT_H5, "w") as f:
        f.create_dataset("index", data=np.array([i for i, _ in rows]))
        f.create_dataset("samples", data=S[[i for i, _ in rows]])
        for key in ("dr_a0", "a_a0", "eps_a0", "drift_a0", "volume_a0",
                    "iota_a0", "status_a0", "dr_lcfs", "a_lcfs", "eps_lcfs",
                    "drift_lcfs", "volume_lcfs", "iota_lcfs", "status_lcfs"):
            f.create_dataset(key, data=np.array(
                [r.get(key, np.nan) for _, r in rows]))
        f.create_dataset("eps_a0_walk",
                         data=eps_a0_walk[[i for i, _ in rows]])
        f.attrs["a0"] = A0
        f.attrs["nturn"] = FULL_NTURN
        f.attrs["npart"] = FULL_NPART
        f.attrs["nphi"] = NPHI
    print(f"  saved -> {OUT_H5}")


if __name__ == "__main__":
    main()
