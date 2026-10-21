#!/usr/bin/env python3
"""Stage 3: choose the reference surface a0, filter the islands, measure eps.

Reads the stage-2 depth map.  Two independent modes, so changing a0 only re-runs
the cheap part:

  MODE = "filter"   seconds.  Keeps every sample whose column reaches a0
                    (``lcfs_a >= A0``), reports how many there are, where they
                    sit in the box and how densely they cover it.  No oracle
                    call, no file written.

  MODE = "eps"      one full-tier eps measurement per selected sample, on the
                    surface whose minor radius is a0 (interpolated from the
                    stored profile).  The walk is NOT repeated: stage 2 already
                    recorded the axis and the a(dr) table.

eps is measured here and nowhere else.  It is reproducible in the inner 80% of
a column (1e-5 relative jitter 0.2%) and not near the edge (up to 66%), which is
why the reference surface is a fixed minor radius rather than the LCFS.

  python -u tests/h1_eps_scan.py            # filter mode
  python -u tests/h1_eps_scan.py --eps      # filter + eps
"""

from __future__ import annotations

import multiprocessing
import time
from pathlib import Path

import h5py
import numpy as np

from ripplepy.radial_scan import (ScanConfig, _worker_init, fmt_duration,
                                  load_scan_h5, measure_reference)

BASE = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
MGRID_PATH = BASE / "tests" / "test_file" / "mgrid_h1_design.nc"
NFP = 3
FULL_TORUS = False

BOUNDS = np.array([
    [50000.0, 50000.0],     # coil 0: TF, fixed
    [0.0, 10000.0],         # coil 1
    [0.0, 10000.0],         # coil 2
    [-100000.0, 0.0],       # coil 3
    [-100000.0, 0.0],       # coil 4
])

DEPTH_H5 = BASE / "tests" / "h1_radial_scan" / "depth_map.h5"
OUTPUT_H5 = BASE / "tests" / "h1_radial_scan" / "eps_ref.h5"

# Reference surface.  Fixed ABSOLUTE minor radius, so every configuration is
# measured at the same plasma size (eps grows steeply with a, so a fixed
# fraction of each LCFS would mix "plasma size" into the ranking).
A0 = 0.14

# Optional extra filters on the selected set.
MIN_LCFS_DR = 0.0          # keep columns at least this deep
GAP_MAX = 12.0                  # reject samples whose LCFS face sits close to
                                # the knife edge (gate is 15)

# Full tier (eps).  nturn=400 is the practical ceiling: 800 turns already fails
# with a negative LSODE istate on some faces.
NPHI = 360
FULL_NTURN = 400
FULL_NPART = 2000

PROCESSES = 64             # 0 = serial
PROGRESS_INTERVAL = 64


# ---------------------------------------------------------------------------
def _select(depth: dict, a0: float) -> np.ndarray:
    s = depth["scalars"]
    ok = np.isfinite(s["lcfs_a"]) & (s["lcfs_a"] >= a0)
    ok &= np.isfinite(s["lcfs_dr"]) & (s["lcfs_dr"] >= MIN_LCFS_DR)
    ok &= np.isfinite(s["lcfs_gap_norm"]) & (s["lcfs_gap_norm"] <= GAP_MAX)
    return np.flatnonzero(ok)


def _interp_dr(prof: dict, i: int, a0: float):
    """dr on sample i whose measured minor radius equals a0."""
    a = prof["a"][i]
    dr = prof["dr"][i]
    ok = np.isfinite(a) & np.isfinite(dr) & (a > 0)
    ok &= (prof["status"][i] == 0)
    if ok.sum() < 2:
        return None
    a, dr = a[ok], dr[ok]
    o = np.argsort(a)
    a, dr = a[o], dr[o]
    if not (a[0] <= a0 <= a[-1]):
        return None
    return float(np.interp(a0, a, dr))


def _report_selection(depth: dict, idx: np.ndarray, a0: float):
    s = depth["scalars"]
    n = len(depth["samples"])
    print(f"  {len(idx)}/{n} samples reach a >= {a0}")
    if len(idx) == 0:
        return
    for name in ("lcfs_a", "lcfs_dr", "lcfs_volume", "lcfs_iota",
                 "lcfs_gap_norm"):
        v = s[name][idx]
        q = np.percentile(v, [0, 5, 50, 95, 100])
        print(f"    {name:14s} min {q[0]:.4f} p5 {q[1]:.4f} med {q[2]:.4f}"
              f" p95 {q[3]:.4f} max {q[4]:.4f}")

    # fill distance in the normalised free-dim box: how big is the largest hole
    # in the selected set.  Compare it with the overall sampling resolution.
    span = BOUNDS[:, 1] - BOUNDS[:, 0]
    free = span > 0
    x = np.asarray(depth["samples"])[idx]
    u = (x[:, free] - BOUNDS[free, 0]) / span[free]
    from scipy.spatial import cKDTree
    d, _ = cKDTree(u).query(u, k=2)
    nn = d[:, 1]
    print(f"    nearest-neighbour (normalised box): med {np.median(nn):.4f}"
          f" p95 {np.percentile(nn, 95):.4f} max {nn.max():.4f}")


def _map_ref(args):
    i, x, axis_rz, dr_ref, cfg = args
    row = measure_reference(x, cfg, axis_rz, dr_ref)
    return i, row


def _run_eps(depth: dict, idx: np.ndarray, a0: float, cfg: ScanConfig):
    prof = depth["profile"]
    s = depth["scalars"]
    tasks, skipped = [], 0
    for i in idx:
        dr_ref = _interp_dr(prof, i, a0)
        if dr_ref is None:
            skipped += 1
            continue
        tasks.append((int(i), np.asarray(depth["samples"][i], dtype=float),
                      np.array([s["axis_R"][i], s["axis_Z"][i]], dtype=float),
                      dr_ref, cfg))
    print(f"  {len(tasks)} eps measurements ({skipped} skipped: a0 outside"
          f" the profile)", flush=True)
    if not tasks:
        return

    t0 = time.perf_counter()
    out: dict[int, dict] = {}
    if PROCESSES == 0:
        _worker_init(cfg)
        for k, t in enumerate(tasks, start=1):
            i, row = _map_ref(t)
            out[i] = row
            if PROGRESS_INTERVAL and k % PROGRESS_INTERVAL == 0:
                el = time.perf_counter() - t0
                print(f"    {k}/{len(tasks)}  {k / max(el, 1e-9):.2f}/s"
                      f"  {el:.0f}s", flush=True)
    else:
        with multiprocessing.Pool(processes=PROCESSES,
                                  initializer=_worker_init,
                                  initargs=(cfg,)) as pool:
            for k, (i, row) in enumerate(
                    pool.imap_unordered(_map_ref, tasks, chunksize=1), start=1):
                out[i] = row
                if PROGRESS_INTERVAL and k % PROGRESS_INTERVAL == 0:
                    el = time.perf_counter() - t0
                    print(f"    {k}/{len(tasks)}  {k / max(el, 1e-9):.2f}/s"
                          f"  {fmt_duration(el)}"
                          f"  ETA {fmt_duration(el / k * (len(tasks) - k))}",
                          flush=True)

    keys = ("dr", "a", "eps", "eps_defect", "gradpsi_drift", "residual",
            "gap_norm", "iota", "volume", "ripple", "status", "istate")
    arr = {k: np.full(len(tasks), np.nan) for k in keys}
    samp = np.full((len(tasks), depth["samples"].shape[1]), np.nan)
    for j, t in enumerate(tasks):
        i = t[0]
        samp[j] = depth["samples"][i]
        row = out.get(i)
        if row is None:
            continue
        for k in keys:
            arr[k][j] = row.get(k, np.nan)
    path = Path(OUTPUT_H5)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("samples", data=samp, compression="gzip")
        f.create_dataset("source_index", data=np.array([t[0] for t in tasks]))
        g = f.create_group("ref")
        for k, v in arr.items():
            g.create_dataset(k, data=v, compression="gzip")
        f.attrs["a0"] = a0
        f.attrs["nturn"] = FULL_NTURN
        f.attrs["nphi"] = NPHI
        f.attrs["npart"] = FULL_NPART
        f.attrs["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
    good = np.isfinite(arr["eps"])
    print(f"  saved -> {path}   ({int(good.sum())}/{len(tasks)} with eps)",
          flush=True)
    if good.any():
        q = np.percentile(arr["eps"][good], [0, 5, 50, 95, 100])
        print(f"    eps @ a={a0}: min {q[0]:.6f} p5 {q[1]:.6f} med {q[2]:.6f}"
              f" p95 {q[3]:.6f} max {q[4]:.6f}")
        qc = arr["gradpsi_drift"][good]
        qc = qc[np.isfinite(qc)]
        if qc.size:
            print(f"    gradpsi drift: med {np.median(qc):.2f}"
                  f"  (clean ~0.9, corrupted >1.5)")
        order = np.flatnonzero(good)[np.argsort(arr["eps"][good])][:5]
        print("    best by eps:")
        for j in order:
            print(f"      eps={arr['eps'][j]:.6f} a={arr['a'][j]:.4f}"
                  f" V={arr['volume'][j]:.4f} iota={arr['iota'][j]:.3f}"
                  f" QC={arr['gradpsi_drift'][j]:.2f}"
                  f"  extcur=[{', '.join(f'{c:.1f}' for c in samp[j])}]")


def main():
    import sys
    do_eps = "--eps" in sys.argv
    width = 22
    print("=" * width + " stage-3 reference eps " + "=" * width)
    print(f"  depth map {DEPTH_H5}")
    print(f"  reference a0 = {A0} m (absolute minor radius)")
    print(f"  filters   lcfs_dr >= {MIN_LCFS_DR}, lcfs_gap_norm <= {GAP_MAX}")
    print(f"  mode      {'filter + eps' if do_eps else 'filter only'}")
    print("=" * (2 * width + 26))

    depth = load_scan_h5(Path(DEPTH_H5))
    idx = _select(depth, A0)
    _report_selection(depth, idx, A0)
    if not do_eps:
        print("  (filter only; re-run with --eps to measure)")
        return

    cfg = ScanConfig(mgrid_path=str(MGRID_PATH), nfp=NFP, full_torus=FULL_TORUS,
                     nphi=NPHI, full_nturn=FULL_NTURN, full_npart=FULL_NPART)
    _run_eps(depth, idx, A0, cfg)


if __name__ == "__main__":
    main()
