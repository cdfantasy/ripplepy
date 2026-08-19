"""Feasibility-island mapping for coil-current optimisation.

Phase 1+2 helper: sample coil-current space, filter with a hierarchy of
physical oracles, cluster the surviving points into islands, save the state
to HDF5, and generate the next delt_r layer's samples from the previous
layer's islands (hot start).

Oracles (in order of increasing cost and fidelity):
  L1  axis scan       : R-scan find_axis, any |Z| <= axis_z_tol
  L2  short trace     : trace nturn=short_nturn from axis + delt_r
  L3  full trace      : trace nturn=full_nturn from axis + delt_r and check
                        the phi=0 Poincare section with the FFT smoothness test
"""

from __future__ import annotations

import contextlib
import json
import multiprocessing
import os
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats.qmc import Sobol

from .optimize import OptimizationConfig
from .ripple import (
    fieldline_smoothness_poincare,
    find_axis,
    find_axis_multi_guess,
    initialize_mgrid_field,
    set_extcur,
    trace_fieldline,
)

# Worker globals (set once per process by _worker_init)
_worker_cfg: OptimizationConfig | None = None
_worker_params: dict = {}


def _n_workers(processes: int | None) -> int:
    return int(processes) if processes and processes > 0 else multiprocessing.cpu_count()


def _worker_init(cfg: OptimizationConfig, params: dict):
    global _worker_cfg, _worker_params
    _worker_cfg = cfg
    _worker_params = params
    # One worker prints one "Loaded mgrid" line; suppress it so large pools
    # do not flood the console log.
    with open(os.devnull, "w") as sink:
        with contextlib.redirect_stdout(sink):
            initialize_mgrid_field(
                str(cfg.mgrid_path), nfp=cfg.nfp, full_torus=cfg.full_torus)


def _map_point(point: np.ndarray) -> dict:
    """Run the full oracle cascade for one coil-current vector."""
    cfg = _worker_cfg
    p = dict(_worker_params)
    extcur = np.asarray(point, dtype=np.float64)
    out = {
        "extcur": extcur,
        "axis_feasible": False,
        "axis_count": 0,
        "axis_multi": False,
        "axis_used_RZ": np.array([np.nan, np.nan]),
        "short_feasible": False,
        "short_istate": -9999,
        "full_feasible": False,
        "full_istate": -9999,
        "smooth_residual": np.nan,
        "smooth_max_gap": np.nan,
    }

    set_extcur(extcur)

    # L1: multi-guess axis scan.  Coarse nphi=180 keeps the scan cheap; the
    # selected axis is refined below with the full-resolution find_axis.
    axes = find_axis_multi_guess(
        rmin=p["rmin"], rmax=p["rmax"], rstep=p["rstep"], z0=0.0,
        xtol=1e-6, max_iter=100, delta_r=0.01,
        axis_z_tol=cfg.axis_z_tol, nphi=180)
    out["axis_count"] = len(axes)
    if not axes:
        return out
    out["axis_feasible"] = True
    out["axis_multi"] = len(axes) > 1
    if out["axis_multi"]:
        # Multi-axis configurations are recorded but NOT treated as true
        # islands (they may be degenerate / doublet configurations).
        return out

    # Refine the single axis with full poloidal resolution.
    guess = np.array([axes[0][0], axes[0][1]], dtype=np.float64)
    axis_rz, _, _, ok = find_axis(guess, xtol=1e-5, max_iter=100,
                                  delta_r=0.01, nphi=360)
    if not ok or abs(axis_rz[1]) > cfg.axis_z_tol:
        return out
    out["axis_used_RZ"] = np.asarray(axis_rz, dtype=np.float64)

    start_rz = np.array([axis_rz[0] + p["delt_r"], axis_rz[1]],
                        dtype=np.float64, order="F")

    # L2: short trace.  initial_gradpsi left at zero: the RZ field-line path
    # is independent of the grad-psi variables, so this is exact for tracing
    # and avoids an extra B-interpolation call.
    _, short_ist = trace_fieldline(
        initial_rz=start_rz, nturn=p["short_nturn"], nphi=p["short_nphi"],
        verbose=False)
    out["short_istate"] = short_ist
    if short_ist != 0:
        return out
    out["short_feasible"] = True

    # L3: full-resolution trace + Poincare smoothness check.  The trace and
    # the smoothness test are done separately so the smoothness metrics can be
    # recorded in the HDF5 state.
    fld, full_ist = trace_fieldline(
        initial_rz=start_rz, nturn=p["full_nturn"], nphi=p["full_nphi"],
        verbose=False)
    out["full_istate"] = full_ist
    if full_ist != 0:
        return out
    smooth, metrics = fieldline_smoothness_poincare(
        fld, p["full_nturn"], p["full_nphi"], axis_rz=axis_rz,
        n_harmonics=p["smooth_n_harmonics"],
        residual_rms_frac_tol=p["smooth_residual_tol"],
        max_angular_gap=p["smooth_max_gap"],
        min_points=p["smooth_min_points"])
    out["smooth_residual"] = metrics.get("residual_rms_frac", np.nan)
    out["smooth_max_gap"] = metrics.get("max_gap_rad", np.nan)
    if not smooth:
        out["full_istate"] = -2001
        return out
    out["full_feasible"] = True
    return out


def sample_bounds(bounds: np.ndarray, n_samples: int, seed: int = 0) -> np.ndarray:
    """Sobol-sample absolute bounds; locked coils (lo==hi) stay fixed."""
    bounds = np.asarray(bounds, dtype=np.float64)
    lo, hi = bounds[:, 0], bounds[:, 1]
    n_dim = len(lo)
    u = Sobol(d=n_dim, scramble=True, seed=seed).random(n_samples)
    return lo + u * (hi - lo)


def _dbscan_lite(X: np.ndarray, eps: float, min_samples: int):
    """Small DBSCAN implementation using a KD-tree (no sklearn dependency)."""
    n = X.shape[0]
    labels = np.full(n, -1, dtype=int)
    if n == 0:
        return labels
    tree = cKDTree(X)
    counts = tree.query_ball_point(X, r=eps, return_length=True)
    core = counts >= min_samples
    cluster_id = 0
    for i in range(n):
        if labels[i] != -1 or not core[i]:
            continue
        labels[i] = cluster_id
        queue = [i]
        while queue:
            q = queue.pop()
            for nb in tree.query_ball_point(X[q], r=eps):
                if labels[nb] == -1:
                    labels[nb] = cluster_id
                    if core[nb]:
                        queue.append(nb)
        cluster_id += 1
    return labels


def _cluster_full_feasible(samples: np.ndarray, full_mask: np.ndarray,
                           axis_used_RZ: np.ndarray,
                           eps: float, min_samples: int):
    """Cluster full-feasible samples in normalised free-coil space."""
    islands = []
    n_free = 0
    bounds = np.column_stack([samples.min(axis=0), samples.max(axis=0)])
    free_dims = np.flatnonzero(bounds[:, 1] - bounds[:, 0] > 1e-12)
    if len(free_dims) == 0 or not full_mask.any():
        return islands, free_dims

    X = samples[full_mask][:, free_dims].astype(np.float64)
    lo = bounds[free_dims, 0]
    hi = bounds[free_dims, 1]
    span = hi - lo
    span[span <= 1e-12] = 1.0
    Xn = (X - lo) / span

    labels = _dbscan_lite(Xn, eps=eps, min_samples=min_samples)

    for cid in sorted(set(labels.tolist()) - {-1}):
        inds = np.flatnonzero(labels == cid)
        pts = X[inds]
        pts_n = Xn[inds]
        center_free = pts_n.mean(axis=0)
        cov_free = np.cov(pts_n, rowvar=False) if len(inds) >= 2 else np.zeros(
            (len(free_dims), len(free_dims)))
        full_center = samples[full_mask][inds].mean(axis=0)
        q02 = np.percentile(samples[full_mask][inds], 2.0, axis=0)
        q98 = np.percentile(samples[full_mask][inds], 98.0, axis=0)
        islands.append({
            "island_id": int(cid),
            "n_points": int(len(inds)),
            "sample_indices": np.flatnonzero(full_mask)[inds],
            "center": full_center,
            "center_free": center_free,
            "cov_free": cov_free,
            "bounds": np.column_stack([q02, q98]),
            "mean_axis_R": float(np.nanmean(axis_used_RZ[full_mask][inds, 0])),
        })
    islands.sort(key=lambda d: -d["n_points"])
    for rank, island in enumerate(islands):
        island["island_id"] = rank
    return islands, free_dims




def full_feasible_suggested_bounds(res: dict, broad_bounds: np.ndarray,
                                  min_points: int = 8,
                                  percentile: float = 2.0):
    """Borrow the q02/q98 idea from survey_feasibility.

    Given a low-resolution `map_feasible_islands(..., do_cluster=False)`
    result on a very wide engineering box, return a tighter absolute box that
    just covers the full-feasible (non multi-axis) samples.  Falls back to
    `broad_bounds` when too few feasible points were found.
    """
    broad_bounds = np.asarray(broad_bounds, dtype=np.float64)
    mask = res["full_feasible"].astype(bool) & ~res["axis_multi"].astype(bool)
    if int(mask.sum()) < int(min_points):
        return broad_bounds.copy()
    pts = res["samples"][mask]
    q02 = np.percentile(pts, percentile, axis=0)
    q98 = np.percentile(pts, 100.0 - percentile, axis=0)
    sug = np.column_stack([q02, q98])
    # clamp to broad box
    lo = np.maximum(sug[:, 0], broad_bounds[:, 0])
    hi = np.minimum(sug[:, 1], broad_bounds[:, 1])
    # keep locked coils fixed
    locked = broad_bounds[:, 0] == broad_bounds[:, 1]
    lo[locked] = broad_bounds[locked, 0]
    hi[locked] = broad_bounds[locked, 1]
    return np.column_stack([lo, hi])


def map_feasible_islands(
    cfg: OptimizationConfig,
    bounds: np.ndarray,
    delt_r: float,
    n_samples: int = 8192,
    rmin: float = 1.00,
    rmax: float = 1.35,
    rstep: float = 0.05,
    short_nturn: int = 20,
    short_nphi: int = 72,
    full_nturn: int = 200,
    full_nphi: int = 360,
    smooth_n_harmonics: int = 4,
    smooth_residual_tol: float = 0.05,
    smooth_max_gap: float = 1.0,
    smooth_min_points: int = 16,
    cluster_eps: float = 0.15,
    cluster_min_samples: int = 5,
    processes: int | None = None,
    seed: int = 0,
    samples: Optional[np.ndarray] = None,
    output_h5: Optional[Path] = None,
    do_cluster: bool = True,
) -> dict:
    """Map feasible-current islands at one delt_r.

    bounds : ndarray, shape (n_coils, 2), absolute [lo, hi]; fixed coils are
             rows with lo == hi.
    samples : optional pre-generated sample array.  When omitted, a Sobol set
              of size `n_samples` is drawn inside `bounds`.
    """
    bounds = np.asarray(bounds, dtype=np.float64)
    if samples is None:
        samples = sample_bounds(bounds, n_samples, seed=seed)
    else:
        samples = np.asarray(samples, dtype=np.float64)
        n_samples = int(samples.shape[0])

    params = dict(
        delt_r=float(delt_r), rmin=float(rmin), rmax=float(rmax),
        rstep=float(rstep), short_nturn=int(short_nturn),
        short_nphi=int(short_nphi), full_nturn=int(full_nturn),
        full_nphi=int(full_nphi),
        smooth_n_harmonics=int(smooth_n_harmonics),
        smooth_residual_tol=float(smooth_residual_tol),
        smooth_max_gap=float(smooth_max_gap),
        smooth_min_points=int(smooth_min_points),
    )

    n_workers = _n_workers(processes)
    with multiprocessing.Pool(processes=n_workers, initializer=_worker_init,
                              initargs=(cfg, params)) as pool:
        results = pool.map(_map_point, samples)

    n = len(results)
    axis_feasible = np.zeros(n, dtype=bool)
    axis_count = np.zeros(n, dtype=np.uint8)
    axis_multi = np.zeros(n, dtype=bool)
    axis_used_RZ = np.full((n, 2), np.nan, dtype=np.float64)
    short_feasible = np.zeros(n, dtype=bool)
    short_istate = np.full(n, -9999, dtype=np.int32)
    full_feasible = np.zeros(n, dtype=bool)
    full_istate = np.full(n, -9999, dtype=np.int32)
    smooth_residual = np.full(n, np.nan, dtype=np.float64)
    smooth_max_gap = np.full(n, np.nan, dtype=np.float64)

    for k, r in enumerate(results):
        axis_feasible[k] = r["axis_feasible"]
        axis_count[k] = r["axis_count"]
        axis_multi[k] = r["axis_multi"]
        axis_used_RZ[k] = r["axis_used_RZ"]
        short_feasible[k] = r["short_feasible"]
        short_istate[k] = r["short_istate"]
        full_feasible[k] = r["full_feasible"]
        full_istate[k] = r["full_istate"]
        smooth_residual[k] = r["smooth_residual"]
        smooth_max_gap[k] = r["smooth_max_gap"]

    if do_cluster:
        islands, free_dims = _cluster_full_feasible(
            samples, full_feasible, axis_used_RZ,
            eps=cluster_eps, min_samples=cluster_min_samples)
    else:
        islands, free_dims = [], np.flatnonzero(
            bounds[:, 1] - bounds[:, 0] > 1e-12)

    res = {
        "delt_r": float(delt_r),
        "bounds": bounds,
        "samples": samples,
        "axis_feasible": axis_feasible,
        "axis_count": axis_count,
        "axis_multi": axis_multi,
        "axis_used_RZ": axis_used_RZ,
        "short_feasible": short_feasible,
        "short_istate": short_istate,
        "full_feasible": full_feasible,
        "full_istate": full_istate,
        "smooth_residual": smooth_residual,
        "smooth_max_gap": smooth_max_gap,
        "free_dims": np.asarray(free_dims, dtype=int),
        "islands": islands,
        "params": params,
        "n_samples": n_samples,
        "seed": seed,
    }

    if output_h5 is not None:
        save_island_mapping_h5(Path(output_h5), res)
    return res


# ---------------------------------------------------------------------------
# HDF5 persistence / hot start
# ---------------------------------------------------------------------------

def save_island_mapping_h5(path: Path, res: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["delt_r"] = res["delt_r"]
        f.attrs["n_samples"] = int(res["n_samples"])
        f.attrs["seed"] = int(res["seed"])
        f.attrs["params"] = json.dumps(res["params"])
        f.attrs["bounds"] = json.dumps(np.asarray(res["bounds"]).tolist())

        f.create_dataset("samples", data=res["samples"])
        f.create_dataset("axis_feasible", data=res["axis_feasible"].astype(np.uint8))
        f.create_dataset("axis_count", data=res["axis_count"])
        f.create_dataset("axis_multi", data=res["axis_multi"].astype(np.uint8))
        f.create_dataset("axis_used_RZ", data=res["axis_used_RZ"])
        f.create_dataset("short_feasible", data=res["short_feasible"].astype(np.uint8))
        f.create_dataset("short_istate", data=res["short_istate"])
        f.create_dataset("full_feasible", data=res["full_feasible"].astype(np.uint8))
        f.create_dataset("full_istate", data=res["full_istate"])
        f.create_dataset("smooth_residual", data=res["smooth_residual"])
        f.create_dataset("smooth_max_gap", data=res["smooth_max_gap"])
        if "free_dims" in res:
            f.create_dataset("free_dims", data=np.asarray(res["free_dims"], dtype=int))

        for island in res.get("islands", []):
            g = f.create_group(f"islands/island_{island['island_id']}")
            g.attrs["n_points"] = int(island["n_points"])
            g.attrs["mean_axis_R"] = float(island["mean_axis_R"])
            for key in ("sample_indices", "center", "center_free",
                        "cov_free", "bounds"):
                g.create_dataset(key, data=np.asarray(island[key]))


def load_island_mapping_h5(path: Path) -> dict:
    path = Path(path)
    res = {}
    with h5py.File(path, "r") as f:
        res["delt_r"] = float(f.attrs["delt_r"])
        res["n_samples"] = int(f.attrs["n_samples"])
        res["seed"] = int(f.attrs["seed"])
        res["params"] = json.loads(f.attrs["params"])
        res["bounds"] = np.asarray(json.loads(f.attrs["bounds"]))
        res["samples"] = f["samples"][()]
        res["axis_feasible"] = f["axis_feasible"][()].astype(bool)
        res["axis_count"] = f["axis_count"][()]
        res["axis_multi"] = f["axis_multi"][()].astype(bool)
        res["axis_used_RZ"] = f["axis_used_RZ"][()]
        res["short_feasible"] = f["short_feasible"][()].astype(bool)
        res["short_istate"] = f["short_istate"][()]
        res["full_feasible"] = f["full_feasible"][()].astype(bool)
        res["full_istate"] = f["full_istate"][()]
        res["smooth_residual"] = f["smooth_residual"][()]
        res["smooth_max_gap"] = f["smooth_max_gap"][()]
        if "free_dims" in f:
            res["free_dims"] = f["free_dims"][()]
        else:
            res["free_dims"] = np.asarray([], dtype=int)

        islands = []
        if "islands" in f:
            for name in f["islands"].keys():
                g = f["islands"][name]
                islands.append({
                    "island_id": int(name.split("_")[-1]),
                    "n_points": int(g.attrs["n_points"]),
                    "mean_axis_R": float(g.attrs["mean_axis_R"]),
                    "sample_indices": g["sample_indices"][()],
                    "center": g["center"][()],
                    "center_free": g["center_free"][()],
                    "cov_free": g["cov_free"][()],
                    "bounds": g["bounds"][()],
                })
        res["islands"] = islands
    return res


def generate_next_samples(prev_mapping: dict, broad_bounds: np.ndarray,
                          n_local_per_island: int = 4000,
                          n_global: int = 800,
                          alpha: float = 1.5,
                          seed: int = 0) -> np.ndarray:
    """Generate samples for the next delt_r from the previous mapping.

    For each previous island, draw from a Gaussian with the island's
    normalised covariance scaled by alpha**2 (covers the expected island
    shrinkage/motion between adjacent delt_r values).  A small global Sobol
    set is added only to verify that no island was missed.
    """
    broad_bounds = np.asarray(broad_bounds, dtype=np.float64)
    free_dims = np.asarray(prev_mapping.get("free_dims", []), dtype=int)
    lo = broad_bounds[:, 0]
    hi = broad_bounds[:, 1]
    rng = np.random.default_rng(seed)

    blocks = []
    for island in prev_mapping.get("islands", []):
        if island["n_points"] < 2 or len(free_dims) == 0:
            continue
        center_free = np.asarray(island["center_free"], dtype=np.float64)
        cov_free = np.asarray(island["cov_free"], dtype=np.float64)
        # Add a small ridge for numerical stability.
        cov = (alpha ** 2) * cov_free + np.eye(len(free_dims)) * 1e-9
        pts_free = rng.multivariate_normal(
            center_free, cov, size=n_local_per_island)
        pts_free = np.clip(pts_free, 0.0, 1.0)
        pts = np.tile(broad_bounds.mean(axis=1), (n_local_per_island, 1))
        pts[:, free_dims] = lo[free_dims] + pts_free * (hi[free_dims] - lo[free_dims])
        blocks.append(pts)

    if n_global > 0:
        blocks.append(sample_bounds(broad_bounds, n_global, seed=seed + 999))

    if blocks:
        return np.vstack(blocks)
    return sample_bounds(broad_bounds, 1, seed=seed)
