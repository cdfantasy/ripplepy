#!/usr/bin/env python3
"""Optimise epsilon_eff inside a manually specified island box.

Edit the constants in the CONFIGURATION block below, then run:

    python tests/h1_optimise_island.py
"""

from __future__ import annotations

import logging
import multiprocessing
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
MGRID_PATH = BASE / "tests" / "test_file" / "mgrid_h1_design.nc"
OUTPUT_ROOT = BASE / "tests" / "h1_island_optimisation"

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION — edit these values directly
# ═══════════════════════════════════════════════════════════════════════════
DELT_R = 0.08

# 5 coils: [lo, hi] per coil, in coil order.  lo == hi locks that coil.
BOUNDS = np.array([
    [ 50000.0,   50000.0],  # coil 0: TF, fixed
    [  5803.0,   10000.0],  # coil 1
    [     0.0,    7460.0],  # coil 2
    [-156374.0,  -40000.0], # coil 3
    [-100000.0,  -71715.0], # coil 4
], dtype=np.float64)

# find_axis starting guess; use the island's mean axis_R from the mapping.
INITIAL_RZ = np.array([1.3038, 0.0], dtype=np.float64)

# DE settings
N_POP = 64
MAX_GEN = 60
FTOL = 1e-8
PATIENCE = 15
SEED = 42
DE_NTURN = 200
DE_NPHI = 360
DE_NPART = 5000

# Full-resolution verification settings
VERIFY_NTURN = 400
VERIFY_NPHI = 720
VERIFY_NPART = 5000
MIN_MINOR_RADIUS = 0.02
PROCESSES = None   # None -> all CPU cores; or set e.g. 64

# PCA-frame DE (方案 A): when USE_PCA is True the manual BOUNDS above are
# ignored.  The search box is built from the mapping HDF5 island's
# center_free/cov_free: y-space = island principal axes, DE box =
# [-k*sigma, +k*sigma] per axis, initial population uniform in that box.
# Individuals are mapped back to absolute currents (coil 0 fixed) for every
# evaluation via the de.to_x hook.
USE_PCA = True
PCA_H5 = BASE / "tests" / "h1_islands" / "islands_dr0.12.h5"
PCA_ISLAND_ID = 0
PCA_K = 3.0
PCA_MIN_WIDTH = 0.02   # minimum half-width per axis (normalised units)
# ═══════════════════════════════════════════════════════════════════════════

NOMINAL_EXTCUR = np.array([50000.0, 5000.0, 3000.0, -80000.0, -40000.0])
NFP = 3
FULL_TORUS = False


def make_run_dir() -> Path:
    tag = "pca_" if USE_PCA else ""
    tag += "_".join(
        [f"dr{DELT_R:g}"]
        + [f"{lo:.0f}_{hi:.0f}" for lo, hi in BOUNDS]
    )
    base = OUTPUT_ROOT / tag
    candidate = base
    idx = 2
    while candidate.exists():
        candidate = OUTPUT_ROOT / f"{base.name}_run{idx}"
        idx += 1
    candidate.mkdir(parents=True, exist_ok=True)
    return candidate


def verify_full_res(extcur, label, initial_rz):
    from ripplepy import (
        compute_epstot,
        compute_initial_gradpsi_nemov,
        find_axis,
        set_extcur,
        set_trace_parameters,
    )

    set_extcur(extcur)
    axis_rz, _, _, ok = find_axis(initial_rz, xtol=1e-5, max_iter=100,
                                  delta_r=0.01, verbose=False)
    if not ok:
        print(f"  {label}: magnetic axis not found at full res")
        return None
    start_rz = [axis_rz[0] + DELT_R, axis_rz[1]]
    gradpsi = compute_initial_gradpsi_nemov(extcur, start_rz[0], start_rz[1],
                                            verbose=False)
    set_trace_parameters(VERIFY_NTURN, VERIFY_NPHI, npart=VERIFY_NPART,
                         verbose=False)
    result = compute_epstot(start_rz, initial_gradpsi=gradpsi,
                            return_fieldline=False, verbose=False)
    eps = result[0]
    if eps is None or np.isnan(eps):
        print(f"  {label}: epsilon_eff failed at full res")
        return None
    print(f"  {label}: eps_eff^(3/2) = {eps:.6e}  "
          f"(full res: nturn={VERIFY_NTURN}, nphi={VERIFY_NPHI}, npart={VERIFY_NPART})")
    return float(eps)


def build_pca_frame(h5_path, island_id=0, k=3.0, min_width=0.02):
    """Build a principal-axis (PCA) DE frame from a mapping HDF5 island.

    Uses the island's center_free (centroid) + cov_free (shape/tilt) in
    normalised free-coil space.  y-space = the island's own axes, DE box =
    [-k*sigma, +k*sigma] per axis (uniform initial population = 方案 A).

    Returns
    -------
    y_bounds   : (5,2) DE box in y-space (coil 0 locked at [50000, 50000])
    to_x       : 5-vector y -> 5-vector absolute current x (coil 0 = 50000)
    to_y       : 5-vector current x -> 5-vector y
    mean_axis_R: island's mean magnetic axis R (fallback axis warm-start)
    """
    from ripplepy.islands import load_island_mapping_h5

    res = load_island_mapping_h5(Path(h5_path))
    bounds = np.asarray(res["bounds"], dtype=np.float64)
    free = np.flatnonzero(bounds[:, 1] - bounds[:, 0] > 1e-12)
    isl = res["islands"][island_id]
    mu = np.asarray(isl["center_free"], dtype=np.float64)      # (n_free,)
    cov = np.asarray(isl["cov_free"], dtype=np.float64)        # (n_free, n_free)
    lo = bounds[free, 0]
    span = bounds[free, 1] - bounds[free, 0]
    span[span <= 1e-12] = 1.0

    w, V = np.linalg.eigh(cov)                 # ascending eigenvalues
    w, V = w[::-1], V[:, ::-1]                 # descending (principal axes)
    sigma = np.sqrt(np.maximum(w, 0.0))
    half = np.maximum(k * sigma, min_width)

    y_bounds = np.array([[50000.0, 50000.0]] * 5, dtype=np.float64)
    y_bounds[free, 0] = -half
    y_bounds[free, 1] = +half

    def to_x(y):
        y = np.asarray(y, dtype=np.float64)
        u = mu + V @ y[free]                   # normalised free coords
        x = np.array([50000.0] * 5, dtype=np.float64)
        x[free] = lo + u * span
        return x

    def to_y(x):
        x = np.asarray(x, dtype=np.float64)
        u = (x[free] - lo) / span
        y = np.array([50000.0] * 5, dtype=np.float64)
        y[free] = V.T @ (u - mu)
        return y

    return y_bounds, to_x, to_y, float(isl["mean_axis_R"])


class Tee:
    """Mirror stdout/stderr to the console and a run log file."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            try:
                stream.write(data)
                stream.flush()
            except Exception:
                pass

    def flush(self):
        for stream in self.streams:
            try:
                stream.flush()
            except Exception:
                pass


def main():
    from ripplepy import initialize_mgrid_field, set_extcur
    from ripplepy.optimize import DifferentialEvolution, OptimizationConfig

    run_dir = make_run_dir()

    # Save all console output into this run's folder.
    _orig_stdout, _orig_stderr = sys.stdout, sys.stderr
    console_log = open(run_dir / "console.log", "w", buffering=1)
    sys.stdout = Tee(_orig_stdout, console_log)
    sys.stderr = Tee(_orig_stderr, console_log)

    print(f"Output -> {run_dir}")

    # PCA frame first (needed for the axis warm-start guess when enabled).
    pca_frame = None
    if USE_PCA:
        pca_frame = build_pca_frame(
            PCA_H5, island_id=PCA_ISLAND_ID, k=PCA_K,
            min_width=PCA_MIN_WIDTH)
        y_bounds, to_x, to_y, mean_axis_R = pca_frame
        initial_rz = np.array([mean_axis_R, 0.0], dtype=np.float64)
        print(f"PCA frame from {PCA_H5} (island {PCA_ISLAND_ID}, "
              f"k={PCA_K}, axis_R={mean_axis_R:.4f})")
        for i, (lo, hi) in enumerate(y_bounds):
            print(f"  y coil {i}: [{lo:.4f}, {hi:.4f}]"
                  f"{'  (fixed)' if lo == hi else ''}")
    else:
        initial_rz = INITIAL_RZ
        print("Bounds:")
        for i, (lo, hi) in enumerate(BOUNDS):
            print(f"  coil {i}: [{lo:.1f}, {hi:.1f}]"
                  f"{'  (fixed)' if lo == hi else ''}")

    n_cores = int(PROCESSES) if PROCESSES else multiprocessing.cpu_count()
    print(f"Processes = {n_cores}\n")

    initialize_mgrid_field(str(MGRID_PATH), nfp=NFP, full_torus=FULL_TORUS)
    set_extcur(NOMINAL_EXTCUR)

    # Baseline is always the standard NOMINAL_EXTCUR, even though it lies
    # outside this island's box.  The DE search box is overridden below.
    cfg = OptimizationConfig(
        mgrid_path=str(MGRID_PATH),
        nfp=NFP,
        full_torus=FULL_TORUS,
        initial_rz=initial_rz,
        initial_bounds=NOMINAL_EXTCUR,
        nturn=DE_NTURN,
        nphi=DE_NPHI,
        npart=DE_NPART,
        delt_r=DELT_R,
        n_pop=N_POP,
        max_gen=MAX_GEN,
        F=0.5,
        CR=0.7,
        processes=n_cores,
        output_dir=run_dir,
        csv_filename="h1_island_optimisation_log.csv",
        device_name="H1",
        log_level=logging.INFO,
        ftol=FTOL,
        patience=PATIENCE,
        seed=SEED,
        min_minor_radius=MIN_MINOR_RADIUS,
        adapt_bounds=False,   # stay strictly inside the search box
    )
    # Override the symmetric config bounds: PCA y-box or the manual island box.
    cfg._abs_bounds = y_bounds.copy() if USE_PCA else BOUNDS.copy()

    de = DifferentialEvolution(cfg)
    if USE_PCA:
        de.to_x = to_x   # DE searches y-space; every evaluation maps y -> x
    best_individual, best_fitness, all_infos = de.run()

    best_x = to_x(best_individual) if USE_PCA else best_individual
    print(f"\nBest (DE): eps_eff^(3/2) = {best_fitness:.6e}")
    for c, val in enumerate(best_x):
        print(f"  coil {c}: {val:10.1f} A")
    if USE_PCA:
        print(f"  (y-space: {np.round(best_individual, 4).tolist()})")

    # Full-resolution verification from the DE-recorded axis branch.
    best_axis = None
    for info in all_infos:
        if (info.get("axis_rz") is not None
                and np.allclose(info.get("extcur"), best_x)
                and abs(info.get("epsilon_eff", np.inf) - best_fitness) < 1e-12):
            best_axis = np.asarray(info["axis_rz"], dtype=np.float64)
            break
    if best_axis is None:
        best_axis = initial_rz
    verify_full_res(best_x, "best    ", initial_rz=best_axis)

    sys.stdout = _orig_stdout
    sys.stderr = _orig_stderr
    console_log.close()
    print(f"\nConsole output saved to {run_dir / 'console.log'}")
    print("Done.")


if __name__ == "__main__":
    main()
