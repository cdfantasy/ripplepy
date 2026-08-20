#!/usr/bin/env python3
"""H1 Phase 1+2: map feasible-current islands as delt_r is increased.

Each delt_r layer is saved as an HDF5 file under tests/h1_islands; later
layers reuse the previous layer's islands to generate their samples (hot
start), so the mapping can be resumed or extended without re-running earlier
layers.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
MGRID_PATH = BASE / "tests" / "test_file" / "mgrid_h1_design.nc"
OUTPUT_ROOT = BASE / "tests" / "h1_islands"

NOMINAL_EXTCUR = np.array([50000.0, 5000.0, 3000.0, -80000.0, -40000.0])
INITIAL_RZ = np.array([1.26, 0.0], dtype=np.float64)
NFP = 3
FULL_TORUS = False

# Coil 0 is the TF coil and is kept fixed.
FIXED_COILS = [0]

# Very wide engineering hard-box used ONLY by the first layer's low-resolution
# pre-survey.  The actual dense-mapping bounds are data-driven: q02/q98 of the
# full-feasible points found in this box (borrowing survey_feasibility's idea).
ENGINEERING_BOUNDS = np.array([
    [ 50000.0,   50000.0],  # coil 0: TF, fixed
    [     0.0,   10000.0],  # coil 1
    [     0.0,   10000.0],  # coil 2
    [-220000.0,  -40000.0], # coil 3
    [-100000.0,  -10000.0], # coil 4
])

DELT_R_LIST = [0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12]

# Previously found good staged-delt_r solutions.  Used as a safety prior when
# deriving the first-layer mapping bounds: the data-driven q02/q98 box is
# expanded, if necessary, so none of these known-good currents are clipped.
KNOWN_GOOD_SOLUTIONS = np.array([
    [50000.0, 2401.9, 1419.6, -107178.6, -61004.8],  # delt_r = 0.06
    [50000.0, 2527.9, 1036.3, -133371.7, -67991.6],  # delt_r = 0.07
    [50000.0, 2304.1,  829.0, -153739.6, -63354.3],  # delt_r = 0.08
    [50000.0, 2123.8,  735.0, -153850.7, -58941.5],  # delt_r = 0.09
    [50000.0, 2053.3,  792.5, -175575.9, -51580.6],  # delt_r = 0.10
    [50000.0, 2078.4,  855.9, -174436.2, -50388.2],  # delt_r = 0.11
    [50000.0, 2494.1,  980.6, -139549.0, -40340.2],  # delt_r = 0.12
])

# Phase 1 sampling budget (per layer)
# First layer maps directly on the engineering box.  Subsequent layers draw
# per-island local samples + a small global verification set; keep these
# modest so a 7-layer overnight run stays feasible.
N_PRE_SURVEY = 4096
N_LOCAL_PER_ISLAND = 1000
N_GLOBAL = 200
ALPHA = 1.5

# Oracle parameters
RMIN, RMAX, RSTEP = 1.00, 1.35, 0.05
SHORT_NTURN, SHORT_NPHI = 20, 72
FULL_NTURN, FULL_NPHI = 200, 360
SMOOTH_N_HARMONICS = 4
SMOOTH_RESIDUAL_TOL = 0.05
SMOOTH_MAX_GAP = 1.0
SMOOTH_MIN_POINTS = 16

CLUSTER_EPS = 0.15
CLUSTER_MIN_SAMPLES = 5
PROGRESS_INTERVAL = 500


def ensure_known_good_inside(bounds, hard_bounds, margin_frac=0.05):
    """Expand `bounds` so every known-good solution is inside it.

    The q02/q98 pre-survey box can clip a known-good island when the
    pre-survey has few full-feasible samples.  This prior only widens the box;
    it never shrinks it and never exceeds `hard_bounds`.
    """
    bounds = np.asarray(bounds, dtype=np.float64).copy()
    hard = np.asarray(hard_bounds, dtype=np.float64)
    for good in KNOWN_GOOD_SOLUTIONS:
        good = np.asarray(good, dtype=np.float64)
        for d in range(bounds.shape[0]):
            margin = margin_frac * max(abs(good[d]), 1.0)
            bounds[d, 0] = min(bounds[d, 0], good[d] - margin)
            bounds[d, 1] = max(bounds[d, 1], good[d] + margin)
    # clamp back to the engineering hard-box; keep fixed coils fixed
    bounds[:, 0] = np.maximum(bounds[:, 0], hard[:, 0])
    bounds[:, 1] = np.minimum(bounds[:, 1], hard[:, 1])
    fixed = hard[:, 0] == hard[:, 1]
    bounds[fixed, 0] = hard[fixed, 0]
    bounds[fixed, 1] = hard[fixed, 1]
    return bounds


def survey_first_bounds(cfg, dr, processes):
    """Borrow survey_feasibility's q02/q98 idea for the first layer.

    A low-resolution mapping is run on the engineering box with clustering
    disabled; the returned box is the q02/q98 extent of its full-feasible
    samples, clamped to the engineering box.
    """
    from ripplepy.islands import (
        full_feasible_suggested_bounds,
        map_feasible_islands as map_islands,
    )

    print("  first layer pre-survey on engineering box ...")
    pre = map_islands(
        cfg, ENGINEERING_BOUNDS, dr,
        n_samples=N_PRE_SURVEY,
        rmin=RMIN, rmax=RMAX, rstep=RSTEP,
        short_nturn=SHORT_NTURN, short_nphi=SHORT_NPHI,
        full_nturn=FULL_NTURN, full_nphi=FULL_NPHI,
        smooth_n_harmonics=SMOOTH_N_HARMONICS,
        smooth_residual_tol=SMOOTH_RESIDUAL_TOL,
        smooth_max_gap=SMOOTH_MAX_GAP,
        smooth_min_points=SMOOTH_MIN_POINTS,
        processes=processes,
        seed=42 + int(dr * 100),
        do_cluster=False,
    )
    bounds = full_feasible_suggested_bounds(
        pre, ENGINEERING_BOUNDS, min_points=8)
    bounds = ensure_known_good_inside(bounds, ENGINEERING_BOUNDS)
    print(f"  pre-survey: axis_feasible={pre['axis_feasible'].sum()}, "
          f"full_feasible={pre['full_feasible'].sum()} -> mapping bounds "
          f"(known-good safety prior applied):")
    for i, (lo, hi) in enumerate(bounds):
        print(f"    coil {i}: [{lo:.1f}, {hi:.1f}]")
    return bounds


def main():
    parser = argparse.ArgumentParser(description="H1 feasibility-island mapping")
    parser.add_argument("--processes", type=int, default=None)
    parser.add_argument("--force", action="store_true",
                        help="recompute layers even if an HDF5 file exists")
    parser.add_argument("--smoke", action="store_true",
                        help="tiny single-layer run into tests/h1_islands_smoke")
    parser.add_argument("--cluster-eps", type=float, default=CLUSTER_EPS)
    parser.add_argument("--cluster-min-samples", type=int,
                        default=CLUSTER_MIN_SAMPLES)
    args = parser.parse_args()

    global OUTPUT_ROOT, DELT_R_LIST, N_PRE_SURVEY
    global N_LOCAL_PER_ISLAND, N_GLOBAL, SHORT_NTURN, SHORT_NPHI
    global FULL_NTURN, FULL_NPHI, SMOOTH_MIN_POINTS, PROGRESS_INTERVAL

    if args.smoke:
        OUTPUT_ROOT = BASE / "tests" / "h1_islands_smoke"
        DELT_R_LIST = [0.06]
        N_PRE_SURVEY = 32
        N_LOCAL_PER_ISLAND = 16
        N_GLOBAL = 8
        SHORT_NTURN, SHORT_NPHI = 5, 36
        FULL_NTURN, FULL_NPHI = 20, 72
        SMOOTH_MIN_POINTS = 8
        PROGRESS_INTERVAL = 8
        args.force = True
        print("SMOKE MODE: tiny budgets, output -> tests/h1_islands_smoke")

    from ripplepy import OptimizationConfig
    from ripplepy.islands import (
        generate_next_samples,
        load_island_mapping_h5,
        map_feasible_islands,
        sample_bounds,
    )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    cfg = OptimizationConfig(
        mgrid_path=str(MGRID_PATH),
        nfp=NFP,
        full_torus=FULL_TORUS,
        initial_rz=INITIAL_RZ,
        initial_bounds=NOMINAL_EXTCUR,
    )

    print("Engineering hard-box:")
    for i, (lo, hi) in enumerate(ENGINEERING_BOUNDS):
        print(f"  coil {i}: [{lo:.1f}, {hi:.1f}]"
              f"{'  (fixed)' if lo == hi else ''}")

    prev = None
    for dr in DELT_R_LIST:
        h5 = OUTPUT_ROOT / f"islands_dr{dr:g}.h5"
        print(f"\n{'='*60}\ndelt_r = {dr:g}  ->  {h5.name}")
        if h5.exists() and not args.force:
            print("  HDF5 exists; loading for hot start (use --force to redo).")
            prev = load_island_mapping_h5(h5)
            n_islands = len(prev.get("islands", []))
            print(f"  loaded {n_islands} island(s) from {h5.name}")
            continue

        if prev is None:
            # First layer: map islands directly on the engineering box with
            # N_PRE_SURVEY samples.  This is the same cascade as the old
            # pre-survey, but it is clustered and saved immediately, so the
            # first useful island map is available as soon as this finishes.
            bounds = ENGINEERING_BOUNDS
            samples = sample_bounds(bounds, N_PRE_SURVEY, seed=42 + int(dr*100))
            print(f"  first layer: engineering-box mapping, samples = {samples.shape[0]}")
        else:
            bounds = np.asarray(prev["bounds"], dtype=np.float64)
            samples = generate_next_samples(
                prev, bounds,
                n_local_per_island=N_LOCAL_PER_ISLAND,
                n_global=N_GLOBAL,
                alpha=ALPHA,
                seed=42 + int(dr*100))
            print(f"  hot-start layer: generated samples = {samples.shape[0]} "
                  f"({len(prev.get('islands', []))} previous island(s))")

        res = map_feasible_islands(
            cfg, bounds, dr,
            samples=samples,
            rmin=RMIN, rmax=RMAX, rstep=RSTEP,
            short_nturn=SHORT_NTURN, short_nphi=SHORT_NPHI,
            full_nturn=FULL_NTURN, full_nphi=FULL_NPHI,
            smooth_n_harmonics=SMOOTH_N_HARMONICS,
            smooth_residual_tol=SMOOTH_RESIDUAL_TOL,
            smooth_max_gap=SMOOTH_MAX_GAP,
            smooth_min_points=SMOOTH_MIN_POINTS,
            cluster_eps=args.cluster_eps,
            cluster_min_samples=args.cluster_min_samples,
            processes=args.processes,
            seed=42 + int(dr*100),
            output_h5=h5,
            progress_interval=PROGRESS_INTERVAL,
        )

        print(f"  axis_feasible    : {res['axis_feasible'].sum()}/{len(samples)}")
        print(f"  short_feasible   : {res['short_feasible'].sum()}")
        print(f"  full_feasible    : {res['full_feasible'].sum()}")
        print(f"  islands          : {len(res['islands'])}")
        for isl in res["islands"]:
            print(f"    island {isl['island_id']}: n={isl['n_points']}, "
                  f"axis_R~{isl['mean_axis_R']:.4f}")

        prev = res

    print("\nDone.")


if __name__ == "__main__":
    main()
