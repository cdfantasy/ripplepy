#!/usr/bin/env python3
"""Optimise epsilon_eff inside a manually specified island box.

Edit the constants in the CONFIGURATION block below, then run:

    python tests/h1_optimise_island.py
"""

from __future__ import annotations

import logging
import multiprocessing
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
# ═══════════════════════════════════════════════════════════════════════════

NOMINAL_EXTCUR = np.array([50000.0, 5000.0, 3000.0, -80000.0, -40000.0])
NFP = 3
FULL_TORUS = False


def make_run_dir() -> Path:
    tag = "_".join(
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


def main():
    from ripplepy import initialize_mgrid_field, set_extcur
    from ripplepy.optimize import DifferentialEvolution, OptimizationConfig

    run_dir = make_run_dir()
    print(f"Output -> {run_dir}")
    print("Bounds:")
    for i, (lo, hi) in enumerate(BOUNDS):
        print(f"  coil {i}: [{lo:.1f}, {hi:.1f}]"
              f"{'  (fixed)' if lo == hi else ''}")

    n_cores = int(PROCESSES) if PROCESSES else multiprocessing.cpu_count()
    print(f"Processes = {n_cores}\n")

    initialize_mgrid_field(str(MGRID_PATH), nfp=NFP, full_torus=FULL_TORUS)
    set_extcur(NOMINAL_EXTCUR)

    baseline = BOUNDS.mean(axis=1)

    cfg = OptimizationConfig(
        mgrid_path=str(MGRID_PATH),
        nfp=NFP,
        full_torus=FULL_TORUS,
        initial_rz=INITIAL_RZ,
        initial_bounds=np.column_stack([baseline, np.zeros(len(baseline))]),
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
    )
    # Override the symmetric config bounds by the manual absolute box.
    cfg._abs_bounds = BOUNDS.copy()

    de = DifferentialEvolution(cfg)
    best_individual, best_fitness, all_infos = de.run()

    print(f"\nBest (DE): eps_eff^(3/2) = {best_fitness:.6e}")
    for c, val in enumerate(best_individual):
        print(f"  coil {c}: {val:10.1f} A")

    # Full-resolution verification from the DE-recorded axis branch.
    best_axis = None
    for info in all_infos:
        if (info.get("axis_rz") is not None
                and np.allclose(info.get("extcur"), best_individual)
                and abs(info.get("epsilon_eff", np.inf) - best_fitness) < 1e-12):
            best_axis = np.asarray(info["axis_rz"], dtype=np.float64)
            break
    if best_axis is None:
        best_axis = INITIAL_RZ
    verify_full_res(best_individual, "best    ", initial_rz=best_axis)

    print("\nDone.")


if __name__ == "__main__":
    main()
