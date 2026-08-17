#!/usr/bin/env python3
"""
H1 stellarator coil-current optimisation — end-to-end pipeline
===============================================================

Chain:
  1. auto-bounds  : nominal coil currents -> +/-bounds_fraction search box
  2. exploration  : Sobol + find_axis survey, adaptively widen until the
                    feasible-region boundary is bracketed -> extent_bounds
  3. optimisation : JADE differential evolution within extent_bounds
  4. verification : re-evaluate the best solution at full resolution

Simply run it.
"""

from __future__ import annotations

import logging
from pathlib import Path
import multiprocessing

import numpy as np

# ---------------------------------------------------------------------------
# Paths  (safe outside __main__ — these are just constant definitions)
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent.parent   # tests/ -> ripplepy/
MGRID_PATH = BASE / "tests" / "test_file" / "mgrid_h1_design.nc"
OUTPUT_DIR = BASE / "tests" / "h1_optimisation"

# ---------------------------------------------------------------------------
# Problem / pipeline settings
# ---------------------------------------------------------------------------
NOMINAL_EXTCUR = np.array([50000.0, 5000.0, 3000.0, -80000.0, -40000.0])
INITIAL_RZ = np.array([1.26, 0.0], dtype=np.float64)
NFP = 3
FULL_TORUS = False
DELT_R = 0.05

# 1-D nominal currents -> automatic bounds (see OptimizationConfig)
BOUNDS_FRACTION = 0.2

# Adaptive feasible-region exploration
EXPLORE = True                # set False to skip and use BOUNDS_FRACTION directly
SURVEY_N_SAMPLES = 256
EXPLORE_MAX_ROUNDS = 5
EXPLORE_EXPAND = 1.5

# Differential Evolution (JADE) settings
N_POP = 40
MAX_GEN = 30
FTOL = 1e-8
PATIENCE = 15
SEED = 42
DE_NTURN = 200
DE_NPHI = 360
DE_NPART = 5000

# Full-resolution verification (single evaluations)
VERIFY_NTURN = 400
VERIFY_NPHI = 720
VERIFY_NPART = 5000


def verify_extcur(extcur, label):
    """Re-evaluate one extcur at full resolution in the main process."""
    from ripplepy import (
        set_extcur, find_axis, compute_initial_gradpsi_nemov,
        set_trace_parameters, compute_epstot,
    )
    set_extcur(extcur)
    axis_rz, _, _, ok = find_axis(INITIAL_RZ, xtol=1e-6, max_iter=100,
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
    """Run the end-to-end H1 optimisation (wrapped for multiprocessing safety)."""
    from ripplepy import initialize_mgrid_field, set_extcur
    from ripplepy.optimize import (
        OptimizationConfig, DifferentialEvolution, explore_feasible_region,
    )

    print("=" * 60)
    print("H1 Coil-Current Optimisation — end-to-end pipeline")
    print("=" * 60)

    if OUTPUT_DIR.exists():
        print(f"\nOutput directory {OUTPUT_DIR} already exists — deleting it …")
        import shutil
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1.  Initialise the magnetic field ──
    print(f"\nLoading mgrid from {MGRID_PATH}")
    initialize_mgrid_field(str(MGRID_PATH), nfp=NFP, full_torus=FULL_TORUS)
    set_extcur(NOMINAL_EXTCUR)
    print(f"  Nominal coil currents: {NOMINAL_EXTCUR}")

    # ── 2.  Search box: auto bounds + adaptive feasible-region exploration ──
    config = OptimizationConfig(
        mgrid_path=str(MGRID_PATH),
        nfp=NFP,
        full_torus=FULL_TORUS,
        initial_rz=INITIAL_RZ,
        initial_bounds=NOMINAL_EXTCUR,       # 1-D -> auto +/-bounds_fraction
        delt_r=DELT_R,
        bounds_fraction=BOUNDS_FRACTION,
        output_dir=OUTPUT_DIR,
    )

    if EXPLORE:
        print("\n=== [explore] adaptive feasible-region survey (Sobol + find_axis) ===")
        res = explore_feasible_region(
            config, n_samples=SURVEY_N_SAMPLES, seed=0,
            expand_factor=EXPLORE_EXPAND, max_rounds=EXPLORE_MAX_ROUNDS,
        )
        search_bounds = res["extent_bounds"]
        print("  -> using extent_bounds (explored feasible region) for the DE run")
        print("  -> core_bounds (inner reliable region): "
              f"{np.round(res['core_bounds'][:, 1], 3).tolist()}")
    else:
        print("\n=== [explore] skipped (EXPLORE=False) — using auto ±"
              f"{int(BOUNDS_FRACTION * 100)}% bounds ===")
        search_bounds = np.column_stack(
            [NOMINAL_EXTCUR, np.full(len(NOMINAL_EXTCUR), BOUNDS_FRACTION)])

    n_cores = multiprocessing.cpu_count()
    print(f"\nDetected {n_cores} CPU cores — using all for optimisation.")

    # ── 3.  Configure & run JADE ──
    de_config = OptimizationConfig(
        mgrid_path=str(MGRID_PATH),
        nfp=NFP,
        full_torus=FULL_TORUS,
        initial_rz=INITIAL_RZ,
        initial_bounds=search_bounds,
        nturn=DE_NTURN,
        nphi=DE_NPHI,
        npart=DE_NPART,
        delt_r=DELT_R,
        n_pop=N_POP,
        max_gen=MAX_GEN,
        F=0.5,
        CR=0.7,
        processes=n_cores,
        output_dir=OUTPUT_DIR,
        csv_filename="h1_optimisation_log.csv",
        device_name="H1",
        log_file=OUTPUT_DIR / "h1_optimisation.log",
        log_level=logging.INFO,
        ftol=FTOL,
        patience=PATIENCE,
        seed=SEED,
    )

    print("\nStarting JADE optimisation …")
    print(f"  Coils        : {len(search_bounds)} (all free, auto bounds)")
    print("  Search box   : [nominal, fraction] -> [lo, hi]:")
    for i, (nom, frac) in enumerate(search_bounds):
        lo = nom - abs(nom) * frac
        hi = nom + abs(nom) * frac
        print(f"    coil {i}: [{nom:8.1f}, {frac:.2f}]  ->  [{lo:10.1f}, {hi:10.1f}]")
    print(f"  Population   : {de_config.n_pop}")
    print(f"  Max gen      : {de_config.max_gen} (ftol={FTOL}, patience={PATIENCE})")
    print(f"  Strategy     : jade (adaptive F/CR)")
    print(f"  Output       : {OUTPUT_DIR}")
    print()

    de = DifferentialEvolution(de_config)
    best_individual, best_fitness, all_infos = de.run()

    # ── 4.  Results ──
    print("\n" + "=" * 60)
    print("OPTIMISATION COMPLETE")
    print("=" * 60)
    print(f"\nBest fitness (eps_eff^(3/2), DE resolution): {best_fitness:.6e}")

    print("\nOptimal coil currents:")
    for i, val in enumerate(best_individual):
        nom = NOMINAL_EXTCUR[i]
        change_pct = (val - nom) / nom * 100 if abs(nom) > 1e-12 else float("nan")
        print(f"  coil {i}: {val:10.1f} A   (nominal {nom:7.1f} A,  "
              f"Δ = {change_pct:+.1f}%)")

    # ── 5.  Full-resolution verification ──
    print("\n=== [verify] full-resolution re-evaluation ===")
    verify_extcur(NOMINAL_EXTCUR, "nominal")
    verify_extcur(best_individual, "best    ")

    # ── 6.  Convergence plot ──
    try:
        import matplotlib.pyplot as plt

        gen_best = {}
        for info in all_infos:
            g = info["Generation"]
            if g == "start":
                continue
            eps = info["epsilon_eff"]
            if g not in gen_best or eps < gen_best[g]:
                gen_best[g] = eps

        gens = sorted(gen_best.keys())
        vals = [gen_best[g] for g in gens]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

        ax1.semilogy(gens, vals, "b.-", linewidth=1.5, markersize=4)
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("eps_eff^(3/2)  (best)")
        ax1.set_title("H1 Optimisation — Convergence (JADE)")
        ax1.grid(True, alpha=0.3)

        labels = [f"Coil {i}" for i in range(len(NOMINAL_EXTCUR))]
        x = np.arange(len(labels))
        width = 0.35
        ax2.bar(x - width / 2, NOMINAL_EXTCUR, width, alpha=0.6,
                label="Nominal", color="gray")
        ax2.bar(x + width / 2, best_individual, width, alpha=0.8,
                label="Optimised", color="C0")
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels)
        ax2.set_ylabel("Coil current (A)")
        ax2.set_title("H1 — Coil currents")
        ax2.legend()
        ax2.grid(True, axis="y", alpha=0.3)

        fig.tight_layout()
        plot_path = OUTPUT_DIR / "h1_convergence.png"
        fig.savefig(plot_path, dpi=150)
        print(f"\nConvergence plot saved -> {plot_path}")
        plt.close()

    except ImportError:
        print("\n(matplotlib not available — skipping convergence plot)")

    print("\nDone.")


if __name__ == "__main__":
    main()
