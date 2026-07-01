#!/usr/bin/env python3
"""
H1 stellarator coil-current optimisation
=========================================

Optimises 4 of the 5 coil-group currents of the H1 heliac to minimise
neoclassical transport (ε_eff^(3/2)).

With 5 coils total the physical degree of freedom is 4 (n-1), so the
first coil (coil 0, nominal 50000 A) is held fixed and coils 1-4 are
varied.  Simply run it — checkpoint/restart is automatic.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths  (safe outside __main__ — these are just constant definitions)
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent.parent   # tests/ → ripplepy/
MGRID_PATH = BASE / "tests" / "test_file" / "mgrid_h1_design.nc"
OUTPUT_DIR = BASE / "tests" / "h1_optimisation"


def main():
    """Run the H1 optimisation (wrapped for multiprocessing safety)."""
    # ── 1.  Initialise the magnetic field ──
    from ripplepy import initialize_mgrid_field, set_extcur

    print("=" * 60)
    print("H1 Coil-Current Optimisation via Differential Evolution")
    print("=" * 60)

    if OUTPUT_DIR.exists():
        print(f"\nOutput directory {OUTPUT_DIR} already exists — deleting it and all contents … ")
        import shutil
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading mgrid from {MGRID_PATH}")
    initialize_mgrid_field(
        str(MGRID_PATH),
        nfp=3,
        full_torus=False,
    )
    nominal_extcur = np.array([50000, 5000, 1, -80000, -40000], dtype=np.float64)
    set_extcur(nominal_extcur)
    print(f"  Nominal coil currents: {nominal_extcur}")

    # ── 2.  Define the optimisation problem ──
    # 5 coils total → degree of freedom = 4 (= n-1).
    # Coil 0 is fixed via lo == hi; coils 1-4 are free.
    #
    #   Coil      Nominal       Status               Bounds
    #   ─────     ───────       ──────               ─────────────────
    #   0          50000        FIXED (lo=hi)        [ 50000,   50000]
    #   1           5000        FREE                  [ -5000,   15000]
    #   2              1        FREE                  [-10000,   10000]
    #   3         -80000        FREE                  [-100000, -60000]
    #   4         -40000        FREE                  [ -50000,  -30000]

    extcur_fixed = np.array([50000], dtype=np.float64)   # coil 0 fixed

    # BOUNDS = np.array([            # coils 1-4 free
    #     [ 5000,  5000],          # coil 1
    #     [0,  5000],          # coil 2
    #     [-80000, -80000],         # coil 3
    #     [-40000, -40000],          # coil 4
    # ], dtype=np.float64)
    BOUNDS = np.array([           
        [ 4500,  5500],          
        [0,  5500],          
        [-88000, -72000],         
        [-44000, -36000],         
    ], dtype=np.float64)

    INITIAL_RZ = np.array([1.26, 0.0], dtype=np.float64)

    # ── 3.  Configure & run ──
    from ripplepy.optimize import OptimizationConfig, DifferentialEvolution

    config = OptimizationConfig(
        mgrid_path=str(MGRID_PATH),
        nfp=3,
        full_torus=False,
        initial_rz=INITIAL_RZ,
        initial_bounds=BOUNDS,
        nturn=200,
        nphi=360,
        npart=5000,
        delt_r=0.05,
        n_pop=20,
        max_gen=5,
        F=0.5,
        CR=0.7,
        processes=8,
        output_dir=OUTPUT_DIR,
        csv_filename="h1_optimisation_log.csv",
        device_name="H1",
        log_file=OUTPUT_DIR / "h1_optimisation.log",
        log_level=logging.INFO,
        checkpoint_interval=10,
        ftol=1e-8,
        patience=15,
        restart_best=None,
        seed=42,
    )

    print("\nStarting optimisation …")
    print(f"  Coils total  : {len(BOUNDS)}  (n)")
    n_fixed = int(np.sum(BOUNDS[:, 0] == BOUNDS[:, 1]))
    print(f"  Fixed         : {n_fixed}  (coils with lo == hi)")
    print(f"  Free (dof)    : {len(BOUNDS) - n_fixed}  (= n-1)")
    print(f"  Population    : {config.n_pop}")
    print(f"  Max gen       : {config.max_gen}")
    fixed_indices = [i for i in range(len(BOUNDS)) if BOUNDS[i, 0] == BOUNDS[i, 1]]
    print(f"  Fixed coils   : {fixed_indices}")
    print(f"  Bounds:")
    for i, (lo, hi) in enumerate(BOUNDS):
        tag = "FIXED" if lo == hi else "FREE "
        print(f"    coil {i}: [{lo:8.1f}, {hi:8.1f}]  ({tag})")
    print(f"  Output → {OUTPUT_DIR}")
    print()

    de = DifferentialEvolution(config)
    best_individual, best_fitness, all_infos = de.run()

    # ── 4.  Results ──
    print("\n" + "=" * 60)
    print("OPTIMISATION COMPLETE")
    print("=" * 60)
    print(f"\nBest fitness (ε_eff^(3/2)): {best_fitness:.6e}")

    # best_individual is the full extcur (all 5 coils).
    print(f"\nOptimal coil currents:")
    for i, val in enumerate(best_individual):
        is_fixed = BOUNDS[i, 0] == BOUNDS[i, 1]
        tag = "FIXED" if is_fixed else "FREE "
        change_pct = (val - nominal_extcur[i]) / abs(nominal_extcur[i]) * 100
        print(f"  coil {i}:  {val:10.1f} A   ({tag})   "
              f"nominal {nominal_extcur[i]:7.1f} A,  Δ = {change_pct:+.1f}%")

    # ── 5.  Convergence plot ──
    try:
        import matplotlib.pyplot as plt

        gen_best = {}
        for info in all_infos:
            g = info["Generation"]
            eps = info["epsilon_eff"]
            if g not in gen_best or eps < gen_best[g]:
                gen_best[g] = eps

        gens = sorted(gen_best.keys())
        vals = [gen_best[g] for g in gens]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

        ax1.semilogy(gens, vals, "b.-", linewidth=1.5, markersize=4)
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("ε_eff^(3/2)  (best)")
        ax1.set_title("H1 Optimisation — Convergence")
        ax1.grid(True, alpha=0.3)

        labels = [f"Coil {i}" for i in range(len(nominal_extcur))]
        x = np.arange(len(labels))
        width = 0.35
        ax2.bar(x - width / 2, nominal_extcur, width, alpha=0.6, label="Nominal",
                color="gray")
        ax2.bar(x + width / 2, best_individual, width, alpha=0.8,
                label="Optimised", color="C0")
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels)
        ax2.set_ylabel("Coil current (A)")
        ax2.set_title("H1 — Coil currents  (coil 0 fixed)")
        ax2.legend()
        ax2.grid(True, axis="y", alpha=0.3)

        fig.tight_layout()
        plot_path = OUTPUT_DIR / "h1_convergence.png"
        fig.savefig(plot_path, dpi=150)
        print(f"\nConvergence plot saved → {plot_path}")
        plt.close()

    except ImportError:
        print("\n(matplotlib not available — skipping convergence plot)")

    print("\nDone.")


if __name__ == "__main__":
    main()
