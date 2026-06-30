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
    # 5 coils total → degree of freedom = 4.
    # Coil 0 (main coil, 50000 A) is held fixed; coils 1-4 are free.
    #
    #   Coil      Nominal       Fixed / Free       Bounds
    #   ─────     ───────       ─────────────       ─────────────────
    #   0          50000        FIXED
    #   1           5000        FREE                [ -5000,  15000]
    #   2              1        FREE                [-10000,  10000]
    #   3         -80000        FREE                [-100000, -60000]
    #   4         -40000        FREE                [ -50000, -30000]

    extcur_fixed = np.array([50000], dtype=np.float64)   # coil 0 fixed

    BOUNDS = np.array([            # coils 1-4 free
        [ 5000,  5000],          # coil 1
        [0,  5000],          # coil 2
        [-80000, -80000],         # coil 3
        [-40000, -40000],          # coil 4
    ], dtype=np.float64)

    INITIAL_RZ = np.array([1.26, 0.0], dtype=np.float64)

    # ── 3.  Configure & run ──
    from ripplepy.optimize import OptimizationConfig, DifferentialEvolution

    config = OptimizationConfig(
        mgrid_path=str(MGRID_PATH),
        nfp=3,
        full_torus=False,
        extcur_fixed=extcur_fixed,
        initial_rz=INITIAL_RZ,
        initial_bounds=BOUNDS,
        nturn=200,
        nphi=360,
        npart=5000,
        delt_r=0.05,
        n_pop=8,
        max_gen=3,
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
    print(f"  Coils total  : {len(extcur_fixed) + len(BOUNDS)}  (n)")
    print(f"  Fixed        : {len(extcur_fixed)}  (extcur_fixed)")
    print(f"  Free (dof)   : {len(BOUNDS)}  (= n-1)")
    print(f"  Population   : {config.n_pop}")
    print(f"  Max gen      : {config.max_gen}")
    print(f"  Bounds (free coils 1–4):")
    for i, (lo, hi) in enumerate(BOUNDS, start=1):
        print(f"    coil {i}: [{lo:8.1f}, {hi:8.1f}]")
    print(f"  Output → {OUTPUT_DIR}")
    print()

    de = DifferentialEvolution(config)
    best_individual, best_fitness, all_infos = de.run()

    # ── 4.  Results ──
    print("\n" + "=" * 60)
    print("OPTIMISATION COMPLETE")
    print("=" * 60)
    print(f"\nBest fitness (ε_eff^(3/2)): {best_fitness:.6e}")

    full_extcur = np.concatenate([extcur_fixed, best_individual])
    print(f"\nOptimal coil currents  (coil 0 fixed, coils 1-4 optimised):")
    for i in range(len(nominal_extcur)):
        tag = "FIXED" if i == 0 else "FREE "
        change_pct = (full_extcur[i] - nominal_extcur[i]) / abs(nominal_extcur[i]) * 100
        print(f"  coil {i}:  {full_extcur[i]:10.1f} A   ({tag})   "
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
        ax2.bar(x + width / 2, full_extcur, width, alpha=0.8,
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
