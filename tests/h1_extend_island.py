#!/usr/bin/env python3
"""Extend an island-mapping HDF5 to a new (wider / adjusted) current box.

The oracle is deterministic, so the old samples' feasibility and eps values
are reused as-is (old samples outside the new box are dropped if a coil is
narrowed).  Only the region newly exposed by the wider bounds is re-mapped:
per-coil slabs where the new bounds extend beyond the old box, sampled at the
base run's adaptive samples-per-volume density.  The new box must OVERLAP the
old box per coil (continuous extension); a disjoint jump is rejected.  The
combined sample set is re-clustered over the new box.

Output: <stem>_ext.h5 next to the input (auto-incremented _ext2, _ext3 ... for
repeated extensions); override with --out.

Edit NEW_BOUNDS (the adjusted full ENGINEERING_BOUNDS) in the CONFIGURATION
block, then:

    python tests/h1_extend_island.py islands_dr0.1.h5 \
        [--n-extension 20000] [--processes 64] [--out extended.h5]
"""

import argparse
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
MGRID_PATH = BASE / "tests" / "test_file" / "mgrid_h1_design.nc"
NFP = 3
FULL_TORUS = False
NOMINAL_EXTCUR = np.array([50000.0, 5000.0, 3000.0, -80000.0, -40000.0])
INITIAL_RZ = np.array([1.26, 0.0], dtype=np.float64)
CLUSTER_EPS = 0.15
CLUSTER_MIN_SAMPLES = 5
MIN_PER_SLAB = 100          # floor for the adaptive per-slab sample count

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION — edit NEW_BOUNDS to the adjusted full ENGINEERING_BOUNDS
# (can widen AND narrow any coil; the extension region = new box minus old
# box is computed automatically, old samples outside the new box are dropped).
# ═══════════════════════════════════════════════════════════════════════
NEW_BOUNDS = np.array([
    [ 50000.0,   50000.0],  # coil 0: TF, fixed
    [     0.0,   10000.0],  # coil 1
    [     0.0,   10000.0],  # coil 2
    [-220000.0,  -40000.0], # coil 3
    [-140000.0,  -10000.0], # coil 4
])
# ═══════════════════════════════════════════════════════════════════════


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("h5", help="base island-mapping HDF5 to extend")
    ap.add_argument("--n-extension", type=int, default=None,
                    help="manual total sample override (default: adaptive, "
                         "matching the base run's samples-per-volume)")
    ap.add_argument("--processes", type=int, default=None)
    ap.add_argument("--cluster-eps", type=float, default=CLUSTER_EPS)
    ap.add_argument("--cluster-min-samples", type=int,
                    default=CLUSTER_MIN_SAMPLES)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from ripplepy import OptimizationConfig
    from ripplepy.islands import (
        _cluster_full_feasible,
        load_island_mapping_h5,
        map_feasible_islands,
        sample_bounds,
        save_island_mapping_h5,
    )

    old = load_island_mapping_h5(Path(args.h5))
    old_bounds = np.asarray(old["bounds"], dtype=np.float64)
    new_bounds = NEW_BOUNDS.astype(np.float64)
    p = old.get("params", {})
    gp = lambda k, d: d if p.get(k) is None else p[k]

    print(f"base HDF5 : {args.h5} (delt_r={old.get('delt_r')}, "
          f"samples={old['samples'].shape[0]})")
    print("old box   :", np.round(old_bounds, 1).tolist())
    print("new box   :", np.round(new_bounds, 1).tolist())

    # The adjusted box must OVERLAP the old box per coil (continuous
    # extension).  A disjoint jump (no overlap on some coil) is rejected;
    # narrowing a coil is allowed -- its old samples outside the new box are
    # dropped in the merge below.
    tol = 1e-9
    disjoint = [d for d in range(5)
                if new_bounds[d, 1] < old_bounds[d, 0] - tol or
                   new_bounds[d, 0] > old_bounds[d, 1] + tol]
    if disjoint:
        for d in disjoint:
            print(f"  coil {d}: old [{old_bounds[d, 0]:.1f}, "
                  f"{old_bounds[d, 1]:.1f}] vs new "
                  f"[{new_bounds[d, 0]:.1f}, {new_bounds[d, 1]:.1f}] "
                  f"-- no overlap")
        raise SystemExit(
            "ERROR: the new box must overlap the old box per coil "
            "(continuous extension); a disjoint jump is rejected.")

    # Per-coil exposed slabs where the new box extends beyond the old box.
    slabs = []
    for d in range(5):
        if new_bounds[d, 0] < old_bounds[d, 0] - 1e-9:
            b = new_bounds.copy()
            b[d, 1] = old_bounds[d, 0]
            slabs.append(b)
        if new_bounds[d, 1] > old_bounds[d, 1] + 1e-9:
            b = new_bounds.copy()
            b[d, 0] = old_bounds[d, 1]
            slabs.append(b)
    if not slabs:
        print("New bounds do not extend beyond the old box; nothing to do.")
        return

    # Adaptive sampling density: the extension region is sampled at the base
    # run's samples-per-volume (n_old / V_old), so the new region has the same
    # average density as the original box.  --n-extension overrides the total
    # (still distributed over the slabs by volume).
    free = np.flatnonzero(old_bounds[:, 1] - old_bounds[:, 0] > 1e-12)
    v_old = float(np.prod(old_bounds[free, 1] - old_bounds[free, 0]))
    rho = old["samples"].shape[0] / v_old if v_old > 0 else 0.0
    slab_vols = [float(np.prod([max(s[d, 1] - s[d, 0], 1e-12)
                                for d in free]))
                 for s in slabs]
    tot_vol = sum(slab_vols)
    if args.n_extension:
        totals = [args.n_extension * v / tot_vol for v in slab_vols]
    else:
        totals = [rho * v for v in slab_vols]
    n_per_slab = [max(MIN_PER_SLAB, int(round(t))) for t in totals]
    ext = np.vstack([sample_bounds(s, n, seed=args.seed + k)
                     for k, (s, n) in enumerate(zip(slabs, n_per_slab))])
    print(f"extension : {len(slabs)} slab(s) -> {ext.shape[0]} new samples "
          f"(density {rho:.2e}/A^4, {n_per_slab} per slab)")

    # Re-map only the extension samples with the base run's oracle parameters.
    cfg = OptimizationConfig(
        mgrid_path=str(MGRID_PATH), nfp=NFP, full_torus=FULL_TORUS,
        initial_rz=INITIAL_RZ, initial_bounds=NOMINAL_EXTCUR,
    )
    new_res = map_feasible_islands(
        cfg, new_bounds, float(old["delt_r"]), samples=ext,
        rmin=gp("rmin", 1.0), rmax=gp("rmax", 1.35), rstep=gp("rstep", 0.05),
        short_nturn=gp("short_nturn", 20), short_nphi=gp("short_nphi", 360),
        full_nturn=gp("full_nturn", 400), full_nphi=gp("full_nphi", 360),
        full_npart=gp("full_npart", 2000), compute_eps=True,
        smooth_n_harmonics=gp("smooth_n_harmonics", 4),
        smooth_residual_tol=gp("smooth_residual_tol", 0.05),
        smooth_max_gap=gp("smooth_max_gap", 1.0),
        smooth_min_points=gp("smooth_min_points", 16),
        processes=args.processes, seed=args.seed, do_cluster=False,
    )

    # Merge: keep old samples inside the new box (narrowed-away strips are
    # dropped); the oracle is deterministic, so kept samples are reused as-is.
    o_s = old["samples"]
    inside = np.all((o_s >= new_bounds[:, 0]) & (o_s <= new_bounds[:, 1]),
                    axis=1)
    print(f"merge     : keeping {int(inside.sum())}/{len(o_s)} old samples "
          f"(inside new box)")

    def cat(a, b):
        return np.concatenate([np.asarray(a)[inside], np.asarray(b)])

    samples = cat(o_s, new_res["samples"])
    full = cat(old["full_feasible"].astype(bool),
               new_res["full_feasible"].astype(bool))
    eps = cat(old.get("eps", np.full(len(o_s), np.nan)), new_res["eps"])
    axis_used = cat(old["axis_used_RZ"], new_res["axis_used_RZ"])

    def get_arr(name):
        return old.get(name, np.full(len(o_s), np.nan))

    iota = cat(get_arr("iota"), new_res["iota"])
    volume = cat(get_arr("volume"), new_res["volume"])
    minor_radius = cat(get_arr("minor_radius"), new_res["minor_radius"])
    aspect_ratio = cat(get_arr("aspect_ratio"), new_res["aspect_ratio"])
    param_values = np.column_stack(
        [eps, iota, volume, minor_radius, aspect_ratio])
    param_names = ["eps", "iota", "volume", "minor_radius", "aspect_ratio"]

    combined = {
        "delt_r": float(old["delt_r"]),
        "bounds": new_bounds,
        "samples": samples,
        "axis_feasible": cat(old["axis_feasible"].astype(bool),
                             new_res["axis_feasible"].astype(bool)),
        "axis_count": cat(old["axis_count"], new_res["axis_count"]),
        "axis_used_RZ": axis_used,
        "short_feasible": cat(old["short_feasible"].astype(bool),
                              new_res["short_feasible"].astype(bool)),
        "short_istate": cat(old["short_istate"], new_res["short_istate"]),
        "full_feasible": full,
        "full_istate": cat(old["full_istate"], new_res["full_istate"]),
        "smooth_residual": cat(old["smooth_residual"], new_res["smooth_residual"]),
        "smooth_max_gap": cat(old["smooth_max_gap"], new_res["smooth_max_gap"]),
        "eps": eps,
        "iota": iota,
        "volume": volume,
        "minor_radius": minor_radius,
        "aspect_ratio": aspect_ratio,
        "param_names": param_names,
        "params": p,
        "n_samples": int(samples.shape[0]),
        "seed": args.seed,
    }
    islands, free_dims = _cluster_full_feasible(
        samples, full, axis_used, eps=args.cluster_eps,
        min_samples=args.cluster_min_samples,
        param_values=param_values, param_names=param_names)
    combined["islands"] = islands
    combined["free_dims"] = np.asarray(free_dims, dtype=int)

    stem = Path(args.h5).stem
    out = Path(args.out) if args.out else \
        Path(args.h5).parent / f"{stem}_ext.h5"
    idx = 2
    while out.exists():
        out = Path(args.h5).parent / f"{stem}_ext{idx}.h5"
        idx += 1
    save_island_mapping_h5(out, combined)
    print(f"\nsaved -> {out}: total samples={len(samples)}, "
          f"full_feasible={int(full.sum())}, islands={len(islands)}")
    for isl in islands:
        print(f"  island {isl['island_id']}: n={isl['n_points']}, "
              f"axis_R~{isl['mean_axis_R']:.4f}")


if __name__ == "__main__":
    main()
