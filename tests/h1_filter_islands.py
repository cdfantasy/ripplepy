#!/usr/bin/env python3
"""Interactively remove clusters (islands) from a mapping HDF5.

Reads a mapping HDF5, prints the STORED per-island info (no re-verification
computations), asks which island ids to remove, marks those samples as
non-full-feasible (eps -> NaN), renumbers the remaining islands and saves a
filtered copy.

Usage:
    python tests/h1_filter_islands.py islands_dr0.12.h5
    python tests/h1_filter_islands.py islands_dr0.12.h5 --out kept.h5
"""

import argparse
from pathlib import Path

import numpy as np

from ripplepy.islands import load_island_mapping_h5, save_island_mapping_h5


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("h5", help="mapping HDF5 to filter")
    ap.add_argument("--out", default=None,
                    help="output HDF5 (default: <stem>_filtered.h5)")
    args = ap.parse_args()

    res = load_island_mapping_h5(Path(args.h5))
    n = int(res["samples"].shape[0])
    eps = np.asarray(res.get("eps", np.full(n, np.nan)), dtype=np.float64)
    full = res["full_feasible"].astype(bool).copy()
    islands = res.get("islands", [])

    print(f"HDF5: {args.h5} (delt_r={res['delt_r']}, samples={n}, "
          f"full_feasible={full.sum()}, islands={len(islands)})")
    for isl in islands:
        inds = isl["sample_indices"]
        e = eps[inds]
        valid = int(np.isfinite(e).sum())
        best = float(e[np.isfinite(e)].min()) if valid else float("nan")
        b = np.asarray(isl["bounds"])
        print(f"\n[island {isl['island_id']}] members={isl['n_points']}, "
              f"eps-valid={valid}, min_eps={best:.6f}, "
              f"axis_R={isl['mean_axis_R']:.4f}")
        for c, (lo, hi) in enumerate(b):
            print(f"    coil {c}: [{lo:.1f}, {hi:.1f}]")

    if not islands:
        print("\nNo islands to filter; nothing done.")
        return

    choice = input("\nRemove islands (comma/space separated ids, "
                   "e.g. 1 2; Enter = keep all): ").strip()
    ids = {isl["island_id"] for isl in islands}
    remove = []
    for tok in choice.replace(",", " ").split():
        if not tok:
            continue
        try:
            remove.append(int(tok))
        except ValueError:
            print(f"  ignoring non-integer '{tok}'")
    bad = [r for r in remove if r not in ids]
    if bad:
        print(f"Unknown island ids {bad}; aborting (nothing saved).")
        return

    if not remove:
        print("No islands removed; nothing saved.")
        return

    n_full_before = int(full.sum())
    removed_set = set(remove)
    for isl in islands:
        if isl["island_id"] in removed_set:
            inds = isl["sample_indices"]
            full[inds] = False
            eps[inds] = np.nan
    res["full_feasible"] = full
    res["eps"] = eps
    res["islands"] = [isl for isl in islands
                      if isl["island_id"] not in removed_set]
    for rank, isl in enumerate(res["islands"]):
        isl["island_id"] = rank

    out = Path(args.out) if args.out else \
        Path(args.h5).with_name(Path(args.h5).stem + "_filtered.h5")
    save_island_mapping_h5(out, res)
    print(f"\nSaved {out}: removed {len(remove)} island(s), "
          f"{n_full_before - int(full.sum())} samples de-flagged, "
          f"remaining full_feasible={full.sum()}, "
          f"islands={len(res['islands'])}")


if __name__ == "__main__":
    main()
