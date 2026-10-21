#!/usr/bin/env python3
"""One-pass current scan for H1.

Everything happens in a single loop over the samples:

    Sobol sample -> find the magnetic axis (rank every candidate branch)
                 -> walk outward to the LCFS
                 -> eps on EVERY surface that passes the gates

The per-sample axis position is stored with the results, so a later re-walk
(different eps resolution, radial step or gates) can skip the axis search,
which dominates the total cost.

The run is chunked: each chunk of CHUNK_SIZE samples is written to its own file
as soon as it finishes, and existing chunks are skipped when the script is
relaunched, so an interrupted job resumes where it stopped.  At the end the
chunks are merged into a single HDF5 file.

  nohup python -u tests/h1_full_scan.py > tests/h1_full_scan.log 2>&1 &
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ripplepy.radial_scan import (ScanConfig, merge_chunks, merge_dual_chunks,
                                  scan_all_box)

BASE = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
MGRID_PATH = BASE / "tests" / "test_file" / "mgrid_h1_design.nc"
NFP = 3
FULL_TORUS = False

BOUNDS = np.array([
    [50000.0, 50000.0],   # coil 0: TF, fixed at 50 kA
    [0.0, 10000.0],       # coil 1
    [0.0, 10000.0],       # coil 2
    [-100000.0, 0.0],     # coil 3
    [-100000.0, 0.0],     # coil 4
])

N_SAMPLES = 65536
SEED = 11
SAMPLES = None            # optional explicit (N, 5) array of absolute currents

# axis search
RMIN, RMAX, RSTEP = 1.10, 1.35, 0.05
AXIS_NPHI = 180
N_AXIS_TRIES = 3

# radial walk (eps is measured on every walked surface at this tier; the
# official numbers come from h1_refine_measure.py at nturn=400)
WALK_EPS = True
PROFILE_NTURN = 200       # walk-tier trace resolution
PROFILE_NPART = 500       # eps quadrature at the walk tier
DR0, DR_STEP, DR_MAX = 0.02, 0.02, 0.60

# gates
RESIDUAL_TOL = 0.05
GAP_NORM_TOL = 15.0

# dual-axis side product
RECORD_DUAL_AXIS = True

PROCESSES = 64
CHUNK_SIZE = 1000

OUT_DIR = BASE / "tests" / "h1_full_scan"
CHUNK_DIR = OUT_DIR / "chunks"
OUT_H5 = OUT_DIR / "scan_all.h5"
DUAL_H5 = OUT_DIR / "dual_axis.h5"


# ---------------------------------------------------------------------------
def main():
    cfg = ScanConfig(
        mgrid_path=str(MGRID_PATH), nfp=NFP, full_torus=FULL_TORUS,
        rmin=RMIN, rmax=RMAX, rstep=RSTEP, axis_nphi=AXIS_NPHI,
        n_axis_tries=N_AXIS_TRIES, axis_record_branches=RECORD_DUAL_AXIS,
        walk_eps=WALK_EPS, profile_nturn=PROFILE_NTURN,
        profile_npart=PROFILE_NPART, dr0=DR0, dr_step=DR_STEP,
        dr_max=DR_MAX, residual_tol=RESIDUAL_TOL,
        gap_norm_tol=GAP_NORM_TOL,
    )
    width = 26
    print("=" * width + " one-pass H1 scan " + "=" * width)
    print(f"  mgrid    {MGRID_PATH}")
    print(f"  box      " + ", ".join(f"[{lo:.0f},{hi:.0f}]" for lo, hi in BOUNDS))
    print(f"  samples  Sobol {N_SAMPLES}, seed {SEED}")
    print(f"  axis     R {RMIN}..{RMAX} step {RSTEP}, up to {N_AXIS_TRIES}"
          f" branches, dual-axis {'recorded' if RECORD_DUAL_AXIS else 'off'}")
    print(f"  walk     dr {DR0}..{DR_MAX} step {DR_STEP};"
          f" eps on every surface = {WALK_EPS}"
          f" (nturn={PROFILE_NTURN}, npart={PROFILE_NPART})")
    print(f"  output   {OUT_DIR}")
    print("=" * (2 * width + 24), flush=True)

    t0 = time.perf_counter()
    scan_all_box(cfg, BOUNDS, n_samples=N_SAMPLES, seed=SEED,
                 samples=SAMPLES, processes=PROCESSES, out_dir=CHUNK_DIR,
                 chunk_size=CHUNK_SIZE,
                 dual_axis_h5=DUAL_H5 if RECORD_DUAL_AXIS else None)
    print(f"  chunks done in {time.perf_counter() - t0:.0f}s", flush=True)

    merge_chunks(CHUNK_DIR, OUT_H5)
    if RECORD_DUAL_AXIS:
        merge_dual_chunks(CHUNK_DIR, DUAL_H5)
    print(f"  total {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
