#!/usr/bin/env python3
"""H1 radial scan: sample the coil box and measure every surface from the axis
out to the LCFS (no `delt_r` reference surface any more).

New pipeline (per sample):
  axis  ->  dr = 0.02, 0.04, ... outward (one trace per face, eps and every
  1D quantity measured on that same trace)  ->  one bisection  ->  full-tier
  run on the true LCFS face.

Everything that used to be a post-processing knob (delt_r, ladder step, gap
gate in radians) is now a property of the scan itself, and the radial axis of
every recorded profile is the volume-equivalent minor radius `a`, not `dr`.

  nohup python -u tests/h1_radial_scan.py > tests/h1_radial_scan.log 2>&1 &
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ripplepy.radial_scan import ScanConfig, scan_box

BASE = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
MGRID_PATH = BASE / "tests" / "test_file" / "mgrid_h1_design.nc"
NFP = 3
FULL_TORUS = False

# Coil-current box (absolute Amperes).  Coil 0 is the TF coil: fixed.
BOUNDS = np.array([
    [50000.0, 50000.0],   # coil 0: TF, fixed at 50 kA
    [0.0, 10000.0],       # coil 1
    [0.0, 10000.0],       # coil 2
    [-100000.0, 0.0],     # coil 3
    [-100000.0, 0.0],     # coil 4
])

# Sampling: Sobol inside BOUNDS.  Set SAMPLES to an (N, 5) array of absolute
# currents to scan an explicit list instead (e.g. a local densification).
N_SAMPLES = 65536
SEED = 7
SAMPLES = None

# Axis scan window (R of the one-turn fixed point) and resolution.
RMIN, RMAX, RSTEP = 1.10, 1.35, 0.05
AXIS_NPHI = 180

# Radial walk.  dr is only the construction step of the sweep; the recorded
# radial coordinate of every profile row is the minor radius `a` of that face.
DR0 = 0.02
DR_STEP = 0.02
DR_MAX = 0.60
# The single full-tier eps measurement happens on the face whose minor radius
# is closest to EPS_AT_A (metres).  None = screening only (no eps anywhere).
EPS_AT_A = 0.14

# Resolutions.  nphi is full for both tiers: the ODE cost does not depend on
# nphi (measured), only the eps quadrature does.
NPHI = 360
PROFILE_NTURN = 150     # per-face tier: locates the LCFS, gives the profile
PROFILE_NPART = 500
FULL_NTURN = 400        # LCFS tier: the one number used for ranking
FULL_NPART = 2000

# Gates, evaluated on the same trace as eps.
# residual: resolution independent (0.0445 -> 0.0450 over 5x nturn).
# gap_norm: gap_rad * nturn / (2pi); resolution free (healthy 3-6, bad ~55).
RESIDUAL_TOL = 0.05
GAP_NORM_TOL = 15.0

# Execution
PROCESSES = 64          # 0 = serial
PROGRESS_INTERVAL = 64
OUTPUT_H5 = BASE / "tests" / "h1_radial_scan" / "scan.h5"


# ---------------------------------------------------------------------------
def main():
    cfg = ScanConfig(
        mgrid_path=str(MGRID_PATH), nfp=NFP, full_torus=FULL_TORUS,
        rmin=RMIN, rmax=RMAX, rstep=RSTEP, axis_nphi=AXIS_NPHI,
        nphi=NPHI,
        profile_nturn=PROFILE_NTURN, profile_npart=PROFILE_NPART,
        full_nturn=FULL_NTURN, full_npart=FULL_NPART,
        dr0=DR0, dr_step=DR_STEP, dr_max=DR_MAX,
        residual_tol=RESIDUAL_TOL, gap_norm_tol=GAP_NORM_TOL,
    )
    width = 22
    print("=" * width + " radial scan " + "=" * width)
    print(f"  mgrid      {MGRID_PATH}")
    print(f"  box        " + ", ".join(f"[{lo:.0f},{hi:.0f}]"
                                     for lo, hi in BOUNDS))
    if SAMPLES is None:
        print(f"  sampling   Sobol {N_SAMPLES} points, seed {SEED}")
    else:
        print(f"  sampling   explicit array, {len(SAMPLES)} points")
    print(f"  tiers      profile nturn={PROFILE_NTURN} npart={PROFILE_NPART}"
          f" | full nturn={FULL_NTURN} npart={FULL_NPART} | nphi={NPHI}")
    print(f"  walk       dr {DR0}..{DR_MAX} step {DR_STEP},"
          f" one bisection")
    print(f"  gates      residual <= {RESIDUAL_TOL},"
          f" gap_norm <= {GAP_NORM_TOL}")
    print("=" * (2 * width + 14))

    t0 = time.perf_counter()
    scan_box(cfg, BOUNDS, n_samples=N_SAMPLES, seed=SEED, eps_at_a=EPS_AT_A,
             processes=PROCESSES, samples=SAMPLES,
             output_h5=Path(OUTPUT_H5),
             progress_interval=PROGRESS_INTERVAL)
    print(f"  done in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
