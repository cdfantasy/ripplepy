#!/usr/bin/env python3
"""Stage 1: map the region of the current box that has a REAL magnetic axis.

"Real" matters: a one-turn fixed point also exists at the centre of an island
chain, so axis existence alone over-counts (98.6% of a previous full-box survey
had one).  The probe here walks each candidate branch out to AXIS_PROBE_DR at
the nturn=40 tier and keeps the branch that survives furthest, so an
island-centre fixed point (which breaks within its island width) is rejected.

Cost is ~0.8 s/sample, about 5x cheaper than the screening walk, so this can be
run over the whole box (20k-65k samples) before any depth work.

  nohup python -u tests/h1_axis_pilot.py > tests/h1_axis_pilot.log 2>&1 &
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ripplepy.radial_scan import ScanConfig, probe_box

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

N_SAMPLES = 4096          # pilot (power of 2 keeps the Sobol balanced);
                          # 65k costs ~80 min on 64 cores
SEED = 11
SAMPLES = None            # optional explicit (N, 5) array of absolute currents

# Axis scan window (R of the one-turn fixed point).
RMIN, RMAX, RSTEP = 1.10, 1.35, 0.05
AXIS_NPHI = 180

# Stage-1 "real axis" test: walk each candidate branch out to this dr at the
# cheap tier. 0.04 = two faces (0.02, 0.04); raise it if island centres still
# get through.
AXIS_PROBE_DR = 0.04
AXIS_PROBE_NTURN = 40
N_AXIS_TRIES = 3

PROCESSES = 64            # 0 = serial
PROGRESS_INTERVAL = 200
OUTPUT_H5 = BASE / "tests" / "h1_radial_scan" / "axis_pilot.h5"


# ---------------------------------------------------------------------------
def main():
    cfg = ScanConfig(
        mgrid_path=str(MGRID_PATH), nfp=NFP, full_torus=FULL_TORUS,
        rmin=RMIN, rmax=RMAX, rstep=RSTEP, axis_nphi=AXIS_NPHI,
        n_axis_tries=N_AXIS_TRIES,
        axis_probe_dr=AXIS_PROBE_DR, axis_probe_nturn=AXIS_PROBE_NTURN,
    )
    width = 22
    print("=" * width + " stage-1 axis pilot " + "=" * width)
    print(f"  mgrid    {MGRID_PATH}")
    print(f"  box      " + ", ".join(f"[{lo:.0f},{hi:.0f}]" for lo, hi in BOUNDS))
    print(f"  sampling " + (f"Sobol {N_SAMPLES}, seed {SEED}"
                             if SAMPLES is None else
                             f"explicit array, {len(SAMPLES)} points"))
    print(f"  axis     R {RMIN}..{RMAX} step {RSTEP}, nphi {AXIS_NPHI},"
          f" up to {N_AXIS_TRIES} branches per sample")
    print(f"  real-axis test: column must reach dr={AXIS_PROBE_DR}"
          f" at nturn={AXIS_PROBE_NTURN}")
    print("=" * (2 * width + 20))

    t0 = time.perf_counter()
    probe_box(cfg, BOUNDS, n_samples=N_SAMPLES, seed=SEED,
              processes=PROCESSES, samples=SAMPLES,
              output_h5=Path(OUTPUT_H5),
              progress_interval=PROGRESS_INTERVAL)
    print(f"  done in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
