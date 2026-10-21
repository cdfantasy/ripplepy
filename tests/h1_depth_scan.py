#!/usr/bin/env python3
"""Stage 2: depth map of the real-axis region (screening tier, no eps).

Reads the stage-1 pilot file, keeps the samples whose column actually reached a
few faces (``reach > 0``; an island-centre fixed point does not), and walks each
one from the axis outward to its first failure.  Every face is traced once and
everything (a, V, iota, <B>, residual, gap_norm) comes from that trace.  No eps
is computed here: eps is only meaningful in the inner 80% of a column, and it is
deferred to stage 3 where the reference surface a0 is chosen from this map.

Output: one H5 with the per-face profile and the per-sample LCFS geometry.
Change a0 later and only stage 3 has to be re-run -- nothing here is invalidated.

  nohup python -u tests/h1_depth_scan.py > tests/h1_depth_scan.log 2>&1 &
"""

from __future__ import annotations

import time
from pathlib import Path

import h5py
import numpy as np

from ripplepy.radial_scan import ScanConfig, fmt_duration, scan_box

BASE = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
MGRID_PATH = BASE / "tests" / "test_file" / "mgrid_h1_design.nc"
NFP = 3
FULL_TORUS = False

INPUT_H5 = BASE / "tests" / "h1_radial_scan" / "axis_pilot.h5"
OUTPUT_H5 = BASE / "tests" / "h1_radial_scan" / "depth_map.h5"

# Which stage-1 samples go on to the depth walk.  reach > 0 keeps every column
# that has at least one nested face (reach = 0.02) or two (0.04); reach >= 0.04
# keeps only the fully probed ones (~72% of the real-axis set).
MIN_REACH = 0.0

BOUNDS = np.array([
    [50000.0, 50000.0],     # coil 0: TF, fixed
    [0.0, 10000.0],         # coil 1
    [0.0, 10000.0],         # coil 2
    [-100000.0, 0.0],       # coil 3
    [-100000.0, 0.0],       # coil 4
])

# Radial walk.  nturn=100 is enough for the gates: the residual metric is
# resolution independent and gap_norm is already normalised by nturn.
DR0 = 0.02
DR_STEP = 0.02
DR_MAX = 0.60
NPHI = 360
PROFILE_NTURN = 100
PROFILE_NPART = 200

# Gates (same thresholds as the pilot / the calibrated values).
RESIDUAL_TOL = 0.05
GAP_NORM_TOL = 15.0

# Axis scan + branch ranking (unchanged from stage 1).
RMIN, RMAX, RSTEP = 1.10, 1.35, 0.05
AXIS_NPHI = 180
N_AXIS_TRIES = 3
AXIS_PROBE_DR = 0.04
AXIS_PROBE_NTURN = 40

PROCESSES = 64              # 0 = serial
PROGRESS_INTERVAL = 128


# ---------------------------------------------------------------------------
def load_input(path: Path):
    with h5py.File(path, "r") as f:
        keep = f["reach"][:] > MIN_REACH
        samples = f["samples"][:][keep]
        reach = f["reach"][:][keep]
    return samples, reach


def main():
    cfg = ScanConfig(
        mgrid_path=str(MGRID_PATH), nfp=NFP, full_torus=FULL_TORUS,
        rmin=RMIN, rmax=RMAX, rstep=RSTEP, axis_nphi=AXIS_NPHI,
        n_axis_tries=N_AXIS_TRIES,
        axis_probe_dr=AXIS_PROBE_DR, axis_probe_nturn=AXIS_PROBE_NTURN,
        nphi=NPHI,
        profile_nturn=PROFILE_NTURN, profile_npart=PROFILE_NPART,
        dr0=DR0, dr_step=DR_STEP, dr_max=DR_MAX,
        residual_tol=RESIDUAL_TOL, gap_norm_tol=GAP_NORM_TOL,
    )
    width = 22
    print("=" * width + " stage-2 depth map " + "=" * width)
    print(f"  input    {INPUT_H5}  (reach > {MIN_REACH})")
    print(f"  tier     screening, nturn={PROFILE_NTURN}, nphi={NPHI}, no eps")
    print(f"  walk     dr {DR0}..{DR_MAX} step {DR_STEP}, one bisection")
    print(f"  gates    residual <= {RESIDUAL_TOL}, gap_norm <= {GAP_NORM_TOL}")
    print("=" * (2 * width + 22))

    samples, reach = load_input(Path(INPUT_H5))
    print(f"  {len(samples)} columns to walk"
          f"  (reach 0.02: {int(np.sum(reach <= 0.02 + 1e-9))},"
          f" 0.04: {int(np.sum(reach > 0.02 + 1e-9))})", flush=True)
    if len(samples) == 0:
        print("  nothing to do")
        return

    t0 = time.perf_counter()
    scan_box(cfg, BOUNDS, samples=samples, processes=PROCESSES,
             output_h5=Path(OUTPUT_H5), progress_interval=PROGRESS_INTERVAL,
             eps_at_a=None)
    print(f"  done in {fmt_duration(time.perf_counter() - t0)}")


if __name__ == "__main__":
    main()
