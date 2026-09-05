#!/usr/bin/env python3
"""Simple smoke test for the ripplepy H1 pipeline.

Loads the H1 mgrid, finds the magnetic axis, computes eps_eff^(3/2) on a
surface offset from the axis, and (optionally) plots the traced field line.

Fails loudly with a clear reason instead of crashing on a None result.

Run:  python tests/simple_test.py
"""

import sys
import time
from pathlib import Path

import numpy as np

from ripplepy import (
    compute_initial_gradpsi_nemov,
    compute_epstot,
    find_axis,
    initialize_mgrid_field,
    plot_fieldline_3d,
    set_extcur,
    set_trace_parameters,
    calculate_plasma_params,
)

# ---------------------------------------------------------------------------
# Configuration (H1)
# ---------------------------------------------------------------------------
DEVICE = "H1"
BASE = Path(__file__).resolve().parent.parent
MGRID_PATH = str(BASE / "tests" / "test_file" / "mgrid_h1_design.nc")
NFP = 3
FULL_TORUS = False
EXTCUR = [50000.0, 5000.0, 0, -80000.0, -40000.0]
INITIAL_RZ = (1.25, 0.0)
# EXTCUR = [50000.0, 1027.7, 933.0, -64940.2, -15620.6]
# INITIAL_RZ = (1.0986, 0.0)
DELTA_R = 0.1        # radial offset of the traced surface from the axis
NTURN = 400
NPHI = 360
NPART = 5000
PLOT = True              # set False on a headless machine (needs plotly)
AXIS_Z_TOL =1e-4        # |Z_axis| tolerance (stellarator-symmetry check)


def main():
    print("=" * 60)
    print(f"ripplepy simple test — {DEVICE}")
    print("=" * 60)

    print("\n[1] Loading mgrid + initialising field ...")
    initialize_mgrid_field(MGRID_PATH, NFP, full_torus=FULL_TORUS)
    set_extcur(EXTCUR)

    print("\n[2] Searching for the magnetic axis ...")
    axis_rz, R0, axis_fieldline, ok = find_axis(
        INITIAL_RZ, xtol=1e-8, max_iter=200,delta_r=0.01, verbose=True)
    if not ok or axis_rz is None:
        print(f"  ✗ Magnetic axis not found for extcur={EXTCUR}. "
              "Check the coil currents / mgrid file.")
        sys.exit(1)
    # Own copy — never mutate find_axis' result array in place.
    axis_rz = np.asarray(axis_rz, dtype=np.float64)
    print(f"  ✓ Axis: R={axis_rz[0]:.4f}, Z={axis_rz[1]:.4f}, R0={R0:.4f}")
    if abs(axis_rz[1]) > AXIS_Z_TOL:
        print(f"  ⚠ |Z_axis| = {abs(axis_rz[1]):.4f} > tol = {AXIS_Z_TOL}: "
              "configuration is off the symmetry plane — result may be unreliable")

    # Field-line start point: copy + offset.
    start_rz = np.array([axis_rz[0] + DELTA_R, axis_rz[1]], dtype=np.float64)
    initial_gradpsi = compute_initial_gradpsi_nemov(
        EXTCUR, start_rz[0], start_rz[1], verbose=False)

    print(f"\n[3] Tracing field line + computing eps_eff^(3/2) "
          f"(nturn={NTURN}, nphi={NPHI}, npart={NPART}) ...")
    set_trace_parameters(NTURN, NPHI, npart=NPART, verbose=False)
    t0 = time.time()
    eps, bnd, fieldline_data, istate = compute_epstot(
        start_rz, initial_gradpsi=initial_gradpsi,
        return_fieldline=True, verbose=False,
    )
    elapsed = time.time() - t0
    if istate != 0 or eps is None:
        print(f"  ✗ Field-line tracing / eps_eff failed (istate={istate}). "
              "Try a different DELTA_R or extcur.")
        sys.exit(1)
    print(f"  ✓ eps_eff^(3/2) = {eps:.6e}  (time: {elapsed:.2f}s)")
    print(f"  start_rz = ({start_rz[0]:.4f}, {start_rz[1]:.4f})")

    vol,am,iota = calculate_plasma_params(fieldline_data,axis_fieldline,NTURN,NPHI,R0)
    # am = sqrt(V / (2 pi^2 R0)) is the volume-equivalent MINOR radius; R0
    # (the mean axis radius) is the major radius.
    print(f"Volume  of plasma: {vol:.3e},minor radius = {am:.3e},iota = {iota:.3e}")

    if PLOT:
        print("\n[4] Plotting the field line in 3D ...")
        try:
            fig = plot_fieldline_3d(fieldline_data, color_by_b=True)
            # Headless-safe: save an interactive HTML (fig.show() opens a
            # browser and can hang on a server without a display).
            fig.show()
        except ImportError:
            print("  (plotly not installed — install with 'pip install plotly')")
        except Exception as exc:
            print(f"  (3D plot failed: {exc})")

    print("\nDone.")


if __name__ == "__main__":
    main()
