#!/usr/bin/env python3
"""Calibrate the radial-ladder rung test for the H1 mapping redesign.

Question: how many turns per rung are needed to reliably tell a good
magnetic surface from a broken/chaotic one, using the phi=0 Poincare FFT
smoothness test?

Config used: the basin-B candidate [50000, 1027.7, 933, -64940.2, -15620.6],
where the region R in [1.05, 1.1] is chaotic and real surfaces exist near
~1.125 and ~1.2.

Method
------
For every distinct magnetic axis found by a multi-start scan, trace field
lines at radial offsets dr = 0.02 .. 0.12 (one 160-turn RZ trace each) and
evaluate the FFT smoothness on prefixes of the SAME trace at nturn =
20/40/80/160 (a phi=0 section after nt turns is exactly the nt-turn trace,
so no extra integration is needed).  Also report a cheap iota estimate per
rung and a "closes after 1 toroidal turn" flag for each axis (iota=1
rational-axis / island-O-point indicator).

Run:  python tests/diag_rung_calibration.py [extcur_csv]
"""

import sys
from pathlib import Path

import numpy as np

from ripplepy import (
    find_axis,
    fieldline_smoothness_poincare,
    initialize_mgrid_field,
    set_extcur,
    trace_fieldline,
)

BASE = Path(__file__).resolve().parent.parent
MGRID_PATH = str(BASE / "tests" / "test_file" / "mgrid_h1_design.nc")
NFP = 3
FULL_TORUS = False

if len(sys.argv) > 1:
    EXTCUR = [float(x) for x in sys.argv[1].split(",")]
else:
    EXTCUR = [50000.0, 1027.7, 933.0, -64940.2, -15620.6]

DR_LIST = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12]   # ladder rungs
MAX_NTURN = 160                                  # single trace per rung, sliced
NTURN_PROBES = [20, 40, 80, 160]                 # prefixes evaluated
NPHI = 360
AXIS_STARTS = [1.04, 1.06, 1.08, 1.10, 1.125, 1.15, 1.20, 1.25]
# Same tolerances as the mapping pipeline.
RESIDUAL_TOL = 0.05
GAP_TOL = 1.0
MIN_POINTS = 16
TARGET_DR = 0.10      # the mapping's target delt_r


def find_distinct_axes():
    """Multi-start axis scan; return distinct converged (R, Z) tuples."""
    axes = []
    for r0 in AXIS_STARTS:
        try:
            ax, _, _, ok = find_axis(
                np.array([r0, 0.0]), xtol=1e-6, max_iter=100, delta_r=0.01,
                nphi=180, verbose=False)
        except Exception:
            ax, ok = None, False
        if ok and ax is not None and abs(ax[1]) <= 1e-4:
            if not any(np.hypot(ax[0] - a[0], ax[1] - a[1]) < 1e-3
                       for a in axes):
                axes.append((float(ax[0]), float(ax[1])))
                print(f"  start R={r0:.3f} -> axis R={ax[0]:.5f} Z={ax[1]:+.2e}")
            else:
                print(f"  start R={r0:.3f} -> axis R={ax[0]:.5f} (dup)")
        else:
            print(f"  start R={r0:.3f} -> none")
    return sorted(axes)


def axis_one_turn_closure(axis_rz):
    """Trace 2 turns from the axis; report if it closes after exactly 1 turn.

    A 1-turn-closed axis line means the rotational transform on axis is an
    integer (iota=1 is the classic rational/fake-axis case).
    """
    fld, ist = trace_fieldline(initial_rz=np.asarray(axis_rz, order="F"),
                               nturn=2, nphi=NPHI, verbose=False)
    if ist != 0 or fld is None:
        return np.nan
    d = float(np.hypot(fld[NPHI, 0] - fld[0, 0], fld[NPHI, 1] - fld[0, 1]))
    return d


def iota_from_section(fld, axis_rz, nt):
    """Cheap iota estimate: unwrap the poloidal angle of the phi=0 section."""
    n = int(nt) * int(NPHI)
    idx = np.arange(0, n, NPHI)
    R, Z = fld[idx, 0], fld[idx, 1]
    th = np.unwrap(np.arctan2(Z - axis_rz[1], R - axis_rz[0]))
    if len(th) < 3:
        return np.nan
    return float((th[-1] - th[0]) / (2.0 * np.pi * (len(th) - 1)))


def rung_table(axis_rz, label):
    print(f"\n=== ladder from axis R={axis_rz[0]:.5f} ({label}) ===")
    closure = axis_one_turn_closure(axis_rz)
    flag = ("1-TURN CLOSED (iota integer?)" if closure < 1e-4
            else "open orbit")
    print(f"  axis 1-turn closure dist = {closure:.2e}  ->  {flag}")
    hdr = ("dr     | " + " | ".join(
        f"{'nt=' + str(nt):<22}" for nt in NTURN_PROBES)
        + " | iota@160")
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for dr in DR_LIST:
        start = np.array([axis_rz[0] + dr, axis_rz[1]], dtype=np.float64,
                         order="F")
        fld, ist = trace_fieldline(initial_rz=start, nturn=MAX_NTURN,
                                   nphi=NPHI, verbose=False)
        cells = []
        if ist != 0 or fld is None:
            cells = [f"ist={ist}" for _ in NTURN_PROBES]
        else:
            for nt in NTURN_PROBES:
                smooth, m = fieldline_smoothness_poincare(
                    fld, nt, NPHI, axis_rz=axis_rz,
                    n_harmonics=4,
                    residual_rms_frac_tol=RESIDUAL_TOL,
                    max_angular_gap=GAP_TOL,
                    min_points=MIN_POINTS)
                res = m.get("residual_rms_frac", np.nan)
                gap = m.get("max_gap_rad", np.nan)
                if np.isnan(res):
                    cells.append(f"PASS(n={m.get('n_points', 0)})" if smooth
                                 else f"FAIL(np={m.get('n_points', 0)})")
                else:
                    tag = "PASS" if smooth else "FAIL"
                    cells.append(f"{tag} res={res:.3f} gap={gap:.2f}")
        iota = iota_from_section(fld, axis_rz, MAX_NTURN) if ist == 0 and \
            fld is not None else np.nan
        print(f"{dr:5.2f}  | " + " | ".join(f"{c:<22}" for c in cells)
              + f" | {iota:+.4f}")
        rows.append((dr, ist))
    return rows


def main():
    print(f"extcur = {EXTCUR}")
    print("initialising mgrid ...")
    initialize_mgrid_field(MGRID_PATH, NFP, full_torus=FULL_TORUS)
    set_extcur(EXTCUR)

    print("\n[1] multi-start axis scan ...")
    axes = find_distinct_axes()
    print(f"  distinct axes: {axes}")
    if not axes:
        print("  no axis found - abort")
        sys.exit(1)

    print("\n[2] radial ladder from each axis ...")
    for i, ax in enumerate(axes):
        rows = rung_table(ax, f"axis#{i}")
        ok80 = [dr for dr, ist in rows
                if ist == 0 and dr <= TARGET_DR]
        print(f"  -> candidate verdict: clean rungs(ist) <= {TARGET_DR}: "
              f"{ok80}")

    print("\nDone.")


if __name__ == "__main__":
    main()
