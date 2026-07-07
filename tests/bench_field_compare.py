#!/usr/bin/env python3
"""Compare pyneo (Boozer) vs ripplepy (mgrid) |B| profiles and well structure.

For a single flux surface on CFQS, extract |B| along each field line and
run identical well-detection + statistics in pure Python to isolate whether
the ε_eff discrepancy comes from |B| differences or from elsewhere.

Usage:
    python tests/bench_field_compare.py
"""

import numpy as np
from pathlib import Path
from simsopt.mhd import Boozer, Vmec
from simsopt.geo import SurfaceRZFourier
import neo
from neo import NeoContext, neo_surfaces_from_simsopt_boozer
from ripplepy import (
    set_extcur, initialize_mgrid_field, set_trace_parameters,
    trace_fieldline, find_axis,
)
from ripplepy.boozer_eps_verify import (
    sample_fieldline_from_boozer, _boozer_obj_to_dict,
)

# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

# BASE = str(Path(__file__).resolve().parent.parent)
# DEVICE = "CFQS"
# VMEC_PATH = f"{BASE}/tests/test_file/wout_cfqs_test_m10_n5_fixed.nc"
# MGRID_PATH = f"{BASE}/tests/test_file/mgrid_2b40R1mB01.nc"
# extcur = None
# INITIAL_RZ = (1.21, 0.0)
# NFP = 2
# FULL_TORUS = False

BASE = str(Path(__file__).resolve().parent.parent)
DEVICE = "H1"
VMEC_PATH = f"{BASE}/tests/test_file/wout_h1_design.nc"
MGRID_PATH = f"{BASE}/tests/test_file/mgrid_h1_design.nc"
extcur = [50000, 5000, 2000, -80000, -40000]
INITIAL_RZ = (1.26, 0.0)
NFP = 3
FULL_TORUS = False

# Surface to analyze (s-value)
SURF_S = 0.5
SURF_IDX_LIST = np.linspace(0.1, 1.0, 11)

# Tracing parameters (must match bench_pyneo_style.py)
NTURN = 400
NPHI = 360
NPART = 5000


# ═══════════════════════════════════════════════════════════════
# Well detection
# ═══════════════════════════════════════════════════════════════

def find_local_minima(B: np.ndarray) -> np.ndarray:
    """Return indices of local minima in 1D array B (periodic boundary)."""
    n = len(B)
    minima = []
    for i in range(n):
        if B[i] < B[(i - 1) % n] and B[i] < B[(i + 1) % n]:
            minima.append(i)
    if not minima:
        return np.array([0, n], dtype=np.int32)
    idx = np.array(minima, dtype=np.int32)
    # Append wrap-around closure
    return np.concatenate([idx, [idx[0] + n]])


def well_statistics(B: np.ndarray) -> dict:
    """Extract well structure from |B| array.

    Returns
    -------
    dict with keys:
        n_wells : int
        depths  : ndarray  — ΔB/B_ref for each well
        b_mins  : ndarray  — B at each well bottom
        b_maxs  : ndarray  — B at each well top
        b_range : tuple    — (B_min, B_max) global
    """
    minima = find_local_minima(B)
    n_wells = len(minima) - 1
    b_ref = np.max(B)

    depths = np.zeros(n_wells)
    b_mins = np.zeros(n_wells)
    b_maxs = np.zeros(n_wells)

    for k in range(n_wells):
        i1 = minima[k]
        i2_raw = minima[k + 1]
        # Segment between consecutive minima
        if i2_raw < len(B):
            seg = B[i1:i2_raw + 1]
        else:
            # Wrap-around segment
            seg = np.concatenate([B[i1:], B[:i2_raw - len(B) + 1]])
        b_mins[k] = np.min(seg)
        b_maxs[k] = np.max(seg)
        depths[k] = (b_maxs[k] - b_mins[k]) / b_ref

    return {
        "n_wells": n_wells,
        "depths": depths,
        "b_mins": b_mins,
        "b_maxs": b_maxs,
        "b_range": (np.min(B), np.max(B)),
    }


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    print(f"\n{'='*60}")
    print(f"  {DEVICE} — Field Comparison (s={SURF_S})")
    print(f"{'='*60}")

    # ── Load VMEC + Boozer ──
    print("\n[1] Loading VMEC + Boozer …")
    vmec = Vmec(str(VMEC_PATH))
    R0_vmec = float(vmec.wout.Rmajor_p)

    boozer = Boozer(vmec)
    boozer.mpol = 72
    boozer.ntor = 36
    boozer.register(SURF_IDX_LIST)
    boozer.run()
    print("  Computed Boozer from VMEC.")

    booz_dict = _boozer_obj_to_dict(boozer)

    # Find surface index closest to SURF_S
    sur_idx_arr = np.asarray(SURF_IDX_LIST)
    k_surf = np.argmin(np.abs(sur_idx_arr - SURF_S))

    # ── Run pyneo full pipeline ──
    print("\n[2] Running pyneo full pipeline …")
    neoclass = neo.from_simsopt_boozer(boozer)
    ctx = NeoContext()
    ctx.set_boozer(neoclass)
    ctx.set_flux_surfaces(neo_surfaces_from_simsopt_boozer(boozer).tolist())
    ctx.set_resolution(theta_n=100, phi_n=100)
    ctx.set_transport_options(
        npart=NPART, multra=1, acc_req=0.01, no_bins=100,
        nstep_per=50, nstep_min=500, nstep_max=5000, calc_nstep_max=0,
    )
    ctx.set_switches(ref_swi=2, eout_swi=2, calc_cur=0)
    ctx.set_output_options(
        write_progress=0, write_output_files=0,
        write_integrate=0, write_diagnostic=0, suppress_file_io=True,
    )
    ctx.setup_grids()
    ctx.run_all()
    py_eps_profile = ctx.epstot_profile()
    print(f"  pyneo ε_eff^(3/2) = {py_eps_profile[k_surf]:.6e}")

    # ── Compute common starting point at φ=0 ──
    print("\n[3] Computing φ=0 starting point …")
    surf = SurfaceRZFourier.from_wout(str(VMEC_PATH), SURF_S)
    rpz = surf.cross_section(phi=0.0)[0]
    rz_start = rpz[[0, 2]]   # (R_target, Z_target) at φ=0
    R_target, Z_target = float(rz_start[0]), float(rz_start[1])

    # Invert Boozer transform: at φ=0, Z=0 ⇒ θ ∈ {0, π} (stellarator symmetry)
    # R(θ,0) = Σ rmnc[m]·cos(xm[m]·θ), pick the θ that matches R_target
    xm = booz_dict["ixm_b"].astype(np.int32)
    rmnc = booz_dict["rmnc_b"][k_surf, :].astype(np.float64)
    R0_val = float(np.dot(rmnc, np.cos(xm.astype(np.float64) * 0.0)))   # θ=0
    Rpi_val = float(np.dot(rmnc, np.cos(xm.astype(np.float64) * np.pi))) # θ=π
    theta0 = 0.0 if abs(R0_val - R_target) < abs(Rpi_val - R_target) else np.pi
    print(f"  φ=0 start: R={R_target:.4f}, Z={Z_target:.4f} "
          f"→ R(0)={R0_val:.4f}, R(π)={Rpi_val:.4f} → θ₀={theta0:.4f} rad")

    # ── Get Boozer |B| along analytic field line ──
    print("\n[4] Sampling Boozer field line …")
    fl = sample_fieldline_from_boozer(
        booz_dict, k_surf,
        theta0=theta0, nzeta=NPHI, nturn=NTURN,
    )
    B_booz = fl.B.copy()
    npts_booz = len(B_booz)
    print(f"  Boozer field line: {npts_booz} points")

    # ── Run ripplepy full pipeline ──
    print("\n[5] Running ripplepy full pipeline …")
    initialize_mgrid_field(MGRID_PATH, NFP, full_torus=FULL_TORUS)
    set_extcur(extcur) 

    # Find axis
    axis_rz, R0_rp, axis_fl, ok = find_axis(INITIAL_RZ, xtol=1e-5, max_iter=100)
    print(f"  Axis: R={axis_rz[0]:.4f}, Z={axis_rz[1]:.4f}, R0={R0_rp:.4f}")
    print(f"  R0: VMEC={R0_vmec:.4f}, ripplepy={R0_rp:.4f}")

    set_trace_parameters(NTURN, NPHI, npart=NPART, verbose=False)

    # Run ripplepy ε_eff for all surfaces (like bench_pyneo_style.py)
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from ripplepy import compute_epstot

    RZ_points = []
    for s_val in SURF_IDX_LIST:
        srf = SurfaceRZFourier.from_wout(str(VMEC_PATH), s_val)
        rpz_s = srf.cross_section(phi=0.0)[0]
        RZ_points.append(rpz_s[[0, 2]])
    RZ_points = np.asarray(RZ_points)

    def _compute_one(rz):
        eps, bnd, ist = compute_epstot(
            rz,
            initial_gradpsi=np.array([1, 0, 0], dtype=np.float64),
            verbose=False,
        )
        return eps if eps is not None else np.nan

    rp_eps_profile = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(_compute_one, rz): i for i, rz in enumerate(RZ_points)}
        results = [np.nan] * len(RZ_points)
        for fut in as_completed(futures):
            i = futures[fut]
            results[i] = fut.result()
    rp_eps_profile = results
    print(f"  ripplepy ε_eff^(3/2) = {rp_eps_profile[k_surf]:.6e}")

    # ── Get ripplepy |B| along traced field line ──
    print(f"\n[6] Tracing ripplepy field line (R={rz_start[0]:.4f}, Z={rz_start[1]:.4f}) …")
    fld, ist = trace_fieldline(
        initial_rz=rz_start,
        nturn=NTURN, nphi=NPHI, verbose=False,
    )
    if ist != 0:
        print(f"  ERROR: trace_fieldline failed with istate={ist}")
        return
    B_rp = fld[:, 6].copy()  # column 7 (1-based) = |B|
    npts_rp = len(B_rp)
    print(f"  ripplepy field line: {npts_rp} points")

    # ── Well statistics ──
    print("\n[7] Well detection & statistics …")
    wells_booz = well_statistics(B_booz)
    wells_rp = well_statistics(B_rp)

    # ── Output ──
    bmin_b, bmax_b = wells_booz["b_range"]
    bmin_r, bmax_r = wells_rp["b_range"]
    db_booz = bmax_b - bmin_b
    db_rp = bmax_r - bmin_r

    # Align lengths for direct comparison (take min)
    n_compare = min(npts_booz, npts_rp)
    B_b_cmp = B_booz[:n_compare]
    B_r_cmp = B_rp[:n_compare]
    rms_diff = np.sqrt(np.mean((B_b_cmp - B_r_cmp) ** 2))
    corr = np.corrcoef(B_b_cmp, B_r_cmp)[0, 1]
    mean_ratio = np.mean(B_r_cmp / B_b_cmp)
    max_abs_diff = np.max(np.abs(B_b_cmp - B_r_cmp))

    print(f"\n  {'─'*60}")
    print(f"  |B| comparison")
    print(f"  {'─'*60}")
    print(f"  Boozer:  [{bmin_b:.6f}, {bmax_b:.6f}]  "
          f"ΔB = {db_booz:.6f} T  ({100*db_booz/(0.5*(bmin_b+bmax_b)):.2f}%)")
    print(f"  mgrid:   [{bmin_r:.6f}, {bmax_r:.6f}]  "
          f"ΔB = {db_rp:.6f} T  ({100*db_rp/(0.5*(bmin_r+bmax_r)):.2f}%)")
    print(f"  ΔB ratio (mgrid/booz): {db_rp/db_booz:.6f}")
    print(f"  mean(B_m/B_b): {mean_ratio:.6f}")
    print(f"  correlation:   {corr:.6f}")
    print(f"  RMS diff:      {rms_diff:.6f} T  ({100*rms_diff/np.mean(B_b_cmp):.4f}%)")
    print(f"  max |ΔB|:      {max_abs_diff:.6f} T  ({100*max_abs_diff/np.max(B_b_cmp):.4f}%)")

    print(f"\n  {'─'*60}")
    print(f"  Well structure")
    print(f"  {'─'*60}")
    print(f"  Boozer wells: {wells_booz['n_wells']}")
    print(f"  mgrid  wells: {wells_rp['n_wells']}")
    print(f"  wells ratio:  {wells_rp['n_wells']/wells_booz['n_wells']:.3f}")
    print(f"  Boozer well depths: [{wells_booz['depths'].min():.6f}, "
          f"{wells_booz['depths'].max():.6f}]  "
          f"mean={wells_booz['depths'].mean():.6f}")
    print(f"  mgrid  well depths: [{wells_rp['depths'].min():.6f}, "
          f"{wells_rp['depths'].max():.6f}]  "
          f"mean={wells_rp['depths'].mean():.6f}")

    # Interpolate mgrid wells to boozer well count for depth comparison
    if wells_booz["n_wells"] > 0 and wells_rp["n_wells"] > 0:
        nw_b = wells_booz["n_wells"]
        nw_r = wells_rp["n_wells"]
        # Compare depth distributions via sorted depths
        depths_b_sorted = np.sort(wells_booz["depths"])
        depths_r_sorted = np.sort(wells_rp["depths"])
        # Interpolate to common grid
        x_common = np.linspace(0, 1, min(nw_b, nw_r))
        depths_b_interp = np.interp(x_common, np.linspace(0, 1, nw_b), depths_b_sorted)
        depths_r_interp = np.interp(x_common, np.linspace(0, 1, nw_r), depths_r_sorted)
        rms_depth_diff = np.sqrt(np.mean((depths_b_interp - depths_r_interp) ** 2))
        print(f"  RMS well-depth diff (interp): {rms_depth_diff:.6f}")

    print(f"\n  {'─'*60}")
    print(f"  ε_eff^(3/2) comparison (full pipeline)")
    print(f"  {'─'*60}")
    print(f"  {'s':>8s}  {'pyneo':>12s}  {'ripplepy':>12s}  {'rp/py':>8s}")
    print(f"  {'─'*8}  {'─'*12}  {'─'*12}  {'─'*8}")
    for i in range(len(SURF_IDX_LIST)):
        s_val = SURF_IDX_LIST[i]
        ratio = rp_eps_profile[i] / py_eps_profile[i]
        print(f"  {s_val:8.3f}  {py_eps_profile[i]:12.4e}  "
              f"{rp_eps_profile[i]:12.4e}  {ratio:8.4f}")
    print()

    # ── Optional: save plot ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # |B| overlay (first 2000 points for visibility)
        n_plot = min(2000, n_compare)
        phi_plot = np.arange(n_plot) * (2 * np.pi / NPHI)
        ax = axes[0, 0]
        ax.plot(phi_plot, B_b_cmp[:n_plot], alpha=0.7, linewidth=0.5, label="Boozer")
        ax.plot(phi_plot, B_r_cmp[:n_plot], alpha=0.7, linewidth=0.5, label="mgrid")
        ax.set_xlabel("φ")
        ax.set_ylabel("|B| (T)")
        ax.set_title(f"{DEVICE} s={SURF_S} — |B| along field line")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # |B| relative error
        ax = axes[0, 1]
        rel_err = (B_b_cmp - B_r_cmp)[:n_plot] / B_b_cmp[:n_plot] * 100
        ax.plot(phi_plot, rel_err, linewidth=0.5, color="red")
        ax.set_xlabel("φ")
        ax.set_ylabel("(Boozer − mgrid) / Boozer  (%)")
        ax.set_title("|B| relative error")
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)

        # Well depth histogram
        ax = axes[1, 0]
        ax.hist(wells_booz["depths"], bins=40, alpha=0.5, label="Boozer")
        ax.hist(wells_rp["depths"], bins=40, alpha=0.5, label="mgrid")
        ax.set_xlabel("well depth ΔB/B₀")
        ax.set_ylabel("count")
        ax.set_title("Well depth distribution")
        ax.legend()

        # ε_eff profile
        ax = axes[1, 1]
        ax.plot(SURF_IDX_LIST, py_eps_profile, "o-", label="pyneo")
        ax.plot(SURF_IDX_LIST, rp_eps_profile, "x-", label="ripplepy")
        ax.axvline(x=SURF_S, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("s")
        ax.set_ylabel("ε_eff^(3/2)")
        ax.set_title(f"{DEVICE} ε_eff profile")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

        plt.tight_layout()
        out_path = f"{BASE}/tests/bench_field_compare_{DEVICE}_s{SURF_S:.2f}.png"
        plt.savefig(out_path, dpi=150)
        print(f"  Plot saved to {out_path}")
    except Exception as e:
        print(f"  (plot skipped: {e})")


if __name__ == "__main__":
    main()
