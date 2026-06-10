#!/usr/bin/env python3
"""Diagnose the ~3.9x discrepancy between boozer integration and pyneo.

Runs pyneo with diagnostic output enabled (write_integrate=1) to dump
internal y2, y3, bigint values, then compares with our boozer-integration
counterparts for the SAME field line.
"""
import numpy as np
from simsopt.mhd import Boozer, Vmec
import neo
from neo import NeoContext, neo_surfaces_from_simsopt_boozer
from ripplepy.boozer_eps_verify import (
    sample_fieldline_from_boozer, eps_eff_from_boozer,
    _boozer_obj_to_dict, _find_local_minima, _integrate_bounce_segment,
)

# ---- config ----
vmec_path = "/Users/zkgao/ripplepy/tests/test_file/wout_h1_design.nc"
surf_idx = 0
theta0 = 0.0
nzeta = 512
nturn = 64

# ---- run pyneo with diagnostics ----
vmec = Vmec(str(vmec_path))
boozer = Boozer(vmec)
boozer.mpol = 24; boozer.ntor = 72
boozer.register(np.linspace(0.1, 0.3, 1))
boozer.run()

neoclass = neo.from_simsopt_boozer(boozer)
ctx = NeoContext()
ctx.set_boozer(neoclass)
surfaces = neo_surfaces_from_simsopt_boozer(boozer)
ctx.set_flux_surfaces(surfaces.tolist())
ctx.set_resolution(theta_n=200, phi_n=200)
ctx.set_transport_options(
    npart=100, multra=1, acc_req=0.01, no_bins=100,
    nstep_per=50, nstep_min=500, nstep_max=2000, calc_nstep_max=0,
)
ctx.set_switches(ref_swi=2, eout_swi=2, calc_cur=0)
# Enable write_integrate to dump internal variables
ctx.set_output_options(
    write_progress=0, write_output_files=0,
    write_integrate=1, write_diagnostic=0,
    suppress_file_io=False,
)
ctx.setup_grids()
ctx.run_all()

py_epstot = ctx.epstot_profile()
print(f"pyneo epstot = {py_epstot[0]:.6e}")

# ---- our boozer integration ----
booz_dict = _boozer_obj_to_dict(boozer)
fl = sample_fieldline_from_boozer(booz_dict, surf_idx, theta0, nzeta, nturn)

dphi_me = 2 * np.pi / nzeta
e2_me = np.sum(dphi_me / fl.B**2)
e3_me = np.sum(dphi_me * np.abs(fl.gradpsi) / fl.B**2)

print(f"\n--- Our integration ---")
print(f"  npoints = {len(fl.B)}, dphi = {dphi_me:.6f}")
print(f"  B range: [{fl.B.min():.4f}, {fl.B.max():.4f}]")
print(f"  e2 = {e2_me:.4f}")
print(f"  e3 = {e3_me:.4f}  (abs)")
print(f"  e2/e3² = {e2_me/e3_me**2:.4f}")

# Compute e1 for a subset of bp values to compare structure
bp_test = [0.65, 0.75, 0.85, 0.95]
b0 = fl.B.max()
minima = _find_local_minima(fl.B)
print(f"  wells detected: {len(minima)-1}")

for bp in bp_test:
    Htot, Itot = 0.0, 0.0
    for k in range(len(minima) - 1):
        H, I = _integrate_bounce_segment(
            bp, b0, minima[k], minima[k+1],
            fl.B, np.abs(fl.gradpsi), fl.kg_gradpsi,
            np.full(len(fl.B), dphi_me),
        )
        Htot += H; Itot += I
    print(f"  bp={bp:.2f}: ΣH={Htot:.4e}, ΣI={Itot:.4e}, ΣH²/I={Htot**2/Itot:.4e}")

# ---- Compare with pyneo's internal values ----
# Read the conver.dat file if pyneo wrote it
import os
conver_file = "conver.dat"
if os.path.exists(conver_file):
    data = np.loadtxt(conver_file)
    print(f"\n--- pyneo conver.dat: {data.shape[0]} lines ---")
    if data.ndim == 2:
        cols = data.shape[1]
        # Format: n, epstot_check, p_bm3ge, p_i/p_bm2, aditot/p_bm2
        # See flint_bo.f90:217-222
        print(f"  columns: {cols}")
        # epstot_check uses p_bm2 / p_bm2gv**2 at intermediate point
        py_y2 = float(data[-1, 2]) if cols > 2 else None  # p_bm3ge? No...
        print(f"  last line: {data[-1]}")
else:
    print("\n--- conver.dat NOT found ---")

# ---- Summary ----
R0 = float(vmec.wout.Rmajor_p)
rt0 = booz_dict['rmnc_b'][surf_idx, 0]  # m=n=0
bmin, bmax = fl.B.min(), fl.B.max()
heta = (1.0 - bmin/bmax) / 99.0  # (npart-1)

print(f"\n--- Parameters ---")
print(f"  R0 (VMEC) = {R0:.4f}")
print(f"  rt0 (rmnc m=n=0) = {rt0:.4f}")
print(f"  B_min/B_max = {bmin/bmax:.4f}")
print(f"  heta = {heta:.6f}")
print(f"  coeps = π·rt0²·heta/(8√2) = {np.pi*rt0**2*heta/(8*np.sqrt(2)):.6f}")

# Our raw and scaled eps
r = eps_eff_from_boozer(booz_dict, surf_idx, theta0, nzeta, nturn, n_b=2000, use_gauss=False)
eps_raw = r['eps_eff_32']
eps_R02 = eps_raw * R0**2
eps_rt02 = eps_raw * rt0**2

print(f"\n--- Final comparison ---")
print(f"  pyneo epstot         = {py_epstot[0]:.6e}")
print(f"  our eps_raw          = {eps_raw:.6e}")
print(f"  our eps × R₀²        = {eps_R02:.6e}  (×{eps_R02/py_epstot[0]:.2f})")
print(f"  our eps × rt0²       = {eps_rt02:.6e}  (×{eps_rt02/py_epstot[0]:.2f})")
print(f"  our eps_raw (no ×R²) = {eps_raw:.6e}  (×{eps_raw/py_epstot[0]:.2f})")

# Diagnostic: what factor would make them match?
calibration_factor = py_epstot[0] / eps_raw
print(f"\n  Calibration factor (pyneo / our_raw) = {calibration_factor:.4f}")
print(f"  1/calibration = {1/calibration_factor:.4f}")
print(f"  sqrt(calibration) = {np.sqrt(calibration_factor):.4f}")
