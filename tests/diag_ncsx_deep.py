#!/usr/bin/env python3
"""Deep diagnostic: why ripplepy/pyneo ratio is ~2x for NCSX vs ~1x for H1."""
import os, sys
os.chdir('/Users/zkgao/ripplepy')
sys.path.insert(0, 'python')

import numpy as np
from simsopt.mhd import Boozer, Vmec
from simsopt.geo import SurfaceRZFourier
import neo
from neo import NeoContext, neo_surfaces_from_simsopt_boozer

vmec_path = "tests/test_file/wout_ncsx_c09r00_free.nc"
mgrid_path = "tests/test_file/mgrid_c09r00.nc"
nfp = 3

# ── Boozer + pyneo (just one surface) ──
sur_idx = np.array([0.1])
vmec = Vmec(str(vmec_path))
boozer = Boozer(vmec)
boozer.mpol = 72; boozer.ntor = 72
boozer.register(sur_idx)
boozer.run()

neoclass = neo.from_simsopt_boozer(boozer)
ctx = NeoContext()
ctx.set_boozer(neoclass)
surfaces = neo_surfaces_from_simsopt_boozer(boozer)
ctx.set_flux_surfaces(surfaces.tolist())
ctx.set_resolution(theta_n=100, phi_n=100)
ctx.set_transport_options(npart=100, multra=1, acc_req=0.01, no_bins=100,
    nstep_per=50, nstep_min=500, nstep_max=2000, calc_nstep_max=0)
ctx.set_switches(ref_swi=2, eout_swi=2, calc_cur=0)
ctx.set_output_options(write_progress=0, write_output_files=0,
    write_integrate=0, write_diagnostic=1, suppress_file_io=False)
ctx.setup_grids()
ctx.run_all()
py_epstot = ctx.epstot_profile()[0]

# Read pyneo internal vars
with open('diagnostic_add.dat') as f:
    parts = f.read().split()
# Format: psi_ind, istepc, npart, max_class, b_min, b_max, bmref, coeps, y2, y3
idx = 10  # last surface
py_y2 = float(parts[-2]); py_y3 = float(parts[-1])
py_bmin = float(parts[-8]); py_bmax = float(parts[-7])
py_bmref = float(parts[-6]); py_coeps = float(parts[-5])
py_npart = int(parts[-9])
heta = (1.0 - py_bmin/py_bmax) / (py_npart - 1)

# ── Get Boozer B along one field line ──
from ripplepy.boozer_eps_verify import sample_fieldline_from_boozer, _boozer_obj_to_dict
booz_dict = _boozer_obj_to_dict(boozer)
fl = sample_fieldline_from_boozer(booz_dict, 0, theta0=0.0, nzeta=512, nturn=32)

# ── ripplepy field line ──
from ripplepy import (set_extcur, initialize_mgrid_field, set_trace_parameters,
                      compute_epstot, find_axis)
initialize_mgrid_field(mgrid_path, nfp, full_torus=False)
extcur_arr = set_extcur(None)
axis_rz, R0, axis_fl, istate = find_axis((1.57, 0), xtol=1e-5, max_iter=100)

surf = SurfaceRZFourier.from_wout(str(vmec_path), 0.1)
rpz = surf.cross_section(phi=0)[0]
rz_start = rpz[[0, 2]]

initial_gradpsi = np.array([1, 0, 0], dtype=np.float64)
set_trace_parameters(60, 360)  # nturn, nphi
npoints = 60 * 360
fieldline_data = np.zeros((npoints, 20), dtype=np.float64, order='F')
result = compute_epstot(R0, extcur_arr, rz_start, initial_gradpsi,
                        fieldline_data, return_fieldline=True)
eps_rp = result[0]

# ── Analysis ──
print("=" * 70)
print("NCSX DIAGNOSTIC")
print("=" * 70)

print(f"\n── pyneo ──")
print(f"  epstot = {py_epstot:.4e}")
print(f"  B range: [{py_bmin:.4f}, {py_bmax:.4f}]  bmref={py_bmref:.4f}")
print(f"  coeps = {py_coeps:.6f}  heta = {heta:.6f}")
print(f"  y2 = {py_y2:.2f}  |y3| = {abs(py_y3):.2f}  y2/y3² = {py_y2/py_y3**2:.4e}")
print(f"  npart={py_npart}")

print(f"\n── Boozer field line (our Fourier eval) ──")
dphi_b = 2*np.pi/512
e2_b = np.sum(dphi_b / fl.B**2)
e3_b = np.sum(dphi_b * np.abs(fl.gradpsi) / fl.B**2)
print(f"  B range: [{fl.B.min():.4f}, {fl.B.max():.4f}]")
print(f"  |∇ψ| range: [{np.abs(fl.gradpsi).min():.4f}, {np.abs(fl.gradpsi).max():.4f}]")
print(f"  κ_G|∇ψ| range: [{fl.kg_gradpsi.min():.4f}, {fl.kg_gradpsi.max():.4f}]")
print(f"  e2 = {e2_b:.2f}  |e3| = {e3_b:.4f}  e2/e3² = {e2_b/e3_b**2:.4e}")
print(f"  npoints = {len(fl.B)}, nturn = 32")

print(f"\n── ripplepy (coil field) ──")
print(f"  axis: R={axis_rz[0]:.4f}, Z={axis_rz[1]:.4f}, R0={R0:.4f}")
print(f"  start: R={rz_start[0]:.4f}, Z={rz_start[1]:.4f}")
print(f"  eps_eff = {eps_rp:.4e}")

# Extract ripplepy field-line data
fl_data = fieldline_data

B_rp = fl_data[:, 6]  # |B|
gp_rp = fl_data[:, 10]  # |grad_psi|
R_rp = fl_data[:, 0]
Z_rp = fl_data[:, 1]
phi_rp = fl_data[:, 2]
Br_rp = fl_data[:, 3]
Bz_rp = fl_data[:, 4]
Bphi_rp = fl_data[:, 5]

# Remove zeros (failed trace points)
valid = (B_rp > 1e-10) & (gp_rp > 1e-10)
B_rp_v = B_rp[valid]
gp_rp_v = gp_rp[valid]

if len(B_rp_v) > 0:
    print(f"  valid pts: {np.sum(valid)}/{len(valid)}")
    print(f"  B range: [{B_rp_v.min():.4f}, {B_rp_v.max():.4f}]")
    print(f"  gp range: [{gp_rp_v.min():.4f}, {gp_rp_v.max():.4f}]")
    # Estimate e2, e3 from ripplepy trace
    dphi_rp = np.diff(phi_rp[valid])
    dphi_rp = np.append(dphi_rp, dphi_rp[-1])
    e2_rp = np.sum(dphi_rp / B_rp_v**2)
    e3_rp = np.sum(dphi_rp * gp_rp_v / B_rp_v**2)
    print(f"  estimated e2 = {e2_rp:.2f}, e3 = {e3_rp:.4f}, e2/e3² = {e2_rp/e3_rp**2:.4e}")

# ── Compare B fields ──
print(f"\n── Direct comparison ──")
print(f"  Boozer  B range: [{fl.B.min():.4f}, {fl.B.max():.4f}]")
print(f"  ripplepy B range: [{B_rp_v.min():.4f}, {B_rp_v.max():.4f}]")
print(f"  pyneo B range: [{py_bmin:.4f}, {py_bmax:.4f}]")

# Check whether ripplepy's B range matches pyneo
B_range_ratio = (B_rp_v.max() - B_rp_v.min()) / (py_bmax - py_bmin)
print(f"  ripplepy/pyneo B modulation ratio: {B_range_ratio:.3f}")
print(f"  B_max ratio: {B_rp_v.max()/py_bmax:.3f}")
print(f"  B_min ratio: {B_rp_v.min()/py_bmin:.3f}")

# Check how much of the field line is valid
print(f"\n  ripplepy field line: {npoints} total, {np.sum(valid)} valid ({100*np.sum(valid)/npoints:.1f}%)")
if np.sum(valid) < npoints:
    print(f"  ⚠ {npoints - np.sum(valid)} points failed (field line left grid?)")
    # Check where failures occur
    bad_idx = np.where(~valid)[0]
    if len(bad_idx) > 0:
        print(f"  first bad at idx {bad_idx[0]}/{npoints}")
        print(f"  last good R={R_rp[bad_idx[0]-1]:.4f} Z={Z_rp[bad_idx[0]-1]:.4f}")

# Cross-check: evaluate Boozer B at ripplepy field-line points
# (This checks if the coil field matches the equilibrium field at the same (R,Z,φ) locations)
if np.sum(valid) > 100:
    # Sample every 100th point
    sample_idx = np.where(valid)[0][::100]
    R_s = R_rp[sample_idx]
    Z_s = Z_rp[sample_idx]
    B_rp_s = B_rp[sample_idx]

    # Get B from mgrid at same points
    from ripplepy import get_bfield_matrix
    B_at_points = get_bfield_matrix(extcur_arr, R_s, Z_s, np.zeros_like(R_s))
    B_mgrid = np.sqrt(B_at_points[:, 0]**2 + B_at_points[:, 1]**2 + B_at_points[:, 2]**2)

    print(f"\n── Coil vs equilibrium B at same (R,Z,φ=0) ──")
    print(f"  sampled {len(R_s)} points")
    print(f"  mgrid B range: [{B_mgrid.min():.3f}, {B_mgrid.max():.3f}]")
