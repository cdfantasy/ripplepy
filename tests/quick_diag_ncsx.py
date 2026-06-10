#!/usr/bin/env python3
"""Quick diagnostic: compare ripplepy vs pyneo for NCSX on a few surfaces."""
import os, sys
os.chdir('/Users/zkgao/ripplepy')
sys.path.insert(0, 'python')

import numpy as np
from simsopt.mhd import Boozer, Vmec
from simsopt.geo import SurfaceRZFourier
import neo
from neo import NeoContext, neo_surfaces_from_simsopt_boozer
from ripplepy import set_extcur, initialize_mgrid_field, set_trace_parameters, compute_epstot, find_axis

# ── Config ──
vmec_path = "tests/test_file/wout_ncsx_c09r00_free.nc"
mgrid_path = "tests/test_file/mgrid_c09r00.nc"
initial_rz = (1.57, 0)
nfp = 3
full_torus = False
nturn, nphi = 100, 180

sur_idx = np.linspace(0.1, 0.5, 3)  # just 3 surfaces

# ── RZ starting points ──
vmec = Vmec(str(vmec_path))
RZ_points = []
for s in sur_idx:
    surf = SurfaceRZFourier.from_wout(str(vmec_path), s)
    rpz = surf.cross_section(phi=0)[0]
    RZ_points.append(rpz[[0, 2]])
RZ_points = np.asarray(RZ_points)

# ── Boozer + pyneo ──
print("=== Computing Boozer coordinates ===")
boozer = Boozer(vmec)
boozer.mpol = 72; boozer.ntor = 72
boozer.register(sur_idx)
boozer.run()
print("Boozer done.")

neoclass = neo.from_simsopt_boozer(boozer)
ctx = NeoContext()
ctx.set_boozer(neoclass)
surfaces = neo_surfaces_from_simsopt_boozer(boozer)
ctx.set_flux_surfaces(surfaces.tolist())
ctx.set_resolution(theta_n=100, phi_n=100)
ctx.set_transport_options(npart=100, multra=1, acc_req=0.01, no_bins=100,
    nstep_per=50, nstep_min=500, nstep_max=5000, calc_nstep_max=0)
ctx.set_switches(ref_swi=2, eout_swi=2, calc_cur=0)
ctx.set_output_options(write_progress=0, write_output_files=0,
    write_integrate=0, write_diagnostic=1, suppress_file_io=False)
ctx.setup_grids()
print("Running pyneo...")
ctx.run_all()
py_epstot = ctx.epstot_profile()
print(f"pyneo epstot: {py_epstot}")

# ── ripplepy ──
print("\n=== ripplepy ===")
initialize_mgrid_field(mgrid_path, nfp, full_torus=full_torus)
extcur = set_extcur(None)
axis_rz, R0, axis_fl, istate = find_axis(initial_rz, xtol=1e-5, max_iter=100)
print(f"Axis: R={axis_rz[0]:.4f}, Z={axis_rz[1]:.4f}, R0={R0:.4f}")

initial_gradpsi = np.array([1, 0, 0], dtype=np.float64)
set_trace_parameters(nturn, nphi)

ripplepy_results = []
for i, rz in enumerate(RZ_points):
    fieldline_data = np.zeros((nturn * nphi, 20), dtype=np.float64, order='F')
    eps, Bb, ist = compute_epstot(R0, extcur, rz, initial_gradpsi, fieldline_data)
    ripplepy_results.append(eps)
    print(f"  surf {i}: ripplepy eps={eps:.6e}  RZ=({rz[0]:.4f},{rz[1]:.4f})")

# ── Compare ──
print("\n=== Comparison ===")
print(f"{'surf':>4s}  {'pyneo':>12s}  {'ripplepy':>12s}  {'ratio':>8s}")
for i in range(len(sur_idx)):
    r = ripplepy_results[i] / py_epstot[i] if py_epstot[i] != 0 else np.nan
    print(f"  {i:3d}  {py_epstot[i]:12.4e}  {ripplepy_results[i]:12.4e}  {r:8.4f}")

# ── Check pyneo diagnostics ──
for fname in ['diagnostic_add.dat', 'conver.dat']:
    if os.path.exists(fname):
        with open(fname) as f:
            lines = f.readlines()
        print(f"\n{fname}: {len(lines)} lines")
        print(f"  last: {lines[-1].strip()[:200]}")
