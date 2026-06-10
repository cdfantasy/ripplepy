#!/usr/bin/env python3
"""NCSX 4-way comparison: pyneo vs boozer+rect vs boozer+Gauss vs ripplepy."""
import os, sys
os.chdir('/Users/zkgao/ripplepy')
sys.path.insert(0, 'python')

import numpy as np
from simsopt.mhd import Boozer, Vmec
from simsopt.geo import SurfaceRZFourier
import neo
from neo import NeoContext, neo_surfaces_from_simsopt_boozer
from ripplepy.boozer_eps_verify import (
    eps_eff_from_boozer, _boozer_obj_to_dict,
    sample_fieldline_from_boozer, _find_local_minima,
    _integrate_bounce_segment, _compute_H2_over_I_for_bp,
)
from ripplepy import set_extcur, initialize_mgrid_field, set_trace_parameters, compute_epstot, find_axis
import time 

# ── Config ──
vmec_path = "tests/test_file/wout_ncsx_c09r00_free.nc"
mgrid_path = "tests/test_file/mgrid_c09r00.nc"
nfp = 3
sur_idx = np.linspace(0, 0.2, 11)
initial_rz0 = (1.57, 0)

vmec = Vmec(str(vmec_path))
R0_vmec = float(vmec.wout.Rmajor_p)

vmec_surf = SurfaceRZFourier.from_wout(str(vmec_path), 1)



# ── RZ starting points ──
RZ_points = []
for s in sur_idx:
    surf = SurfaceRZFourier.from_wout(str(vmec_path), s)
    rpz = surf.cross_section(phi=0)[0]
    RZ_points.append(rpz[[0, 2]])
    # print(f's = {s:.1e}, R = {rpz[0]:.3e}')
RZ_points = np.asarray(RZ_points)

# ═══════════════════════════════════════════════════════════
# PART 1: pyneo
# ═══════════════════════════════════════════════════════════
print("=== pyneo ===")
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
    nstep_per=50, nstep_min=500, nstep_max=5000, calc_nstep_max=0)
ctx.set_switches(ref_swi=2, eout_swi=2, calc_cur=0)
ctx.set_output_options(write_progress=0, write_output_files=0,
    write_integrate=0, write_diagnostic=0, suppress_file_io=True)
ctx.setup_grids()
ctx.run_all()
py_epstot = ctx.epstot_profile()
print(f"pyneo: {py_epstot}")

# ═══════════════════════════════════════════════════════════
# PART 2: Boozer-coordinate integration (rect + Gauss)
# ═══════════════════════════════════════════════════════════
print("\n=== Boozer integration ===")
booz_dict = _boozer_obj_to_dict(boozer)

booz_rect = []
booz_gauss = []
for i in range(len(sur_idx)):
    time_start = time.time()
    # r = eps_eff_from_boozer(booz_dict, i, theta0=0.0, nzeta=360, nturn=200,
    #                          n_b=500, use_gauss=False)
    # time_end_rect = time.time()
    # print(f"  surf {i}: rect*R0²={r['eps_eff_32'] * R0_vmec**2:.4e}  (time={time_end_rect - time_start:.2f}s)")
    rg = eps_eff_from_boozer(booz_dict, i, theta0=0.0, nzeta=360, nturn=20,
                              n_gauss=64, use_gauss=True)
    time_end_gauss = time.time()
    print(f"  surf {i}: gauss*R0²={rg['eps_eff_32'] * R0_vmec**2:.4e}  (time={time_end_gauss - time_start:.2f}s)")
    # booz_rect.append(r['eps_eff_32'] * R0_vmec**2)
    booz_gauss.append(rg['eps_eff_32'] * R0_vmec**2)

# ═══════════════════════════════════════════════════════════
# PART 3: ripplepy (coil field)
# ═══════════════════════════════════════════════════════════
print("\n=== ripplepy ===")
initialize_mgrid_field(mgrid_path, nfp, full_torus=False)
extcur_arr = set_extcur(None)
axis_rz, R0, axis_fl, istate = find_axis(initial_rz0, xtol=1e-5, max_iter=100)
print(f"Axis: R={axis_rz[0]:.4f}, R0={R0:.4f}")

initial_gradpsi = np.array([1, 0, 0], dtype=np.float64)
set_trace_parameters(400, 360)
ripplepy_res = []
for rz in RZ_points:
    fld = np.zeros((400*360, 20), dtype=np.float64, order='F')
    eps, Bb, ist = compute_epstot(R0, extcur_arr, rz, initial_gradpsi, fld)
    ripplepy_res.append(eps)
    print(f"  ripplepy: eps={eps:.4e} at R={rz[0]:.4f}, Z={rz[1]:.4f} (time={time.time() - time_start:.2f}s)")

# ═══════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 85)
print(f"{'surf':>4s}  {'pyneo':>12s}  "
      f"{'booz_gauss':>12s}  {'b_gauss/py':>9s}  {'ripplepy':>12s}  {'rp/py':>9s}")
print("-" * 85)
for i in range(len(sur_idx)):
    # bp = booz_rect[i] / py_epstot[i]
    bg = booz_gauss[i] / py_epstot[i]
    rp = ripplepy_res[i] / py_epstot[i]
    print(f"  {i:3d}  {py_epstot[i]:12.4e}  "
          f"{booz_gauss[i]:12.4e}  {bg:9.4f}  {ripplepy_res[i]:12.4e}  {rp:9.4f}")

import matplotlib.pyplot as plt

R = RZ_points[:, 0]
plt.figure(figsize=(8, 6))
plt.plot(R, py_epstot, 'o-', label='pyneo')
# plt.plot(R, booz_rect, 's-', label='boozer rect')
# plt.plot(R, booz_gauss, 'd-', label='boozer gauss')
plt.plot(R, ripplepy_res, 'x-', label='ripplepy')
plt.xlabel('R')
plt.ylabel('eps_eff * R0^2')
plt.title('Effective ripple comparison')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
plt.savefig('ncsx_4way_comparison.png', dpi=300)

plt.figure(figsize=(8, 6))
# plt.plot(R, booz_rect / py_epstot, 's-', label='boozer rect / pyneo')
# plt.plot(R, booz_gauss / py_epstot, 'd-', label='boozer gauss / pyneo')
plt.plot(R, ripplepy_res / py_epstot, 'x-', label='ripplepy / pyneo')
plt.xlabel('R')
plt.ylabel('eps_eff / pyneo eps_eff')
plt.title('Relative effective ripple')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
plt.savefig('ncsx_4way_relative.png', dpi=300)