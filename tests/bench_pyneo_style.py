#!/usr/bin/env python3
"""Benchmark: ripplepy (old bp-scan) vs ripplepy (new pyneo-style) vs pyneo for H1 and NCSX."""
import numpy as np, time
from simsopt.mhd import Boozer, Vmec
from simsopt.geo import SurfaceRZFourier
import neo
from neo import NeoContext, neo_surfaces_from_simsopt_boozer
from ripplepy import (
    set_extcur, initialize_mgrid_field, set_trace_parameters,
    compute_epstot, find_axis,
)
from ripplepy.ripple import compute_epstot_pyneo

def run_benchmark(name, vmec_path, mgrid_path, initial_rz, extcur, nfp,
                  sur_idx, nturn, nphi, npart, full_torus=False):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    
    vmec = Vmec(str(vmec_path))
    R0_vmec = float(vmec.wout.Rmajor_p)
    
    # RZ start points
    RZ_points = []
    for s in sur_idx:
        surf = SurfaceRZFourier.from_wout(str(vmec_path), s)
        rpz = surf.cross_section(phi=0)[0]
        RZ_points.append(rpz[[0, 2]])
    RZ_points = np.asarray(RZ_points)
    
    # ── pyneo ──
    print("  Running pyneo...")
    boozer = Boozer(vmec)
    boozer.mpol = 72; boozer.ntor = 36
    boozer.register(sur_idx); boozer.run()
    neoclass = neo.from_simsopt_boozer(boozer)
    ctx = NeoContext(); ctx.set_boozer(neoclass)
    ctx.set_flux_surfaces(neo_surfaces_from_simsopt_boozer(boozer).tolist())
    ctx.set_resolution(theta_n=100, phi_n=100)
    ctx.set_transport_options(npart=npart, multra=1, acc_req=0.01, no_bins=100,
        nstep_per=50, nstep_min=500, nstep_max=5000, calc_nstep_max=0)
    ctx.set_switches(ref_swi=2, eout_swi=2, calc_cur=0)
    ctx.set_output_options(write_progress=0, write_output_files=0,
        write_integrate=0, write_diagnostic=0, suppress_file_io=True)
    ctx.setup_grids(); ctx.run_all()
    py_eps = ctx.epstot_profile()
    
    # ── ripplepy (old) ──
    print("  Running ripplepy (old bp-scan)...")
    initialize_mgrid_field(mgrid_path, nfp, full_torus=full_torus)
    set_extcur(extcur)
    axis_rz, R0_rp, axis_fl, ist = find_axis(initial_rz, xtol=1e-5, max_iter=100)
    print(f"  Axis: R={axis_rz[0]:.4f}, R0={R0_rp:.4f}")
    
    print(f'major radius from vmec: {R0_vmec:.4f}, from ripplepy: {R0_rp:.4f}')

    rp_old = []
    for rz in RZ_points:
        fld = np.zeros((nturn*nphi, 20), dtype=np.float64, order='F')
        eps, Bb, ist = compute_epstot(R0_vmec, rz,
                                       np.array([1,0,0], dtype=np.float64), fld)
        rp_old.append(eps)
    
    # ── ripplepy (new pyneo-style) ──
    print("  Running ripplepy (new pyneo-style)...")
    rp_new = []
    for rz in RZ_points:
        eps, ist = compute_epstot_pyneo(
            R0_vmec, rz,
            initial_gradpsi=np.array([1,0,0], dtype=np.float64),
            npart=npart, nturn=nturn, nphi=nphi, verbose=False,
        )
        rp_new.append(eps if eps is not None else np.nan)
    
    # ── Print ──
    print(f"\n  {'s':>6s}  {'pyneo':>12s}  {'rp_old':>12s}  {'old/py':>8s}  "
          f"{'rp_new':>12s}  {'new/py':>8s}")
    print(f"  {'-'*6}  {'-'*12}  {'-'*12}  {'-'*8}  {'-'*12}  {'-'*8}")
    for i in range(len(sur_idx)):
        o = rp_old[i] / py_eps[i] if py_eps[i] != 0 else np.nan
        n = rp_new[i] / py_eps[i] if py_eps[i] != 0 else np.nan
        print(f"  {sur_idx[i]:6.3f}  {py_eps[i]:12.4e}  {rp_old[i]:12.4e}  {o:8.4f}  "
              f"{rp_new[i]:12.4e}  {n:8.4f}")
    # plot
    from matplotlib import pyplot as plt
    plt.figure(figsize=(8,6))
    plt.plot(sur_idx, py_eps, 'o-', label='pyneo')
    plt.plot(sur_idx, rp_old, 's-', label='ripplepy (old)')
    plt.plot(sur_idx, rp_new, 'x-', label='ripplepy (new pyneo-style)')
    plt.xlabel('s'); plt.ylabel('eps_tot'); plt.title(f'{name} Benchmark'); plt.legend(); plt.grid(True)
    plt.title(f"{name} Benchmark: eps_tot vs s")
    plt.tight_layout(); plt.show()

    
    return py_eps, rp_old, rp_new

# ═══════════════════════════════════════════════════════════════
# NCSX
# ═══════════════════════════════════════════════════════════════
BASE = "/Users/zkgao/ripplepy"

run_benchmark(
    "NCSX",
    f"{BASE}/tests/test_file/wout_ncsx_c09r00_free.nc",
    f"{BASE}/tests/test_file/mgrid_c09r00.nc",
    (1.57, 0), None, 3,
    np.linspace(0.1, 0.2, 11),
    nturn=64, nphi=180, npart=50,
    full_torus=False,
)

# ═══════════════════════════════════════════════════════════════
# CFQS
# ═══════════════════════════════════════════════════════════════
BASE = "/Users/zkgao/ripplepy"

run_benchmark(
    "CFQS",
    f"{BASE}/tests/test_file/wout_cfqs_test_m10_n5_fixed.nc",
    f"{BASE}/tests/test_file/mgrid_2b40R1mB01.nc",
    (1.21, 0), None, 2,
    np.linspace(0.1, 1, 11),
    nturn=200, nphi=360, npart=50,
    full_torus=False,
)

# ═══════════════════════════════════════════════════════════════
# H1
# ═══════════════════════════════════════════════════════════════
# run_benchmark(
#     "H1",
#     f"{BASE}/tests/test_file/wout_h1_design.nc",
#     f"{BASE}/tests/test_file/mgrid_h1_design.nc",
#     (1.26, 0), [50000, 5000, 1, -80000, -40000], 3,
#     np.linspace(0.1, 1, 11),
#     nturn=200, nphi=360, npart=50,
#     full_torus=False,
# )

