#!/usr/bin/env python3
"""Benchmark: Fourier-filtered mgrid vs unfiltered for NCSX and CFQS."""
import numpy as np
from simsopt.mhd import Boozer, Vmec
from simsopt.geo import SurfaceRZFourier
import neo
from neo import NeoContext, neo_surfaces_from_simsopt_boozer
from ripplepy import (
    set_extcur, initialize_mgrid_field, set_trace_parameters,
)
from ripplepy.ripple import compute_epstot_pyneo

BASE = "/Users/zkgao/ripplepy"

def run_bench(name, vmec_path, mgrid_path, sur_idx, initial_rz, extcur, nfp,
              nturn, nphi, npart, filter_modes=None, filter_nphi=None, full_torus=False):
    tag = f"F-{filter_modes}" if filter_modes else "unfiltered"
    print(f"\n{'='*60}")
    print(f"  {name} [{tag}]")
    print(f"{'='*60}")
    
    vmec = Vmec(str(vmec_path))
    R0_vmec = float(vmec.wout.Rmajor_p)
    
    RZ_points = []
    for s in sur_idx:
        surf = SurfaceRZFourier.from_wout(str(vmec_path), s)
        rpz = surf.cross_section(phi=0)[0]
        RZ_points.append(rpz[[0, 2]])
    RZ_points = np.asarray(RZ_points)
    
    # pyneo
    print("  pyneo...")
    boozer = Boozer(vmec); boozer.mpol=48; boozer.ntor=48
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
    
    # ripplepy
    print("  ripplepy...")
    initialize_mgrid_field(mgrid_path, nfp, full_torus=full_torus,
                           filter_modes=filter_modes, filter_nphi_new=filter_nphi)
    extcur_arr = set_extcur(extcur)
    print(f"  using VMEC R0={R0_vmec:.4f}")
    
    rp_new = []
    for rz in RZ_points:
        eps, ist = compute_epstot_pyneo(R0_vmec, rz,
            initial_gradpsi=np.array([1,0,0], dtype=np.float64),
            npart=npart, nturn=nturn, nphi=nphi, verbose=False)
        rp_new.append(eps if eps is not None else np.nan)
    
    print(f"  {'s':>6s}  {'pyneo':>10s}  {'rp_new':>10s}  {'new/py':>7s}")
    for i in range(len(sur_idx)):
        n = rp_new[i]/py_eps[i] if py_eps[i]!=0 else np.nan
        print(f"  {sur_idx[i]:6.3f}  {py_eps[i]:10.4e}  {rp_new[i]:10.4e}  {n:7.4f}")
    return py_eps, rp_new

sur_idx = np.linspace(0.1, 0.5, 5)

# ═══ NCSX ═══
for fm, fn in [(None, None), (18, 72)]:
    run_bench("NCSX",
        f"{BASE}/tests/test_file/wout_ncsx_c09r00_free.nc",
        f"{BASE}/tests/test_file/mgrid_c09r00.nc",
        sur_idx, (1.57, 0), None, 3,
        nturn=64, nphi=180, npart=50,
        filter_modes=fm, filter_nphi=fn, full_torus=False)

# ═══ CFQS ═══
for fm, fn in [(None, None), (30, 120)]:
    run_bench("CFQS",
        f"{BASE}/tests/test_file/wout_cfqs_test_m10_n5_fixed.nc",
        f"{BASE}/tests/test_file/mgrid_2b40R1mB01.nc",
        sur_idx, (1.23, 0), None, 2,
        nturn=64, nphi=180, npart=50,
        filter_modes=fm, filter_nphi=fn, full_torus=False)
