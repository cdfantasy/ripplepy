#!/usr/bin/env python3
"""Benchmark: ripplepy vs pyneo for CFQS and H1."""
import numpy as np, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from simsopt.mhd import Boozer, Vmec
from simsopt.geo import SurfaceRZFourier
import neo
from neo import NeoContext, neo_surfaces_from_simsopt_boozer
from ripplepy import (
    set_extcur, initialize_mgrid_field, set_trace_parameters,
    compute_epstot, find_axis,
)

def run_benchmark(name, vmec_path,boozer_path, mgrid_path, extcur, nfp,
                  sur_idx, nturn, nphi, npart, full_torus=False,py_old = False):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    
    vmec = Vmec(str(vmec_path))
    R0_vmec = float(vmec.wout.Rmajor_p)
    # Magnetic-axis guess from the VMEC axis Fourier coefficients:
    # at phi=0, R = sum(raxis_cc) + sum(raxis_cs); Z = 0 (symmetry plane).
    # Not every wout carries raxis_cs — treat it as zero when absent.
    initial_rz = np.array([
        float(sum(vmec.wout.raxis_cc))
        + float(sum(getattr(vmec.wout, "raxis_cs", [0.0]))),
        0.0,
    ])

    # RZ start points
    RZ_points = []
    for s in sur_idx:
        surf = SurfaceRZFourier.from_wout(str(vmec_path), s)
        rpz = surf.cross_section(phi=0)[0]
        RZ_points.append(rpz[[0, 2]])
    RZ_points = np.asarray(RZ_points)

    print("  Running pyneo...")

    boozer = Boozer(vmec)
    boozer.mpol = 72; boozer.ntor = 36
    try:
        boozer.bx.read_boozmn(str(boozer_path))
        boozer.register(sur_idx)
        print("  Loaded Boozer from cached boozmn netcdf.")
    except Exception:
        boozer.register(sur_idx); boozer.run()
        # boozer.bx.write_boozmn(str(boozer_path))
        print("  Computed Boozer transform and cached to boozmn netcdf.")

    neoclass = neo.from_simsopt_boozer(boozer)

    ctx = NeoContext(); 
    ctx.set_boozer(neoclass)
    ctx.set_flux_surfaces(neo_surfaces_from_simsopt_boozer(boozer).tolist())
    ctx.set_resolution(theta_n=100, phi_n=100)
    ctx.set_transport_options(npart=npart, multra=1, acc_req=0.01, no_bins=100,
        nstep_per=50, nstep_min=500, nstep_max=5000, calc_nstep_max=0)
    ctx.set_switches(ref_swi=2, eout_swi=2, calc_cur=0)
    ctx.set_output_options(write_progress=0, write_output_files=0,
        write_integrate=0, write_diagnostic=0, suppress_file_io=True)
    ctx.setup_grids(); ctx.run_all()
    py_eps = ctx.epstot_profile()
    
    print("  Running ripplepy ...")
    initialize_mgrid_field(mgrid_path, nfp, full_torus=full_torus)
    set_extcur(extcur)
    axis_rz, R0_rp, axis_fl, ist = find_axis(initial_rz, xtol=1e-5, max_iter=100)
    print(f"  Axis: R={axis_rz[0]:.4f}, R0={R0_rp:.4f}")
    
    print(f'major radius from vmec: {R0_vmec:.4f}, from ripplepy: {R0_rp:.4f}')
    print("  Running ripplepy ...")
    set_trace_parameters(nturn, nphi, npart=npart, verbose=False)
    ripplepy = []
    
    def _compute_one(rz):
        eps, bnd, ist = compute_epstot(
            rz,
            initial_gradpsi=np.array([1,0,0], dtype=np.float64),
            verbose=False,
        )
        return eps if eps is not None else np.nan

    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(_compute_one, rz): i for i, rz in enumerate(RZ_points)}
        results = [np.nan] * len(RZ_points)
        for fut in as_completed(futures):
            i = futures[fut]
            results[i] = fut.result()
    ripplepy = results
    
    # ── ripplepy (old) ──
    if py_old:

        rp_old = []
        for rz in RZ_points:
            fld = np.zeros((nturn*nphi, 20), dtype=np.float64, order='F')
            eps, Bb, ist = compute_epstot(rz,
                                        np.array([1,0,0], dtype=np.float64))
            rp_old.append(eps)
        print(f"\n  {'s':>6s}  {'pyneo':>12s} {'py_old':>12s} {'old/py':>8s}"
            f"{'ripplepy':>12s}  {'new/py':>8s}")
        print(f"  {'-'*6}  {'-'*12}  {'-'*12}  {'-'*8} {'-'*12}  {'-'*8}")
        for i in range(len(sur_idx)):
            n = ripplepy[i] / py_eps[i] if py_eps[i] != 0 else np.nan
            o = rp_old[i] / py_eps[i] if py_eps[i] != 0 else np.nan
            print(f"  {sur_idx[i]:6.3f}  {py_eps[i]:12.4e}  {rp_old[i]:12.4e}  {o:8.4f}"
                f"{ripplepy[i]:12.4e}  {n:8.4f}")        
        from matplotlib import pyplot as plt
        plt.figure(figsize=(8,6))
        plt.plot(sur_idx, py_eps, 'o-', label='pyneo')
        plt.plot(sur_idx, rp_old, 'x-', label='ripplepy(old)')
        plt.plot(sur_idx, ripplepy, 'x-', label='ripplepy (new pyneo-style)')
        plt.xlabel('s'); plt.ylabel('eps_tot'); plt.title(f'{name} Benchmark'); plt.legend(); plt.grid(True)
        plt.title(f"{name} Benchmark: eps_tot vs s")
        plt.tight_layout(); plt.show()

        
        return py_eps,rp_old, ripplepy        
    # ── ripplepy (new pyneo-style) ──

    else:
        print(f"\n  {'s':>6s}  {'pyneo':>12s}  "
            f"{'ripplepy':>12s}  {'new/py':>8s}")
        print(f"  {'-'*6}  {'-'*12}  {'-'*12}  {'-'*8}")
        for i in range(len(sur_idx)):
            n = ripplepy[i] / py_eps[i] if py_eps[i] != 0 else np.nan
            print(f"  {sur_idx[i]:6.3f}  {py_eps[i]:12.4e}  "
                f"{ripplepy[i]:12.4e}  {n:8.4f}")        
    # plot
        from matplotlib import pyplot as plt
        plt.figure(figsize=(8,6))
        plt.plot(sur_idx, py_eps, 'o-', label='pyneo')
        plt.plot(sur_idx, ripplepy, 'x-', label='ripplepy (new pyneo-style)')
        plt.xlabel('s'); plt.ylabel('eps_tot'); plt.title(f'{name} Benchmark'); plt.legend(); plt.grid(True)
        plt.title(f"{name} Benchmark: eps_tot vs s")
        plt.tight_layout(); plt.show()

        
        return py_eps, ripplepy


BASE = str(Path(__file__).resolve().parent.parent)

# ═══════════════════════════════════════════════════════════════
# w7x
# ═══════════════════════════════════════════════════════════════
run_benchmark(
    "w7x",
    f"{BASE}/tests/test_file/wout_w7x_test_m10_n5_fixed.nc",
    f"{BASE}/tests/test_file/w7x_boozmn.nc",
    f"{BASE}/tests/test_file/mgrid_w7-x.nc",
    None, 5,
    np.linspace(0.1, 0.2, 11),
    nturn=200, nphi=180, npart=50,
    full_torus=False,
)

# ═══════════════════════════════════════════════════════════════
# CFQS
# ═══════════════════════════════════════════════════════════════


# run_benchmark(
#     "CFQS",
#     f"{BASE}/tests/test_file/wout_cfqs_test_m10_n5_fixed.nc",
#     f"{BASE}/tests/test_file/cfqs_boozmn.nc",
#     f"{BASE}/tests/test_file/mgrid_2b40R1mB01.nc",
#     None, 2,
#     np.linspace(0.1, 1, 11),
#     nturn=100, nphi=100, npart=500,
#     full_torus=False,
# )

# ═══════════════════════════════════════════════════════════════
# H1
# ═══════════════════════════════════════════════════════════════
# run_benchmark(
#     "H1",
#     f"{BASE}/tests/test_file/wout_h1_design.nc",
#     f"{BASE}/tests/test_file/h1_boozmn.nc",
#     f"{BASE}/tests/test_file/mgrid_h1_design.nc",
#     [50000, 5000, 0, -80000, -40000], 3,
#     np.linspace(0.1, 1, 11),
#     nturn=200, nphi=360, npart=5000,
#     full_torus=False,
# )