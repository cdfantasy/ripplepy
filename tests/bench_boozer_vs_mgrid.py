#!/usr/bin/env python3
"""Benchmark: pyneo vs ripplepy_boozer vs ripplepy_pyneo for CFQS and H1.

Three columns, same pyneo η-state-machine integration algorithm:

  pyneo             — VMEC Boozer field, pyneo's own field-line trace + integrator
  ripplepy_boozer   — VMEC Boozer field, analytic θ=θ₀+ιζ trace, ripplepy integrator
  ripplepy_pyneo    — mgrid grid field, DLSODE real-space trace, ripplepy integrator

Differences isolated:
  pyneo vs ripplepy_boozer  → trace/integrator difference (same Boozer field)
  boozer vs ripplepy_pyneo  → field-source difference (same integrator)
"""
import numpy as np
import time
from simsopt.mhd import Boozer, Vmec
from simsopt.geo import SurfaceRZFourier
import neo
from neo import NeoContext, neo_surfaces_from_simsopt_boozer
from ripplepy import initialize_mgrid_field, set_extcur, set_trace_parameters
from ripplepy.ripple import compute_epstot
from ripplepy.boozer_eps_verify import (_boozer_obj_to_dict, eps_eff_pyneo_style,
    _fourier_sum_cos, _fourier_sum_sin)


def _find_theta0_fast(rmnc, zmns, xm, xn, R_target, Z_target, ntheta=20000):
    """Find θ₀ at ζ=0 matching (R_target, Z_target).  1D scan only."""
    th = np.linspace(0, 2*np.pi, ntheta)
    ze = np.zeros(ntheta)
    R = _fourier_sum_cos(rmnc, xm, xn, th, ze)
    Z = _fourier_sum_sin(zmns, xm, xn, th, ze)
    return th[np.argmin((R - R_target)**2 + (Z - Z_target)**2)]


def run_benchmark(name, vmec_path, mgrid_path, extcur, nfp,
                  sur_idx, nturn, nphi, npart, full_torus=False):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    vmec = Vmec(str(vmec_path))
    R0_vmec = float(vmec.wout.Rmajor_p)

    # RZ start points for ripplepy_pyneo
    RZ_points = []
    for s in sur_idx:
        surf = SurfaceRZFourier.from_wout(str(vmec_path), s)
        rpz = surf.cross_section(phi=0)[0]
        RZ_points.append(rpz[[0, 2]])
    RZ_points = np.asarray(RZ_points)

    # ── pyneo ──
    print("  Running pyneo ...")
    boozer = Boozer(vmec)
    boozer.mpol = 72; boozer.ntor = 36
    boozer.register(sur_idx); boozer.run()
    neoclass = neo.from_simsopt_boozer(boozer)
    ctx = NeoContext(); ctx.set_boozer(neoclass)
    ctx.set_flux_surfaces(neo_surfaces_from_simsopt_boozer(boozer).tolist())
    ctx.set_resolution(theta_n=100, phi_n=100)
    ctx.set_transport_options(npart=npart, multra=1, acc_req=0.01,
        no_bins=100, nstep_per=50, nstep_min=500, nstep_max=5000,
        calc_nstep_max=0)
    ctx.set_switches(ref_swi=2, eout_swi=2, calc_cur=0)
    ctx.set_output_options(write_progress=0, write_output_files=0,
        write_integrate=0, write_diagnostic=0, suppress_file_io=True)
    ctx.setup_grids(); ctx.run_all()
    py_eps = ctx.epstot_profile()

    # ── ripplepy_boozer (Boozer field + analytic trace + ripplepy integrator) ──
    print("  Running ripplepy_boozer (Boozer field, analytic trace, η-state-machine)...")
    booz_dict = _boozer_obj_to_dict(boozer)
    # Pre-compute θ₀ for each surface (fast: ζ=0 only)
    theta0_list = []
    for i, s in enumerate(sur_idx):
        rmnc = np.asarray(booz_dict['rmnc_b'][i], dtype=np.float64)
        zmns = np.asarray(booz_dict['zmns_b'][i], dtype=np.float64)
        xm = np.asarray(booz_dict['ixm_b'], dtype=np.int32)
        xn = np.asarray(booz_dict['ixn_b'], dtype=np.int32)
        th0 = _find_theta0_fast(rmnc, zmns, xm, xn,
                                 RZ_points[i, 0], RZ_points[i, 1])
        theta0_list.append(th0)

    rp_booz = []
    t0 = time.time()
    for i, s in enumerate(sur_idx):
        res = eps_eff_pyneo_style(booz_dict, i, theta0=theta0_list[i],
                                  nzeta=nphi, nturn=nturn, npart=npart)
        rp_booz.append(res['eps_eff'])
    t1 = time.time()
    print(f"  ripplepy_boozer done in {t1-t0:.1f}s")

    # ── ripplepy_pyneo (mgrid field + DLSODE trace + ripplepy integrator) ──
    print("  Running ripplepy_pyneo (mgrid field, DLSODE trace, η-state-machine)...")
    initialize_mgrid_field(mgrid_path, nfp, full_torus=full_torus)
    set_extcur(extcur)
    set_trace_parameters(nturn, nphi, npart=npart, verbose=False)
    rp_mgrid = []
    t0 = time.time()
    for s, rz in zip(sur_idx, RZ_points):
        eps, bnd, ist = compute_epstot(
            rz,
            initial_gradpsi=np.array([1, 0, 0], dtype=np.float64),
            verbose=False,
        )
        rp_mgrid.append(eps if eps is not None else np.nan)
    t1 = time.time()
    print(f"  ripplepy_pyneo done in {t1-t0:.1f}s")

    # ── Print table ──
    print(f"\n  {'s':>6s}  {'pyneo':>12s}  {'rp_booz':>12s}  "
          f"{'booz/py':>8s}  {'rp_mgrid':>12s}  {'mgrid/py':>8s}  "
          f"{'mgrid/booz':>10s}")
    print(f"  {'-'*6}  {'-'*12}  {'-'*12}  {'-'*8}  "
          f"{'-'*12}  {'-'*8}  {'-'*10}")
    for i in range(len(sur_idx)):
        booz_py = rp_booz[i] / py_eps[i] if py_eps[i] != 0 else np.nan
        mgrid_py = rp_mgrid[i] / py_eps[i] if py_eps[i] != 0 else np.nan
        mgrid_booz = rp_mgrid[i] / rp_booz[i] if rp_booz[i] != 0 else np.nan
        print(f"  {sur_idx[i]:6.3f}  {py_eps[i]:12.4e}  {rp_booz[i]:12.4e}  "
              f"{booz_py:8.4f}  {rp_mgrid[i]:12.4e}  {mgrid_py:8.4f}  "
              f"{mgrid_booz:10.4f}")

    # ── Plot ──
    from matplotlib import pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: absolute ε_eff
    ax1.plot(sur_idx, py_eps, 'o-', label='pyneo (Boozer, pyneo trace)')
    ax1.plot(sur_idx, rp_booz, 's--', label='rp_booz (Boozer, analytic θ=θ₀+ιζ)')
    ax1.plot(sur_idx, rp_mgrid, 'x-.', label='rp_mgrid (mgrid, DLSODE trace)')
    ax1.set_xlabel('s'); ax1.set_ylabel('ε_eff')
    ax1.set_title(f'{name}: ε_eff')
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    # Right: ratios
    ax2.plot(sur_idx, np.array(rp_booz)/np.array(py_eps), 's--',
             label='rp_booz / pyneo')
    ax2.plot(sur_idx, np.array(rp_mgrid)/np.array(py_eps), 'x-.',
             label='rp_mgrid / pyneo')
    ax2.axhline(y=1.0, color='gray', ls=':', lw=0.8)
    ax2.set_xlabel('s'); ax2.set_ylabel('ratio to pyneo')
    ax2.set_title(f'{name}: ratio to pyneo')
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    plt.suptitle(f'{name} — three-way benchmark', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'/tmp/bench_boozer_vs_mgrid_{name}.png', dpi=150)
    print(f"  Plot saved to /tmp/bench_boozer_vs_mgrid_{name}.png")
    plt.close()

    return py_eps, rp_booz, rp_mgrid


# ═══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    BASE = "/Users/zkgao/ripplepy"

    # ── CFQS ──
    run_benchmark(
        "CFQS",
        f"{BASE}/tests/test_file/wout_cfqs_test_m10_n5_fixed.nc",
        f"{BASE}/tests/test_file/mgrid_2b40R1mB01.nc",
        None, 2,
        np.linspace(0.1, 1.0, 10),
        nturn=200, nphi=360, npart=50,
        full_torus=False,
    )

    # ── H1 ──
    run_benchmark(
        "H1",
        f"{BASE}/tests/test_file/wout_h1_design.nc",
        f"{BASE}/tests/test_file/mgrid_h1_design.nc",
        [50000, 5000, 1, -80000, -40000], 3,
        np.linspace(0.1, 1.0, 10),
        nturn=200, nphi=360, npart=50,
        full_torus=False,
    )
