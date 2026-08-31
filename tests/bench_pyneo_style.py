#!/usr/bin/env python3
"""Benchmark: ripplepy vs pyneo for CFQS and H1.

Results are cached locally (tests/benchmark_results/) after the first run;
plotting-only reruns read from disk and skip the physics (use --recompute
to force a fresh calculation).
"""
import argparse

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

# Publication-quality plotting (headless-safe; call before importing pyplot)
from ripplepy.plotting import setup_publication_style, save_figure, PUB_COLORS
setup_publication_style()
import matplotlib.pyplot as plt

# Directory where computed benchmark results are cached locally, so that
# re-running for plotting only does not recompute the physics.
RESULT_DIR = Path(__file__).resolve().parent / "benchmark_results"


def _result_paths(name):
    """Return (npz, csv) paths for the cached results of ``name``."""
    stem = RESULT_DIR / f"pyneo_style_{name}"
    return stem.with_suffix(".npz"), stem.with_suffix(".csv")


def _print_table(name, Radius, py_eps, ripplepy):
    print(f"\n  {'R':>6s}  {'pyneo':>12s}  "
          f"{'ripplepy':>12s}  {'new/py':>8s}")
    print(f"  {'-'*6}  {'-'*12}  {'-'*12}  {'-'*8}")
    for i in range(len(Radius)):
        n = ripplepy[i] / py_eps[i] if py_eps[i] != 0 else np.nan
        print(f"  {Radius[i]:6.3f}  {py_eps[i]:12.4e}  "
              f"{ripplepy[i]:12.4e}  {n:8.4f}")


def _save_results(name, Radius, py_eps, ripplepy, iota_at_s):
    """Store computed benchmark results locally as npz (data) + csv (readable)."""
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    npz_path, csv_path = _result_paths(name)
    py_ref = np.asarray(py_eps, dtype=np.float64)
    py_arr = np.asarray(ripplepy, dtype=np.float64)
    radius = np.asarray(Radius, dtype=np.float64)
    iota = np.asarray(iota_at_s, dtype=np.float64)
    ratio = np.where(py_ref != 0.0, py_arr / py_ref, np.nan)
    np.savez(npz_path, Radius=radius, pyneo=py_ref, ripplepy=py_arr, iota=iota)
    np.savetxt(
        csv_path,
        np.column_stack([radius, py_ref, py_arr, ratio, iota]),
        header="R,pyneo,ripplepy,ratio_ripplepy_over_pyneo,iota",
        fmt="%.8e", delimiter=",",
    )
    print(f"[cache] Results saved to {npz_path} and {csv_path}")


def _plot_pyneo_style(name, Radius, py_eps, ripplepy):
    """Publication-quality figure: ε_eff^(3/2)(R) for NEO vs ripplepy."""
    R = np.asarray(Radius)
    py = np.asarray(py_eps)
    rp = np.asarray(ripplepy)

    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    fig.suptitle(f"{name} — field-source-level benchmark")

    ax.plot(R, py, "o-", color=PUB_COLORS["blue"], label="NEO (VMEC + Boozer)")
    ax.plot(R, rp, "s--", color=PUB_COLORS["red"], label="ripplepy (mgrid vacuum)")
    ax.set_yscale("log")
    ax.set_xlabel(r"major radius $R$ (m)")
    ax.set_ylabel(r"$\varepsilon_{\mathrm{eff}}^{3/2}$")
    ax.legend(loc="best")

    fig.tight_layout()
    stem = RESULT_DIR / f"pyneo_style_{name}_benchmark"
    save_figure(fig, str(stem))
    print(f"[plot] Figure saved to {stem}.pdf / {stem}.png")
    plt.close(fig)

def run_benchmark(name, vmec_path,boozer_path, mgrid_path, extcur, nfp,
                  sur_idx, nturn, nphi, npart, full_torus=False,py_old = False,
                  recompute=False):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    # Plotting-only rerun: load cached results and skip the physics.
    if not recompute:
        npz_path, _ = _result_paths(name)
        if npz_path.exists():
            d = np.load(npz_path)
            print(f"[cache] Loading results from {npz_path} "
                  f"(use --recompute to rerun)")
            Radius = d["Radius"]
            py_eps = list(d["pyneo"])
            ripplepy = list(d["ripplepy"])
            _print_table(name, Radius, py_eps, ripplepy)
            _plot_pyneo_style(name, Radius, py_eps, ripplepy)
            return py_eps, ripplepy
    
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

    # Iota profile from VMEC (for the benchmark plot), interpolated onto the
    # same s values as the eps points — the s -> R mapping is already done in
    # RZ_points, so iota shares the Radius axis.
    iota_profile = np.asarray(vmec.wout.iotas)
    s_iota = np.linspace(0.0, 1.0, len(iota_profile))
    iota_at_s = np.interp(sur_idx, s_iota, iota_profile)

    # RZ start points
    RZ_points = []
    for s in sur_idx:
        surf = SurfaceRZFourier.from_wout(str(vmec_path), s)
        rpz = surf.cross_section(phi=0)[0]
        RZ_points.append(rpz[[0, 2]])
    RZ_points = np.asarray(RZ_points)
    Radius = RZ_points[:,0]

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
        fig, ax1 = plt.subplots(figsize=(9, 6))
        ax1.plot(Radius, py_eps, 'o-', label='pyneo')
        ax1.plot(Radius, rp_old, 'x-', label='ripplepy(old)')
        ax1.plot(Radius, ripplepy, 'x-', label='ripplepy (new pyneo-style)')
        ax1.set_xlabel('R'); ax1.set_ylabel('eps_tot', color='C0')
        ax1.set_title(f"{name} Benchmark: eps_tot vs R")
        ax1.grid(True)
        ax2 = ax1.twinx()
        ax2.plot(Radius, iota_at_s, 's--', color='C2', label='iota (VMEC)')
        ax2.set_ylabel('iota', color='C2')
        ax2.tick_params(axis='y', labelcolor='C2')
        lines = ax1.get_lines() + ax2.get_lines()
        ax1.legend(lines, [l.get_label() for l in lines], loc='best')
        fig.tight_layout(); plt.show()

        
        return py_eps,rp_old, ripplepy        
    # ── ripplepy (new pyneo-style) ──

    else:
        _print_table(name, Radius, py_eps, ripplepy)
        _save_results(name, Radius, py_eps, ripplepy, iota_at_s)
        _plot_pyneo_style(name, Radius, py_eps, ripplepy)
        return py_eps, ripplepy


BASE = str(Path(__file__).resolve().parent.parent)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Field-source-level benchmark: ripplepy (mgrid vacuum field) "
                    "vs pyneo (VMEC + Boozer) for the same configuration. Results "
                    "are cached locally after the first run; plotting-only reruns "
                    "read from disk.")
    parser.add_argument("--recompute", action="store_true",
                        help="recompute the physics instead of loading cached results")
    return parser.parse_args()


ARGS = _parse_args()


# ═══════════════════════════════════════════════════════════════
# w7x
# ═══════════════════════════════════════════════════════════════


# run_benchmark(
#     "w7x",
#     f"{BASE}/tests/test_file/wout_w7x_test_m10_n5_fixed.nc",
#     f"{BASE}/tests/test_file/w7x_boozmn.nc",
#     f"{BASE}/tests/test_file/mgrid_w7-x.nc",
#     None, 5,
#     np.linspace(0.1, 1, 11),
#     nturn=400, nphi=360, npart=5000,
#     full_torus=False,
#     recompute=ARGS.recompute,
# )

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
#     nturn=400, nphi=360, npart=5000,
#     full_torus=False,
#     recompute=ARGS.recompute,
# )

# ═══════════════════════════════════════════════════════════════
# H1
# ═══════════════════════════════════════════════════════════════
run_benchmark(
    "H1",
    f"{BASE}/tests/test_file/wout_h1_design.nc",
    f"{BASE}/tests/test_file/h1_boozmn.nc",
    f"{BASE}/tests/test_file/mgrid_h1_design.nc",
    [50000, 5000, 0, -80000, -40000], 3,
    np.linspace(0.1, 1, 11),
    nturn=400, nphi=360, npart=5000,
    full_torus=False,
    recompute=ARGS.recompute,
)