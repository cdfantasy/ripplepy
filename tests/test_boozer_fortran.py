#!/usr/bin/env python3
"""Test: Boozer analytic field line → Fortran compute_r0 + effective_ripple_pyneo.

Computes ε_eff from Boozer Fourier harmonics using the same Fortran
η-state-machine as the mgrid pipeline.  Compares with pyneo's native result.

The field line is θ(φ)=θ₀+ι·φ, sampled analytically via Fourier summation.
|B|, |∇ψ|, κ_G, R, Z, Bφ are all evaluated from the Boozer harmonics
without any grid interpolation or field-line tracing.

Set CACHE_FIELDLINE = True to skip Fourier summation on reruns.
"""

import numpy as np
from pathlib import Path
from simsopt.mhd import Boozer, Vmec
import neo
from neo import NeoContext, neo_surfaces_from_simsopt_boozer
from ripplepy.boozer_eps_verify import (
    _boozer_obj_to_dict, _find_bmax_location, _sample_fieldline_fourier,
    eps_eff_pyneo_ode_fast,
)
from ripplepy.ripple import Effective_Ripple, set_trace_parameters

BASE = str(Path(__file__).resolve().parent.parent)

# ═══════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════

DEVICE = "CFQS"
VMEC_PATH = f"{BASE}/tests/test_file/wout_cfqs_test_m10_n5_fixed.nc"

# DEVICE = "H1"
# VMEC_PATH = f"{BASE}/tests/test_file/wout_h1_design.nc"


SURF_IDX_LIST = np.linspace(0.1, 1.0, 10)
NTURN = 100
NPHI = 100    # Fortran trapezoidal: grid pts per turn (match 4 × nstep_per)
NPART = 500
COMPARE_PYTHON = True            # True → also run Python η-state-machine + diagnostics
CACHE_FIELDLINE = False
CACHE_DIR = Path(__file__).resolve().parent / "fieldline_cache"

# ═══════════════════════════════════════════════
# Build fieldline_data + geocur from Boozer
# ═══════════════════════════════════════════════

def build_fieldline_from_boozer(booz_dict, surf_idx, theta0, nzeta, nturn, phi0=0.0):
    s_val = booz_dict.get("_compute_surfs", None)
    s_label = f"{s_val[surf_idx]:06.3f}" if s_val is not None else f"{surf_idx:06.3f}"
    cache_path = CACHE_DIR / f"{DEVICE}_s{s_label}.npz"
    if CACHE_FIELDLINE and cache_path.exists():
        data = np.load(cache_path)
        return data["fld"], data["geocur"]

    iota = float(booz_dict["iota_b"].flat[surf_idx])
    ntot = nzeta * nturn
    dphi = 2.0 * np.pi / nzeta
    zeta = np.arange(ntot, dtype=np.float64) * dphi
    theta = theta0 + iota * (zeta - phi0)

    xm  = booz_dict["ixm_b"].astype(np.int32)
    xn  = booz_dict["ixn_b"].astype(np.int32)
    bmnc = booz_dict["bmnc_b"][surf_idx, :].astype(np.float64)
    rmnc = booz_dict["rmnc_b"][surf_idx, :].astype(np.float64)
    zmns = booz_dict["zmns_b"][surf_idx, :].astype(np.float64)
    pmns = booz_dict.get("pmns_b", None)
    if pmns is not None:
        pmns = pmns[surf_idx, :].astype(np.float64)

    result = _sample_fieldline_fourier(
        bmnc, rmnc, zmns, xm, xn, theta, zeta, pmns=pmns)
    if pmns is not None:
        B, dBdt, dBdz, R, dRdt, dRdz, Z, dZdt, dZdz, Nu, dNdt, dNdz = result
    else:
        B, dBdt, dBdz, R, dRdt, dRdz, Z, dZdt, dZdz = result
        Nu = dNdt = dNdz = np.zeros_like(B)
    nfp = int(booz_dict.get('nfp_b', booz_dict.get('nfp', 1)))
    dNdt_nrm = dNdt * (2.0 * np.pi / nfp)
    dNdz_nrm = 1.0 + dNdz * (2.0 * np.pi / nfp)

    I_ = float(booz_dict["bvco_b"].flat[surf_idx])  # curr_pol
    J_ = float(booz_dict["buco_b"].flat[surf_idx])  # curr_tor
    fac = I_ + iota * J_
    gtb  = dRdt**2 + dZdt**2 + R**2 * dNdt_nrm**2
    gpb  = dRdz**2 + dZdz**2 + R**2 * dNdz_nrm**2
    gtbp = dRdt*dRdz + dZdt*dZdz + R**2 * dNdt_nrm * dNdz_nrm
    sqrg11 = np.sqrt(np.abs(gtb * gpb - gtbp**2)) * B**2 / fac
    kg_gradpsi = (J_ * dBdz - I_ * dBdt) / fac
    kappa_g = kg_gradpsi / np.maximum(sqrg11, 1e-15)
    Bphi = R * B**2

    fld = np.zeros((ntot, 20), dtype=np.float64, order="F")
    fld[:, 0] = R; fld[:, 1] = Z; fld[:, 2] = zeta
    fld[:, 5] = Bphi; fld[:, 6] = B; fld[:, 10] = sqrg11
    geocur = np.asfortranarray(kappa_g.astype(np.float64))

    if CACHE_FIELDLINE:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, fld=fld, geocur=geocur)
    return fld, geocur

# ═══════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════

def main():
    print(f"\n{'='*60}")
    print(f"  {DEVICE} — Boozer → Fortran ε_eff")
    print(f"{'='*60}")

    print("\n[1] Loading VMEC + Boozer …")
    vmec = Vmec(str(VMEC_PATH))
    boozer = Boozer(vmec)
    boozer.mpol = 72; boozer.ntor = 36
    boozer.register(SURF_IDX_LIST)
    boozer.run()
    booz_dict = _boozer_obj_to_dict(boozer)

    print("\n[2] Running pyneo reference …")
    neoclass = neo.from_simsopt_boozer(boozer)
    ctx = NeoContext()
    ctx.set_boozer(neoclass)
    ctx.set_flux_surfaces(neo_surfaces_from_simsopt_boozer(boozer).tolist())
    ctx.set_resolution(theta_n=100, phi_n=100)
    ctx.set_transport_options(
        npart=NPART, multra=1, acc_req=0.01, no_bins=100,
        nstep_per=50, nstep_min=500, nstep_max=5000, calc_nstep_max=0)
    ctx.set_switches(ref_swi=2, eout_swi=2, calc_cur=0)
    ctx.set_output_options(
        write_progress=0, write_output_files=0,
        write_integrate=0, write_diagnostic=0, suppress_file_io=True)
    ctx.setup_grids(); ctx.run_all()
    py_eps = ctx.epstot_profile()

    set_trace_parameters(NTURN, NPHI, npart=NPART, verbose=False)
    k_diag = np.argmin(np.abs(np.asarray(SURF_IDX_LIST) - 0.5))

    print(f"\n[3] Boozer → Fortran (effective_ripple_pyneo) …")
    eps_bf = []; debug_fld = None; debug_geo = None
    xm_all = booz_dict["ixm_b"].astype(np.int32)
    xn_all = booz_dict["ixn_b"].astype(np.int32)
    for k_surf in range(len(SURF_IDX_LIST)):
        bmnc = booz_dict["bmnc_b"][k_surf, :].astype(np.float64)
        theta0, phi0 = _find_bmax_location(bmnc, xm_all, xn_all)
        fld, geocur = build_fieldline_from_boozer(
            booz_dict, k_surf, theta0, NPHI, NTURN, phi0=phi0)
        R0 = Effective_Ripple.compute_r0_from_fieldline(fld)
        eps = Effective_Ripple.effective_ripple_pyneo(fld, geocur, R0)
        eps_bf.append(eps)
        if k_surf == k_diag:
            debug_fld = fld; debug_geo = geocur

    if COMPARE_PYTHON:
        print(f"\n[4] Boozer → Python η-state-machine (npart={NPART}) …")
        eps_bp = []; debug_r = None
        for k_surf in range(len(SURF_IDX_LIST)):
            r = eps_eff_pyneo_ode_fast(booz_dict, k_surf, nturn=NTURN, npart=NPART, nstep_per=50)
            eps_bp.append(r["eps_eff_32"])
            if k_surf == k_diag:
                debug_r = r

        # (debug diagnostic block removed — past debugging phase)

    else:
        eps_bp = [np.nan] * len(SURF_IDX_LIST)

    print(f"\n  {'─'*60}")
    print(f"  {DEVICE} — ε_eff^(3/2) comparison")
    print(f"  {'─'*60}")
    if COMPARE_PYTHON:
        print(f"  {'s':>8s}  {'pyneo':>12s}  {'booz→f90':>12s}  {'booz→py':>12s}  {'f90/pyneo':>9s}  {'py/pyneo':>9s}  {'f90/py':>9s}")
        print(f"  {'─'*8}  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*9}  {'─'*9}  {'─'*9}")
        for i, s_val in enumerate(SURF_IDX_LIST):
            r_f90 = eps_bf[i]/py_eps[i]; r_py = eps_bp[i]/py_eps[i]
            r_f90_py = eps_bf[i]/eps_bp[i]
            print(f"  {s_val:8.3f}  {py_eps[i]:12.4e}  "
                  f"{eps_bf[i]:12.4e}  {eps_bp[i]:12.4e}  "
                  f"{r_f90:9.4f}  {r_py:9.4f}  {r_f90_py:9.4f}")
    else:
        print(f"  {'s':>8s}  {'pyneo':>12s}  {'booz→f90':>12s}  {'f90/pyneo':>8s}")
        print(f"  {'─'*8}  {'─'*12}  {'─'*12}  {'─'*8}")
        for i, s_val in enumerate(SURF_IDX_LIST):
            r_f90 = eps_bf[i]/py_eps[i]
            print(f"  {s_val:8.3f}  {py_eps[i]:12.4e}  "
                  f"{eps_bf[i]:12.4e}  {r_f90:8.4f}")
    print()

if __name__ == "__main__":
    main()
