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
    eps_eff_pyneo_style,
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

SURF_IDX_LIST = np.linspace(0.1, 1.0, 11)
NTURN = 200
NPHI = 360
NPART = 5000
CACHE_FIELDLINE = False            # True → skip Fourier summation on reruns
CACHE_DIR = Path(__file__).resolve().parent / "fieldline_cache"

# ═══════════════════════════════════════════════
# Build fieldline_data + geocur from Boozer
# ═══════════════════════════════════════════════

def build_fieldline_from_boozer(booz_dict, surf_idx, theta0, nzeta, nturn):
    """Build fieldline_data(nturn·nzeta, 20) and geocur(:) from Boozer
    Fourier harmonics along θ(φ)=θ₀+ι·φ."""
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
    theta = theta0 + iota * zeta

    xm  = booz_dict["ixm_b"].astype(np.int32)
    xn  = booz_dict["ixn_b"].astype(np.int32)
    bmnc = booz_dict["bmnc_b"][surf_idx, :].astype(np.float64)
    rmnc = booz_dict["rmnc_b"][surf_idx, :].astype(np.float64)
    zmns = booz_dict["zmns_b"][surf_idx, :].astype(np.float64)

    B, dBdt, dBdz, R, dRdt, dRdz, Z, dZdt, dZdz = _sample_fieldline_fourier(
        bmnc, rmnc, zmns, xm, xn, theta, zeta)

    # |∇ψ|
    I_ = float(booz_dict["buco_b"].flat[surf_idx])
    J_ = float(booz_dict["bvco_b"].flat[surf_idx])
    fac = I_ + iota * J_
    gtb  = dRdt**2 + dZdt**2
    gpb  = dRdz**2 + dZdz**2 + R**2
    gtbp = dRdt * dRdz + dZdt * dZdz
    sqrg11 = np.sqrt(np.abs(gtb * gpb - gtbp**2)) * B**2 / fac

    # κ_G  (|∇ψ|·κ_G / |∇ψ|)
    kg_gradpsi = (J_ * dBdz - I_ * dBdt) / fac
    kappa_g = kg_gradpsi / np.maximum(sqrg11, 1e-15)

    # Bφ trick: set to R·|B|² so that Fortran ds = R·|B|/|Bφ|·dφ
    # becomes ds = dφ/|B|, hence ds/|B| = dφ/B² — matching pyneo's measure.
    Bphi = R * B**2

    # Fill Fortran array  (1-based column indices used by Fortran)
    fld = np.zeros((ntot, 20), dtype=np.float64, order="F")
    fld[:, 0] = R          # col  1 – R
    fld[:, 1] = Z          # col  2 – Z
    fld[:, 2] = zeta       # col  3 – phi
    fld[:, 5] = Bphi       # col  6 – Bφ
    fld[:, 6] = B          # col  7 – |B|
    fld[:, 10] = sqrg11    # col 11 – |∇ψ|

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

    # ── Load VMEC + Boozer ──
    print("\n[1] Loading VMEC + Boozer …")
    vmec = Vmec(str(VMEC_PATH))
    boozer = Boozer(vmec)
    boozer.mpol = 72; boozer.ntor = 36
    boozer.register(SURF_IDX_LIST)
    boozer.run()
    booz_dict = _boozer_obj_to_dict(boozer)

    # ── pyneo reference ──
    print("\n[2] Running pyneo reference …")
    neoclass = neo.from_simsopt_boozer(boozer)
    ctx = NeoContext()
    ctx.set_boozer(neoclass)
    ctx.set_flux_surfaces(neo_surfaces_from_simsopt_boozer(boozer).tolist())
    ctx.set_resolution(theta_n=100, phi_n=100)
    ctx.set_transport_options(
        npart=NPART, multra=1, acc_req=0.01, no_bins=100,
        nstep_per=50, nstep_min=500, nstep_max=5000, calc_nstep_max=0,
    )
    ctx.set_switches(ref_swi=2, eout_swi=2, calc_cur=0)
    ctx.set_output_options(
        write_progress=0, write_output_files=0,
        write_integrate=0, write_diagnostic=0, suppress_file_io=True,
    )
    ctx.setup_grids(); ctx.run_all()
    py_eps = ctx.epstot_profile()

    # ── Set npart for Fortran calls ──
    set_trace_parameters(NTURN, NPHI, npart=NPART, verbose=False)

    # ── Boozer → Fortran for each surface ──
    print(f"\n[3] Boozer → Fortran (effective_ripple_pyneo) …")
    eps_bf = []
    xm_all = booz_dict["ixm_b"].astype(np.int32)
    xn_all = booz_dict["ixn_b"].astype(np.int32)

    for k_surf in range(len(SURF_IDX_LIST)):
        bmnc = booz_dict["bmnc_b"][k_surf, :].astype(np.float64)
        theta0, _ = _find_bmax_location(bmnc, xm_all, xn_all)

        fld, geocur = build_fieldline_from_boozer(
            booz_dict, k_surf, theta0, NPHI, NTURN)

        R0  = Effective_Ripple.compute_r0_from_fieldline(fld)
        eps = Effective_Ripple.effective_ripple_pyneo(fld, geocur, R0)
        eps_bf.append(eps)

    # ── Boozer → pure-Python η-state-machine (verified vs pyneo) ──
    print(f"\n[4] Boozer → Python η-state-machine …")
    eps_bp = []
    for k_surf in range(len(SURF_IDX_LIST)):
        r = eps_eff_pyneo_style(booz_dict, k_surf, nzeta=NPHI, nturn=NTURN, npart=NPART)
        eps_bp.append(r["eps_eff_32"])

    # ── Output ──
    print(f"\n  {'─'*60}")
    print(f"  {DEVICE} — ε_eff^(3/2) comparison")
    print(f"  {'─'*60}")
    print(f"  {'s':>8s}  {'pyneo':>12s}  {'booz→f90':>12s}  {'booz→py':>12s}  {'f90/pyneo':>8s}  {'py/pyneo':>8s}")
    print(f"  {'─'*8}  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*8}  {'─'*8}")
    for i, s_val in enumerate(SURF_IDX_LIST):
        r_f90 = eps_bf[i] / py_eps[i]
        r_py  = eps_bp[i] / py_eps[i]
        print(f"  {s_val:8.3f}  {py_eps[i]:12.4e}  "
              f"{eps_bf[i]:12.4e}  {eps_bp[i]:12.4e}  "
              f"{r_f90:8.4f}  {r_py:8.4f}")
    print()


if __name__ == "__main__":
    main()
