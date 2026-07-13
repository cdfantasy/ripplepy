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


SURF_IDX_LIST = np.linspace(0.1, 1.0, 10)
NTURN = 200
NPHI = 360
NPART = 5000
NPART_PY = 500
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
        npart=NPART_PY, multra=1, acc_req=0.01, no_bins=100,
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
        print(f"\n[4] Boozer → Python η-state-machine (npart={NPART_PY}) …")
        eps_bp = []; debug_r = None
        for k_surf in range(len(SURF_IDX_LIST)):
            r = eps_eff_pyneo_style(booz_dict, k_surf, nzeta=NPHI, nturn=NTURN, npart=NPART_PY)
            eps_bp.append(r["eps_eff_32"])
            if k_surf == k_diag:
                debug_r = r

        if debug_r is not None:
            s_diag = SURF_IDX_LIST[k_diag]
            coeps = np.pi*debug_r["rt0_squared"]*debug_r["heta"]/(8.0*np.sqrt(2.0))
            eps_recomp = coeps*debug_r["bigint_total"]*debug_r["e2"]/debug_r["e3"]**2
            print(f"\n  {'─'*60}")
            print(f"  Diagnostic breakdown  (s={s_diag:.3f})")
            print(f"  {'─'*60}")
            print(f"  {'':>20s}  {'booz→py':>14s}  {'pyneo':>14s}  {'ratio':>10s}")
            print(f"  {'─'*20}  {'─'*14}  {'─'*14}  {'─'*10}")
            print(f"  {'ε_eff^(3/2)':>20s}  {debug_r['eps_eff_32']:14.4e}  "
                  f"{py_eps[k_diag]:14.4e}  {debug_r['eps_eff_32']/py_eps[k_diag]:10.4f}")
            print(f"  {'bmin':>20s}  {debug_r['bmin']:14.6f}")
            print(f"  {'b0 (B_max)':>20s}  {debug_r['b0']:14.6f}")
            print(f"  {'heta':>20s}  {debug_r['heta']:14.6e}")
            print(f"  {'rt0²':>20s}  {debug_r['rt0_squared']:14.6f}")
            print(f"  {'coeps':>20s}  {coeps:14.6e}")
            print(f"  {'e2 = ∫dφ/B²':>20s}  {debug_r['e2']:14.6e}")
            print(f"  {'e3 = ∫dφ·|∇ψ|/B²':>20s}  {debug_r['e3']:14.6e}")
            print(f"  {'e2/e3²':>20s}  {debug_r['e2']/debug_r['e3']**2:14.6e}")
            print(f"  {'bigint':>20s}  {debug_r['bigint_total']:14.6e}")
            print(f"  {'coeps·bigint·e2/e3²':>20s}  {eps_recomp:14.6e}")

            # ── Python η-state-machine on SAME data as Fortran ──
            if debug_fld is not None and debug_geo is not None:
                B_s = debug_fld[:, 6]
                gp_s = debug_fld[:, 10]
                kg_s = debug_geo * gp_s    # |∇ψ|·κ_G
                dphi = 2.0 * np.pi / NPHI
                b0 = np.max(B_s)
                bmin = np.min(B_s)
                invB2 = 1.0 / B_s**2
                e2_s = np.sum(invB2) * dphi
                e3_s = np.sum(invB2 * gp_s) * dphi
                etamin = bmin / b0
                heta = (1.0 - etamin) / (NPART_PY - 1)
                eta_vals = etamin + heta/2.0 + np.arange(NPART_PY) * heta
                xm_s = booz_dict["ixm_b"].astype(np.int32)
                xn_s = booz_dict["ixn_b"].astype(np.int32)
                rmnc_s = booz_dict["rmnc_b"][k_diag,:].astype(np.float64)
                m0 = np.where((xm_s == 0) & (xn_s == 0))[0]
                rt0_s = float(rmnc_s[m0[0]]) if len(m0) > 0 else 1.0
                coeps_s = np.pi * rt0_s**2 * heta / (8.0 * np.sqrt(2.0))
                bra_s = B_s / b0
                sqeta_s = np.sqrt(eta_vals)
                bigint_s = 0.0
                for i_eta, eta in enumerate(eta_vals):
                    isw = 0; iswst = 0; I_acc = 0.0; H_acc = 0.0
                    for k in range(len(B_s)):
                        bra = bra_s[k]
                        subsq = 1.0 - bra / eta
                        if subsq > 0.0:
                            isw = 1
                            sq = np.sqrt(subsq) * invB2[k]
                            I_acc += sq * dphi
                            H_acc += sq * (4.0/bra - 1.0/eta) * kg_s[k] / sqeta_s[i_eta] * dphi
                        elif isw == 1:
                            if I_acc > 1e-15:
                                bigint_s += H_acc*H_acc/I_acc * iswst
                            iswst = 1; I_acc = 0.0; H_acc = 0.0; isw = 0
                eps_same = coeps_s * bigint_s * e2_s / e3_s**2
                print(f"  {'booz→py(same data)':>20s}  {eps_same:14.4e}")
                print(f"  {'booz→f90(diag)':>20s}  {eps_bf[k_diag]:14.4e}  {'ratio':>10s}  {eps_same/eps_bf[k_diag]:10.4f}")

            try:
                from neo import lowlevel
                B_py  = lowlevel.get_b(ctx.handle)
                gp_py = lowlevel.get_sqrg11(ctx.handle)
                kg_py = lowlevel.get_kg(ctx.handle)
                print(f"\n  {'─'*60}")
                print(f"  pyneo 2D-grid values ({B_py.shape[0]}×{B_py.shape[1]})")
                print(f"  {'─'*60}")
                print(f"  {'':>14s}  {'min':>12s}  {'max':>12s}  {'mean':>12s}")
                print(f"  {'─'*14}  {'─'*12}  {'─'*12}  {'─'*12}")
                print(f"  {'|B|':>14s}  {np.min(B_py):12.6f}  {np.max(B_py):12.6f}  {np.mean(B_py):12.6f}")
                print(f"  {'|∇ψ|':>14s}  {np.min(gp_py):12.6f}  {np.max(gp_py):12.6f}  {np.mean(gp_py):12.6f}")
                print(f"  {'|∇ψ|·κ_G':>14s}  {np.min(kg_py):12.6f}  {np.max(kg_py):12.6f}  {np.mean(kg_py):12.6f}")
                our_kg = debug_geo*debug_fld[:,10]
                our_gp = debug_fld[:,10]; our_B = debug_fld[:,6]
                print(f"\n  {'our |B| (fld-line)':>14s}  {np.min(our_B):12.6f}  "
                      f"{np.max(our_B):12.6f}  {np.mean(our_B):12.6f}")
                print(f"  {'our |∇ψ|':>14s}  {np.min(our_gp):12.6f}  "
                      f"{np.max(our_gp):12.6f}  {np.mean(our_gp):12.6f}")
                print(f"  {'our |∇ψ|·κ_G':>14s}  {np.min(our_kg):12.6f}  "
                      f"{np.max(our_kg):12.6f}  {np.mean(our_kg):12.6f}")

                ntheta, nphi = B_py.shape
                th_g = np.linspace(0, 2*np.pi, ntheta)
                ph_g = np.linspace(0, 2*np.pi, nphi)
                TH, PH = np.meshgrid(th_g, ph_g, indexing="ij")
                B_our, _, _, _, _, _, _, _, _ = _sample_fieldline_fourier(
                    booz_dict["bmnc_b"][k_diag,:].astype(np.float64),
                    booz_dict["rmnc_b"][k_diag,:].astype(np.float64),
                    booz_dict["zmns_b"][k_diag,:].astype(np.float64),
                    xm_all, xn_all, TH.ravel(), PH.ravel())
                B_our = B_our.reshape(ntheta, nphi)
                dB = B_our - B_py
                Il = float(booz_dict["bvco_b"].flat[k_diag])  # curr_pol
                Jl = float(booz_dict["buco_b"].flat[k_diag])  # curr_tor
                iota = float(booz_dict["iota_b"].flat[k_diag])
                fac = Il + iota * Jl
                _, _, _, R, dRdt, dRdz, Z, dZdt, dZdz = _sample_fieldline_fourier(
                    booz_dict["bmnc_b"][k_diag,:].astype(np.float64),
                    booz_dict["rmnc_b"][k_diag,:].astype(np.float64),
                    booz_dict["zmns_b"][k_diag,:].astype(np.float64),
                    xm_all, xn_all, TH.ravel(), PH.ravel())
                gp_our = np.sqrt(np.abs(dRdt**2+dZdt**2)*(dRdz**2+dZdz**2+R**2)-(dRdt*dRdz+dZdt*dZdz)**2)
                gp_our = gp_our * B_our.ravel()**2 / fac
                gp_our = gp_our.reshape(ntheta, nphi)
                _, dBdt, dBdz = _sample_fieldline_fourier(
                    booz_dict["bmnc_b"][k_diag,:].astype(np.float64),
                    booz_dict["rmnc_b"][k_diag,:].astype(np.float64),
                    booz_dict["zmns_b"][k_diag,:].astype(np.float64),
                    xm_all, xn_all, TH.ravel(), PH.ravel())[:3]
                kg_our = (Jl*dBdz - Il*dBdt)/fac
                kg_our = kg_our.reshape(ntheta, nphi)
                print(f"\n  {'─'*60}")
                print(f"  Pointwise diff on grid  (Fourier − pyneo)")
                print(f"  {'─'*60}")
                print(f"  {'I,J,fac':>20s}  I={Il:.6f}  J={Jl:.6f}  fac={fac:.6f}")
                print(f"  {'|B| rms/max Δ':>20s}  {np.sqrt(np.mean(dB**2)):12.3e}  {np.max(np.abs(dB)):12.3e}")
                print(f"  {'|∇ψ| rms/max Δ':>20s}  {np.sqrt(np.mean((gp_our-gp_py)**2)):12.3e}  {np.max(np.abs(gp_our-gp_py)):12.3e}")
                print(f"  {'|∇ψ|·κ_G rms/max Δ':>20s}  {np.sqrt(np.mean((kg_our-kg_py)**2)):12.3e}  {np.max(np.abs(kg_our-kg_py)):12.3e}")

                if debug_fld is not None:
                    n_samp = 10
                    step = max(1, len(debug_fld)//n_samp)
                    print(f"\n  {'─'*60}")
                    print(f"  Field-line samples (every {step} pts)")
                    print(f"  {'─'*60}")
                    print(f"  {'idx':>6s}  {'|B|(T)':>10s}  {'|∇ψ|':>10s}  {'|∇ψ|κ_G':>12s}  {'κ_G':>10s}")
                    print(f"  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*12}  {'─'*10}")
                    for j in range(n_samp):
                        i = j*step
                        b_val = debug_fld[i,6]; gp_val = debug_fld[i,10]
                        kg_val = debug_geo[i]; kg_gp_val = gp_val*kg_val
                        print(f"  {i:6d}  {b_val:10.6f}  {gp_val:10.6f}  "
                              f"{kg_gp_val:12.6f}  {kg_val:10.6f}")
            except Exception as exc:
                print(f"  (pyneo lowlevel comparison skipped: {exc})")
            print()

    else:
        eps_bp = [np.nan] * len(SURF_IDX_LIST)

    print(f"\n  {'─'*60}")
    print(f"  {DEVICE} — ε_eff^(3/2) comparison")
    print(f"  {'─'*60}")
    if COMPARE_PYTHON:
        print(f"  {'s':>8s}  {'pyneo':>12s}  {'booz→f90':>12s}  {'booz→py':>12s}  {'f90/pyneo':>8s}  {'py/pyneo':>8s}")
        print(f"  {'─'*8}  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*8}  {'─'*8}")
        for i, s_val in enumerate(SURF_IDX_LIST):
            r_f90 = eps_bf[i]/py_eps[i]; r_py = eps_bp[i]/py_eps[i]
            print(f"  {s_val:8.3f}  {py_eps[i]:12.4e}  "
                  f"{eps_bf[i]:12.4e}  {eps_bp[i]:12.4e}  "
                  f"{r_f90:8.4f}  {r_py:8.4f}")
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
