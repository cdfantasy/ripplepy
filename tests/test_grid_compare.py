#!/usr/bin/env python3
"""Compare Fourier direct summation vs pyneo 2D spline on the same grid.

Answers: is the ε_eff discrepancy caused by |B|, |∇ψ|, or |∇ψ|·κ_G?
"""

import numpy as np
from pathlib import Path
from simsopt.mhd import Boozer, Vmec
import neo
from neo import NeoContext, neo_surfaces_from_simsopt_boozer
from neo import lowlevel
from ripplepy.boozer_eps_verify import (
    _boozer_obj_to_dict, _find_bmax_location, _sample_fieldline_fourier,
)

BASE = str(Path(__file__).resolve().parent.parent)

DEVICE = "CFQS"
VMEC_PATH = f"{BASE}/tests/test_file/wout_cfqs_test_m10_n5_fixed.nc"
SURF_S = 1
THETA_N = 100
PHI_N = 100

def main():
    print(f"\n{'='*60}")
    print(f"  Grid comparison: Fourier vs pyneo  ({DEVICE}, s={SURF_S})")
    print(f"{'='*60}")

    print("\n[1] Loading …")
    vmec = Vmec(str(VMEC_PATH))
    boozer = Boozer(vmec)
    boozer.mpol = 72; boozer.ntor = 72
    surfs = np.array([SURF_S])
    boozer.register(surfs)
    boozer.run()
    booz_dict = _boozer_obj_to_dict(boozer)
    k_diag = 0

    print(f"\n[2] Running pyneo …")
    neoclass = neo.from_simsopt_boozer(boozer)
    ctx = NeoContext()
    ctx.set_boozer(neoclass)
    ctx.set_flux_surfaces(neo_surfaces_from_simsopt_boozer(boozer).tolist())
    ctx.set_resolution(theta_n=THETA_N, phi_n=PHI_N)
    ctx.set_transport_options(
        npart=100, multra=1, acc_req=0.01, no_bins=100,
        nstep_per=50, nstep_min=500, nstep_max=5000, calc_nstep_max=0)
    ctx.set_switches(ref_swi=2, eout_swi=2, calc_cur=0)
    ctx.set_output_options(
        write_progress=0, write_output_files=0,
        write_integrate=0, write_diagnostic=0, suppress_file_io=True)
    ctx.setup_grids(); ctx.run_all()

    print("\n[3] Extracting pyneo 2D grid …")
    B_py  = lowlevel.get_b(ctx.handle)
    gp_py = lowlevel.get_sqrg11(ctx.handle)
    kg_py = lowlevel.get_kg(ctx.handle)
    ntheta, nphi = B_py.shape

    print(f"[4] Evaluating Fourier at {ntheta}×{nphi} grid …")
    xm = booz_dict["ixm_b"].astype(np.int32)
    xn = booz_dict["ixn_b"].astype(np.int32)
    bmnc = booz_dict["bmnc_b"][k_diag, :].astype(np.float64)
    rmnc = booz_dict["rmnc_b"][k_diag, :].astype(np.float64)
    zmns = booz_dict["zmns_b"][k_diag, :].astype(np.float64)

    # Match pyneo's grid: arange (excludes 2π), not linspace (includes 2π)
    th = np.arange(ntheta, dtype=np.float64) * (2*np.pi / ntheta)
    ph = np.arange(nphi,   dtype=np.float64) * (2*np.pi / nphi)
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    thal = TH.ravel(); phal = PH.ravel()

    pmns = booz_dict.get("pmns_b", None)
    if pmns is not None:
        pmns = pmns[k_diag, :].astype(np.float64)
    result = _sample_fieldline_fourier(
        bmnc, rmnc, zmns, xm, xn, thal, phal, pmns=pmns)
    if pmns is not None:
        B_f, dBdt, dBdz, R, dRdt, dRdz, Z, dZdt, dZdz, Nu, dNdt, dNdz = result
    else:
        B_f, dBdt, dBdz, R, dRdt, dRdz, Z, dZdt, dZdz = result
        Nu = dNdt = dNdz = np.zeros_like(B_f)

    I_ = float(booz_dict["bvco_b"].flat[k_diag])
    J_ = float(booz_dict["buco_b"].flat[k_diag])
    iota = float(booz_dict["iota_b"].flat[k_diag])
    fac = I_ + iota * J_
    gp_f = np.sqrt(np.abs((dRdt**2+dZdt**2+R**2*dNdt**2)*(dRdz**2+dZdz**2+R**2*(1+dNdz)**2)
                        -(dRdt*dRdz+dZdt*dZdz+R**2*dNdt*(1+dNdz))**2)) * B_f**2 / fac
    kg_f = (J_*dBdz - I_*dBdt) / fac

    B_f = B_f.reshape(ntheta, nphi)
    gp_f = gp_f.reshape(ntheta, nphi)
    kg_f = kg_f.reshape(ntheta, nphi)

    dB  = B_f  - B_py
    dgp = gp_f - gp_py
    dkg = kg_f - kg_py

    dB_relative  = dB / B_py
    dgp_relative = dgp / gp_py
    dkg_relative = dkg / kg_py
    py_iota = lowlevel.get_iota_profile(ctx.handle)

    from matplotlib import pyplot as plt
    plt.figure(figsize=(6, 4))
    plt.subplot(1, 3, 1)
    plt.title("|B| Fourier − pyneo")
    plt.pcolormesh(th, ph, dB_relative.T, shading="auto")
    plt.colorbar()
    plt.subplot(1, 3, 2)
    plt.title("|∇ψ| Fourier − pyneo")
    plt.pcolormesh(th, ph, dgp_relative.T, shading="auto")
    plt.colorbar()
    plt.subplot(1, 3, 3)
    plt.title("|∇ψ|·κ_G Fourier − pyneo")
    plt.pcolormesh(th, ph, dkg_relative.T, shading="auto")
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    print(f"\n  {'─'*60}")
    print(f"  Pointwise difference  (Fourier − pyneo)")
    print(f"  {'─'*60}")
    print(f"  {'I =':>10s} {I_:.6f}     {'J =':>8s} {J_:.6f}     {'fac =':>8s} {fac:.6f}")
    print(f"  {'iota_py =':>10s} {py_iota[k_diag]:.6f}     {'iota_our =':>10s} {iota:.6f}")
    print(f"  {'─'*60}")
    print(f"  {'':>14s}  {'rms Δ':>12s}  {'max |Δ|':>12s}  {'pyneo mean':>12s}  {'ratio':>10s}")
    print(f"  {'─'*14}  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*10}")
    for name, dv, pv in [("|B|", dB, B_py), ("|∇ψ|", dgp, gp_py), ("|∇ψ|·κ_G", dkg, kg_py)]:
        rms = np.sqrt(np.mean(dv**2))
        mx = np.max(np.abs(dv))
        mn_py = np.mean(pv)
        mn_fo = np.mean(pv+dv)
        ratio = mn_fo/mn_py if abs(mn_py)>1e-15 else float("nan")
        s = f"  {name:>14s}  {rms:12.3e}  {mx:12.3e}  {mn_py:12.6f}"
        s += f"  {ratio:10.4f}" if abs(mn_py)>1e-15 else f"  {'--':>10s}"
        print(s)
    print()

if __name__ == "__main__":
    main()
