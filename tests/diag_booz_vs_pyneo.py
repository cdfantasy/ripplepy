#!/usr/bin/env python3
"""Systematic comparison: booz_gauss vs pyneo for NCSX.
Checks e2 vs y2, e3 vs y3, e1 vs bigint*heta, well counts, bp/η ranges."""
import os, sys
os.chdir('/Users/zkgao/ripplepy')
sys.path.insert(0, 'python')

import numpy as np
from simsopt.mhd import Boozer, Vmec
import neo
from neo import NeoContext, neo_surfaces_from_simsopt_boozer
from ripplepy.boozer_eps_verify import (
    eps_eff_from_boozer, _boozer_obj_to_dict,
    sample_fieldline_from_boozer, _find_local_minima,
    _integrate_bounce_segment,
)

# ── config ──
vmec_path = "tests/test_file/wout_ncsx_c09r00_free.nc"
sur_idx_list = np.linspace(0.1, 0.5, 11)
nfp = 3

vmec = Vmec(str(vmec_path))
R0 = float(vmec.wout.Rmajor_p)

# ═══════════════════════════════════════════════════
# pyneo (with diagnostics)
# ═══════════════════════════════════════════════════
boozer = Boozer(vmec); boozer.mpol=72; boozer.ntor=72
boozer.register(sur_idx_list); boozer.run()

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
    write_integrate=0, write_diagnostic=1, suppress_file_io=False)
ctx.setup_grids()
ctx.run_all()
py_epstot = ctx.epstot_profile()

# Read diagnostic_add.dat for each surface
# Format: psi_ind, istepc, npart, max_class, b_min, b_max, bmref, coeps, y2, y3
diag_data = []
with open('diagnostic_add.dat') as f:
    for line in f:
        parts = line.split()
        diag_data.append([float(p) for p in parts])
diag_data = np.array(diag_data)

# ═══════════════════════════════════════════════════
# booz_gauss
# ═══════════════════════════════════════════════════
booz_dict = _boozer_obj_to_dict(boozer)

print(f"{'s':>4s}  {'pyneo':>10s}  {'boozG':>10s}  {'G/py':>7s}  "
      f"{'y2':>10s}  {'e2':>10s}  {'e2r':>7s}  "
      f"{'y2/y3²':>10s}  {'e2/e3²':>10s}  {'Dr':>7s}  "
      f"{'nwells':>7s}  {'Bmod%':>7s}  {'rt0²':>7s}")
print("-" * 120)

for s in range(len(sur_idx_list)):
    # pyneo data
    py = py_epstot[s]
    drow = diag_data[s] if s < len(diag_data) else None
    if drow is not None:
        py_y2 = drow[8]; py_y3 = drow[9]
        py_bmin = drow[4]; py_bmax = drow[5]
        py_bmref = drow[6]; py_coeps = drow[7]
        py_y2y3sq = py_y2 / py_y3**2
    else:
        py_y2 = py_y3 = py_y2y3sq = np.nan; py_bmin=py_bmax=np.nan
    
    # booz_gauss
    rg = eps_eff_from_boozer(booz_dict, s, theta0=0.0, nzeta=256, nturn=32,
                              n_gauss=64, use_gauss=True, return_debug=False)
    bg = rg['eps_eff_32'] * R0**2
    e2 = rg['e2']; e3 = rg['e3']; b0 = rg['b0']; bmin = rg['bmin']
    e2e3sq = e2 / e3**2
    
    # intermediate quantities
    fl = sample_fieldline_from_boozer(booz_dict, s, theta0=0.0, nzeta=256, nturn=32)
    minima = _find_local_minima(fl.B)
    nwells = len(minima) - 1
    bmod_pct = 100 * (fl.B.max() - fl.B.min()) / fl.B.mean()
    
    # rt0 for this surface
    rmnc0 = np.asarray(booz_dict['rmnc_b'][s])
    xm = np.asarray(booz_dict['ixm_b'], dtype=np.int32)
    xn = np.asarray(booz_dict['ixn_b'], dtype=np.int32)
    m0_idx = np.where((xm == 0) & (xn == 0))[0]
    rt0_s = rmnc0[m0_idx[0]] if len(m0_idx) > 0 else np.nan
    
    print(f"  {s:2d}  {py:10.4e}  {bg:10.4e}  {bg/py:7.2f}  "
          f"{py_y2:10.2f}  {e2:10.2f}  {e2/py_y2:7.3f}  "
          f"{py_y2y3sq:10.4e}  {e2e3sq:10.4e}  {e2e3sq/py_y2y3sq:7.3f}  "
          f"{nwells:7d}  {bmod_pct:7.2f}  {rt0_s**2:7.3f}")

# Extra: check ratio of e1 to pyneo equivalent
print(f"\n── Ratios ──")
for s in range(len(sur_idx_list)):
    rg = eps_eff_from_boozer(booz_dict, s, theta0=0.0, nzeta=256, nturn=32,
                              n_gauss=64, use_gauss=True)
    # pyneo epstot = π·rt0²·Δη/(8√2) · Σ H²/I · y2/y3²
    # booz_raw    = π/(8√2) · e1 · e2/e3²
    # If e2/e3² = y2/y3², then:
    # bg_raw/py = e1 / (rt0²·Δη·Σ H²/I)
    drow = diag_data[s]
    py_coeps = drow[7]
    py_y2 = drow[8]; py_y3 = drow[9]
    py_bmin = drow[4]; py_bmax = drow[5]
    py_bmref = drow[6]
    heta = (1.0 - py_bmin/py_bmax) / 99.0
    rt0_s = np.nan  # compute below
    rmnc0 = np.asarray(booz_dict['rmnc_b'][s])
    xm = np.asarray(booz_dict['ixm_b'], dtype=np.int32)
    xn = np.asarray(booz_dict['ixn_b'], dtype=np.int32)
    m0_idx = np.where((xm == 0) & (xn == 0))[0]
    rt0_s = rmnc0[m0_idx[0]] if len(m0_idx) > 0 else np.nan
    
    # pyneo: epstot = coeps * bigint * y2/y3² where coeps = π·rt0²·Δη/(8√2)
    # So bigint_total = py_epstot / (coeps * y2/y3²)
    bigint_total = py_epstot[s] / (py_coeps * py_y2/py_y3**2)
    
    # booz: eps_raw = π/(8√2) * e1 * e2/e3²
    # So e1 = eps_raw * 8√2/π / (e2/e3²)
    eps_raw = rg['eps_eff_32']
    e1 = eps_raw * 8*np.sqrt(2)/np.pi / (rg['e2']/rg['e3']**2)
    
    # In pyneo: Σ Δη · Σ H²/I → Δη times the η-sum of H²/I
    # bigint_total = Σ_η Σ_bounce H²/I (without Δη)
    # So pyneo equivalent of e1 = heta * bigint_total
    py_e1 = heta * bigint_total
    
    print(f"  s={s}: e1={e1:.4e}  py_e1(heta*bigint)={py_e1:.4e}  ratio={e1/py_e1:.2f}  "
          f"heta={heta:.6f}  bigint_total={bigint_total:.2f}")
