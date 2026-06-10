#!/usr/bin/env python3
"""Verify booz_gauss vs pyneo with matched defaults: theta0=B_max, nturn≥500, rt0² scaling."""
import os, sys
os.chdir('/Users/zkgao/ripplepy')
sys.path.insert(0, 'python')

import numpy as np, time
from simsopt.mhd import Boozer, Vmec
import neo
from neo import NeoContext, neo_surfaces_from_simsopt_boozer
from ripplepy.boozer_eps_verify import (
    eps_eff_from_boozer, _boozer_obj_to_dict,
    sample_fieldline_from_boozer, _find_local_minima,
)

vmec_path = "tests/test_file/wout_ncsx_c09r00_free.nc"
sur_idx = np.array([0.1, 0.3, 0.5])
vmec = Vmec(str(vmec_path))

# ── pyneo ──
boozer = Boozer(vmec); boozer.mpol=72; boozer.ntor=72
boozer.register(sur_idx); boozer.run()
neoclass = neo.from_simsopt_boozer(boozer)
ctx = NeoContext(); ctx.set_boozer(neoclass)
ctx.set_flux_surfaces(neo_surfaces_from_simsopt_boozer(boozer).tolist())
ctx.set_resolution(theta_n=100, phi_n=100)
ctx.set_transport_options(npart=100, multra=1, acc_req=0.01, no_bins=100,
    nstep_per=50, nstep_min=500, nstep_max=2000, calc_nstep_max=0)
ctx.set_switches(ref_swi=2, eout_swi=2, calc_cur=0)
ctx.set_output_options(write_progress=0, write_output_files=0,
    write_integrate=0, write_diagnostic=1, suppress_file_io=False)
ctx.setup_grids(); ctx.run_all()
py_ep = ctx.epstot_profile()

diag = []
with open('diagnostic_add.dat') as f:
    for line in f: diag.append([float(x) for x in line.split()])
diag = np.array(diag)

# ── booz_gauss (NEW defaults: theta0=None, nturn=500, built-in rt0²) ──
booz_dict = _boozer_obj_to_dict(boozer)

print(f"\n{'s':>3s}  {'pyneo':>10s}  {'boozG':>10s}  {'G/py':>7s}  "
      f"{'y2':>9s}  {'e2':>9s}  {'e2r':>6s}  "
      f"{'y2/y3²':>10s}  {'e2/e3²':>10s}  {'Dr':>6s}  "
      f"{'e1':>10s}  {'py_e1':>10s}  {'e1r':>6s}  {'nwells':>6s}")
print("-" * 135)

for s in range(3):
    t0 = time.time()
    rg = eps_eff_from_boozer(booz_dict, s, theta0=None, nzeta=256, nturn=500,
                              n_gauss=64, use_gauss=True)
    dt = time.time() - t0
    
    dr = diag[s]
    py_y2=dr[8]; py_y3=dr[9]; py_coeps=dr[7]
    py_y2y3 = py_y2 / py_y3**2
    heta = (1.0 - dr[4]/dr[5]) / 99.0
    
    e2=rg['e2']; e3=rg['e3']; e2e3 = e2/e3**2
    eps_raw_no_rt0 = (np.pi/(8*np.sqrt(2))) * rg['e1'] * e2e3
    bg_rt0 = rg['eps_eff_32']
    
    bigint_total = py_ep[s] / (py_coeps * py_y2y3) if py_coeps*py_y2y3>1e-30 else np.nan
    py_e1 = heta * bigint_total
    e1 = rg['e1']
    
    fl = sample_fieldline_from_boozer(booz_dict, s, nzeta=256, nturn=500)
    nwells = len(_find_local_minima(fl.B)) - 1
    
    print(f"  {s:2d}  {py_ep[s]:10.4e}  {bg_rt0:10.4e}  {bg_rt0/py_ep[s]:7.2f}  "
          f"{py_y2:9.1f}  {e2:9.1f}  {e2/py_y2:6.3f}  "
          f"{py_y2y3:10.4e}  {e2e3:10.4e}  {e2e3/py_y2y3:6.3f}  "
          f"{e1:10.4e}  {py_e1:10.4e}  {e1/py_e1:6.2f}  {nwells:6d}  "
          f"[{dt:.0f}s]")
