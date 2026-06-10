#!/usr/bin/env python3
"""Quick diagnostic: booz_gauss vs pyneo for 3 NCSX surfaces."""
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
)

vmec_path = "tests/test_file/wout_ncsx_c09r00_free.nc"
sur_idx = np.array([0.1, 0.3, 0.5])
vmec = Vmec(str(vmec_path)); R0 = float(vmec.wout.Rmajor_p)

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

# Read pyneo internal data
diag = []
with open('diagnostic_add.dat') as f:
    for line in f: diag.append([float(x) for x in line.split()])
diag = np.array(diag)

# ── booz_gauss ──
booz_dict = _boozer_obj_to_dict(boozer)
xm = np.asarray(booz_dict['ixm_b'], dtype=np.int32)
xn = np.asarray(booz_dict['ixn_b'], dtype=np.int32)

for s in range(3):
    py = py_ep[s]; dr = diag[s]
    py_y2=dr[8]; py_y3=dr[9]; py_bmin=dr[4]; py_bmax=dr[5]
    py_coeps=dr[7]; py_bmref=dr[6]
    heta = (1.0 - py_bmin/py_bmax) / 99.0
    
    rg = eps_eff_from_boozer(booz_dict, s, theta0=0.0, nzeta=256, nturn=32,
                              n_gauss=64, use_gauss=True)
    bg = rg['eps_eff_32'] * R0**2
    e2=rg['e2']; e3=rg['e3']; e2e3 = e2/e3**2
    py_y2y3 = py_y2/py_y3**2
    
    # e1 vs py equivalent
    eps_raw = rg['eps_eff_32']
    e1 = eps_raw * 8*np.sqrt(2)/np.pi / e2e3
    bigint_total = py / (py_coeps * py_y2y3) if py_coeps*py_y2y3 > 1e-30 else np.nan
    py_e1 = heta * bigint_total
    
    # Well count
    fl = sample_fieldline_from_boozer(booz_dict, s, theta0=0.0, nzeta=256, nturn=32)
    nwells = len(_find_local_minima(fl.B)) - 1
    bmod = 100*(fl.B.max()-fl.B.min())/fl.B.mean()
    
    # rt0
    rmnc = np.asarray(booz_dict['rmnc_b'][s])
    m0 = np.where((xm==0)&(xn==0))[0]
    rt0 = rmnc[m0[0]] if len(m0)>0 else np.nan
    
    print(f"\ns={s}: py={py:.4e}  boozG={bg:.4e}  ratio={bg/py:.2f}")
    print(f"  B: [{py_bmin:.3f},{py_bmax:.3f}]  mod={bmod:.2f}%  wells={nwells}")
    print(f"  y2={py_y2:.1f}  e2={e2:.1f}  r={e2/py_y2:.3f}  |  y3={py_y3:.2f}  e3={e3:.2f}")
    print(f"  y2/y3²={py_y2y3:.4e}  e2/e3²={e2e3:.4e}  r={e2e3/py_y2y3:.3f}")
    print(f"  e1={e1:.4e}  py_e1={py_e1:.4e}  r={e1/py_e1:.2f}")
    print(f"  heta={heta:.6f}  bigint_total={bigint_total:.2f}  coeps={py_coeps:.6f}  rt0²={rt0**2:.3f}")
    print(f"  Δη*bigint={heta*bigint_total:.4e}  e1/(Δη*bigint)={e1/(heta*bigint_total):.2f}")
