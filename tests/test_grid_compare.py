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
SURF_S = 0.5
THETA_N = 100
PHI_N = 100

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
# py_iota_profile = lowlevel.get_iota_profile(ctx.handle)
# iota = py_iota_profile[k_diag]
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

# Surface-mapping diagnostic: compare bmnc and iota
py_iota_profile = lowlevel.get_iota_profile(ctx.handle)
bmnc_py = np.asarray(getattr(neoclass, 'bmnc', None)) if hasattr(neoclass, 'bmnc') else None
iota_py_full = np.asarray(getattr(neoclass, 'iota', None)) if hasattr(neoclass, 'iota') else None
cs = int(booz_dict.get('_compute_surfs', [0])[k_diag])
print(f"\n  {'─'*60}")
print(f"  Surface-mapping check  (compute_surfs[{k_diag}] = {cs})")
print(f"  {'─'*60}")
print(f"  {'iota pyneo (profile)':>20s} = {py_iota_profile[k_diag]:.6f}")
if iota_py_full is not None and cs < len(iota_py_full):
    print(f"  {'iota pyneo (full)':>20s} = {iota_py_full[cs]:.6f}")
print(f"  {'iota ours (bx.iota_b)':>20s} = {iota:.6f}")
if bmnc_py is not None and cs < bmnc_py.shape[0]:
    print(f"  {'bmnc[0:3] pyneo':>20s} = {bmnc_py[cs, :3]}")
    print(f"  {'bmnc[0:3] ours':>20s} = {bmnc[:3]}")
    print(f"  {'bmnc shape pyneo':>20s} = {bmnc_py.shape}")
    print(f"  {'bmnc shape ours':>20s} = {booz_dict['bmnc_b'].shape}")

# Three iota sources
bx = getattr(boozer, 'bx', boozer)
iota_bx_b = getattr(bx, 'iota_b', None)
iota_bx = getattr(bx, 'iota', None)
equil = getattr(boozer, 'equil', None)
wout = getattr(equil, 'wout', None) if equil is not None else None
iotas_wout = getattr(wout, 'iotas', None) if wout is not None else None
if iota_bx_b is not None:
    print(f"  {'bx.iota_b[{cs}]':>20s} = {np.asarray(iota_bx_b).flat[cs]:.6f}")
if iota_bx is not None:
    print(f"  {'bx.iota[{cs}]':>20s} = {np.asarray(iota_bx).flat[cs]:.6f}")
if iotas_wout is not None and cs < len(iotas_wout):
    print(f"  {'wout.iotas[{cs}]':>20s} = {iotas_wout[cs]:.6f}" if hasattr(iotas_wout, '__getitem__') else f"  {'wout.iotas':>20s} = {iotas_wout:.6f}")
print(f"  {'─'*60}")

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def _plot_diff(ax, data, title, cmap="RdBu_r", vlim=0.05):
    """Plot 2D difference with symmetric colorbar clipped to ±vlim."""
    im = ax.pcolormesh(th, ph, data.T, shading="auto",
                       cmap=cmap, vmin=-vlim, vmax=vlim)
    plt.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("θ"); ax.set_ylabel("ζ")

fig, axes = plt.subplots(1, 3, figsize=(30, 10))
_plot_diff(axes[0], dB / np.maximum(B_py, 1e-15), "|B| (F − P) / P")
_plot_diff(axes[1], dgp / np.maximum(gp_py, 1e-15), "|∇ψ| (F − P) / P")
_plot_diff(axes[2], dkg, "|∇ψ|·κ_G  (F − P)", cmap="RdBu_r", vlim=0.02)
plt.tight_layout()
# plt.savefig(str(Path(__file__).resolve().parent / f"grid_compare_{DEVICE}_s{SURF_S:.2f}.png"), dpi=150)
# plt.close()
# print(f"  Plot saved.")
plt.show()


print(f"\n  {'─'*60}")
print(f"  Pointwise difference  (Fourier − pyneo)")
print(f"  {'─'*60}")
print(f"  {'I =':>10s} {I_:.6f}     {'J =':>8s} {J_:.6f}     {'fac =':>8s} {fac:.6f}")
# print(f"  {'iota_py =':>10s} {py_iota_profile[k_diag]:.6f}     {'iota_our =':>10s} {iota:.6f}")
print(f"  {'─'*60}")
print(f"  {'':>14s}  {'rms Δ':>12s}  {'max |Δ|':>12s}  {'pyneo mean':>12s}  {'fourier mean':>12s}  {'ratio':>10s}")
print(f"  {'─'*14}  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*10}")
for name, dv, pv in [("|B|", dB, B_py), ("|∇ψ|", dgp, gp_py), ("|∇ψ|·κ_G", dkg, kg_py)]:
    rms = np.sqrt(np.mean(dv**2))
    mx = np.max(np.abs(dv))
    mn_py = np.mean(pv)
    mn_fo = np.mean(pv+dv)
    ratio = mn_fo/mn_py if abs(mn_py)>1e-15 else float("nan")
    s = f"  {name:>14s}  {rms:12.3e}  {mx:12.3e}  {mn_py:12.6f}  {mn_fo:12.6f}"
    s += f"  {ratio:10.4f}" if abs(mn_py)>1e-15 else f"  {'--':>10s}"
    print(s)

# ══════════════════════════════════
# mgrid comparison: Fourier |B| vs mgrid |B| at same (R,Z,φ_geo)
# ══════════════════════════════════

phi_geo = phal - Nu if pmns is not None else phal
nfp = int(np.asarray(booz_dict.get('nfp_b', 2)).flat[0])
phi_geo = np.mod(phi_geo, 2*np.pi/nfp)  # restrict to one field period
# R_line = R.ravel()
# Z_line = Z.ravel()

# X = R_line * np.cos(phi_geo)
# Y = R_line * np.sin(phi_geo)

# fig = plt.figure(figsize=(24, 24))
# ax = fig.add_subplot(111, projection='3d')

# ax.scatter(X, Y, Z_line, color='blue', s=1)  # s控制点大小

# ax.set_title("Fieldline in 3D (R, Z, φ_geo)")
# ax.set_xlabel("R")
# ax.set_ylabel("Z")
# ax.set_zlabel("φ_geo")

# plt.tight_layout()
# plt.show()


from ripplepy import initialize_mgrid_field, set_extcur, get_bfield_matrix
MGRID_PATH = f"{BASE}/tests/test_file/mgrid_2b40R1mB01.nc"
initialize_mgrid_field(MGRID_PATH, nfp=int(np.asarray(booz_dict.get('nfp_b',2)).flat[0]), full_torus=False)
set_extcur(None)
B_mgrid = np.sqrt(np.sum(get_bfield_matrix(None, R.ravel(), Z.ravel(), phi_geo)[:, :3]**2, axis=1))
B_mgrid = B_mgrid.reshape(ntheta, nphi)
dB_mg = B_f.ravel() - B_mgrid.ravel()
rms_mg = np.sqrt(np.mean(dB_mg**2))
mx_mg  = np.max(np.abs(dB_mg))

plt.figure(figsize=(24, 24))
_plot_diff(plt.gca(), dB_mg.reshape(ntheta, nphi) /np.maximum(B_mgrid, 1e-15), "|B| (F − M) / M")
plt.tight_layout()
plt.show()
print(f"\n  {'─'*60}")
print(f"  Fourier |B| vs mgrid |B|  (same real-space points)")
print(f"  {'─'*60}")
print(f"  {'|B| rms Δ':>14s}  {rms_mg:12.3e}  {'max |Δ|':>12s}  {mx_mg:12.3e}")
print(f"  {'Fourier mean':>14s}  {np.mean(B_f):12.6f}  {'mgrid mean':>14s}  {np.mean(B_mgrid):12.6f}")
print()
print()

