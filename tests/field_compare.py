#!/usr/bin/env python3
"""High-precision Boozer vs mgrid B-field comparison for NCSX and H1."""
import os, sys
os.chdir('/Users/zkgao/ripplepy')
sys.path.insert(0, 'python')

import numpy as np
from simsopt.mhd import Boozer, Vmec
from ripplepy.boozer_eps_verify import (
    _boozer_obj_to_dict, _fourier_sum_cos, _fourier_sum_sin,
    _fourier_sum_deriv_theta_cos, _fourier_sum_deriv_zeta_cos,
    _fourier_sum_deriv_theta_sin, _fourier_sum_deriv_zeta_sin,
    _find_local_minima,
)
from ripplepy import set_extcur, initialize_mgrid_field,set_trace_parameters,trace_fieldline
import importlib
import matplotlib.pyplot as plt

nfp = 3

# ═══════════════════════════════════════════════════════════════
def compare_device(name, vmec_path, mgrid_path, sur_idx, extcur, mpol, ntor):
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")
    
    vmec = Vmec(str(vmec_path))
    boozer = Boozer(vmec); boozer.mpol=mpol; boozer.ntor=ntor
    boozer.register(np.array([sur_idx])); boozer.run()
    d = _boozer_obj_to_dict(boozer)
    R0 = float(vmec.wout.Rmajor_p)
    
    xm = np.asarray(d['ixm_b'], dtype=np.int32)
    xn = np.asarray(d['ixn_b'], dtype=np.int32)
    rmnc = np.asarray(d['rmnc_b'][0]); zmns = np.asarray(d['zmns_b'][0])
    bmnc = np.asarray(d['bmnc_b'][0])
    iota = float(np.asarray(d['iota_b']).flat[0])
    I_   = float(np.asarray(d['buco_b']).flat[0])
    J_   = float(np.asarray(d['bvco_b']).flat[0])
    fac = I_ + iota * J_
    
    # ── dense field-line sampling (many turns for statistics) ──
    nzeta, nphi = 360, 50
    ntot = nzeta * nphi
    dp = 2*np.pi/nzeta
    ph = np.arange(ntot) * dp
    th = 0.0 + iota * ph
    
    # Boozer evaluation
    B_b  = _fourier_sum_cos(bmnc, xm, xn, th, ph)
    dBdt = _fourier_sum_deriv_theta_cos(bmnc, xm, xn, th, ph)
    dBdz = _fourier_sum_deriv_zeta_cos(bmnc, xm, xn, th, ph)
    R_b  = _fourier_sum_cos(rmnc, xm, xn, th, ph)
    Z_b  = _fourier_sum_sin(zmns, xm, xn, th, ph)
    dRdt = _fourier_sum_deriv_theta_cos(rmnc, xm, xn, th, ph)
    dRdz = _fourier_sum_deriv_zeta_cos(rmnc, xm, xn, th, ph)
    dZdt = _fourier_sum_deriv_theta_sin(zmns, xm, xn, th, ph)
    dZdz = _fourier_sum_deriv_zeta_sin(zmns, xm, xn, th, ph)
    
    # Boozer |∇ψ| and |∇ψ|κ_G
    gtb  = dRdt**2 + dZdt**2
    gpb  = dRdz**2 + dZdz**2 + R_b**2
    gtbp = dRdt*dRdz + dZdt*dZdz
    isqrg  = B_b**2 / fac
    gp_b   = np.sqrt(np.abs(gtb*gpb - gtbp**2)) * isqrg
    kggp_b = (J_*dBdz - I_*dBdt) / fac
    
    # ── mgrid evaluation at same (R,Z,φ) ──
    initialize_mgrid_field(mgrid_path, nfp, full_torus=False)
    eff = importlib.import_module('ripplepy.effective_ripple').Effective_Ripple
    extcur_arr = set_extcur(extcur)
    eff.sum_bfield_internal(extcur_arr)
    phi_period = 2*np.pi/nfp
    
    B_m  = np.zeros(ntot); Br_m = np.zeros(ntot)
    Bz_m = np.zeros(ntot); Bp_m = np.zeros(ntot)
    for j in range(ntot):
        r = eff.interpolate_field(float(R_b[j]), float(Z_b[j]), 
                                  float(ph[j] % phi_period))
        Br_m[j]=r[0]; Bz_m[j]=r[1]; Bp_m[j]=r[2]
        B_m[j] = np.sqrt(r[0]**2 + r[1]**2 + r[2]**2)
    
        ph_grid = ph.reshape(nphi, nzeta)
        th_grid = th.reshape(nphi, nzeta)

    initial_rz = (R_b[0], Z_b[0])
    initial_gradpsi = [1,0,0]
    fieldline_data,trace_istate=trace_fieldline(initial_rz,initial_gradpsi,nturn=nzeta,nphi=nphi,extcur=extcur_arr)
    R_m_line = fieldline_data[:,0]; Z_m_line = fieldline_data[:,1]; ph_m_line = fieldline_data[:,2]; B_m_line =fieldline_data[:,6]

    # plot fieldline in 3d view with axis equal and color by |B|
    X_b = R_b * np.cos(ph)
    Y_b = R_b * np.sin(ph)
    Z_b = Z_b
    X_m = R_m_line * np.cos(ph_m_line)
    Y_m = R_m_line * np.sin(ph_m_line)
    Z_m = Z_m_line
    B_line_err = (B_m_line - B_b)/B_m_line

    npoints = 360
    print(f'len of X_b = {len(X_b[:npoints])}, len of X_m = {len(X_m[:npoints])}')

    import plotly.graph_objects as go

    fig = go.Figure()

    # 第一条线：线条模式 - 固定颜色，不显示 color bar
    fig.add_trace(go.Scatter3d(
        x=X_b[:npoints], y=Y_b[:npoints], z=Z_b[:npoints],
        mode='lines',
        line=dict(color='blue', width=4),          # 固定颜色（或 'red', 'green' 等）
        name='曲线 B'
        # showscale=False                            # 确保不显示 color bar
    ))

    # 第二条线：点模式 - 使用 B_line_err 作为颜色映射，显示 color bar
    fig.add_trace(go.Scatter3d(
        x=X_m[:npoints], y=Y_m[:npoints], z=Z_m[:npoints],
        mode='markers',
        marker=dict(
            color=B_line_err[:npoints],            # 用 B_line_err 控制颜色
            colorscale='Viridis',                  # 颜色映射
            size=4,
            colorbar=dict(title="Error Value")     # 显示 color bar，可以自定义标题
        ),
        name='曲线 M'
    ))

    fig.update_layout(scene_aspectmode="data")
    fig.show()

    # plt.figure(figsize=(8, 5))
    # plt.plot(ph, B_b, 'k-', label='B (Boozer)')
    # plt.plot(ph, B_m, 'r--', label='B (mgrid)')
    # plt.xlabel('toroidal angle φ (rad)')
    # plt.ylabel('|B| (T)')
    # plt.title(f'{name} |B| comparison along field line')
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    # plt.tight_layout()
    # plt.savefig(f"comparison_{name.replace(' ', '_')}_B.png", dpi=150)
    # print(f"✓ Saved comparison_{name.replace(' ', '_')}_B.png")
    # plt.show()

    # # B_error plot
    # B_error = (B_m - B_b) / B_b * 100
    # plt.figure(figsize=(8, 5))
    # plt.plot(ph, B_error, 'm-', label='(B_mgrid - B_boozer) / B_boozer (%)')
    # plt.xlabel('toroidal angle φ (rad)')
    # plt.ylabel('Relative error (%)')
    # plt.title(f'{name} |B| relative error along field line')
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    # plt.tight_layout()
    # plt.savefig(f"comparison_{name.replace(' ', '_')}_B_error.png", dpi=150)
    # print(f"✓ Saved comparison_{name.replace(' ', '_')}_B_error.png")
    # plt.show()



    # ── Basic statistics ──
    ΔB_b = B_b.max() - B_b.min(); ΔB_m = B_m.max() - B_m.min()
    corr_B  = np.corrcoef(B_b, B_m)[0,1]
    rms_B   = np.sqrt(np.mean((B_b - B_m)**2))
    maxerr_B = np.max(np.abs(B_b - B_m))
    mean_ratio = np.mean(B_m / B_b)
    
    print(f"\n  ── |B| comparison ──")
    print(f"  Boozer:  [{B_b.min():.6f}, {B_b.max():.6f}]  ΔB = {ΔB_b:.6f} T  ({100*ΔB_b/B_b.mean():.2f}%)")
    print(f"  mgrid:   [{B_m.min():.6f}, {B_m.max():.6f}]  ΔB = {ΔB_m:.6f} T  ({100*ΔB_m/B_m.mean():.2f}%)")
    print(f"  ΔB ratio (mgrid/booz): {ΔB_m/ΔB_b:.6f}")
    print(f"  mean(B_m/B_b): {mean_ratio:.6f}")
    print(f"  correlation:   {corr_B:.6f}")
    print(f"  RMS diff:      {rms_B:.6f} T  ({100*rms_B/B_b.mean():.4f}%)")
    print(f"  max |ΔB|:      {maxerr_B:.6f} T  ({100*maxerr_B/B_b.mean():.4f}%)")
    
    # ── Component-wise comparison ──
    # Compute Boozer components from mgrid at field-line points
    # (Br, Bz, Bphi from mgrid; compare with reconstructed B from Boozer)
    Bmag_b = B_b
    Bmag_m = B_m
    
    
    # ── Well structure comparison ──
    minima_b = _find_local_minima(B_b)
    minima_m = _find_local_minima(B_m)
    nwells_b = len(minima_b) - 1
    nwells_m = len(minima_m) - 1
    
    print(f"\n  ── Well structure ──")
    print(f"  Boozer wells: {nwells_b}")
    print(f"  mgrid  wells: {nwells_m}")
    print(f"  wells ratio:  {nwells_m/nwells_b:.3f}")
    
    # Check well depth statistics
    well_depths_b = []
    well_depths_m = []
    for k in range(len(minima_b)-1):
        i1, i2 = minima_b[k], minima_b[k+1]
        seg = B_b[i1 % ntot:(i2 % ntot)+1] if i2-i1 < ntot else B_b[i1 % ntot:]
        if len(seg) > 1:
            bmax_local = max(seg[0], seg[-1]) if len(seg) < 3 else max(np.max(seg[:5]), np.max(seg[-5:]))
            bmin_local = np.min(seg)
            well_depths_b.append(bmax_local - bmin_local)
    
    for k in range(len(minima_m)-1):
        i1, i2 = minima_m[k], minima_m[k+1]
        seg = B_m[i1 % ntot:(i2 % ntot)+1] if i2-i1 < ntot else B_m[i1 % ntot:]
        if len(seg) > 1:
            bmax_local = max(seg[0], seg[-1]) if len(seg) < 3 else max(np.max(seg[:5]), np.max(seg[-5:]))
            bmin_local = np.min(seg)
            well_depths_m.append(bmax_local - bmin_local)
    
    if well_depths_b and well_depths_m:
        wd_b = np.array(well_depths_b); wd_m = np.array(well_depths_m)
        print(f"  Boozer well depths: [{wd_b.min():.6f}, {wd_b.max():.6f}]  mean={wd_b.mean():.6f}")
        print(f"  mgrid  well depths: [{wd_m.min():.6f}, {wd_m.max():.6f}]  mean={wd_m.mean():.6f}")
        if len(wd_m) > 0 and len(wd_b) > 0:
            # Simple comparison: sort and compare
            wd_b_sort = np.sort(wd_b)
            wd_m_sort = np.sort(wd_m)
            # interpolate to same length
            from scipy.interpolate import interp1d
            xb = np.linspace(0, 1, len(wd_b_sort))
            xm = np.linspace(0, 1, len(wd_m_sort))
            common_x = np.linspace(0, 1, min(len(wd_b_sort), len(wd_m_sort)))
            fb = interp1d(xb, wd_b_sort)(common_x)
            fm = interp1d(xm, wd_m_sort)(common_x)
            rms_depth = np.sqrt(np.mean((fb - fm)**2))
            print(f"  RMS well-depth diff (interp): {rms_depth:.6f} T")
    
    # ── Error distribution ──
    delta_B = B_m - B_b
    print(f"\n  ── ΔB = mgrid - Boozer distribution ──")
    print(f"  mean:  {delta_B.mean():+.6f} T")
    print(f"  std:   {delta_B.std():.6f} T")
    print(f"  skew:  {np.mean((delta_B-delta_B.mean())**3)/delta_B.std()**3:+.4f}")
    pcts = [1, 5, 25, 50, 75, 95, 99]
    pvals = np.percentile(delta_B, pcts)
    for p, v in zip(pcts, pvals):
        print(f"  {p:2d}%: {v:+.6f} T  ({100*abs(v)/B_b.mean():.4f}%)")
    
    # ── ε_eff sensitivity estimate ──
    # For a quasi-symmetric device, ε_eff ∝ (ΔB)^(3/2) approximately
    eps_sensitivity = 1.5 * (ΔB_m/ΔB_b - 1.0)
    print(f"\n  ── ε_eff sensitivity ──")
    print(f"  Estimated ε_eff ratio (from ΔB scaling): {1+eps_sensitivity:.4f}")
    print(f"  (Assuming ε ∝ (ΔB)^1.5; actual may differ due to well structure)")
    
    return {
        'name': name, 'ΔB_b': ΔB_b, 'ΔB_m': ΔB_m,
        'corr': corr_B, 'rms': rms_B, 'maxerr': maxerr_B,
        'nwells_b': nwells_b, 'nwells_m': nwells_m,
        'mean_ratio': mean_ratio,
    }

# ═══════════════════════════════════════════════════════════════
# Run both devices
# ═══════════════════════════════════════════════════════════════

r1 = compare_device(
    "NCSX (quasi-axisymmetric)",
    "tests/test_file/wout_ncsx_c09r00_free.nc",
    "tests/test_file/mgrid_c09r00.nc",
    0.5, None, 128, 128,
)

r2 = compare_device(
    "H1 (classical stellarator)",
    "tests/test_file/wout_h1_design.nc",
    "tests/test_file/mgrid_h1_design.nc",
    0.5, [50000, 5000, 1, -80000, -40000], 128, 128,
)

print(f"\n{'='*70}")
print(f"  FINAL SUMMARY")
print(f"{'='*70}")
print(f"  {'':20s}  {'ΔB ratio':>10s}  {'corr':>8s}  {'RMS':>10s}  {'max|Δ|':>10s}  {'wells m/b':>10s}")
print(f"  {'─'*20}  {'─'*10}  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*10}")
for r in [r1, r2]:
    print(f"  {r['name']:20s}  {r['ΔB_m']/r['ΔB_b']:10.6f}  {r['corr']:8.6f}  "
          f"{r['rms']:10.6f}  {r['maxerr']:10.6f}  {r['nwells_m']/r['nwells_b']:10.3f}")
