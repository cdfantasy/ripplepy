#!/usr/bin/env python3
"""Compare κ_G from ripplepy (mgrid interpolation) vs analytic Boozer formula.

For each flux surface, trace field lines in both representations from the same
starting (R,Z) at φ=0, then compare κ_G, |B|, and |∇ψ| point-by-point (aligned
by step index, since both use the same nphi × nturn stepping in toroidal angle).

Goal: determine whether B-field differences (mgrid vs VMEC/Boozer) account for
the ε_eff gap between compute_epstot_pyneo and pyneo.
"""
import numpy as np
from simsopt.mhd import Boozer, Vmec
from simsopt.geo import SurfaceRZFourier
from ripplepy.boozer_eps_verify import (
    _boozer_obj_to_dict, _fourier_sum_cos, _fourier_sum_sin,
    _fourier_sum_deriv_theta_cos, _fourier_sum_deriv_zeta_cos,
    _fourier_sum_deriv_theta_sin, _fourier_sum_deriv_zeta_sin,
)
from ripplepy import initialize_mgrid_field, set_extcur, set_trace_parameters
from ripplepy.ripple import Effective_Ripple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def find_theta0_for_rz(rmnc, zmns, xm, xn, R_target, Z_target, ntheta=20000):
    """Find θ₀ ∈ [0,2π) such that Boozer (R(θ₀,0), Z(θ₀,0)) ≈ (R_target, Z_target)."""
    theta = np.linspace(0, 2 * np.pi, ntheta)
    zeta = np.zeros(ntheta)
    R = _fourier_sum_cos(rmnc, xm, xn, theta, zeta)
    Z = _fourier_sum_sin(zmns, xm, xn, theta, zeta)
    dist = np.sqrt((R - R_target) ** 2 + (Z - Z_target) ** 2)
    return theta[np.argmin(dist)]


def trace_ripplepy_kg(mgrid_path, nfp, initial_rz, nturn, nphi, extcur):
    """Trace a field line in ripplepy (mgrid B) and return κ_G and related quantities."""
    initialize_mgrid_field(mgrid_path, nfp, full_torus=False)
    set_extcur(extcur)
    set_trace_parameters(nturn, nphi, verbose=False)

    npoints = nturn * nphi
    fld = np.zeros((npoints, 20), dtype=np.float64, order='F')
    ist = Effective_Ripple.trace_gradpsi_internal(
        fld,
        np.asfortranarray(np.array(initial_rz, dtype=np.float64)),
        np.asfortranarray(np.array([1.0, 0.0, 0.0], dtype=np.float64)),
    )

    geocur = np.zeros(npoints, dtype=np.float64, order='F')
    Bb = np.zeros(1, dtype=np.float64, order='F')
    Effective_Ripple.geodesic_curvature_internal(fld, geocur, Bb)

    return {
        'phi': fld[:, 2].copy(),
        'R': fld[:, 0].copy(),
        'Z': fld[:, 1].copy(),
        'B': fld[:, 6].copy(),       # |B|
        'gradpsi': fld[:, 10].copy(), # |∇ψ|
        'kg': geocur.copy(),          # κ_G (already divided by |∇ψ|)
        'ist': ist,
    }


def trace_boozer_kg(booz_dict, surf_idx, theta0, nzeta, nturn):
    """Sample a Boozer field line analytically and return κ_G and related quantities."""
    xm = np.asarray(booz_dict['ixm_b'], dtype=np.int32)
    xn = np.asarray(booz_dict['ixn_b'], dtype=np.int32)
    bmnc = np.asarray(booz_dict['bmnc_b'][surf_idx], dtype=np.float64)
    rmnc = np.asarray(booz_dict['rmnc_b'][surf_idx], dtype=np.float64)
    zmns = np.asarray(booz_dict['zmns_b'][surf_idx], dtype=np.float64)
    iota = float(np.asarray(booz_dict['iota_b']).flat[surf_idx])
    I_ = float(np.asarray(booz_dict['buco_b']).flat[surf_idx])
    J_ = float(np.asarray(booz_dict['bvco_b']).flat[surf_idx])
    fac = I_ + iota * J_

    ntot = nzeta * nturn
    dphi = 2 * np.pi / nzeta
    zeta = np.arange(ntot) * dphi
    theta = theta0 + iota * zeta

    B = _fourier_sum_cos(bmnc, xm, xn, theta, zeta)
    dBdt = _fourier_sum_deriv_theta_cos(bmnc, xm, xn, theta, zeta)
    dBdz = _fourier_sum_deriv_zeta_cos(bmnc, xm, xn, theta, zeta)
    R = _fourier_sum_cos(rmnc, xm, xn, theta, zeta)
    Z_b = _fourier_sum_sin(zmns, xm, xn, theta, zeta)
    dRdt = _fourier_sum_deriv_theta_cos(rmnc, xm, xn, theta, zeta)
    dRdz = _fourier_sum_deriv_zeta_cos(rmnc, xm, xn, theta, zeta)
    dZdt = _fourier_sum_deriv_theta_sin(zmns, xm, xn, theta, zeta)
    dZdz = _fourier_sum_deriv_zeta_sin(zmns, xm, xn, theta, zeta)

    gtb = dRdt ** 2 + dZdt ** 2
    gpb = dRdz ** 2 + dZdz ** 2 + R ** 2
    gtbp = dRdt * dRdz + dZdt * dZdz
    isqrg = B ** 2 / fac
    gradpsi = np.sqrt(np.abs(gtb * gpb - gtbp ** 2)) * isqrg
    kg_gradpsi = (J_ * dBdz - I_ * dBdt) / fac
    kg = kg_gradpsi / (gradpsi + 1e-30)

    return {
        'zeta': zeta,
        'R': R,
        'Z': Z_b,
        'B': B,
        'gradpsi': gradpsi,
        'kg': kg,
    }


def compare_surface(vmec_path, booz_dict, surf_idx, mgrid_path, nfp, s_val, extcur,
                    nturn, nphi):
    """Run both traces and return comparison data for a single flux surface."""
    # Starting point: VMEC (R,Z) at φ=0 on the flux surface
    surf = SurfaceRZFourier.from_wout(str(vmec_path), s_val)
    rpz = surf.cross_section(phi=0)[0]
    initial_rz = rpz[[0, 2]].copy()

    # Match θ₀
    rmnc = np.asarray(booz_dict['rmnc_b'][surf_idx], dtype=np.float64)
    zmns = np.asarray(booz_dict['zmns_b'][surf_idx], dtype=np.float64)
    xm = np.asarray(booz_dict['ixm_b'], dtype=np.int32)
    xn = np.asarray(booz_dict['ixn_b'], dtype=np.int32)
    theta0 = find_theta0_for_rz(rmnc, zmns, xm, xn, initial_rz[0], initial_rz[1])

    # Trace
    rp = trace_ripplepy_kg(mgrid_path, nfp, initial_rz, nturn, nphi, extcur)
    bz = trace_boozer_kg(booz_dict, surf_idx, theta0, nphi, nturn)

    npoints = nturn * nphi
    # Both have exactly npoints entries; align by index
    kg_rp = rp['kg'][:npoints]
    kg_bz = bz['kg'][:npoints]
    B_rp = rp['B'][:npoints]
    B_bz = bz['B'][:npoints]
    gp_rp = rp['gradpsi'][:npoints]
    gp_bz = bz['gradpsi'][:npoints]

    # κ_G statistics
    dkg = kg_rp - kg_bz
    rms_dkg = float(np.sqrt(np.mean(dkg ** 2)))
    mean_abs_kg = float(np.mean(np.abs(kg_bz)))
    corr_kg = float(np.corrcoef(kg_rp, kg_bz)[0, 1])

    # |B| statistics
    rms_dB = float(np.sqrt(np.mean((B_rp - B_bz) ** 2)))
    mean_B = float(np.mean(B_bz))
    corr_B = float(np.corrcoef(B_rp, B_bz)[0, 1])

    # |∇ψ| statistics
    rms_dgp = float(np.sqrt(np.mean((gp_rp - gp_bz) ** 2)))
    mean_gp = float(np.mean(gp_bz))

    return {
        's': s_val,
        'theta0': float(theta0),
        'npoints': npoints,
        # κ_G
        'rms_dkg': rms_dkg,
        'mean_abs_kg': mean_abs_kg,
        'rel_rms_kg': rms_dkg / (mean_abs_kg + 1e-30),
        'corr_kg': corr_kg,
        # |B|
        'rms_dB': rms_dB,
        'mean_B': mean_B,
        'rel_rms_B': rms_dB / (mean_B + 1e-30),
        'corr_B': corr_B,
        # |∇ψ|
        'rms_dgp': rms_dgp,
        'mean_gp': mean_gp,
        'rel_rms_gp': rms_dgp / (mean_gp + 1e-30),
        # raw data (first turn only, for plotting)
        'phi': rp['phi'][:nphi],
        'kg_rp': kg_rp,
        'kg_bz': kg_bz,
        'B_rp': B_rp,
        'B_bz': B_bz,
        'gp_rp': gp_rp,
        'gp_bz': gp_bz,
    }


def plot_comparison(results, device_name, out_dir='/tmp'):
    """Generate diagnostic plots for all surfaces of one device."""
    n_surfs = len(results)
    fig, axes = plt.subplots(n_surfs, 3, figsize=(18, 3.5 * n_surfs))
    if n_surfs == 1:
        axes = axes.reshape(1, -1)

    for row, r in enumerate(results):
        s = r['s']
        nphi = len(r['phi'])

        # Column 1: κ_G vs φ (first turn)
        ax = axes[row, 0]
        ax.plot(r['phi'], r['kg_rp'][:nphi], 'r-', lw=0.6, alpha=0.8, label='mgrid')
        ax.plot(r['phi'], r['kg_bz'][:nphi], 'b--', lw=1.0, alpha=0.8, label='Boozer')
        ax.set_ylabel('κ_G')
        ax.set_title(f's={s:.2f}  corr={r["corr_kg"]:.3f}  rms={r["rms_dkg"]:.2e}')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # Column 2: κ_G scatter
        ax = axes[row, 1]
        # Downsample for scatter
        step = max(1, r['npoints'] // 5000)
        ax.scatter(r['kg_bz'][::step], r['kg_rp'][::step], s=2, alpha=0.4, c='k')
        lo = min(r['kg_bz'].min(), r['kg_rp'].min())
        hi = max(r['kg_bz'].max(), r['kg_rp'].max())
        ax.plot([lo, hi], [lo, hi], 'r--', lw=0.8)
        ax.set_xlabel('κ_G Boozer')
        ax.set_ylabel('κ_G mgrid')
        ax.set_title(f's={s:.2f}  rms/mean={r["rel_rms_kg"]:.3f}')
        ax.grid(True, alpha=0.3)

        # Column 3: |B| vs φ (first turn)
        ax = axes[row, 2]
        ax.plot(r['phi'], r['B_rp'][:nphi], 'r-', lw=0.6, alpha=0.8, label='mgrid')
        ax.plot(r['phi'], r['B_bz'][:nphi], 'b--', lw=1.0, alpha=0.8, label='Boozer')
        ax.set_ylabel('|B| [T]')
        ax.set_title(f's={s:.2f}  corr_B={r["corr_B"]:.4f}  rms_B={r["rms_dB"]:.2e}')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'{device_name}: κ_G and |B| comparison (mgrid vs Boozer)', fontsize=13)
    plt.tight_layout()
    path = f'{out_dir}/diag_kg_compare_{device_name}.png'
    plt.savefig(path, dpi=150)
    print(f'  Plot saved to {path}')
    plt.close()


def run_device(name, vmec_path, mgrid_path, nfp, s_vals, extcur, nturn, nphi):
    """Run κ_G comparison for a list of flux surfaces on one device."""
    print(f"\n{'=' * 65}")
    print(f"  {name}")
    print(f"{'=' * 65}")

    vmec = Vmec(str(vmec_path))
    boozer = Boozer(vmec)
    boozer.mpol = 72
    boozer.ntor = 36
    boozer.register(list(s_vals))
    boozer.run()
    booz_dict = _boozer_obj_to_dict(boozer)

    results = []
    for i, s in enumerate(s_vals):
        print(f"  s={s:.3f} ...", end=' ', flush=True)
        r = compare_surface(vmec_path, booz_dict, i, mgrid_path, nfp, s, extcur,
                            nturn, nphi)
        results.append(r)
        print(f"κ_G: corr={r['corr_kg']:.4f}, "
              f"rms={r['rms_dkg']:.2e}, "
              f"rms/mean|κ|={r['rel_rms_kg']:.3f}  |  "
              f"B: corr={r['corr_B']:.4f}, "
              f"rms={r['rms_dB']:.2e} "
              f"({100*r['rms_dB']/r['mean_B']:.3f}%)")

    # ── Summary table ──
    print(f"\n  {'s':>6s}  {'corr_kg':>8s}  {'rms_kg':>10s}  "
          f"{'rms/|kg|':>9s}  {'corr_B':>8s}  {'rms_B':>10s}  {'rms/B%':>8s}")
    print(f"  {'-' * 6}  {'-' * 8}  {'-' * 10}  {'-' * 9}  "
          f"{'-' * 8}  {'-' * 10}  {'-' * 8}")
    for r in results:
        print(f"  {r['s']:6.3f}  {r['corr_kg']:8.4f}  {r['rms_dkg']:10.2e}  "
              f"{r['rel_rms_kg']:9.3f}  {r['corr_B']:8.4f}  "
              f"{r['rms_dB']:10.2e}  {100*r['rms_dB']/r['mean_B']:7.3f}")

    plot_comparison(results, name)
    return results


# ═══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    BASE = "/Users/zkgao/ripplepy"
    NTURN, NPHI = 200, 360

    # ── NCSX: surfaces covering the benchmark range ──
    run_device(
        "NCSX",
        f"{BASE}/tests/test_file/wout_ncsx_c09r00_free.nc",
        f"{BASE}/tests/test_file/mgrid_c09r00.nc",
        nfp=3,
        s_vals=[0.10, 0.12, 0.15, 0.18, 0.20],
        extcur=None,
        nturn=NTURN, nphi=NPHI,
    )

    # ── CFQS: surfaces covering the benchmark range ──
    run_device(
        "CFQS",
        f"{BASE}/tests/test_file/wout_cfqs_test_m10_n5_fixed.nc",
        f"{BASE}/tests/test_file/mgrid_2b40R1mB01.nc",
        nfp=2,
        s_vals=[0.10, 0.28, 0.46, 0.64, 0.82, 1.00],
        extcur=None,
        nturn=NTURN, nphi=NPHI,
    )
