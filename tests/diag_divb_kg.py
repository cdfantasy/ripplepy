#!/usr/bin/env python3
"""Diagnostic: ∇·B error and its correlation with κ_G in ripplepy vs pyneo.

For a given surface, trace the same field line in both ripplepy (mgrid) and
pyneo (Boozer), then compare κ_G and ∇·B point-by-point.
"""
import numpy as np
from ripplepy import initialize_mgrid_field, set_extcur, set_trace_parameters, find_axis
from ripplepy.ripple import compute_epstot_pyneo
import neo
from neo import NeoContext, neo_surfaces_from_simsopt_boozer
from simsopt.mhd import Boozer, Vmec
from simsopt.geo import SurfaceRZFourier
import matplotlib.pyplot as plt


def diag_divb_kg(vmec_path, mgrid_path, nfp, s_val, extcur=None,
                 nturn=200, nphi=360, full_torus=False):
    """Diagnose ∇·B and κ_G on a single flux surface.

    Parameters
    ----------
    vmec_path, mgrid_path : str
    nfp : int
    s_val : float — normalised toroidal flux label
    extcur : array or None
    nturn, nphi : int — trace resolution
    """
    # ── 1. Get initial R,Z from VMEC ──
    vmec = Vmec(str(vmec_path))
    surf = SurfaceRZFourier.from_wout(str(vmec_path), s_val)
    rpz = surf.cross_section(phi=0)[0]
    initial_rz = rpz[[0, 2]].copy()  # (R, Z)

    # ── 2. ripplepy field line trace ──
    initialize_mgrid_field(mgrid_path, nfp, full_torus=full_torus)
    set_extcur(extcur)
    set_trace_parameters(nturn, nphi, verbose=False)

    npoints = nturn * nphi
    fld = np.zeros((npoints, 20), dtype=np.float64, order='F')
    eps, ist = compute_epstot_pyneo(
        1.0, initial_rz,
        initial_gradpsi=np.array([1, 0, 0], dtype=np.float64),
        npart=100, nturn=nturn, nphi=nphi, verbose=False,
    )

    # Re-trace to get full fieldline_data (compute_epstot_pyneo doesn't return it)
    from ripplepy.ripple import Effective_Ripple as ER
    set_trace_parameters(nturn, nphi, verbose=False)
    fld = np.zeros((npoints, 20), dtype=np.float64, order='F')
    ist2 = ER.trace_gradpsi_internal(
        fld,
        np.asfortranarray(initial_rz.astype(np.float64)),
        np.asfortranarray(np.array([1.0, 0.0, 0.0], dtype=np.float64)),
    )

    # Compute κ_G via Fortran
    geocur = np.zeros(npoints, dtype=np.float64, order='F')
    Bboundary = np.zeros(1, dtype=np.float64, order='F')
    ER.geodesic_curvature_internal(fld, geocur, Bboundary)

    # ── 3. Compute ∇·B from stored derivatives ──
    # Columns in fld: 1=R,2=Z,3=phi, 4=Br,5=Bz,6=Bphi, 7=|B|,
    #   8=P,9=G,10=Q,11=|∇ψ|,
    #   12=dBr_dr,13=dBr_dz,14=dBr_dphi,
    #   15=dBz_dr,16=dBz_dz,17=dBz_dphi,
    #   18=dBphi_dr,19=dBphi_dz,20=dBphi_dphi
    R   = fld[:, 0]
    Br  = fld[:, 3]
    Bz  = fld[:, 4]
    Bphi = fld[:, 5]
    Bmag = fld[:, 6]

    dBr_dr   = fld[:, 11]
    dBr_dz   = fld[:, 12]
    dBr_dphi = fld[:, 13]
    dBz_dr   = fld[:, 14]
    dBz_dz   = fld[:, 15]
    dBz_dphi = fld[:, 16]
    dBphi_dr   = fld[:, 17]
    dBphi_dz   = fld[:, 18]
    dBphi_dphi = fld[:, 19]

    # ∇·B = (1/R) ∂(R*Br)/∂R + (1/R) ∂Bφ/∂φ + ∂Bz/∂z
    divB = (Br + R * dBr_dr) / R + dBphi_dphi / R + dBz_dz
    divB_rel = np.abs(divB) / (Bmag + 1e-30)

    # ── 4. pyneo reference κ_G ──
    boozer = Boozer(vmec)
    boozer.mpol = 72
    boozer.ntor = 36
    boozer.register([s_val])
    boozer.run()
    neoclass = neo.from_simsopt_boozer(boozer)
    ctx = NeoContext()
    ctx.set_boozer(neoclass)
    ctx.set_flux_surfaces(neo_surfaces_from_simsopt_boozer(boozer).tolist())
    ctx.set_resolution(theta_n=100, phi_n=100)
    ctx.set_transport_options(npart=100, multra=1, acc_req=0.01, no_bins=100,
                              nstep_per=50, nstep_min=500, nstep_max=5000,
                              calc_nstep_max=0)
    ctx.set_switches(ref_swi=2, eout_swi=2, calc_cur=0)
    ctx.set_output_options(write_progress=0, write_output_files=0,
                           write_integrate=0, write_diagnostic=0,
                           suppress_file_io=True)
    ctx.setup_grids()
    ctx.run_all()
    py_eps = ctx.epstot_profile()[0]

    # ── 5. Report ──
    print(f"\n{'='*60}")
    print(f"  ∇·B Diagnostic: s = {s_val:.3f}")
    print(f"{'='*60}")
    print(f"  ε_eff (pyneo):    {py_eps:.4e}")
    print(f"  ε_eff (rp_new):   {eps:.4e}")
    print(f"  ratio rp/py:      {eps/py_eps:.4f}")
    print(f"  ---")
    print(f"  ∇·B stats along field line ({npoints} points):")
    print(f"    max |∇·B|:        {np.max(np.abs(divB)):.4e}")
    print(f"    mean |∇·B|:       {np.mean(np.abs(divB)):.4e}")
    print(f"    mean |∇·B|/|B|:   {np.mean(divB_rel):.4e}")
    print(f"    mean ∇·B (signed):{np.mean(divB):.4e}  ← should be ~0 if random")
    print(f"    std ∇·B:          {np.std(divB):.4e}")
    print(f"  ---")
    print(f"  κ_G stats:")
    print(f"    max |κ_G|:        {np.max(np.abs(geocur)):.4e}")
    print(f"    mean |κ_G|:       {np.mean(np.abs(geocur)):.4e}")
    print(f"    mean κ_G (signed):{np.mean(geocur):.4e}")

    # ── 6. Plots ──
    phi = fld[:, 2]
    sort_idx = np.argsort(phi)
    phi_s = phi[sort_idx]

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    # Panel 1: |B| vs φ
    ax = axes[0]
    ax.plot(phi_s, Bmag[sort_idx], 'b-', lw=0.5)
    ax.set_ylabel('|B| [T]')
    ax.set_title(f'Field line trace: s={s_val:.3f}, nturn={nturn}')

    # Panel 2: κ_G vs φ
    ax = axes[1]
    ax.plot(phi_s, geocur[sort_idx], 'r-', lw=0.5, label='ripplepy κ_G')
    ax.set_ylabel('κ_G')
    ax.legend(loc='upper right', fontsize=8)
    ax.axhline(y=0, color='gray', ls='--', lw=0.5)

    # Panel 3: ∇·B vs φ
    ax = axes[2]
    ax.plot(phi_s, divB[sort_idx], 'g-', lw=0.5)
    ax.set_ylabel('∇·B')
    ax.axhline(y=0, color='gray', ls='--', lw=0.5)

    # Panel 4: |∇·B|/|B| vs φ (log scale)
    ax = axes[3]
    ax.semilogy(phi_s, divB_rel[sort_idx] + 1e-30, 'm-', lw=0.5)
    ax.set_ylabel('|∇·B|/|B|')
    ax.set_xlabel('φ')

    plt.tight_layout()
    plt.savefig(f'/tmp/diag_divb_s{s_val:.2f}.png', dpi=150)
    print(f"\n  Plot saved to /tmp/diag_divb_s{s_val:.2f}.png")
    plt.close()

    # ── 7. Bounce-well analysis ──
    # Find bounce wells (regions between B maxima)
    b = Bmag
    b_sorted_idx = np.argsort(b)
    bmax_global = b[b_sorted_idx[-1]]
    bmin_global = b[b_sorted_idx[0]]

    # Simple peak detection
    peaks = []
    for i in range(1, npoints - 1):
        if b[i] > b[i-1] and b[i] > b[i+1]:
            peaks.append(i)

    print(f"\n  Detected {len(peaks)} B-peaks along field line")
    if len(peaks) >= 2:
        # Per-well averages
        well_divB_means = []
        well_kg_means = []
        for j in range(len(peaks) - 1):
            i1, i2 = peaks[j], peaks[j+1]
            well_divB_means.append(np.mean(divB[i1:i2]))
            well_kg_means.append(np.mean(np.abs(geocur[i1:i2])))

        well_divB_means = np.array(well_divB_means)
        well_kg_means = np.array(well_kg_means)

        print(f"  Per-well mean ∇·B:  mean={np.mean(well_divB_means):.4e}, "
              f"std={np.std(well_divB_means):.4e}")
        print(f"  Per-well mean |κ_G|: mean={np.mean(well_kg_means):.4e}, "
              f"std={np.std(well_kg_means):.4e}")

        if len(well_divB_means) > 4:
            corr = np.corrcoef(well_divB_means, well_kg_means)[0, 1]
            print(f"  Correlation ∇·B vs |κ_G| per well: {corr:.4f}")

    return divB, geocur, fld, py_eps, eps


if __name__ == '__main__':
    BASE = "/Users/zkgao/ripplepy"

    # ── CFQS: low ripple, big gap ──
    print("\n" + "█" * 60)
    print("  CFQS — 低 ripple，gap 最大")
    print("█" * 60)
    diag_divb_kg(
        f"{BASE}/tests/test_file/wout_cfqs_test_m10_n5_fixed.nc",
        f"{BASE}/tests/test_file/mgrid_2b40R1mB01.nc",
        nfp=2, s_val=0.5, extcur=None,
        nturn=200, nphi=360, full_torus=False,
    )

    # ── NCSX: moderate ripple, smaller gap ──
    print("\n" + "█" * 60)
    print("  NCSX — 中低 ripple，gap 较小")
    print("█" * 60)
    diag_divb_kg(
        f"{BASE}/tests/test_file/wout_ncsx_c09r00_free.nc",
        f"{BASE}/tests/test_file/mgrid_c09r00.nc",
        nfp=3, s_val=0.15, extcur=None,
        nturn=200, nphi=180, full_torus=False,
    )
