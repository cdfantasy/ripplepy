from simsopt.mhd import Boozer,Vmec
from simsopt.geo import Surface,SurfaceRZFourier
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import neo
from neo import NeoContext,neo_surfaces_from_simsopt_boozer
import time
import numpy as np
from pathlib import Path


mgrid_candidates = [
    Path("tests/test_file/mgrid_c09r00.nc"),
    Path("test_file/mgrid_c09r00.nc"),
]
mgrid_path = next((p for p in mgrid_candidates if p.exists()), None)
if mgrid_path is None:
    raise FileNotFoundError(
        "Cannot find mgrid file. Tried: " + ", ".join(str(p) for p in mgrid_candidates)
    )


initial_rz = (1.57,0)
mgrid_filename = str(mgrid_path)
extcur = None
vmec_candidates = [
    Path("tests/test_file/wout_ncsx_c09r00_free.nc"),
    Path("test_file/wout_ncsx_c09r00_free.nc"),
]
vmec_path = next((p for p in vmec_candidates if p.exists()), None)
if vmec_path is None:
    raise FileNotFoundError(
        "Cannot find VMEC wout file. Tried: " + ", ".join(str(p) for p in vmec_candidates)
    )
# initial_rz = (1.26,0)
# mgrid_filename = '/Users/zkgao/ripplepy/tests/test_file/mgrid_h1_design.nc'
# extcur = [50000, 5000, 1, -80000, -40000]
# vmec_path = "/Users/zkgao/ripplepy/tests/test_file/wout_h1_design.nc"

# ---- unified surface selection ----
sur_idx = np.linspace(0.1, 0.5, 10)      # normalized toroidal flux s

vmec = Vmec(str(vmec_path))

RZ_points = []  # (R, Z) starting points for ripplepy field-line tracing
for s in sur_idx:
    surface = SurfaceRZFourier.from_wout(str(vmec_path), s)
    RphiZ = surface.cross_section(phi=0)[0]
    RZ = RphiZ[[0, 2]]
    RZ_points.append(RZ)
RZ_points = np.asarray(RZ_points)

# Boozer: use the SAME surface list as RZ_points
boozer = Boozer(vmec)
boozer.mpol = 72
boozer.ntor = 72
boozer.register(sur_idx)                 # ← same as sur_idx
boozer.bx.verbose =True
boozer.run()
print("Boozer coefficients computed.")
# boozer.bx.write_boozmn("test_file/boozmn_h1_design.nc")

neoclass = neo.from_simsopt_boozer(boozer)
ctx = NeoContext()
ctx.set_boozer(neoclass)
surfaces = neo_surfaces_from_simsopt_boozer(boozer)
print('Surfaces from simsopt Boozer:', surfaces)
ctx.set_flux_surfaces(surfaces.tolist())
ctx.set_resolution(theta_n=200, phi_n=200)
ctx.set_mode_limits(max_m_mode=0, max_n_mode=0)
ctx.set_transport_options(
    npart=100,

    multra=1,
    acc_req=0.01,
    no_bins=100,
    nstep_per=50,
    nstep_min=500,
    nstep_max=5000,
    calc_nstep_max=0,
)
ctx.set_switches(ref_swi=2, eout_swi=2, calc_cur=0)
ctx.set_output_options(
    write_progress=0,
    write_output_files=0,
    write_integrate=0,
    write_diagnostic=0,
    suppress_file_io=True,
)

ctx.setup_grids()
ctx.run_all()

got_surfaces = ctx.surface_map()
got_epstot = ctx.epstot_profile()


import numpy as np
from ripplepy import MGrid

# from ripplepy.effective_ripple import Effective_Ripple
from ripplepy import set_extcur, initialize_mgrid_field,set_trace_parameters,compute_epstot,find_axis
import time
from pathlib import Path

full_torus = False
nfp = 3


# mgrid_filename = str(mgrid_path)
# # extcur = [6.52271941985300E+05, 6.51868569367400E+05, 5.37743588647300E+05, 2.50000000000000E-07, 2.50000000000000E-07, 2.80949750000000E+04, -5.48049500000000E+04, 3.01228950000000E+04, 9.42409100000000E+04, 4.55138737653200E+04]
# # extcur = np.ones(10)
# extcur = None




nturn = 200
nphi = 360
initialize_mgrid_field(mgrid_filename, nfp,full_torus=full_torus)

extcur=set_extcur(extcur)
# sum_bfield_internal = True

axis_rz, R0, axis_fieldline, trace_istate = find_axis(initial_rz, xtol=1e-5, max_iter=100)
print(f"✓ Magnetic axis found at R={axis_rz[0]:.10f}, Z={axis_rz[1]:.10f}, R0={R0:.10f}")

# initial_gradpsi = compute_initial_gradpsi(extcur,initial_rz[0], initial_rz[1], phi0=0.0,verbose=False)
initial_gradpsi = [1,0,0]
initial_gradpsi = np.array(initial_gradpsi, dtype=np.float64, order='F')
set_trace_parameters(nturn, nphi)
ripplepy_results = []
for i in RZ_points:

    fieldline_data = np.zeros((nturn*nphi, 20), dtype=np.float64, order='F')
    geocur = np.zeros((nturn*nphi, 3), dtype=np.float64, order='F')
    R0 = np.array(R0, dtype=np.float64, order='F')
    Bboundary = np.array(0.0, dtype=np.float64, order='F')
    initial_rz = np.array(i, dtype=np.float64, order='F')
    # # 预分配轨线数据缓冲区并传入 Fortran 例程（Fortran 连续）

    starttime = time.time()
    epsilon_eff, Bboundary,trace_istate = compute_epstot(R0,extcur, initial_rz, initial_gradpsi, fieldline_data)
    endtime = time.time()
    time_elapsed = endtime - starttime
    print(f"✓ Ripple fieldline computed @ {initial_rz[0]:.10f}, {initial_rz[1]:.10f}. epsilon_eff={epsilon_eff:.6e}, Bboundary={Bboundary:.6e},time={time_elapsed:.3f} s")
    ripplepy_results.append(( epsilon_eff))

print("ripplepy_results:", ripplepy_results)
ripplepy_200nturn_results = ripplepy_results

print("\n" + "=" * 70)
ref_surfaces = RZ_points[:,0]  # 假设 RZ_points 的第一列是表面索引或半径

plt.figure(figsize=(8, 5))
plt.plot(ref_surfaces, got_epstot, "ko-", label="pyneo (NEO)")
plt.plot(ref_surfaces, ripplepy_results, "rs:", label="ripplepy (coils)")
plt.xlabel("reference surface (R or s)")
plt.ylabel(r"$\varepsilon_{\mathrm{eff}}^{3/2}$")
plt.title("Effective ripple: pyneo vs ripplepy")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("benchmark_pyneo_vs_ripplepy.png", dpi=150)
print("✓ Saved benchmark_pyneo_vs_ripplepy.png")
plt.show()

error_ratios = np.array(ripplepy_results) / got_epstot
plt.figure(figsize=(8, 5))
plt.plot(ref_surfaces, error_ratios, "rs-", label="ripplepy / pyneo")
plt.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
plt.xlabel("reference surface (R or s)")
plt.ylabel("ratio to pyneo")
plt.title("Ratio of ripplepy to pyneo")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("benchmark_pyneo_vs_ripplepy_ratio.png", dpi=150)
print("✓ Saved benchmark_pyneo_vs_ripplepy_ratio.png")
plt.show()

# ═══════════════════════════════════════════════════════════════════════
#  PART 3: Boozer-coordinate direct integration benchmark
#  Same Boozer data as pyneo, ripplepy-style bp-scan algorithm.
#  This isolates integration-algorithm differences from B-field differences.
# ═══════════════════════════════════════════════════════════════════════

# from ripplepy.boozer_eps_verify import (
#     sample_fieldline_from_boozer,
#     eps_eff_from_boozer,
#     _boozer_obj_to_dict,
#     _find_local_minima,
#     _integrate_bounce_segment,
#     _compute_H2_over_I_for_bp,
# )
# from ripplepy.boozer_eps_verify import (
#     _fourier_sum_cos,
#     _fourier_sum_deriv_theta_cos,
#     _fourier_sum_deriv_zeta_cos,
#     _fourier_sum_sin,
#     _fourier_sum_deriv_theta_sin,
#     _fourier_sum_deriv_zeta_sin,
# )

# # Boozer and pyneo surfaces are in 1-1 correspondence.
# # got_surfaces are 1-based full-VMEC indices; boozer uses 0..ns_b-1 local indices.
# n_booz_surfs = len(got_epstot)   # = ns_b

# # R₀ from VMEC
# R0_vmec = float(vmec.wout.Rmajor_p)

# print("\n" + "=" * 70)
# print("PART 3: Boozer-coordinate direct integration benchmark")
# print("=" * 70)

# booz_dict = _boozer_obj_to_dict(boozer)

# n_gauss = 64

# print(f"\n{'surf':>4s}  {'pyneo':>12s}  {'booz_rect':>12s}  {'ratio':>8s}  "
#       f"{'booz_gauss':>12s}  {'ratio':>8s}  {'ripplepy':>12s}  {'r/r':>8s}")
# print("-" * 90)

# booz_rect_results = []
# booz_gauss_results = []

# for i in range(n_booz_surfs):
#     py_eps = got_epstot[i]

#     # Boozer-coordinate integration — use boozer-local index i
#     r_rect = eps_eff_from_boozer(
#         booz_dict, i, theta0=0.0, nzeta=512, nturn=64,
#         n_b=500, use_gauss=False,
#     )
#     r_gauss = eps_eff_from_boozer(
#         booz_dict, i, theta0=0.0, nzeta=512, nturn=64,
#         n_gauss=n_gauss, use_gauss=True,
#     )

#     # Apply R₀² scaling to match pyneo convention
#     eps_rect = r_rect["eps_eff_32"] * R0_vmec**2
#     eps_gauss = r_gauss["eps_eff_32"] * R0_vmec**2

#     booz_rect_results.append(eps_rect)
#     booz_gauss_results.append(eps_gauss)

#     rp_val = ripplepy_results[i] if i < len(ripplepy_results) else np.nan

#     print(
#         f"  {i:3d}  {py_eps:12.4e}  {eps_rect:12.4e}  {eps_rect/py_eps:8.4f}  "
#         f"{eps_gauss:12.4e}  {eps_gauss/py_eps:8.4f}  "
#         f"{rp_val:12.4e}  {rp_val/py_eps:8.4f}"
#     )

# # ═══════════════════════════════════════════════════════════════════════
# #  PART 4: Summary plot
# # ═══════════════════════════════════════════════════════════════════════
# print("\n" + "=" * 70)
# print("PART 4: Summary plot")
# print("=" * 70)

# fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ax = axes[0]
# s_vals = np.asarray(sur_idx)
# ax.plot(s_vals, got_epstot, "ko-", label="pyneo (NEO)")
# ax.plot(s_vals, booz_rect_results, "bD--", label="boozer+rect (this work)")
# ax.plot(s_vals, booz_gauss_results, "g^--", label=f"boozer+Gauss{n_gauss} (this work)")
# ax.plot(s_vals, ripplepy_results, "rs:", label="ripplepy (coils)")
# ax.set_xlabel("normalized toroidal flux s")
# ax.set_ylabel(r"$\varepsilon_{\mathrm{eff}}^{3/2}$")
# ax.set_title("Effective ripple: 3-method comparison")
# ax.legend(fontsize=8)
# ax.grid(True, alpha=0.3)

# ax = axes[1]
# ax.plot(s_vals, np.array(booz_rect_results) / got_epstot, "bD--", label="booz+rect / pyneo")
# ax.plot(s_vals, np.array(booz_gauss_results) / got_epstot, "g^--", label=f"booz+Gauss / pyneo")
# ax.plot(s_vals, np.array(ripplepy_results) / got_epstot, "rs:", label="ripplepy / pyneo")
# ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
# ax.set_xlabel("normalized toroidal flux s")
# ax.set_ylabel("ratio to pyneo")
# ax.set_title("Ratio to pyneo")
# ax.legend(fontsize=8)
# ax.grid(True, alpha=0.3)

# plt.tight_layout()
# plt.savefig("benchmark_boozer_comparison.png", dpi=150)
# print("✓ Saved benchmark_boozer_comparison.png")
# plt.show()

# print("\n✓ Benchmark complete.")
# print(f"  pyneo surfaces:         {len(got_epstot)}")
# print(f"  boozer+rect results:    {len(booz_rect_results)}")
# print(f"  boozer+gauss results:   {len(booz_gauss_results)}")
# print(f"  ripplepy (coils) results: {len(ripplepy_results)}")
