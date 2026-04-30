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

# vmec_path = "/home/zkg/ripplepy/tests/test_file/wout_ncsx_c09r00_free.nc"
vmec_path = "/home/zkg/ripplepy/tests/test_file/wout_h1_design.nc"

sur_idx = np.linspace(0.1, 0.5, 10)

RZ_points = []  # 每个元素是 [R, phi, Z]
for s in sur_idx:
    surface = SurfaceRZFourier.from_wout(vmec_path, s)
    RphiZ = surface.cross_section(phi=0)[0]
    RZ = RphiZ[[0, 2]]   # shape: (Npts, 3)
    RZ_points.append(RZ)             # 第一个点: (3,)

RZ_points = np.asarray(RZ_points)       # shape: (len(sur_idx), 3) [R1,Z1], ...]

vmec = Vmec(str(vmec_path))
boozer = Boozer(vmec)
boozer.mpol = 72
boozer.ntor = 36
# ns_list =  np.array([2, 3, 4, 5, 6, 7, 8, 9, 10])
# boozer.register(ns_list/100)
boozer.register(np.linspace(0.1, 1.0, 10))
boozer.bx.verbose =True
boozer.run()

neoclass = neo.from_simsopt_boozer(boozer)
ctx = NeoContext()
ctx.set_boozer(neoclass)
surfaces = neo_surfaces_from_simsopt_boozer(boozer)
print('Surfaces from simsopt Boozer:', surfaces)
ctx.set_flux_surfaces(surfaces.tolist())
ctx.set_resolution(theta_n=200, phi_n=200)
ctx.set_mode_limits(max_m_mode=0, max_n_mode=0)
ctx.set_transport_options(
    npart=50,
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



# Test importing the MGrid module
# %reset -f
import numpy as np
from ripplepy import MGrid

# from ripplepy.effective_ripple import Effective_Ripple
from ripplepy import set_extcur, initialize_mgrid_field,set_trace_parameters,compute_epstot,find_axis
import time
from func_timeout import func_timeout, FunctionTimedOut
from pathlib import Path

full_torus = False
nfp = 3

mgrid_candidates = [
    Path("tests/test_file/mgrid_c09r00.nc"),
    Path("test_file/mgrid_c09r00.nc"),
]
mgrid_path = next((p for p in mgrid_candidates if p.exists()), None)
if mgrid_path is None:
    raise FileNotFoundError(
        "Cannot find mgrid file. Tried: " + ", ".join(str(p) for p in mgrid_candidates)
    )

# mgrid_filename = str(mgrid_path)
# extcur = None

mgrid_filename = '/home/zkg/CN_H1_scan_fieldlines/H1_design/coils/mgrid_h1_design.nc'
extcur = [50000, 5000, 1, -80000, -40000]
nturn = 100
nphi = 360
initialize_mgrid_field(mgrid_filename, nfp,full_torus=full_torus)

set_extcur(extcur)
# sum_bfield_internal = True
initial_rz = (1.26,0)
axis_rz, R0, axis_fieldline, trace_error_flag = find_axis(initial_rz, timeout=10.0, xtol=1e-5, max_iter=100)
# print(f"✓ Magnetic axis found at R={axis_rz[0]:.10f}, Z={axis_rz[1]:.10f}, R0={R0:.10f}, time={time_elapsed:.3f} s")

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
    epsilon_eff, Bboundary = compute_epstot(extcur, initial_rz, initial_gradpsi, fieldline_data)
    epsilon_eff = epsilon_eff*R0**2
    endtime = time.time()
    time_elapsed = endtime - starttime
    print(f"✓ Ripple fieldline computed. epsilon_eff={epsilon_eff:.6e}, Bboundary={Bboundary:.6e},time={time_elapsed:.3f} s")
    ripplepy_results.append(( epsilon_eff))

print("ripplepy_results:", ripplepy_results)
# starttime = time.time()
# vol, a_minor, iota = calculate_plasma_params(fieldline_data, axis_fieldline[:nphi+1,:3], nturn, nphi, R0, nfp)
# endtime = time.time()
# time_elapsed = endtime - starttime
# print(f"✓ Physics parameters calculated in {time_elapsed:.3f} s")
# print(f"✓ Physics parameters calculated: Volume={vol:.6f}, a_minor={a_minor:.6f}, Iota={iota:.6f}")