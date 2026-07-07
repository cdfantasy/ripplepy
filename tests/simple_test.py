from ripplepy import (
    set_extcur, initialize_mgrid_field, set_trace_parameters,
    trace_fieldline, find_axis,compute_epstot
)
import numpy as np
import time
from pathlib import Path

# BASE = str(Path(__file__).resolve().parent.parent)
# DEVICE = "CFQS"
# VMEC_PATH = f"{BASE}/tests/test_file/wout_cfqs_test_m10_n5_fixed.nc"
# MGRID_PATH = f"{BASE}/tests/test_file/mgrid_2b40R1mB01.nc"
# extcur = None
# INITIAL_RZ = (1.21, 0.0)
# NFP = 2
# FULL_TORUS = False

BASE = str(Path(__file__).resolve().parent.parent)
DEVICE = "H1"
VMEC_PATH = f"{BASE}/tests/test_file/wout_h1_design.nc"
MGRID_PATH = f"{BASE}/tests/test_file/mgrid_h1_design.nc"
extcur = [50000, 5000, 2000, -80000, -40000]
INITIAL_RZ = (1.26, 0.0)
NFP = 3
FULL_TORUS = False
initial_rz = (1.26, 0.0)


NTURN = 200
NPHI = 360
NPART = 5000

initialize_mgrid_field(MGRID_PATH, NFP, full_torus=FULL_TORUS)
set_extcur(extcur) 

axis_rz, R0_rp, axis_fl, ok = find_axis(INITIAL_RZ, xtol=1e-5, max_iter=100)
print(f"  Axis: R={axis_rz[0]:.4f}, Z={axis_rz[1]:.4f}, R0={R0_rp:.4f}")

axis_rz[0]+= 0.05

set_trace_parameters(NTURN, NPHI, npart=NPART, verbose=False)
start_time = time.time()
eps, bnd, ist = compute_epstot(
    axis_rz,
    initial_gradpsi=np.array([1, 0, 0], dtype=np.float64),
    verbose=False,
)
end_time = time.time()
print(f"  Time taken: {end_time - start_time:.4f} seconds")
print(f" axis_rz = ({axis_rz[0]:.4f}, {axis_rz[1]:.4f}), eps_tot = {eps:.6e}")