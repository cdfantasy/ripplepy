# Test importing the MGrid module
# %reset -f
import numpy as np
from ripplepy import MGrid

# from ripplepy.effective_ripple import Effective_Ripple
from ripplepy import set_extcur, initialize_mgrid_field,set_trace_parameters,compute_epstot,find_axis,trace_fieldline,plot_fieldline_3d,calculate_plasma_params
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

mgrid_filename = str(mgrid_path)
# extcur = [6.52271941985300E+05, 6.51868569367400E+05, 5.37743588647300E+05, 2.50000000000000E-07, 2.50000000000000E-07, 2.80949750000000E+04, -5.48049500000000E+04, 3.01228950000000E+04, 9.42409100000000E+04, 4.55138737653200E+04]
# extcur = [1.0]*10
extcur = None


# mgrid_filename = '/home/zkg/CN_H1_scan_fieldlines/H1_design/coils/mgrid_h1_design.nc'
# extcur = [50000, 5000, 1, -80000, -40000]
# initial_rz = (1.3,0)

initialize_mgrid_field(mgrid_filename, nfp,full_torus=full_torus)
extcur=set_extcur(extcur)

nturn = 200
nphi = 360
initial_rz = (1.8,0)
initial_rz = np.array(initial_rz, dtype=np.float64, order='F')
axis_rz, R0, axis_fieldline, trace_error_flag = find_axis(initial_rz, timeout=10.0, xtol=1e-5, max_iter=100)
print(f"✓ Magnetic axis found at R={axis_rz[0]:.10f}, Z={axis_rz[1]:.10f}, R0={R0:.10f}")