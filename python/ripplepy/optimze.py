import numpy as np
from scipy.optimize import differential_evolution
from .ripple import find_axis, compute_epstot,calculate_plasma_params,set_extcur,compute_initial_gradpsi_nemov,set_trace_parameters



def objective_function(extcur, initial_rz, nturn, nphi,delt_r=0.05):
    
    extcur=set_extcur(extcur)
    axis_rz, R0, axis_fieldline, success = find_axis(initial_rz, xtol=1e-5, max_iter=100,delta_r=0.01, verbose=False)
    if not success:
        print("Axis optimization did not converge")
        return np.inf  # 返回一个很大的值，表示优化失败
    else:
        print(f"✓ Magnetic axis found at R={axis_rz[0]:.10f}, Z={axis_rz[1]:.10f}, R0={R0:.10f}")
    RZ = np.array([axis_rz[0]+delt_r, axis_rz[1]], dtype=np.float64, order='F')
    initial_gradpsi = compute_initial_gradpsi_nemov(extcur, RZ[0],RZ[1],verbose=True)
    set_trace_parameters(nturn, nphi, verbose=False)
    fieldline_data = np.zeros((nturn*nphi, 20), dtype=np.float64, order='F')
    epsilon_eff, bboundary ,fieldline_data,trace_istate= compute_epstot(R0, extcur, RZ, initial_gradpsi, fieldline_data, return_fieldline=True)
    if trace_istate == 0:
        vol,Am,iota = calculate_plasma_params(fieldline_data, axis_fieldline, nturn, nphi, R0)
        print(f"✓ Plasma parameters calculated: Volume={vol:.3f}, Major radius={Am:.3f}, iota={iota:.3f}")
        return epsilon_eff
    else:
        print("Fieldline tracing failed during compute_epstot")
        return np.inf  # 返回一个很大的值，表示优化失败


