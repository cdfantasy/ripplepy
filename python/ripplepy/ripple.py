
import numpy as np
from scipy.optimize import root
from scipy.integrate import simpson
from importlib import import_module
from .mgrid import MGrid
from func_timeout import func_timeout, FunctionTimedOut

try:
    # Prefer the f90wrap-generated high-level API.
    _effective_ripple_mod = import_module(".effective_ripple", __package__)
    Effective_Ripple = _effective_ripple_mod.Effective_Ripple
except ImportError as e:
    print(f"Failed to import Effective_Ripple wrapper: {e}")
    Effective_Ripple = None


_CURRENT_N_EXT_CUR = None


_EPS_R = 1e-14
_EPS_B = 1e-15
_EPS_GRAD = 1e-14


def initialize_mgrid_field(mgrid_filename, nfp, full_torus=True):
    """Initialize the Fortran backend with an mgrid file and return the loaded grid."""
    if Effective_Ripple is None:
        raise ImportError("Effective_Ripple was not imported successfully.")

    mgrid = MGrid.from_file(mgrid_filename)
    print(f"✓ Loaded mgrid from '{mgrid_filename}' with shape (nr={mgrid.nr}, nz={mgrid.nz}, nphi={mgrid.nphi})")
    mgrid.expand_to_full_torus(nfp=nfp, full_torus=full_torus)

    phimin = 0.0
    phimax = 2 * np.pi if full_torus else 2 * np.pi / nfp

    Effective_Ripple.initialize_field(
        mgrid.br_arr,
        mgrid.bz_arr,
        mgrid.bp_arr,
        mgrid.rmin,
        mgrid.rmax,
        mgrid.nr,
        mgrid.zmin,
        mgrid.zmax,
        mgrid.nz,
        phimin,
        phimax,
        mgrid.nphi,
        mgrid.n_ext_cur,
    )
    global _CURRENT_N_EXT_CUR
    _CURRENT_N_EXT_CUR = int(mgrid.n_ext_cur)
    return mgrid


def set_extcur(extcur):
    """Write the current set into the Fortran backend."""
    if Effective_Ripple is None:
        raise ImportError("Effective_Ripple was not imported successfully.")
    if extcur is None:
        if _CURRENT_N_EXT_CUR is None:
            raise ValueError("Field not initialized. Call initialize_mgrid_field(...) before set_extcur(None).")
        extcur = np.ones(_CURRENT_N_EXT_CUR, dtype=np.float64)
        print(f"✓ No extcur provided; using raw")
    else:
        print(f"✓ Setting extcur: {extcur}")
    extcur_array = np.asarray(extcur, dtype=np.float64)
    Effective_Ripple.sum_bfield_internal(extcur_array)
    return extcur_array


def get_bfield_matrix(extcur, r, z, phi):
    """Return [Br, Bz, Bphi, derivatives] at one point or many points."""
    set_extcur(extcur)

    if np.isscalar(r):
        result = Effective_Ripple.interpolate_field(float(r), float(z), float(phi))
        return np.array(result, dtype=np.float64)

    r_arr = np.atleast_1d(r).astype(np.float64)
    z_arr = np.atleast_1d(z).astype(np.float64)
    phi_arr = np.atleast_1d(phi).astype(np.float64)

    if not (len(r_arr) == len(z_arr) == len(phi_arr)):
        raise ValueError("r, z, and phi arrays must have the same length")

    results = np.zeros((len(r_arr), 12), dtype=np.float64)
    for index in range(len(r_arr)):
        results[index, :] = Effective_Ripple.interpolate_field(
            r_arr[index], z_arr[index], phi_arr[index]
        )
    return results

def set_trace_parameters(nturn, nphi):
    """Set the tracing parameters in the Fortran backend."""
    if Effective_Ripple is None:
        raise ImportError("Effective_Ripple was not imported successfully.")
    Effective_Ripple.set_trace_parameters(int(nturn), int(nphi))
    print(f"✓ Trace parameters set: nturn={nturn}, nphi={nphi}")

def trace_fieldline(initial_rz=None, initial_gradpsi=None,nturn=100, nphi=360, extcur=None):
    """Trace a field line directly, without object wrappers."""
    if Effective_Ripple is None:
        raise ImportError("Effective_Ripple was not imported successfully.")

    initial_rz = np.asarray(initial_rz, dtype=np.float64)
    if initial_rz.shape != (2,):
        raise ValueError("initial_rz must contain exactly two values: (R, Z)")

    if initial_gradpsi is None:
        initial_gradpsi = np.zeros(3, dtype=np.float64, order="F")
    else:
        initial_gradpsi = np.asarray(initial_gradpsi, dtype=np.float64, order="F")
        if initial_gradpsi.shape != (3,):
            raise ValueError("initial_gradpsi must contain exactly three values")
    
    if extcur is not None:
        set_extcur(extcur)
    
    set_trace_parameters(nturn, nphi)

    fieldline_data = np.zeros((int(nturn) * int(nphi), 20), dtype=np.float64, order="F")
    Effective_Ripple.trace_gradpsi_internal(fieldline_data, initial_rz, initial_gradpsi)
    return fieldline_data
    
def compute_kg_cylindrical(r, Br, Bz, Bphi, B, 
                           dBr_dr, dBr_dz, dBr_dphi,
                           dBz_dr, dBz_dz, dBz_dphi,
                           dBphi_dr, dBphi_dz, dBphi_dphi,
                           gradpsi_mag):
    """
    在柱坐标下计算测地曲率 k_G = [h × ((h·∇)h)] · (∇ψ / |∇ψ|)
    使用你 fieldline_data 中提供的全部偏导数。
    """
    h_r = Br / B
    h_phi = Bphi / B
    h_z = Bz / B

    # 计算 (h · ∇)h 的各分量（柱坐标需考虑基矢变化）
    # 这里给出数值安全的实现（可进一步优化为矢量形式）
    dh_r_dr = (dBr_dr * B - Br * (Br*dBr_dr + Bphi*dBphi_dr + Bz*dBz_dr)/B) / B**2
    dh_r_dz = (dBr_dz * B - Br * (Br*dBr_dz + Bphi*dBphi_dz + Bz*dBz_dz)/B) / B**2
    dh_r_dphi = (dBr_dphi * B - Br * (Br*dBr_dphi + Bphi*dBphi_dphi + Bz*dBz_dphi)/B) / B**2

    dh_phi_dr = (dBphi_dr * B - Bphi * (Br*dBr_dr + Bphi*dBphi_dr + Bz*dBz_dr)/B) / B**2
    dh_phi_dz = (dBphi_dz * B - Bphi * (Br*dBr_dz + Bphi*dBphi_dz + Bz*dBz_dz)/B) / B**2
    dh_phi_dphi = (dBphi_dphi * B - Bphi * (Br*dBr_dphi + Bphi*dBphi_dphi + Bz*dBz_dphi)/B) / B**2

    dh_z_dr = (dBz_dr * B - Bz * (Br*dBr_dr + Bphi*dBphi_dr + Bz*dBz_dr)/B) / B**2
    dh_z_dz = (dBz_dz * B - Bz * (Br*dBr_dz + Bphi*dBphi_dz + Bz*dBz_dz)/B) / B**2
    dh_z_dphi = (dBz_dphi * B - Bz * (Br*dBr_dphi + Bphi*dBphi_dphi + Bz*dBz_dphi)/B) / B**2

    # h · ∇ 操作子（柱坐标下对 h_r, h_phi, h_z 的贡献）
    h_dot_grad_h_r = h_r * dh_r_dr + (h_phi / r) * dh_r_dphi + h_z * dh_r_dz - (h_phi**2 / r)
    h_dot_grad_h_phi = h_r * dh_phi_dr + (h_phi / r) * dh_phi_dphi + h_z * dh_phi_dz + (h_r * h_phi / r)
    h_dot_grad_h_z = h_r * dh_z_dr + (h_phi / r) * dh_z_dphi + h_z * dh_z_dz

    # 矢量叉乘 h × (h·∇)h
    cross_r = h_phi * h_dot_grad_h_z - h_z * h_dot_grad_h_phi
    cross_phi = h_z * h_dot_grad_h_r - h_r * h_dot_grad_h_z
    cross_z = h_r * h_dot_grad_h_phi - h_phi * h_dot_grad_h_r

    # k_G = [cross · ∇ψ] / |∇ψ|   （这里近似用 |∇ψ| 归一化方向）
    # 注意：实际中 ∇ψ 方向需与法向一致，你的 gradpsi_mag 已提供
    k_G = (cross_r * (Br / B) + cross_phi * (Bphi / B) + cross_z * (Bz / B)) / gradpsi_mag   # 简化投影

    return k_G   # 返回与数据同长度的数组


def compute_effective_ripple(fieldline_data, R0, B0=None, num_b_prime=5000):
    """
    计算 effective ripple ε_eff^{3/2} 和 ε_eff。
    
    参数:
        fieldline_data: np.ndarray, 形状 (N, >=20)，列顺序与你提供的一致
        R0: float, 装置平均大半径 (m)
        B0: float or None, 参考磁场强度 (默认取数据中 B 的平均值)
        num_b_prime: int, 对 b' 的采样点数（捕获参数扫描）
        discard_fraction: float, 丢弃初始瞬态部分比例
    
    返回: dict {'eps_eff_32': float, 'eps_eff': float, 'converged': bool}
    """
    from scipy.integrate import cumulative_trapezoid
    # 1. 提取数据
    r       = fieldline_data[:, 0]
    phi     = fieldline_data[:, 2]
    Br      = fieldline_data[:, 3]
    Bz      = fieldline_data[:, 4]
    Bphi    = fieldline_data[:, 5]
    B       = fieldline_data[:, 6]
    gradpsi = np.abs(fieldline_data[:, 10])
    npoints = len(r)
    # 其余偏导数（按你列索引）
    dBr_dr   = fieldline_data[:, 11]
    dBr_dz   = fieldline_data[:, 12]
    dBr_dphi = fieldline_data[:, 13]
    dBz_dr   = fieldline_data[:, 14]
    dBz_dz   = fieldline_data[:, 15]
    dBz_dphi = fieldline_data[:, 16]
    dBphi_dr = fieldline_data[:, 17]
    dBphi_dz = fieldline_data[:, 18]
    dBphi_dphi = fieldline_data[:, 19]
     
    
    if B0 is None:
        B0 = np.mean(B)

    # 2. 计算弧长 ds（以 φ 参数化，最准确）


    dphi = np.diff(phi)
    dphi = np.insert(dphi, 0, 0)
    ds = (B / Bphi) * r * dphi                     # dl = R dφ * B / B_φ
    ds_invB = r*dphi/Bphi                               # dl / B 用于后续积分权重
    bmax = np.max(B)
    bmin = np.min(B)
    b_prime = np.linspace(bmin/B0, bmax/B0, num_b_prime)

    # L = cumulative_trapezoid(ds, initial=0)        # 累计弧长

    # 3. 计算 k_G
    k_G = compute_kg_cylindrical(r, Br, Bz, Bphi, B,
                                 dBr_dr, dBr_dz, dBr_dphi,
                                 dBz_dr, dBz_dz, dBz_dphi,
                                 dBphi_dr, dBphi_dz, dBphi_dphi,
                                 gradpsi)
    # db = (bmax - bmin)/num_b_prime/B0
    H_I = np.zeros(num_b_prime)
    for i in range(num_b_prime):
        k = 0
        for j in range(npoints):
            H_j = np.zeros(npoints)
            I_j = np.zeros(npoints)
            if b_prime[i] > B[j]:
                H_sqrt_term = np.sqrt(b_prime[i] - B[j]/B0)
                I_sqrt_term = np.sqrt(1-B[j]/B0*b_prime[i])
                H_j[j] = ds[j]/(b_prime[i]*B[j])*H_sqrt_term*((4*B0/B[j])-(1/b_prime[i]))*gradpsi[j]*k_G[j]
                I_j[j] = ds[j]/B[j]*I_sqrt_term
            else:
                K = +1

        H_I[i] = np.sum(b_prime[i]*H_j**2/I_j)

    e1 = (np.pi*R0*R0)/(8*np.sqrt(2))*np.sum(ds_invB)
    e2 = np.sqrt(np.sum(ds_invB*gradpsi))
    e3 = np.sum(H_I)
    eps_eff = e1*e2*e3
    
    return e1, e2, e3, eps_eff


def plot_fieldline_3d(
    fieldline_data,
    color_by_b=True,
    colorscale="Viridis",
    line_width=4,
    title=None,
    ):
    """Plot a traced fieldline in 3D Cartesian coordinates.

    Parameters
    ----------
    fieldline_data : array_like, shape (n, >=7)
        Trace output whose columns include [R, Z, phi, ..., |B|].
    color_by_b : bool, optional
        If True, color the line by |B| (column 7 in Fortran 1-based indexing).
    colorscale : str, optional
        Plotly colorscale name used when ``color_by_b=True``.
    line_width : int or float, optional
        3D line width.
    title : str or None, optional
        Figure title. Uses a default title when None.
    show : bool, optional
        If True, call ``fig.show()`` before returning.

    Returns
    -------
    plotly.graph_objects.Figure
        The generated Plotly figure.
    """
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError("plotly is required for plot_fieldline_3d. Install with: pip install plotly") from exc

    data = np.asarray(fieldline_data, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] < 7:
        raise ValueError("fieldline_data must be a 2D array with at least 7 columns")

    r_lines = data[:, 0]
    z_lines = data[:, 1]
    phi_lines = data[:, 2]
    b_line = data[:, 6]

    x_lines = r_lines * np.cos(phi_lines)
    y_lines = r_lines * np.sin(phi_lines)

    line_kwargs = {"width": line_width}
    if color_by_b:
        line_kwargs.update(
            {
                "color": b_line,
                "colorscale": colorscale,
                "colorbar": {"title": "|B| (T)"},
            }
        )

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x_lines,
                y=y_lines,
                z=z_lines,
                mode="lines",
                line=line_kwargs,
            )
        ]
    )

    fig.update_layout(
        scene={
            "xaxis_title": "X (m)",
            "yaxis_title": "Y (m)",
            "zaxis_title": "Z (m)",
        },
        title=title or ("Fieldline Colored by |B|" if color_by_b else "Ripple Fieldline in 3D"),
    )
    fig.update_layout(scene_aspectmode="data")
    return fig


def compute_initial_gradpsi_nemov(extcur, R0, Z0, phi0=0.0, verbose=True):
    """Compute the initial Nemov grad-psi vector using direct backend calls."""
    try:
        b_matrix = get_bfield_matrix(extcur, R0, Z0, phi0)
        Br0 = b_matrix[0]
        Bz0 = b_matrix[1]
        Bphi0 = b_matrix[2]
    except Exception as exc:
        if verbose:
            print(f"Error getting magnetic field: {exc}")
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)

    B_mag0 = np.sqrt(Br0 ** 2 + Bz0 ** 2 + Bphi0 ** 2)

    N_R = -Bphi0
    N_phi = Br0
    N_Z = 0.0

    norm_N = np.sqrt(N_R ** 2 + N_phi ** 2 + N_Z ** 2)
    if norm_N < 1e-15:
        n_R = 1.0
        n_phi = 0.0
        n_Z = 0.0
    else:
        n_R = N_R / norm_N
        n_phi = N_phi / norm_N
        n_Z = N_Z / norm_N

    grad_psi_R_phys = n_R
    grad_psi_phi_phys = n_phi
    grad_psi_Z_phys = n_Z

    P0 = grad_psi_R_phys
    G0 = grad_psi_Z_phys
    Q0 = R0 ** 2 * grad_psi_phi_phys

    if verbose:
        print("=" * 60)
        print("AUTO-COMPUTED INITIAL ∇ψ")
        print("=" * 60)
        print(f"Initial point: R={R0:.6f}, Z={Z0:.6f}, φ={phi0:.6f}")
        print(f"Magnetic field: Br={Br0:.6f}, Bφ={Bphi0:.6f}, Bz={Bz0:.6f}")
        print(f"Magnetic field magnitude: {B_mag0:.6f}")
        print(f"Nemov variables: P={P0:.6f}, G={G0:.6f}, Q={Q0:.6f}")
        print("=" * 60)

    return np.array([P0, G0, Q0], dtype=np.float64)


def find_axis(initial_rz, timeout=10.0, xtol=1e-10, max_iter=200):
    """
    包装 find_axis 函数,设置超时限制
    
    Parameters
    ----------
    initial_rz : tuple
        初始 (R, Z) 猜测值
    timeout : float, optional
        超时时间(秒),默认 10.0
    xtol : float, optional
        收敛容差
    max_iter : int, optional
        最大迭代次数
        
    Returns
    -------
    tuple
        (axis_position, major_radius, fieldline_data, trace_error_flag)
        超时时返回 (nan, nan, None, True)
    """
    
    def _find_axis_worker():
        from scipy.optimize import root
        
        print(f"\nSearching for magnetic axis...")
        print(f"  Initial guess: R={initial_rz[0]:.6f}, Z={initial_rz[1]:.6f}")
        print(f"  Timeout: {timeout:.1f} s")

        Effective_Ripple.set_trace_parameters(2, 360)
        initial_gradpsi = [0,0,0]
        initial_gradpsi = np.array(initial_gradpsi, dtype=np.float64, order='F')

        final_fieldline_data = None

        def axis_residual(initial_rz):
            nonlocal final_fieldline_data
            initial_rz = np.array(initial_rz, dtype=np.float64, order='F')
            fieldline_data = np.zeros((2*360, 20), dtype=np.float64, order='F')
            try:
                Effective_Ripple.trace_gradpsi_internal(fieldline_data, initial_rz, initial_gradpsi)
                final_R = fieldline_data[360, 0]
                final_Z = fieldline_data[360, 1]
                residual = np.array([final_R - initial_rz[0], final_Z - initial_rz[1]])
                final_fieldline_data = fieldline_data.copy()
                return residual
            except Exception as e:
                print(f"  Warning: Fieldline tracing failed: {e}")
                return np.array([1e10, 1e10])

        # 单线程优化求解
        result = root(
            axis_residual,
            initial_rz,
            method='hybr',
            tol=xtol,
            options={
                'maxfev': max_iter * (len(initial_rz) + 1),
                'factor': 100
            }
        )
        
        # 计算主半径 R0
        if final_fieldline_data is not None:
            R0 = np.mean(np.sqrt(final_fieldline_data[:360, 0]**2 + final_fieldline_data[:360, 1]**2))
        else:
            R0 = np.nan
            raise RuntimeError("No valid fieldline data")
        
        distance = np.linalg.norm(result.fun)
        
        print("  Optimization completed:")
        print(f"    Axis position: R={result.x[0]:.10f}, Z={result.x[1]:.10f}")
        print(f"    Major radius R0: {R0:.10f}")
        print(f"    Distance error: {distance:.2e}")
        print(f"    Converged: {result.success}")
        
        return result.x, R0, final_fieldline_data, False

    # 使用 func_timeout 包装
    try:
        result = func_timeout(timeout, _find_axis_worker)
        return result
    
    except FunctionTimedOut:
        print(f"  ✗ Timeout: Magnetic axis search exceeded {timeout:.1f} s", flush=True)
        return np.array([np.nan, np.nan]), np.nan, None, True
    
    except Exception as e:
        print(f"  ✗ Error during axis search: {e}", flush=True)
        return np.array([np.nan, np.nan]), np.nan, None, True
