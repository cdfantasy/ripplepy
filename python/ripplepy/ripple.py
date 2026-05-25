
import numpy as np
from scipy.optimize import root
from scipy.integrate import simpson
from importlib import import_module
from .mgrid import MGrid

try:
    # Prefer the f90wrap-generated high-level API.
    _effective_ripple_mod = import_module(".effective_ripple", __package__)
    Effective_Ripple = _effective_ripple_mod.Effective_Ripple
except ImportError as e:
    print(f"Failed to import Effective_Ripple wrapper: {e}")
    Effective_Ripple = None


_CURRENT_N_EXT_CUR = None


def set_trace_verbose(flag, verbose=True):
    """Set or clear Fortran module verbose writes.

    Parameters
    ----------
    flag : bool or int
        If truthy, enable Fortran writes; if falsy, disable them.
    verbose : bool, optional
        If True (default) print a confirmation message.
    """
    if Effective_Ripple is None:
        raise ImportError("Effective_Ripple was not imported successfully.")

    val = 1 if bool(flag) else 0
    Effective_Ripple.set_trace_verbose(int(val))
    if verbose:
        # Query back to ensure the wrapper applied the value
        try:
            cur = int(Effective_Ripple.get_trace_verbose())
        except Exception:
            cur = val
        print(f"✓ trace_verbose set to {cur}")


def get_trace_verbose():
    """Return current value of Fortran module `trace_verbose` (0 or 1)."""
    if Effective_Ripple is None:
        raise ImportError("Effective_Ripple was not imported successfully.")
    return int(Effective_Ripple.get_trace_verbose())

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
        print(f"✓ No extcur provided; using raw.")
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

def set_trace_parameters(nturn, nphi,verbose=True):
    """Set the tracing parameters in the Fortran backend."""
    if Effective_Ripple is None:
        raise ImportError("Effective_Ripple was not imported successfully.")
    Effective_Ripple.set_trace_parameters(int(nturn), int(nphi))
    if verbose:
        print(f"✓ Trace parameters set: nturn={nturn}, nphi={nphi}")


def trace_fieldline(initial_rz=None, initial_gradpsi=None,nturn=400, nphi=360, extcur=None, verbose=False):
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
    
    # configure Fortran backend and allocate Fortran-contiguous buffers
    set_trace_parameters(nturn, nphi, verbose=False)
    npoints = int(nturn) * int(nphi)
    fieldline_data = np.zeros((npoints, 20), dtype=np.float64, order="F")

    # Ensure small input arrays are Fortran-contiguous as well
    initial_rz_f = np.asfortranarray(np.asarray(initial_rz, dtype=np.float64))
    initial_gradpsi_f = np.asfortranarray(np.asarray(initial_gradpsi, dtype=np.float64))

    # Call the Fortran wrapper. Different f2py/f90wrap wrappers return
    # different shapes: could be None, an int status, or a tuple that
    # includes the status. Handle all robustly.
    trace_istate = Effective_Ripple.trace_gradpsi_internal(fieldline_data, initial_rz_f, initial_gradpsi_f)
    if trace_istate != 0 and verbose:
        print(f"✗ Field line tracing failed with istate={trace_istate}")
    return fieldline_data, trace_istate

def compute_epstot(R0, extcur, initial_rz, initial_gradpsi=None,
                   fieldline_data=None, return_fieldline=False):
    """
    Compute total effective ripple (epsilon_eff) and boundary B field.
    
    This is a high-level wrapper around Effective_Ripple.compute_ripple that 
    handles input conversion and data transfer.
    
    Parameters
    ----------
    R0 : float
        Major radius.
    # extcur : array_like, shape (n_ext_cur,)
    #     External coil currents.
    initial_rz : array_like, shape (2,)
        Initial position (R, Z) on the field line.
    initial_gradpsi : array_like, shape (3,), optional
        Initial Nemov gradient-psi vector [P, G, Q]. If None, defaults to [1, 0, 0].
    fieldline_data : ndarray, optional
        Preallocated Fortran-contiguous array with shape (npoints, 20). If not
        provided and return_fieldline=True, a buffer is allocated using the
        current Fortran backend trace settings.
    return_fieldline : bool, optional
        If True, also return the traced field line data (default: False).
    
    Returns
    -------
    epsilon_eff : float
        Total effective ripple.
    bboundary : float
        Boundary magnetic field strength.
    fieldline_data : ndarray, shape (nturn*nphi, 20), optional
        Traced field line data if return_fieldline=True. Columns:
        [R, Z, phi, Br, Bz, Bphi, |B|, |grad_psi|, ..., derivatives]
    
    Examples
    --------
    >>> eps_eff, b_bound = compute_epstot(
    ...     extcur=np.array([1.0, -1.0, 0.5]),
    ...     initial_rz=np.array([1.5, 0.0])
    ... )
    
    >>> eps_eff, b_bound, fline = compute_epstot(
    ...     extcur=extcur,
    ...     initial_rz=[1.5, 0.0],
    ...     initial_gradpsi=[1.0, 0.0, 0.0],
    ...     fieldline_data=fieldline_data,
    ...     return_fieldline=True
    ... )
    """
    if Effective_Ripple is None:
        raise ImportError("Effective_Ripple was not imported successfully.")
    
    # Prepare inputs
    extcur_array = np.asarray(extcur, dtype=np.float64)
    initial_rz_array = np.asarray(initial_rz, dtype=np.float64)
    
    if initial_rz_array.shape != (2,):
        raise ValueError("initial_rz must contain exactly two values: (R, Z)")
    
    if initial_gradpsi is None:
        initial_gradpsi_array = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        initial_gradpsi_array = np.asarray(initial_gradpsi, dtype=np.float64)
        if initial_gradpsi_array.shape != (3,):
            raise ValueError("initial_gradpsi must contain exactly three values [P, G, Q]")
    
    if fieldline_data is None:
        npoints = int(getattr(Effective_Ripple, "npoints", 0))
        if npoints <= 0:
            raise ValueError(
                "fieldline_data was not provided and the Fortran backend trace "
                "parameters are not set. Call set_trace_parameters(...) first or "
                "pass a preallocated fieldline_data array."
            )
        fieldline_data = np.zeros((npoints, 20), dtype=np.float64, order='F')
    else:
        fieldline_data = np.asarray(fieldline_data, dtype=np.float64)
        if fieldline_data.ndim != 2 or fieldline_data.shape[1] != 20:
            raise ValueError("fieldline_data must have shape (npoints, 20)")
        if not fieldline_data.flags.f_contiguous:
            fieldline_data = np.asfortranarray(fieldline_data)
    
    # Call the Fortran compute_ripple
    epsilon_eff, bboundary, trace_istate = Effective_Ripple.compute_ripple(
        extcur_array, 
        initial_rz_array, 
        initial_gradpsi_array,
        fieldline_data
    )
    if trace_istate != 0:
        epsilon_eff = None
        bboundary = None
        print(f"✗ compute_ripple failed with istate={trace_istate}")
    else:
        epsilon_eff = epsilon_eff*R0**2  
        print(f"✓ Effective ripple computed: ε_eff={epsilon_eff:.6e}, B_boundary={bboundary:.6f} T")
    
    if return_fieldline:
        return epsilon_eff, bboundary, fieldline_data,trace_istate
    else:
        return epsilon_eff, bboundary,trace_istate

def calculate_plasma_params(fieldline_data, axis_data, nturn, nphi, Rm):
    """
    针对 R, Z, Phi 数据的体积及 Iota 计算
    :param fieldline_data: (nphi*nturn, 3) -> [R, Z, Phi]
    :param axis_data: (nphi+1, 3) -> [R, Z, Phi] 磁轴轨迹
    :param nturn: 极向采样点数
    :param nphi: 环向采样点数
    :param Rm: 大半径
    :param nfp: 场周期数
    :return: (volume, a_minor, iota)
    """

    # 1. 数据重塑 (按照 R, Z, Phi 分离)
    # 假设 fieldline_data 是按 [nturn, nphi] 排列的场线点
    R = fieldline_data[:, 0].reshape((nturn, nphi)).T # (nphi, nturn)
    Z = fieldline_data[:, 1].reshape((nturn, nphi)).T # (nphi, nturn)
    Phi = fieldline_data[:, 2].reshape((nturn, nphi)).T # (nphi, nturn)

    # 2. 获取磁轴参考点
    # 取对应环向位置的磁轴坐标。假设 axis_data 长度与 nphi 匹配
    # 如果 axis_data 是 (nphi+1, 3)，取前 nphi 个点
    R_axis = axis_data[:nphi, 0].reshape(-1, 1) # 变成列向量以便广播
    Z_axis = axis_data[:nphi, 1].reshape(-1, 1)

    # 3. 极向排序与体积计算
    # 计算极向角进行排序
    thetas = np.arctan2(Z - Z_axis, R - R_axis)
    
    # 预准备排序后的数组
    R_sorted = np.zeros_like(R)
    Z_sorted = np.zeros_like(Z)
    
    for i in range(nphi):
        idx = np.argsort(thetas[i, :])
        R_sorted[i, :] = R[i, idx]
        Z_sorted[i, :] = Z[i, idx]

    # 利用滚位计算截面积分 (Green公式)
    R_next = np.roll(R_sorted, -1, axis=1)
    Z_next = np.roll(Z_sorted, -1, axis=1)
    # 体积元累加
    vol_sum = np.sum((R_next - R_sorted) * (R_next + R_sorted) * (Z_next + Z_sorted))
    volume = abs(vol_sum) * (np.pi / nphi)
    
    # 有效小半径
    am = np.sqrt(volume / (2 * np.pi**2 * Rm))

    # 4. 计算 Iota (Rotational Transform)
    # 注意：计算 iota 通常需要追踪场线在极向角上的连续变化
    # 我们计算第一条场线（或平均）在环向一周内的极向角跨度
    
    # 取第一条追踪场线的坐标轨迹 (假设 fieldline_data 的排列允许提取连续轨迹)
    # 如果 data 是由平衡代码生成的磁面点，iota 通常是输入参数或通过磁通量导出的。
    # 这里提供一种基于坐标演化的几何估计方法：
    
    # 计算相对于磁轴的极向角演化（不排序，使用原始轨迹顺序）
    # 假设 R[0, :] 是一条场线在不同环向角下的 R 坐标
    # 但根据你的 nphi/nturn 结构，通常 R[:, 0] 是固定极向位置在环向的分布
    
    # 几何法估算：iota \approx d_theta / d_phi
    # 提取一条连续场线轨迹（这里假设 fieldline_data 的原始顺序即场线追踪顺序）
    # 如果数据只是磁面上的点阵而非轨迹，请注意该 iota 仅为几何近似
    d_theta = np.unwrap(thetas[:, 0]) # 展开相位，避免 +-pi 跳变
    d_phi = np.unwrap(Phi[:, 0])     # 环向角展开
    
    # 拟合斜率即为 iota
    if len(d_phi) > 1:
        iota = (d_theta[-1] - d_theta[0]) / (d_phi[-1] - d_phi[0])
    else:
        iota = 0.0

    return volume, am, abs(iota)

# def compute_kg_cylindrical(r, Br, Bz, Bphi, B, 
#                            dBr_dr, dBr_dz, dBr_dphi,
#                            dBz_dr, dBz_dz, dBz_dphi,
#                            dBphi_dr, dBphi_dz, dBphi_dphi,
#                            gradpsi_mag):
#     """
#     在柱坐标下计算测地曲率 k_G = [h × ((h·∇)h)] · (∇ψ / |∇ψ|)
#     使用你 fieldline_data 中提供的全部偏导数。
#     """
#     h_r = Br / B
#     h_phi = Bphi / B
#     h_z = Bz / B

#     # 计算 (h · ∇)h 的各分量（柱坐标需考虑基矢变化）
#     # 这里给出数值安全的实现（可进一步优化为矢量形式）
#     dh_r_dr = (dBr_dr * B - Br * (Br*dBr_dr + Bphi*dBphi_dr + Bz*dBz_dr)/B) / B**2
#     dh_r_dz = (dBr_dz * B - Br * (Br*dBr_dz + Bphi*dBphi_dz + Bz*dBz_dz)/B) / B**2
#     dh_r_dphi = (dBr_dphi * B - Br * (Br*dBr_dphi + Bphi*dBphi_dphi + Bz*dBz_dphi)/B) / B**2

#     dh_phi_dr = (dBphi_dr * B - Bphi * (Br*dBr_dr + Bphi*dBphi_dr + Bz*dBz_dr)/B) / B**2
#     dh_phi_dz = (dBphi_dz * B - Bphi * (Br*dBr_dz + Bphi*dBphi_dz + Bz*dBz_dz)/B) / B**2
#     dh_phi_dphi = (dBphi_dphi * B - Bphi * (Br*dBr_dphi + Bphi*dBphi_dphi + Bz*dBz_dphi)/B) / B**2

#     dh_z_dr = (dBz_dr * B - Bz * (Br*dBr_dr + Bphi*dBphi_dr + Bz*dBz_dr)/B) / B**2
#     dh_z_dz = (dBz_dz * B - Bz * (Br*dBr_dz + Bphi*dBphi_dz + Bz*dBz_dz)/B) / B**2
#     dh_z_dphi = (dBz_dphi * B - Bz * (Br*dBr_dphi + Bphi*dBphi_dphi + Bz*dBz_dphi)/B) / B**2

#     # h · ∇ 操作子（柱坐标下对 h_r, h_phi, h_z 的贡献）
#     h_dot_grad_h_r = h_r * dh_r_dr + (h_phi / r) * dh_r_dphi + h_z * dh_r_dz - (h_phi**2 / r)
#     h_dot_grad_h_phi = h_r * dh_phi_dr + (h_phi / r) * dh_phi_dphi + h_z * dh_phi_dz + (h_r * h_phi / r)
#     h_dot_grad_h_z = h_r * dh_z_dr + (h_phi / r) * dh_z_dphi + h_z * dh_z_dz

#     # 矢量叉乘 h × (h·∇)h
#     cross_r = h_phi * h_dot_grad_h_z - h_z * h_dot_grad_h_phi
#     cross_phi = h_z * h_dot_grad_h_r - h_r * h_dot_grad_h_z
#     cross_z = h_r * h_dot_grad_h_phi - h_phi * h_dot_grad_h_r

#     # k_G = [cross · ∇ψ] / |∇ψ|   （这里近似用 |∇ψ| 归一化方向）
#     # 注意：实际中 ∇ψ 方向需与法向一致，你的 gradpsi_mag 已提供
#     k_G = (cross_r * (Br / B) + cross_phi * (Bphi / B) + cross_z * (Bz / B)) / gradpsi_mag   # 简化投影

#     return k_G   # 返回与数据同长度的数组




# def compute_effective_ripple(fieldline_data, R0, B0=None, num_b_prime=5000):
#     """
#     计算 effective ripple ε_eff^{3/2} 和 ε_eff。
    
#     参数:
#         fieldline_data: np.ndarray, 形状 (N, >=20)，列顺序与你提供的一致
#         R0: float, 装置平均大半径 (m)
#         B0: float or None, 参考磁场强度 (默认取数据中 B 的平均值)
#         num_b_prime: int, 对 b' 的采样点数（捕获参数扫描）
#         discard_fraction: float, 丢弃初始瞬态部分比例
    
#     返回: dict {'eps_eff_32': float, 'eps_eff': float, 'converged': bool}
#     """
#     from scipy.integrate import cumulative_trapezoid
#     # 1. 提取数据
#     r       = fieldline_data[:, 0]
#     phi     = fieldline_data[:, 2]
#     Br      = fieldline_data[:, 3]
#     Bz      = fieldline_data[:, 4]
#     Bphi    = fieldline_data[:, 5]
#     B       = fieldline_data[:, 6]
#     gradpsi = np.abs(fieldline_data[:, 10])
#     npoints = len(r)
#     # 其余偏导数（按你列索引）
#     dBr_dr   = fieldline_data[:, 11]
#     dBr_dz   = fieldline_data[:, 12]
#     dBr_dphi = fieldline_data[:, 13]
#     dBz_dr   = fieldline_data[:, 14]
#     dBz_dz   = fieldline_data[:, 15]
#     dBz_dphi = fieldline_data[:, 16]
#     dBphi_dr = fieldline_data[:, 17]
#     dBphi_dz = fieldline_data[:, 18]
#     dBphi_dphi = fieldline_data[:, 19]
     
    
#     if B0 is None:
#         B0 = np.mean(B)

#     # 2. 计算弧长 ds（以 φ 参数化，最准确）


#     dphi = np.diff(phi)
#     dphi = np.insert(dphi, 0, 0)
#     ds = (B / Bphi) * r * dphi                     # dl = R dφ * B / B_φ
#     ds_invB = r*dphi/Bphi                               # dl / B 用于后续积分权重
#     bmax = np.max(B)
#     bmin = np.min(B)
#     b_prime = np.linspace(bmin/B0, bmax/B0, num_b_prime)

#     # L = cumulative_trapezoid(ds, initial=0)        # 累计弧长

#     # 3. 计算 k_G
#     k_G = compute_kg_cylindrical(r, Br, Bz, Bphi, B,
#                                  dBr_dr, dBr_dz, dBr_dphi,
#                                  dBz_dr, dBz_dz, dBz_dphi,
#                                  dBphi_dr, dBphi_dz, dBphi_dphi,
#                                  gradpsi)
#     # db = (bmax - bmin)/num_b_prime/B0
#     H_I = np.zeros(num_b_prime)
#     for i in range(num_b_prime):
#         k = 0
#         for j in range(npoints):
#             H_j = np.zeros(npoints)
#             I_j = np.zeros(npoints)
#             if b_prime[i] > B[j]:
#                 H_sqrt_term = np.sqrt(b_prime[i] - B[j]/B0)
#                 I_sqrt_term = np.sqrt(1-B[j]/B0*b_prime[i])
#                 H_j[j] = ds[j]/(b_prime[i]*B[j])*H_sqrt_term*((4*B0/B[j])-(1/b_prime[i]))*gradpsi[j]*k_G[j]
#                 I_j[j] = ds[j]/B[j]*I_sqrt_term

#             H_I[i] = b_prime[i]*np.sum(H_j**2/I_j)

#     e1 = (np.pi*R0*R0)/(8*np.sqrt(2))*np.sum(ds_invB)
#     e2 = np.sqrt(np.sum(ds_invB*gradpsi))
#     e3 = np.sum(H_I)
#     eps_eff = e1*e2*e3
    
#     return e1, e2, e3, eps_eff



# def compute_effective_ripple(fieldline_data, R0, B0=None, num_b_prime=3000):
#     """
#     高精度版本：自动识别 bounce 区间 + 分段积分
#     计算 effective ripple ε_eff^{3/2} (Nemov et al. 1999)
    
#     参数:
#         fieldline_data: np.ndarray (N, >=20)，列顺序与你之前一致
#         R0: float，装置平均大半径 (m)
#         B0: float or None，参考磁场强度，默认取平均 B
#         num_b_prime: int，b' 采样点数（推荐 2000~5000）
    
#     返回: dict
#     """
#     # ====================== 1. 数据提取 ======================
#     r       = fieldline_data[:, 0]
#     Br      = fieldline_data[:, 3]
#     Bz      = fieldline_data[:, 4]
#     Bphi    = fieldline_data[:, 5]
#     B       = fieldline_data[:, 6]
#     gradpsi = np.abs(fieldline_data[:, 10])
    
#     dBr_dr   = fieldline_data[:, 11]
#     dBr_dz   = fieldline_data[:, 12]
#     dBr_dphi = fieldline_data[:, 13]
#     dBz_dr   = fieldline_data[:, 14]
#     dBz_dz   = fieldline_data[:, 15]
#     dBz_dphi = fieldline_data[:, 16]
#     dBphi_dr = fieldline_data[:, 17]
#     dBphi_dz = fieldline_data[:, 18]
#     dBphi_dphi = fieldline_data[:, 19]
    
#     if B0 is None:
#         B0 = np.mean(B)
    
#     # ====================== 2. 计算 ds ======================
#     phi = fieldline_data[:, 2]
#     dphi = np.diff(phi)
#     dphi = np.insert(dphi, 0, 0.0)
    
#     ds = (B / Bphi) * r * np.abs(dphi)          # 弧长元
    
#     L = cumulative_trapezoid(ds, initial=0.0)   # 累计弧长（仅用于参考）
    
#     # ====================== 3. 计算 k_G ======================
#     k_G = compute_kg_cylindrical(r, Br, Bz, Bphi, B,
#                                  dBr_dr, dBr_dz, dBr_dphi,
#                                  dBz_dr, dBz_dz, dBz_dphi,
#                                  dBphi_dr, dBphi_dz, dBphi_dphi,
#                                  gradpsi)
    
#     # ====================== 4. 长积分（全场线） ======================
#     int_ds_over_B       = trapezoid(1.0 / B, L)                    # ∫ ds / B
#     int_ds_over_B_gradpsi = trapezoid(gradpsi / B, L)              # ∫ (ds / B) |∇ψ|
    
#     # ====================== 5. b' 扫描 + bounce 区间积分 ======================
#     b_min = np.min(B) / B0 * 1.0001
#     b_max = 1.0
#     b_prime = np.linspace(b_min, b_max, num_b_prime)
#     dbp = b_prime[1] - b_prime[0]
    
#     integral_H2_over_I = 0.0
    
#     for b_p in b_prime:
#         # 被捕获区域：B < bp * B0
#         trapped_mask = B < (b_p * B0)
#         if not np.any(trapped_mask):
#             continue
        
#         # 找 bounce 区间边界（符号变化点）
#         diff_sign = np.diff(np.sign(b_p * B0 - B))
#         crossing_idx = np.where(diff_sign != 0)[0] + 1
        
#         # 所有可能起点（包括第一个被捕获点）
#         starts = np.concatenate(([0], crossing_idx))
#         ends   = np.concatenate((crossing_idx, [len(B)]))
        
#         H2_over_I_bp = 0.0
        
#         for istart, iend in zip(starts, ends):
#             if iend - istart < 3:          # 区间太短，跳过
#                 continue
#             if not np.any(trapped_mask[istart:iend]):
#                 continue
            
#             # 当前 bounce 区间数据
#             s_seg = L[istart:iend]
#             # ds_seg = ds[istart:iend]
#             B_seg = B[istart:iend]
#             gradpsi_seg = gradpsi[istart:iend]
#             k_G_seg = k_G[istart:iend]
            
#             # Ĥ_j 积分项 (Nemov Eq.(30))
#             sqrt_H = np.sqrt(b_p - B_seg / B0)
#             factor_H = (4.0 * B0 / B_seg) - (1.0 / b_p)
#             integrand_H = (1 / (b_p * B_seg)) * sqrt_H * factor_H * gradpsi_seg * k_G_seg
            
#             H_j = trapezoid(integrand_H, s_seg)
            
#             # Î_j 积分项 (Nemov Eq.(31))
#             sqrt_I = np.sqrt(1.0 - B_seg / (B0 * b_p))
#             integrand_I = (1 / B_seg) * sqrt_I
#             I_j = trapezoid(integrand_I, s_seg)
            
#             if I_j > 1e-12:                # 避免除零
#                 H2_over_I_bp += (H_j ** 2) / I_j
        
#         integral_H2_over_I += H2_over_I_bp * dbp
    
#     # ====================== 6. 最终计算 ε_eff^{3/2} ======================
#     prefactor = (np.pi * R0**2) / (8.0 * np.sqrt(2.0))
    
#     eps_eff = prefactor * int_ds_over_B / np.sqrt(int_ds_over_B_gradpsi) * integral_H2_over_I
    
#     return {
#         'eps_eff': float(eps_eff),
#         'int_ds_over_B': float(int_ds_over_B),
#         'int_ds_over_B_gradpsi': float(int_ds_over_B_gradpsi),
#         'integral_H2_over_I': float(integral_H2_over_I),
#         'num_b_prime_used': num_b_prime
#     }


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




def find_axis(initial_rz, xtol=1e-10, max_iter=200, delta_r=0.01, verbose=False):
        
    from scipy.optimize import root
    if verbose:
        print(f"\nSearching for magnetic axis...")
        print(f"  Initial guess: R={initial_rz[0]:.6f}, Z={initial_rz[1]:.6f}")

    # Effective_Ripple.set_trace_parameters(2, 360)
    # initial_gradpsi = [0,0,0]
    # initial_gradpsi = np.array(initial_gradpsi, dtype=np.float64, order='F')

    final_fieldline_data = None

    def axis_residual(candidate_rz):
        candidate_rz = np.array(candidate_rz, dtype=np.float64, order='F')

        fieldline_data, trace_istate = trace_fieldline(
            initial_rz=candidate_rz,
            initial_gradpsi=None,
            nturn=2,
            nphi=360,
            verbose=False
        )

        if trace_istate != 0:
            if verbose:
                print(f" R={candidate_rz[0]:.6f}, Z={candidate_rz[1]:.6f} Warning: trace_fieldline failed with ISTATE={trace_istate}")
            return np.array([1e10, 1e10])

        final_R = fieldline_data[360, 0]
        final_Z = fieldline_data[360, 1]
        return np.array([final_R - candidate_rz[0], final_Z - candidate_rz[1]])

    trial_points = [
        np.array(initial_rz, dtype=np.float64),
        np.array([initial_rz[0] + delta_r, initial_rz[1]], dtype=np.float64),
        np.array([initial_rz[0] - delta_r, initial_rz[1]], dtype=np.float64),
    ]

    trial_results = []
    for candidate_rz in trial_points:
        residual = axis_residual(candidate_rz)
        trial_results.append((np.linalg.norm(residual), candidate_rz, residual))

    trial_results.sort(key=lambda item: item[0])
    start_rz = trial_results[0][1]

    if verbose:
        print("  Initial candidate residuals:")
        for residual_norm, candidate_rz, _ in trial_results:
            print(f"    R={candidate_rz[0]:.6f}, Z={candidate_rz[1]:.6f}, residual={residual_norm:.2e}")
        print(f"  Selected start point: R={start_rz[0]:.6f}, Z={start_rz[1]:.6f}")

    # 单线程优化求解
    result = root(
        axis_residual,
        start_rz,
        method='hybr',
        tol=xtol,
        options={
            'maxfev': max_iter,
            'factor': 100
        }
    )

    if result.success:
        final_fieldline_data, trace_istate = trace_fieldline(
            initial_rz=result.x,
            initial_gradpsi=None,
            nturn=2,
            nphi=360,
            verbose=False
        )

    else:
        # print("Axis optimization did not converge")
        trace_istate = -1

    
    # 计算主半径 R0
    if final_fieldline_data is not None and trace_istate == 0:
        R0 = np.mean(np.sqrt(final_fieldline_data[:360, 0]**2 + final_fieldline_data[:360, 1]**2))
    else:
        R0 = np.nan
        print("No valid fieldline data")
    if trace_istate == 0:
        distance = np.linalg.norm(result.fun)
        if verbose:
            print("  Optimization completed:")
            print(f"    Axis position: R={result.x[0]:.10f}, Z={result.x[1]:.10f}")
            print(f"    Major radius R0: {R0:.10f}")
            print(f"    Distance error: {distance:.2e}")
            print(f"    Converged: {result.success}")
    
        return result.x, R0, final_fieldline_data, True
    else:
        print("  Optimization failed to converge.")
        return None, None, None, False