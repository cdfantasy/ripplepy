
import numpy as np
from scipy.optimize import root
from .mgrid import MGrid


try:
    from . import _effective_ripple as Effective_Ripple
    # print("Successfully imported _effective_ripple.")
except ImportError as e:
    print(f"Failed to import _effective_ripple: {e}")
    Effective_Ripple = None

def initialize_mgrid_field(mgrid_filename,nfp,full_torus=True):

    mgrid_raw = MGrid.from_file(mgrid_filename)
    mgrid_raw.expand_to_full_torus(nfp=nfp, full_torus=full_torus)
    nr, nz, nphi = mgrid_raw.nr, mgrid_raw.nz, mgrid_raw.nphi
    rmin, rmax = mgrid_raw.rmin, mgrid_raw.rmax
    zmin, zmax = mgrid_raw.zmin, mgrid_raw.zmax
    phimin = 0.0
    phimax = 2*np.pi/nfp if not full_torus else 2*np.pi
    print(f"✓ MGrid loaded successfully")
    print(f"  Grid dimensions: nr={nr}, nz={nz}, nphi={nphi}")
    print(f"  R range: [{rmin:.3f}, {rmax:.3f}]")
    print(f"  Z range: [{zmin:.3f}, {zmax:.3f}]")
    br_arr_db = mgrid_raw.br_arr
    bz_arr_db = mgrid_raw.bz_arr
    bp_arr_db = mgrid_raw.bp_arr
    nextcur = mgrid_raw.n_ext_cur

    Effective_Ripple.initialize_field(
        br_arr_db,bz_arr_db,bp_arr_db,
        rmin,rmax,nr,
        zmin,zmax,nz,
        phimin,phimax,nphi,
        nextcur
    )
    print("✓ Effective_Ripple field initialized.")
    return Effective_Ripple

def get_bfield_matrix(extcur, r, z, phi):
    """
    获取给定位置的磁场插值结果
    
    Parameters
    ----------
    extcur : array_like
        外部线圈电流组合
    r, z, phi : float or array_like
        插值位置坐标
        
    Returns
    -------
    b_matrix : ndarray, shape (12,) or (n, 12)
        磁场分量及其导数:
        [Br, Bz, Bphi, dBr/dr, dBr/dz, dBr/dphi, 
         dBz/dr, dBz/dz, dBz/dphi, dBphi/dr, dBphi/dz, dBphi/dphi]
    """
    if Effective_Ripple is None:
        raise ImportError("Effective_Ripple 未成功导入。")
    
    extcur = np.asarray(extcur, dtype=np.float64)
    Effective_Ripple.sum_bfield_internal(extcur)
    
    # 处理标量或数组输入
    if np.isscalar(r):
        # 单点插值
        result = Effective_Ripple.interpolate_field(float(r), float(z), float(phi))
        b_matrix = np.array(result, dtype=np.float64)
        print("✓ Magnetic field interpolated at 1 point.")
    else:
        # 多点插值
        r_arr = np.atleast_1d(r).astype(np.float64)
        z_arr = np.atleast_1d(z).astype(np.float64)
        phi_arr = np.atleast_1d(phi).astype(np.float64)
        
        if not (len(r_arr) == len(z_arr) == len(phi_arr)):
            raise ValueError("r, z, phi 数组长度必须一致")
        
        npts = len(r_arr)
        results = np.zeros((npts, 12), dtype=np.float64)
        
        for i in range(npts):
            result_tuple = Effective_Ripple.interpolate_field(
                r_arr[i], z_arr[i], phi_arr[i]
            )
            results[i, :] = result_tuple
        
        b_matrix = results
        print(f"✓ Magnetic field interpolated at {npts} points.")
    
    return b_matrix

