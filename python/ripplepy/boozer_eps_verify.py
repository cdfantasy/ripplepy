#!/usr/bin/env python3
"""Compute ε_eff^(3/2) from Boozer Fourier harmonics using pure Python.

Two algorithms are provided:
  - ``eps_eff_pyneo_style``: η-particle state machine (matches pyneo's flint_bo + rhs_bo1)
  - ``eps_eff_from_boozer``: bp-scanning (matches ripplepy Fortran backend)
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

# ═══════════════════════════════════════════════════════════════════════
# 1.  Fast Fourier evaluation along a field line
# ═══════════════════════════════════════════════════════════════════════

def _fourier_sum_cos(
    coeff: NDArray[np.float64],
    xm: NDArray[np.int32],
    xn: NDArray[np.int32],
    theta: NDArray[np.float64],
    zeta: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Σ coeff cos(xm θ - xn ζ)"""
    arg = np.outer(xm, theta) - np.outer(xn, zeta)
    return np.dot(coeff, np.cos(arg))

def _fourier_sum_sin(
    coeff: NDArray[np.float64],
    xm: NDArray[np.int32],
    xn: NDArray[np.int32],
    theta: NDArray[np.float64],
    zeta: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Σ coeff sin(xm θ - xn ζ)"""
    arg = np.outer(xm, theta) - np.outer(xn, zeta)
    return np.dot(coeff, np.sin(arg))

def _fourier_sum_deriv_theta_cos(
    coeff: NDArray[np.float64],
    xm: NDArray[np.int32],
    xn: NDArray[np.int32],
    theta: NDArray[np.float64],
    zeta: NDArray[np.float64],
) -> NDArray[np.float64]:
    """∂/∂θ: Σ +xm·coeff cos(xm θ - xn ζ)"""
    arg = np.outer(xm, theta) - np.outer(xn, zeta)
    return np.dot(+xm.astype(np.float64) * coeff, np.cos(arg))

def _fourier_sum_deriv_zeta_sin(
    coeff: NDArray[np.float64],
    xm: NDArray[np.int32],
    xn: NDArray[np.int32],
    theta: NDArray[np.float64],
    zeta: NDArray[np.float64],
) -> NDArray[np.float64]:
    """∂/∂ζ: Σ -xn·coeff cos(xm θ - xn ζ)"""
    arg = np.outer(xm, theta) - np.outer(xn, zeta)
    return np.dot(-xn.astype(np.float64) * coeff, np.sin(arg))

# ═══════════════════════════════════════════════════════════════════════
# 2.  Field-line sampling with analytic Fourier evaluation
# ═══════════════════════════════════════════════════════════════════════

@dataclasses.dataclass
class FieldLineData:
    """All quantities sampled along the analytic Boozer field line."""
    zeta: NDArray[np.float64]         # toroidal angle (same as phi in Boozer)
    B: NDArray[np.float64]            # |B|
    gradpsi: NDArray[np.float64]      # |∇ψ| = sqrg11
    kg_gradpsi: NDArray[np.float64]   # |∇ψ|·κ_G  (geodcu in pyneo)
    pard: NDArray[np.float64]         # ∂|B|/∂ζ + ι·∂|B|/∂θ  (along field line)
    iota: float
    I_val: float                      # bvco = curr_pol
    J_val: float                      # buco = curr_tor
    nfp: int

def _find_bmax_location(
    bmnc: NDArray[np.float64],
    xm: NDArray[np.int32],
    xn: NDArray[np.int32],
    ntheta: int = 360,
    nphi: int = 360,
) -> Tuple[float, float]:
    """Find (θ, φ) location of B maximum from Fourier coefficients."""
    th = np.linspace(0, 2 * np.pi, ntheta)
    ph = np.linspace(0, 2 * np.pi, nphi)
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    B2d = _fourier_sum_cos(bmnc, xm, xn, TH.ravel(), PH.ravel())
    imax = np.argmax(B2d)
    return float(TH.ravel()[imax]), float(PH.ravel()[imax])

def         _sample_fieldline_fourier(
    bmnc: NDArray[np.float64],
    rmnc: NDArray[np.float64],
    zmns: NDArray[np.float64],
    xm: NDArray[np.int32],
    xn: NDArray[np.int32],
    theta: NDArray[np.float64],
    zeta: NDArray[np.float64],
    pmns: Optional[NDArray[np.float64]] = None,
) -> tuple:
    """Single-pass Fourier evaluation: 9 or 12 arrays in one mode loop.

    Returns (B, dBdt, dBdz, R, dRdt, dRdz, Z, dZdt, dZdz, *nu_vars)
    where nu_vars = (l, p_tb, p_pb) if pmns is provided, else empty.
    """
    npts = len(theta)
    active = np.where(
        (np.abs(bmnc) > 0) | (np.abs(rmnc) > 0) | (np.abs(zmns) > 0)
    )[0]
    if pmns is not None:
        active = np.union1d(active, np.where(np.abs(pmns) > 0)[0])

    b = np.zeros(npts, dtype=np.float64)
    b_tb = np.zeros(npts, dtype=np.float64); b_pb = np.zeros(npts, dtype=np.float64)
    r = np.zeros(npts, dtype=np.float64)
    r_tb = np.zeros(npts, dtype=np.float64); r_pb = np.zeros(npts, dtype=np.float64)
    z = np.zeros(npts, dtype=np.float64)
    z_tb = np.zeros(npts, dtype=np.float64); z_pb = np.zeros(npts, dtype=np.float64)
    l = np.zeros(npts, dtype=np.float64) if pmns is not None else None
    p_tb = np.zeros(npts, dtype=np.float64) if pmns is not None else None
    p_pb = np.zeros(npts, dtype=np.float64) if pmns is not None else None

    for m in active:
        xm_m = float(xm[m]); xn_n = float(xn[m])
        arg = xm_m * theta - xn_n * zeta
        cos_a = np.cos(arg); sin_a = np.sin(arg)

        bc = bmnc[m]
        if bc != 0.0:
            b    += bc * cos_a
            b_tb += -xm_m * bc * sin_a
            b_pb +=  xn_n * bc * sin_a

        ri = rmnc[m]
        if ri != 0.0:
            r    += ri * cos_a
            r_tb += -xm_m * ri * sin_a
            r_pb +=  xn_n * ri * sin_a

        zi = zmns[m]
        if zi != 0.0:
            z    += zi * sin_a
            z_tb +=  xm_m * zi * cos_a
            z_pb += -xn_n * zi * cos_a

        if pmns is not None:
            li = pmns[m]
            if li != 0.0:
                l    += li * sin_a
                p_tb += -xm_m * li * cos_a    # p_tb = -m*li*cosv
                p_pb +=  xn_n * li * cos_a    # p_pb = +n*li*cosv

    if pmns is not None:
        return b, b_tb, b_pb, r, r_tb, r_pb, z, z_tb, z_pb, l, p_tb, p_pb
    return b, b_tb, b_pb, r, r_tb, r_pb, z, z_tb, z_pb


def sample_fieldline_from_boozer(
    booz: Dict[str, Any],
    surf_idx: int,
    theta0: Optional[float] = None,
    nzeta: int = 1024,
    nturn: int = 64,
) -> FieldLineData:
    """Sample a single field line θ(ζ)=θ₀+ιζ by direct Fourier summation."""
    xm = np.asarray(booz["ixm_b"], dtype=np.int32)
    xn = np.asarray(booz["ixn_b"], dtype=np.int32)
    bmnc = np.asarray(booz["bmnc_b"][surf_idx, :], dtype=np.float64)
    rmnc = np.asarray(booz["rmnc_b"][surf_idx, :], dtype=np.float64)
    zmns = np.asarray(booz["zmns_b"][surf_idx, :], dtype=np.float64)
    pmns = booz.get("pmns_b", None)
    if pmns is not None:
        pmns = np.asarray(pmns[surf_idx, :], dtype=np.float64)

    iota = float(np.asarray(booz["iota_b"]).flat[surf_idx])
    I_   = float(np.asarray(booz["bvco_b"]).flat[surf_idx])
    J_   = float(np.asarray(booz["buco_b"]).flat[surf_idx])
    nfp  = int(np.asarray(booz.get("nfp_b", booz.get("nfp", 1))).flat[0])

    if theta0 is None:
        theta0, phi0_bmax = _find_bmax_location(bmnc, xm, xn)

    ntot = nzeta * nturn
    dphi = 2.0 * np.pi / nzeta
    zeta = np.arange(ntot, dtype=np.float64) * dphi
    theta = theta0 + iota * (zeta - phi0_bmax)

    result = _sample_fieldline_fourier(bmnc, rmnc, zmns, xm, xn, theta, zeta, pmns=pmns)
    if pmns is not None:
        b, b_tb, b_pb, r, r_tb, r_pb, z, z_tb, z_pb, l, p_tb, p_pb = result
    else:
        b, b_tb, b_pb, r, r_tb, r_pb, z, z_tb, z_pb = result
        l = p_tb = p_pb = np.zeros_like(b)

    # ── neo_fourier.f90:138-139:  p_tb = p_tb * twopi/nfp,  p_pb = ONE + p_pb * twopi/nfp ──
    p_tb = p_tb * (2.0 * np.pi / nfp)
    p_pb = 1.0 + p_pb * (2.0 * np.pi / nfp)

    # --- metric: g_ij (neo_fourier.f90:173-175) ---
    gtbtb = r_tb*r_tb + z_tb*z_tb + r*r * p_tb*p_tb
    gpbpb = r_pb*r_pb + z_pb*z_pb + r*r * p_pb*p_pb
    gtbpb = r_tb*r_pb + z_tb*z_pb + r*r * p_tb*p_pb

    fac = I_ + iota * J_
    isqrg  = b*b / fac
    sqrg11 = np.sqrt(np.abs(gtbtb*gpbpb - gtbpb*gtbpb)) * isqrg
    kg = (J_*b_pb - I_*b_tb) / fac
    pard = b_pb + iota * b_tb

    return FieldLineData(
        zeta=zeta, B=b, gradpsi=sqrg11, kg_gradpsi=kg,
        pard=pard, iota=iota, I_val=I_, J_val=J_, nfp=nfp,
    )


# ═══════════════════════════════════════════════════════════════════════
# 3.  Well detection & bounce-segment integration
# ═══════════════════════════════════════════════════════════════════════

def _find_local_minima(B: NDArray[np.float64]) -> NDArray[np.int32]:
    """Periodic local-minima indices of 1-D array B."""
    n = len(B)
    minima = []
    for i in range(n):
        if B[i] < B[(i - 1) % n] and B[i] < B[(i + 1) % n]:
            minima.append(i)
    if not minima:
        return np.array([0, n], dtype=np.int32)
    idx = np.array(minima, dtype=np.int32)
    return np.concatenate([idx, [idx[0] + n]])

def _integrate_bounce_segment(
    bp: float, b0: float, i1: int, i2: int,
    B: NDArray[np.float64],
    gradpsi: NDArray[np.float64],
    kg_gradpsi: NDArray[np.float64],
    dmeasure: NDArray[np.float64],
) -> Tuple[float, float]:
    """Integrate H and I over one bounce well [i1, i2) for a given bp."""
    n = len(B)
    H, I_val = 0.0, 0.0
    for k in range(i1, i2):
        idx = k % n
        b_loc = B[idx] / b0
        if bp <= b_loc:
            continue
        sqrt_term = np.sqrt(bp - b_loc)
        inv_B2 = 1.0 / B[idx]**2
        dI = sqrt_term * inv_B2 / np.sqrt(bp)
        I_val += dI * dmeasure[idx]
        dH = sqrt_term * inv_B2 * (4.0 / b_loc - 1.0 / bp) * kg_gradpsi[idx] / bp
        H += dH * dmeasure[idx]
    return H, I_val

def _compute_H2_over_I_for_bp(
    bp: float, b0: float,
    B: NDArray[np.float64],
    gradpsi: NDArray[np.float64],
    kg_gradpsi: NDArray[np.float64],
    dmeasure: NDArray[np.float64],
) -> float:
    """For one bp value, sum H²/I over all identified bounce wells."""
    minima = _find_local_minima(B)
    total = 0.0
    for k in range(len(minima) - 1):
        i1, i2 = minima[k], minima[k + 1]
        Hk, Ik = _integrate_bounce_segment(
            bp, b0, i1, i2, B, gradpsi, kg_gradpsi, dmeasure)
        if Ik > 1e-15:
            total += Hk * Hk / Ik
    return total

# ═══════════════════════════════════════════════════════════════════════
# 4.  Main computation
# ═══════════════════════════════════════════════════════════════════════

def eps_eff_from_boozer(
    booz: Dict[str, Any],
    surf_idx: int,
    theta0: float = 0.0,
    nzeta: int = 1024,
    nturn: int = 64,
    n_b: int = 5000,
    use_gauss: bool = False,
    n_gauss: int = 64,
    return_debug: bool = False,
) -> Dict[str, Any]:
    """ε_eff^(3/2) using ripplepy's B-profile-scan algorithm."""
    fl = sample_fieldline_from_boozer(booz, surf_idx, theta0, nzeta, nturn)
    B = fl.B; gp = fl.gradpsi; kg = fl.kg_gradpsi
    npts = len(B)
    dphi = 2.0 * np.pi / nzeta
    dmeasure = np.full(npts, dphi)
    b0 = np.max(B)
    bmin, bmax = np.min(B), b0

    e2 = np.sum(dmeasure / B**2)
    e3 = np.sum(dmeasure * gp / B**2)

    if use_gauss:
        from numpy.polynomial.legendre import leggauss
        nodes, weights = leggauss(n_gauss)
        bp_min, bp_max = bmin / b0, 1.0
        e1 = 0.0
        for node, wgt in zip(nodes, weights):
            bp = 0.5*(bp_max+bp_min) + 0.5*(bp_max-bp_min)*node
            e1 += _compute_H2_over_I_for_bp(bp,b0,B,gp,kg,dmeasure)*wgt*0.5*(bp_max-bp_min)
    else:
        dbp = (bmax - bmin) / (n_b - 1) / b0
        e1 = 0.0
        for j in range(n_b):
            bp = bmin/b0 + j*dbp
            e1 += _compute_H2_over_I_for_bp(bp,b0,B,gp,kg,dmeasure)

    xm = np.asarray(booz["ixm_b"], dtype=np.int32)
    xn = np.asarray(booz["ixn_b"], dtype=np.int32)
    rmnc = np.asarray(booz["rmnc_b"][surf_idx,:], dtype=np.float64)
    m0 = np.where((xm == 0) & (xn == 0))[0]
    rt0 = float(rmnc[m0[0]]) if len(m0) > 0 else 1.0
    rt0_sq = rt0**2

    eps_eff_32 = np.pi * rt0_sq / (8.0 * np.sqrt(2.0)) * e1 * e2 / e3**2
    eps_eff = eps_eff_32 ** (2.0 / 3.0)

    result = {"eps_eff_32": float(eps_eff_32), "eps_eff": float(eps_eff),
              "e1": float(e1), "e2": float(e2), "e3": float(e3),
              "rt0_squared": float(rt0_sq)}
    if return_debug:
        result["B"] = B; result["gp"] = gp; result["kg"] = kg
    return result

# ═══════════════════════════════════════════════════════════════════════
# 5.  pyneo-compatible state machine (matches pyneo's flint_bo + rhs_bo1)
# ═══════════════════════════════════════════════════════════════════════

def eps_eff_pyneo_style(
    booz: Dict[str, Any],
    surf_idx: int,
    theta0: Optional[float] = None,
    nturn: int = 64,
    npart: int = 100,
    nstep_per: int = 50,
    multra: int = 1,
) -> Dict[str, Any]:
    """ε_eff^(3/2) matching pyneo's flint_bo + rhs_bo1 algorithm exactly.

    Uses pyneo-style stepper: nstep_per points per toroidal transit,
    with analytic Fourier evaluation at each (θ, ζ) point inside the loop.
    """
    # --- extract surface data (same as sample_fieldline_from_boozer) ---
    xm = np.asarray(booz["ixm_b"], dtype=np.int32)
    xn = np.asarray(booz["ixn_b"], dtype=np.int32)
    bmnc_s = np.asarray(booz["bmnc_b"][surf_idx, :], dtype=np.float64)
    rmnc_s = np.asarray(booz["rmnc_b"][surf_idx, :], dtype=np.float64)
    zmns_s = np.asarray(booz["zmns_b"][surf_idx, :], dtype=np.float64)
    pmns_s = booz.get("pmns_b", None)
    if pmns_s is not None:
        pmns_s = np.asarray(pmns_s[surf_idx, :], dtype=np.float64)

    iota = float(np.asarray(booz["iota_b"]).flat[surf_idx])
    I_   = float(np.asarray(booz["bvco_b"]).flat[surf_idx])
    J_   = float(np.asarray(booz["buco_b"]).flat[surf_idx])
    nfp  = int(np.asarray(booz.get("nfp_b", booz.get("nfp", 1))).flat[0])

    if theta0 is None:
        theta0, phi0_bmax = _find_bmax_location(bmnc_s, xm, xn)
    else:
        phi0_bmax = 0.0

    # ── pyneo-style stepper and state machine ──

    ntot = nstep_per * nturn
    hphi = 2.0 * np.pi / nstep_per
    zeta_arr = phi0_bmax + np.arange(ntot, dtype=np.float64) * hphi
    theta_arr = theta0 + iota * (zeta_arr - phi0_bmax)

    # Single-pass Fourier (same as boozer_eps_verify)
    result = _sample_fieldline_fourier(bmnc_s, rmnc_s, zmns_s, xm, xn,
                                       theta_arr, zeta_arr, pmns=pmns_s)
    if pmns_s is not None:
        b_arr, b_tb_arr, b_pb_arr, r_arr, r_tb_arr, r_pb_arr, z_arr, z_tb_arr, z_pb_arr, l_arr, p_tb_arr, p_pb_arr = result
    else:
        b_arr, b_tb_arr, b_pb_arr, r_arr, r_tb_arr, r_pb_arr, z_arr, z_tb_arr, z_pb_arr = result
        l_arr = p_tb_arr = p_pb_arr = np.zeros_like(b_arr)

    # neo_fourier.f90:138-139 normalization
    p_tb_arr = p_tb_arr * (2.0 * np.pi / nfp)
    p_pb_arr = 1.0 + p_pb_arr * (2.0 * np.pi / nfp)

    # metric → |∇ψ|
    fac = I_ + iota * J_
    gtbtb = r_tb_arr*r_tb_arr + z_tb_arr*z_tb_arr + r_arr*r_arr * p_tb_arr*p_tb_arr
    gpbpb = r_pb_arr*r_pb_arr + z_pb_arr*z_pb_arr + r_arr*r_arr * p_pb_arr*p_pb_arr
    gtbpb = r_tb_arr*r_pb_arr + z_tb_arr*z_pb_arr + r_arr*r_arr * p_tb_arr*p_pb_arr
    sqrg11 = np.sqrt(np.abs(gtbtb*gpbpb - gtbpb*gtbpb)) * b_arr*b_arr / fac  # |∇ψ|
    kg_arr = (J_*b_pb_arr - I_*b_tb_arr) / fac                                  # |∇ψ|·κ_G
    pard_arr = b_pb_arr + iota * b_tb_arr                                       # parallel deriv

    b0 = np.max(b_arr); bmin = np.min(b_arr)
    inv_B2 = 1.0 / b_arr**2
    e2 = np.sum(inv_B2) * hphi          # ∫ dφ / B²
    e3 = np.sum(inv_B2 * sqrg11) * hphi # ∫ dφ |∇ψ| / B²

    etamin = bmin / b0
    heta = (1.0 - etamin) / (npart - 1)
    eta_vals = etamin + heta / 2.0 + np.arange(npart) * heta

    rmnc = np.asarray(booz["rmnc_b"][surf_idx,:], dtype=np.float64)
    m0 = np.where((xm == 0) & (xn == 0))[0]
    rt0 = float(rmnc[m0[0]]) if len(m0) > 0 else 1.0
    rt0_sq = rt0**2
    coeps = np.pi * rt0_sq * heta / (8.0 * np.sqrt(2.0))

    bra_arr = b_arr / b0; invB2_arr = inv_B2
    sqrt_eta_cache = np.sqrt(eta_vals)
    bigint_total = 0.0

    for i_eta, eta in enumerate(eta_vals):
        sqeta = sqrt_eta_cache[i_eta]
        isw = 0; iswst = 0; icount = 0; ipa = 0
        H_acc = 0.0; I_acc = 0.0; pard0 = pard_arr[0]

        for k in range(ntot - 1):
            idx0 = k; idx_h = k; idx1 = k + 1

            def _get(kk, frac=0.0):
                """Linear interpolation between kk and kk+1 at fraction frac."""
                if frac == 0.0:
                    return bra_arr[kk], invB2_arr[kk], kg_arr[kk], pard_arr[kk]
                w1 = frac; w0 = 1.0 - w1
                bra = w0*bra_arr[kk] + w1*bra_arr[kk+1]
                invB2 = w0*invB2_arr[kk] + w1*invB2_arr[kk+1]
                kg = w0*kg_arr[kk] + w1*kg_arr[kk+1]
                pard = w0*pard_arr[kk] + w1*pard_arr[kk+1]
                return bra, invB2, kg, pard

            # RK4 stage 1: ζ_k
            bra, invB2_v, kg_v, _ = _get(k, 0.0)
            subsq = 1.0 - bra / eta
            if subsq > 0.0:
                sq = np.sqrt(subsq) * invB2_v
                I1 = sq * hphi
                H1 = sq * (4.0 / bra - 1.0 / eta) * kg_v / sqeta * hphi
            else:
                I1 = H1 = 0.0

            # RK4 stage 2: ζ_k + hphi/2
            bra, invB2_v, kg_v, _ = _get(k, 0.5)
            subsq = 1.0 - bra / eta
            if subsq > 0.0:
                sq = np.sqrt(subsq) * invB2_v
                I2 = sq * hphi
                H2 = sq * (4.0 / bra - 1.0 / eta) * kg_v / sqeta * hphi
            else:
                I2 = H2 = 0.0

            # RK4 stage 3: ζ_k + hphi/2 (same point, different slope in ODE)
            I3 = I2; H3 = H2

            # RK4 stage 4: ζ_{k+1}
            bra, invB2_v, kg_v, _ = _get(k + 1, 0.0)
            subsq = 1.0 - bra / eta
            if subsq > 0.0:
                sq = np.sqrt(subsq) * invB2_v
                I4 = sq * hphi
                H4 = sq * (4.0 / bra - 1.0 / eta) * kg_v / sqeta * hphi
            else:
                I4 = H4 = 0.0

            # State machine: detect entry/exit using ζ_k values
            bra_k1 = bra_arr[k + 1]
            subsq_k1 = 1.0 - bra_k1 / eta
            if pard0 <= 0.0 and pard_arr[k + 1] > 0.0: ipass = 1
            else: ipass = 0

            if subsq_k1 > 0.0:
                isw = 1; icount += 1; ipa += ipass
                I_acc += (I1 + 2.0*I2 + 2.0*I3 + I4) / 6.0
                H_acc += (H1 + 2.0*H2 + 2.0*H3 + H4) / 6.0
            else:
                if isw == 1: isw = 2

            if isw == 2:
                if I_acc > 1.0e-15:
                    bigint_total += H_acc*H_acc/I_acc * iswst
                iswst = 1; H_acc = 0.0; I_acc = 0.0; icount = 0; ipa = 0; isw = 0
            pard0 = pard_arr[k + 1]

    eps_eff_32 = coeps * bigint_total * e2 / e3**2
    eps_eff = eps_eff_32 ** (2.0 / 3.0)
    return {
        "eps_eff_32": float(eps_eff_32), "eps_eff": float(eps_eff),
        "e2": float(e2), "e3": float(e3), "bigint_total": float(bigint_total),
        "heta": float(heta), "iota": iota,
        "b0": float(b0), "bmin": float(bmin), "bmax": float(b0),
        "rt0_squared": float(rt0_sq),
    }

# ═══════════════════════════════════════════════════════════════════════
# 6.  Utility: convert SIMSOPT Boozer → plain dict
# ═══════════════════════════════════════════════════════════════════════

def _boozer_obj_to_dict(boozer) -> Dict[str, Any]:
    """Convert pyneo BoozerData or SIMSOPT Boozer to plain dict."""
    booz: Dict[str, Any] = {}
    bx = getattr(boozer, "bx", boozer)

    for sims_key, pyneo_key in [("xm_b", "ixm_b"), ("xn_b", "ixn_b")]:
        val = getattr(bx, sims_key, None)
        if val is None: val = getattr(bx, pyneo_key, None)
        if val is not None: booz[pyneo_key] = np.asarray(val, dtype=np.int32)

    for key in ("bmnc_b", "rmnc_b", "zmns_b", "pmns_b"):
        val = getattr(bx, key, None)
        if val is not None:
            booz[key] = np.asarray(val, dtype=np.float64)
            if booz[key].ndim == 2 and booz[key].shape[1] < booz[key].shape[0]:
                booz[key] = booz[key].T
    if "pmns_b" not in booz:
        val = getattr(bx, "numns_b", None)
        if val is not None:
            booz["pmns_b"] = -np.asarray(val, dtype=np.float64)
            if booz["pmns_b"].ndim == 2 and booz["pmns_b"].shape[1] < booz["pmns_b"].shape[0]:
                booz["pmns_b"] = booz["pmns_b"].T

    iota = getattr(bx, "iota_b", None)
    if iota is None: iota = getattr(bx, "iota", None)
    if iota is not None: booz["iota_b"] = np.asarray(iota, dtype=np.float64)

    for sims_key, pyneo_key in [("buco_b","buco_b"),("bvco_b","bvco_b"),
                                ("Boozer_I","buco_b"),("Boozer_G","bvco_b")]:
        if pyneo_key in booz: continue
        val = getattr(bx, sims_key, None)
        if val is not None: booz[pyneo_key] = np.asarray(val, dtype=np.float64)

    if "buco_b" not in booz or "bvco_b" not in booz:
        equil = getattr(boozer, "equil", None)
        wout = getattr(equil, "wout", None) if equil is not None else None
        if wout is not None:
            for wout_key, pyneo_key in [("buco","buco_b"),("bvco","bvco_b")]:
                if pyneo_key not in booz and hasattr(wout,wout_key):
                    booz[pyneo_key] = np.asarray(getattr(wout,wout_key), dtype=np.float64)

    if "nfp_b" not in booz:
        nfp = getattr(bx, "nfp_b", None)
        if nfp is None: nfp = getattr(bx, "nfp", None)
        if nfp is not None: booz["nfp_b"] = np.asarray(nfp, dtype=np.int32)

    booz["_compute_surfs"] = np.asarray(getattr(bx, "compute_surfs", np.arange(len(booz.get("bmnc_b",[[]])))), dtype=np.int32)
    return booz


# ═══════════════════════════════════════════════════════════════════════
# 7.  Proper ODE-based RK4 — θ(φ) in the ODE, Fourier eval at each substep
#     (matches pyNEO's flint_bo + rhs_bo1 + rk4d_bo1 exactly)
# ═══════════════════════════════════════════════════════════════════════

def _eval_geometry_at_point(
    theta: float,
    phi: float,
    bmnc: NDArray[np.float64],
    rmnc: NDArray[np.float64],
    zmns: NDArray[np.float64],
    xm: NDArray[np.int32],
    xn: NDArray[np.int32],
    nfp: int,
    iota: float,
    I_val: float,
    J_val: float,
    pmns: Optional[NDArray[np.float64]] = None,
) -> Tuple[float, float, float, float]:
    """Evaluate B, |∇ψ|, |∇ψ|·κ_G, pard at a single (θ, φ).

    This is the single-point version of the Fourier evaluation in
    _sample_fieldline_fourier + metric computation (sample_fieldline_from_boozer).
    No pre-computed arrays are needed — the geometry is evaluated on the fly
    from the Fourier coefficients, exactly as pyNEO's neo_eval does.
    """
    b = b_tb = b_pb = 0.0
    r = r_tb = r_pb = 0.0
    z = z_tb = z_pb = 0.0
    # lambda / pmns variables
    l_val = p_tb = p_pb = 0.0
    has_lmns = pmns is not None

    for m in range(len(bmnc)):
        xmm = float(xm[m])
        xnn = float(xn[m])
        arg = xmm * theta - xnn * phi
        cos_a = np.cos(arg)
        sin_a = np.sin(arg)

        bc = bmnc[m]
        b    += bc * cos_a
        b_tb += -xmm * bc * sin_a
        b_pb +=  xnn * bc * sin_a

        ri = rmnc[m]
        r    += ri * cos_a
        r_tb += -xmm * ri * sin_a
        r_pb +=  xnn * ri * sin_a

        zi = zmns[m]
        z    += zi * sin_a
        z_tb +=  xmm * zi * cos_a
        z_pb += -xnn * zi * cos_a

        if has_lmns:
            li = pmns[m]
            l_val   += li * sin_a
            p_tb    += -xmm * li * cos_a
            p_pb    +=  xnn * li * cos_a

    # neo_fourier.f90:138-139 normalization
    twopi_nfp = 2.0 * np.pi / nfp
    p_tb = p_tb * twopi_nfp
    p_pb = 1.0 + p_pb * twopi_nfp

    # Metric tensor → sqrg11 = |∇ψ|,  kg = |∇ψ|·κ_G
    gtbtb = r_tb * r_tb + z_tb * z_tb + r * r * p_tb * p_tb
    gpbpb = r_pb * r_pb + z_pb * z_pb + r * r * p_pb * p_pb
    gtbpb = r_tb * r_pb + z_tb * z_pb + r * r * p_tb * p_pb

    fac = I_val + iota * J_val
    sqrg11 = np.sqrt(np.abs(gtbtb * gpbpb - gtbpb * gtbpb)) * b * b / fac
    kg = (J_val * b_pb - I_val * b_tb) / fac
    pard = b_pb + iota * b_tb

    return b, sqrg11, kg, pard


def _find_bminmax(
    bmnc: NDArray[np.float64],
    xm: NDArray[np.int32],
    xn: NDArray[np.int32],
    ntheta: int = 180,
    nphi: int = 180,
) -> Tuple[float, float, float, float]:
    """Scan B on a grid to find (θ_max, φ_max, B_max, B_min)."""
    th = np.linspace(0, 2 * np.pi, ntheta)
    ph = np.linspace(0, 2 * np.pi, nphi)
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    B2d = _fourier_sum_cos(bmnc, xm, xn, TH.ravel(), PH.ravel())
    imax = int(np.argmax(B2d))
    imin = int(np.argmin(B2d))
    return (float(TH.ravel()[imax]), float(PH.ravel()[imax]),
            float(B2d[imax]), float(B2d[imin]))


def _ode_rhs_pyneo_bo(
    theta: float,
    phi: float,
    b0: float,
    eta_vals: NDArray[np.float64],
    bmnc: NDArray[np.float64],
    rmnc: NDArray[np.float64],
    zmns: NDArray[np.float64],
    xm: NDArray[np.int32],
    xn: NDArray[np.int32],
    nfp: int,
    iota: float,
    I_val: float,
    J_val: float,
    pmns: Optional[NDArray[np.float64]],
    pard0: float,
    isw: NDArray[np.int32],
    ipa: NDArray[np.int32],
    icount: NDArray[np.int32],
    ipmax: int,
) -> Tuple[
    float, float, float, float,   # dtheta, dy2, dy3, dy4
    NDArray[np.float64], NDArray[np.float64],  # dI, dH
    float,                        # new pard0
    NDArray[np.int32], NDArray[np.int32], NDArray[np.int32],  # isw, ipa, icount
    int,                          # ipmax
]:
    """RHS of the ODE system at (θ, φ) — matches pyNEO's rhs_bo1 exactly.

    ODE variables (matching pyNEO's y vector):
        y[0]   = θ           — field line poloidal angle
        y[1]   = ∫ dφ / B²  — y2
        y[2]   = ∫ |∇ψ|/B² dφ — y3
        y[3]   = ∫ K_G/B³ dφ — y4
        y[4+i] = I_fj         — bounce-averaged I for particle i
        y[4+npart+i] = H_fj  — bounce-averaged H for particle i

    Also updates the bounce state machine (isw, ipa, icount, pard0, ipmax)
    exactly as rhs_bo1.f90 does.
    """
    b, sqrg11, kg, pard = _eval_geometry_at_point(
        theta, phi, bmnc, rmnc, zmns, xm, xn, nfp, iota, I_val, J_val, pmns,
    )

    bra = b / b0
    invB2 = 1.0 / b**2

    # ── ODE derivatives ──
    dtheta = iota           # dθ/dφ
    dy2 = invB2             # d(∫dφ/B²)/dφ = 1/B²
    dy3 = invB2 * sqrg11    # d(∫|∇ψ|/B² dφ)/dφ = |∇ψ|/B²
    dy4 = kg / b**3         # d(∫K_G/B³ dφ)/dφ = K_G/B³

    # ── bounce crossing detection (rhs_bo1.f90:44-49) ──
    # ipass = 1 when pard crosses zero upward: particle is at a bounce point
    # and the field-line derivative is changing from → to ← (or vice vers)
    if pard * pard0 <= 0.0 and pard > 0.0:
        ipass = 1
    else:
        ipass = 0

    # ipmax: detect first downward crossing (used for single-trapped diagnostic)
    if ipmax == 0 and pard * pard0 <= 0.0 and pard < 0.0:
        ipmax = 1

    # ── particle loop (rhs_bo1.f90:57-79) ──
    npart = len(eta_vals)
    dI = np.zeros(npart, dtype=np.float64)
    dH = np.zeros(npart, dtype=np.float64)

    for i in range(npart):
        subsq = 1.0 - bra / eta_vals[i]
        sqeta = np.sqrt(eta_vals[i])
        if subsq > 0.0:
            # Particle is inside the bounce well
            isw[i] = 1
            icount[i] += 1
            ipa[i] += ipass
            sq = np.sqrt(subsq) * invB2
            dI[i] = sq
            dH[i] = sq * (4.0 / bra - 1.0 / eta_vals[i]) * kg / sqeta
        else:
            # Particle is outside — derivative is 0
            # If it just exited a well, mark bounce complete
            if isw[i] == 1:
                isw[i] = 2
            # dI[i], dH[i] stay 0

    return (dtheta, dy2, dy3, dy4, dI, dH,
            pard, isw, ipa, icount, ipmax)


def _rk4_step_pyneo_bo(
    phi: float,
    hphi: float,
    theta: float,
    y2: float,
    y3: float,
    y4: float,
    I_fj: NDArray[np.float64],
    H_fj: NDArray[np.float64],
    b0: float,
    eta_vals: NDArray[np.float64],
    bmnc: NDArray[np.float64],
    rmnc: NDArray[np.float64],
    zmns: NDArray[np.float64],
    xm: NDArray[np.int32],
    xn: NDArray[np.int32],
    nfp: int,
    iota: float,
    I_val: float,
    J_val: float,
    pmns: Optional[NDArray[np.float64]],
    pard0: float,
    isw: NDArray[np.int32],
    iswst: NDArray[np.int32],
    ipa: NDArray[np.int32],
    icount: NDArray[np.int32],
    ipmax: int,
    bigint: float,
) -> Tuple[float, float, float, float, NDArray[np.float64], NDArray[np.float64],
           float, NDArray[np.int32], NDArray[np.int32],
           NDArray[np.int32], NDArray[np.int32], int, float]:
    """One RK4 step from φ to φ + hphi.

    Each of the 4 substeps calls _ode_rhs_pyneo_bo at the intermediate (θ, φ),
    updating θ via the ODE state (dθ/dφ = ι).  No pre-computed arrays, no
    interpolation — exactly like pyNEO's rk4d_bo1 + rhs_bo1.

    After the RK4 combination, performs bounce settlement (checking isw==2)
    exactly as pyNEO's flint_bo.f90:183-203.
    """
    npart = len(eta_vals)
    hh = 0.5 * hphi
    h6 = hphi / 6.0

    # ── k1 at (φ, θ) ──
    (dth1, dy2_1, dy3_1, dy4_1, dI1, dH1,
     pard1, isw1, ipa1, ic1, ipmax1) = _ode_rhs_pyneo_bo(
        theta, phi, b0, eta_vals,
        bmnc, rmnc, zmns, xm, xn, nfp, iota, I_val, J_val, pmns,
        pard0, isw, ipa, icount, ipmax,
    )

    # ── k2 at (φ + h/2, θ + k1θ · h/2) ──
    th2 = theta + hh * dth1
    (dth2, dy2_2, dy3_2, dy4_2, dI2, dH2,
     pard2, isw2, ipa2, ic2, ipmax2) = _ode_rhs_pyneo_bo(
        th2, phi + hh, b0, eta_vals,
        bmnc, rmnc, zmns, xm, xn, nfp, iota, I_val, J_val, pmns,
        pard1, isw1, ipa1, ic1, ipmax1,
    )

    # ── k3 at (φ + h/2, θ + k2θ · h/2) ──
    th3 = theta + hh * dth2
    (dth3, dy2_3, dy3_3, dy4_3, dI3, dH3,
     pard3, isw3, ipa3, ic3, ipmax3) = _ode_rhs_pyneo_bo(
        th3, phi + hh, b0, eta_vals,
        bmnc, rmnc, zmns, xm, xn, nfp, iota, I_val, J_val, pmns,
        pard2, isw2, ipa2, ic2, ipmax2,
    )

    # ── k4 at (φ + h, θ + k3θ · h) ──
    th4 = theta + hphi * dth3
    (dth4, dy2_4, dy3_4, dy4_4, dI4, dH4,
     pard4, isw4, ipa4, ic4, ipmax4) = _ode_rhs_pyneo_bo(
        th4, phi + hphi, b0, eta_vals,
        bmnc, rmnc, zmns, xm, xn, nfp, iota, I_val, J_val, pmns,
        pard3, isw3, ipa3, ic3, ipmax3,
    )

    # ── RK4 combination (rk4d_bo1.f90:40-42) ──
    theta_new = theta + h6 * (dth1 + 2.0 * dth2 + 2.0 * dth3 + dth4)
    y2_new = y2 + h6 * (dy2_1 + 2.0 * dy2_2 + 2.0 * dy2_3 + dy2_4)
    y3_new = y3 + h6 * (dy3_1 + 2.0 * dy3_2 + 2.0 * dy3_3 + dy3_4)
    y4_new = y4 + h6 * (dy4_1 + 2.0 * dy4_2 + 2.0 * dy4_3 + dy4_4)
    I_fj_new = I_fj + h6 * (dI1 + 2.0 * dI2 + 2.0 * dI3 + dI4)
    H_fj_new = H_fj + h6 * (dH1 + 2.0 * dH2 + 2.0 * dH3 + dH4)

    # ── Bounce settlement (flint_bo.f90:183-203) ──
    # After the RK4 step, check each particle: if isw==2 the particle
    # completed a bounce orbit during this step.  Accumulate H²/I and reset.
    for i in range(npart):
        if isw4[i] == 2:
            if I_fj_new[i] > 1.0e-15:
                bigint += H_fj_new[i]**2 / I_fj_new[i] * iswst[i]
            iswst[i] = 1            # Skip first bounce (flint_bo:197)
            H_fj_new[i] = 0.0       # Reset accumulators (flint_bo:198-199)
            I_fj_new[i] = 0.0
            isw4[i] = 0             # Reset state (flint_bo:200)
            ic4[i] = 0              # (flint_bo:201)
            ipa4[i] = 0             # (flint_bo:202)

    return (theta_new, y2_new, y3_new, y4_new, I_fj_new, H_fj_new,
            pard4, isw4, iswst, ipa4, ic4, ipmax4, bigint)


def eps_eff_pyneo_ode(
    booz: Dict[str, Any],
    surf_idx: int,
    theta0: Optional[float] = None,
    nturn: int = 64,
    npart: int = 100,
    nstep_per: int = 50,
) -> Dict[str, Any]:
    """ε_eff^(3/2) using proper θ-in-ODE RK4 integration.

    Unlike ``eps_eff_pyneo_style``, this function does **not** pre-compute
    geometry arrays and interpolate within each step.  Instead θ(φ) is a
    dependent variable in the ODE system, and every RK4 substep evaluates
    the Fourier representation at the exact (θ, φ) point — exactly as
    pyNEO's rk4d_bo1 + rhs_bo1 + flint_bo do.

    Algorithm flow (1:1 with pyNEO):
      1. Find B_max location → (θ₀, φ₀)
      2. Discretise η = B/B₀ at npart particles
      3. Initialise ODE state: θ=θ₀, y2=y3=y4=0, I_fj=H_fj=0
      4. Loop over nturn toroidal transits:
           For each of nstep_per steps:
             - Call RK4 (4 substeps, each evaluating Fourier at current θ,φ)
             - Accumulate I_fj, H_fj per particle via RK4
             - On bounce completion (isw==2): accumulate H²/I and reset
      5. Final result: ε_eff^{3/2} = coeps · bigint · y2 / y3²
    """
    # ═══════════════════════════════════════════════════════════════
    # Extract surface data
    # ═══════════════════════════════════════════════════════════════
    xm = np.asarray(booz["ixm_b"], dtype=np.int32)
    xn = np.asarray(booz["ixn_b"], dtype=np.int32)
    bmnc_s = np.asarray(booz["bmnc_b"][surf_idx, :], dtype=np.float64)
    rmnc_s = np.asarray(booz["rmnc_b"][surf_idx, :], dtype=np.float64)
    zmns_s = np.asarray(booz["zmns_b"][surf_idx, :], dtype=np.float64)
    pmns_s = booz.get("pmns_b", None)
    if pmns_s is not None:
        pmns_s = np.asarray(pmns_s[surf_idx, :], dtype=np.float64)

    iota = float(np.asarray(booz["iota_b"]).flat[surf_idx])
    I_val = float(np.asarray(booz["bvco_b"]).flat[surf_idx])
    J_val = float(np.asarray(booz["buco_b"]).flat[surf_idx])
    nfp = int(np.asarray(booz.get("nfp_b", booz.get("nfp", 1))).flat[0])

    # ═══════════════════════════════════════════════════════════════
    # Find B_max location and reference values
    # ═══════════════════════════════════════════════════════════════
    if theta0 is None:
        theta0, phi0, b0, bmin = _find_bminmax(bmnc_s, xm, xn)
    else:
        phi0 = 0.0
        # Need b0 and bmin — scan a coarse grid
        th_scan = np.linspace(0, 2 * np.pi, 180)
        ph_scan = np.linspace(0, 2 * np.pi, 180)
        TH_s, PH_s = np.meshgrid(th_scan, ph_scan, indexing="ij")
        B_scan = _fourier_sum_cos(bmnc_s, xm, xn, TH_s.ravel(), PH_s.ravel())
        b0 = float(np.max(B_scan))
        bmin = float(np.min(B_scan))

    # Reference radius R₀
    m0 = np.where((xm == 0) & (xn == 0))[0]
    rt0 = float(rmnc_s[m0[0]]) if len(m0) > 0 else 1.0
    rt0_sq = rt0**2

    # ═══════════════════════════════════════════════════════════════
    # Eta particles (flint_bo.f90:125-134)
    # ═══════════════════════════════════════════════════════════════
    etamin = bmin / b0
    heta = (1.0 - etamin) / (npart - 1)
    etamin_shifted = etamin + 0.5 * heta     # flint_bo.f90:131
    eta_vals = etamin_shifted + np.arange(npart, dtype=np.float64) * heta

    coeps = np.pi * rt0_sq * heta / (8.0 * np.sqrt(2.0))  # flint_bo.f90:135

    # ═══════════════════════════════════════════════════════════════
    # Initialise ODE state and bounce state machine
    # ═══════════════════════════════════════════════════════════════
    hphi = 2.0 * np.pi / nstep_per

    theta = float(theta0)
    phi = float(phi0)
    y2, y3, y4 = 0.0, 0.0, 0.0
    I_fj = np.zeros(npart, dtype=np.float64)
    H_fj = np.zeros(npart, dtype=np.float64)

    # Initial geometry evaluation → get initial pard0 (flint_bo.f90:172)
    _, _, _, pard0 = _eval_geometry_at_point(
        theta, phi, bmnc_s, rmnc_s, zmns_s, xm, xn, nfp, iota, I_val, J_val, pmns_s,
    )

    isw = np.zeros(npart, dtype=np.int32)      # 0=outside, 1=inside, 2=just-exited
    iswst = np.zeros(npart, dtype=np.int32)    # 0 → skip first bounce
    ipa = np.zeros(npart, dtype=np.int32)      # bounce count (for multra)
    icount = np.zeros(npart, dtype=np.int32)   # steps inside well
    ipmax = 0
    bigint = 0.0

    # ═══════════════════════════════════════════════════════════════
    # Integration loop (flint_bo.f90:179-315)
    # ═══════════════════════════════════════════════════════════════
    for _ in range(nturn):
        for _ in range(nstep_per):
            (theta, y2, y3, y4, I_fj, H_fj,
             pard0, isw, iswst, ipa, icount, ipmax, bigint) = _rk4_step_pyneo_bo(
                phi, hphi, theta, y2, y3, y4, I_fj, H_fj,
                b0, eta_vals,
                bmnc_s, rmnc_s, zmns_s, xm, xn, nfp, iota, I_val, J_val, pmns_s,
                pard0, isw, iswst, ipa, icount, ipmax, bigint,
            )
            phi += hphi

    # ═══════════════════════════════════════════════════════════════
    # Final result (flint_bo.f90:426-430)
    # ═══════════════════════════════════════════════════════════════
    eps_eff_32 = coeps * bigint * y2 / y3**2
    eps_eff = eps_eff_32 ** (2.0 / 3.0)

    return {
        "eps_eff_32": float(eps_eff_32),
        "eps_eff": float(eps_eff),
        "bigint": float(bigint),
        "y2": float(y2), "y3": float(y3),
        "heta": float(heta), "iota": iota,
        "b0": float(b0), "bmin": float(bmin),
        "rt0_squared": float(rt0_sq),
    }


def _compute_geom_on_arrays(
    bmnc_s: NDArray[np.float64],
    rmnc_s: NDArray[np.float64],
    zmns_s: NDArray[np.float64],
    xm: NDArray[np.int32],
    xn: NDArray[np.int32],
    theta_pts: NDArray[np.float64],
    phi_pts: NDArray[np.float64],
    pmns_s: Optional[NDArray[np.float64]],
    nfp: int,
    iota: float,
    I_val: float,
    J_val: float,
) -> Tuple[NDArray[np.float64], NDArray[np.float64],
           NDArray[np.float64], NDArray[np.float64]]:
    """Vectorised Fourier -> (B, |grad-psi|, |grad-psi|.kappa_G, pard) on arrays."""
    res = _sample_fieldline_fourier(
        bmnc_s, rmnc_s, zmns_s, xm, xn, theta_pts, phi_pts, pmns=pmns_s)
    if pmns_s is not None:
        b_a, b_tb, b_pb, r_a, r_tb, r_pb, z_a, z_tb, z_pb, _, p_tb, p_pb = res
    else:
        b_a, b_tb, b_pb, r_a, r_tb, r_pb, z_a, z_tb, z_pb = res
        p_tb = np.zeros_like(b_a)
        p_pb = np.zeros_like(b_a)

    twopi_nfp = 2.0 * np.pi / nfp
    p_tb *= twopi_nfp
    p_pb = 1.0 + p_pb * twopi_nfp

    gtbtb = r_tb * r_tb + z_tb * z_tb + r_a * r_a * p_tb * p_tb
    gpbpb = r_pb * r_pb + z_pb * z_pb + r_a * r_a * p_pb * p_pb
    gtbpb = r_tb * r_pb + z_tb * z_pb + r_a * r_a * p_tb * p_pb
    fac = I_val + iota * J_val
    sqrg11 = np.sqrt(np.abs(gtbtb * gpbpb - gtbpb * gtbpb)) * b_a * b_a / fac
    kg = (J_val * b_pb - I_val * b_tb) / fac
    pard = b_pb + iota * b_tb
    return b_a, sqrg11, kg, pard


def eps_eff_pyneo_ode_fast(
    booz: Dict[str, Any],
    surf_idx: int,
    theta0: Optional[float] = None,
    nturn: int = 64,
    npart: int = 100,
    nstep_per: int = 50,
) -> Dict[str, Any]:
    """eps_eff^(3/2) -- fast vectorized-precompute variant.

    Precomputes geometry at the EXACT RK4 substep positions using
    vectorised numpy Fourier evaluation (_sample_fieldline_fourier),
    then the integration loop does only array lookups + bounce state machine.

    No interpolation -- the k1/k2/k3/k4 (theta, phi) positions are determined
    analytically from dtheta/dphi = iota and evaluated exactly.

    This is ~50x faster than :func:`eps_eff_pyneo_ode` while producing
    identical results (since dtheta/dphi = iota is constant, the RK4 midpoint
    positions are exact).
    """
    # Extract surface data
    xm = np.asarray(booz["ixm_b"], dtype=np.int32)
    xn = np.asarray(booz["ixn_b"], dtype=np.int32)
    bmnc_s = np.asarray(booz["bmnc_b"][surf_idx, :], dtype=np.float64)
    rmnc_s = np.asarray(booz["rmnc_b"][surf_idx, :], dtype=np.float64)
    zmns_s = np.asarray(booz["zmns_b"][surf_idx, :], dtype=np.float64)
    pmns_s = booz.get("pmns_b", None)
    if pmns_s is not None:
        pmns_s = np.asarray(pmns_s[surf_idx, :], dtype=np.float64)

    iota = float(np.asarray(booz["iota_b"]).flat[surf_idx])
    I_val = float(np.asarray(booz["bvco_b"]).flat[surf_idx])
    J_val = float(np.asarray(booz["buco_b"]).flat[surf_idx])
    nfp = int(np.asarray(booz.get("nfp_b", booz.get("nfp", 1))).flat[0])

    # Find B_max location and reference values
    if theta0 is None:
        theta0, phi0, b0, bmin = _find_bminmax(bmnc_s, xm, xn)
    else:
        phi0 = 0.0
        th_scan = np.linspace(0, 2 * np.pi, 180)
        ph_scan = np.linspace(0, 2 * np.pi, 180)
        TH_s, PH_s = np.meshgrid(th_scan, ph_scan, indexing="ij")
        B_scan = _fourier_sum_cos(bmnc_s, xm, xn, TH_s.ravel(), PH_s.ravel())
        b0 = float(np.max(B_scan))
        bmin = float(np.min(B_scan))

    m0 = np.where((xm == 0) & (xn == 0))[0]
    rt0 = float(rmnc_s[m0[0]]) if len(m0) > 0 else 1.0
    rt0_sq = rt0**2

    # Eta particles
    etamin = bmin / b0
    heta = (1.0 - etamin) / (npart - 1)
    etamin_shifted = etamin + 0.5 * heta
    eta_vals = etamin_shifted + np.arange(npart, dtype=np.float64) * heta
    coeps = np.pi * rt0_sq * heta / (8.0 * np.sqrt(2.0))

    # Precompute geometry at ALL RK4 substep positions (vectorized)
    nstep = nturn * nstep_per
    hphi = 2.0 * np.pi / nstep_per

    # k1/k4 points: (theta_k, phi_k) for k = 0 .. nstep  (nstep+1 values)
    phi_arr = phi0 + np.arange(nstep + 1, dtype=np.float64) * hphi
    theta_arr = theta0 + iota * (phi_arr - phi0)

    # k2/k3 midpoints: (theta_k+iota*h/2, phi_k+h/2) for k = 0 .. nstep-1
    phi_mid = phi0 + (np.arange(nstep, dtype=np.float64) + 0.5) * hphi
    theta_mid = theta0 + iota * (phi_mid - phi0)

    B_a, gp_a, kg_a, pard_a = _compute_geom_on_arrays(
        bmnc_s, rmnc_s, zmns_s, xm, xn,
        theta_arr, phi_arr, pmns_s, nfp, iota, I_val, J_val)
    B_m, gp_m, kg_m, pard_m = _compute_geom_on_arrays(
        bmnc_s, rmnc_s, zmns_s, xm, xn,
        theta_mid, phi_mid, pmns_s, nfp, iota, I_val, J_val)

    invB2_a = 1.0 / B_a**2
    invB2_m = 1.0 / B_m**2
    bra_a = B_a / b0
    bra_m = B_m / b0
    sqeta = np.sqrt(eta_vals)

    # Integration loop — vectorized over particles (no Python particle loop)
    y2, y3 = 0.0, 0.0
    I_fj = np.zeros(npart, dtype=np.float64)
    H_fj = np.zeros(npart, dtype=np.float64)
    isw = np.zeros(npart, dtype=np.int32)
    iswst = np.zeros(npart, dtype=np.int32)
    ipa = np.zeros(npart, dtype=np.int32)
    icount = np.zeros(npart, dtype=np.int32)
    ipmax = 0
    bigint = 0.0
    pard0 = float(pard_a[0])
    h6 = hphi / 6.0

    # Precomputed per-particle constants
    inv_eta = 1.0 / eta_vals
    inv_sqeta = 1.0 / sqeta

    for k in range(nstep):
        # ── k1 at (θₖ, φₖ) ──
        bra  = bra_a[k]; invB = invB2_a[k]; geod = kg_a[k]; pard = pard_a[k]
        ipass = 1 if pard * pard0 <= 0.0 and pard > 0.0 else 0
        if ipmax == 0 and pard * pard0 <= 0.0 and pard < 0.0: ipmax = 1
        pard0 = pard

        subsq = 1.0 - bra * inv_eta
        in_well = subsq > 0.0
        isw[in_well] = 1
        isw[~in_well & (isw == 1)] = 2
        icount[in_well] += 1
        ipa[in_well] += ipass

        dI1 = np.zeros(npart)
        dH1 = np.zeros(npart)
        sqrt_subsq = np.sqrt(subsq[in_well])
        dI1[in_well] = sqrt_subsq * invB
        dH1[in_well] = sqrt_subsq * invB * (4.0/bra - inv_eta[in_well]) * geod * inv_sqeta[in_well]

        # ── k2 at midpoint ──
        bra2  = bra_m[k]; invB2 = invB2_m[k]; geod2 = kg_m[k]; pard2 = pard_m[k]
        ipass2 = 1 if pard2 * pard0 <= 0.0 and pard2 > 0.0 else 0
        if ipmax == 0 and pard2 * pard0 <= 0.0 and pard2 < 0.0: ipmax = 1
        pard0 = pard2

        subsq = 1.0 - bra2 * inv_eta
        in_well = subsq > 0.0
        isw[in_well] = 1
        isw[~in_well & (isw == 1)] = 2
        icount[in_well] += 1
        ipa[in_well] += ipass2

        dI2 = np.zeros(npart)
        dH2 = np.zeros(npart)
        sqrt_subsq = np.sqrt(subsq[in_well])
        dI2[in_well] = sqrt_subsq * invB2
        dH2[in_well] = sqrt_subsq * invB2 * (4.0/bra2 - inv_eta[in_well]) * geod2 * inv_sqeta[in_well]

        # ── k3: same (θ, φ) as k2, ipass=0 always ──
        subsq = 1.0 - bra2 * inv_eta
        in_well = subsq > 0.0
        isw[in_well] = 1
        isw[~in_well & (isw == 1)] = 2
        icount[in_well] += 1

        dI3 = np.zeros(npart)
        dH3 = np.zeros(npart)
        sqrt_subsq = np.sqrt(subsq[in_well])
        dI3[in_well] = sqrt_subsq * invB2
        dH3[in_well] = sqrt_subsq * invB2 * (4.0/bra2 - inv_eta[in_well]) * geod2 * inv_sqeta[in_well]

        # ── k4 at (θₖ₊₁, φₖ₊₁) ──
        bra4  = bra_a[k + 1]; invB4 = invB2_a[k + 1]; geod4 = kg_a[k + 1]
        pard4 = pard_a[k + 1]
        ipass4 = 1 if pard4 * pard0 <= 0.0 and pard4 > 0.0 else 0
        if ipmax == 0 and pard4 * pard0 <= 0.0 and pard4 < 0.0: ipmax = 1
        pard0 = pard4

        subsq = 1.0 - bra4 * inv_eta
        in_well = subsq > 0.0
        isw[in_well] = 1
        isw[~in_well & (isw == 1)] = 2
        icount[in_well] += 1
        ipa[in_well] += ipass4

        dI4 = np.zeros(npart)
        dH4 = np.zeros(npart)
        sqrt_subsq = np.sqrt(subsq[in_well])
        dI4[in_well] = sqrt_subsq * invB4
        dH4[in_well] = sqrt_subsq * invB4 * (4.0/bra4 - inv_eta[in_well]) * geod4 * inv_sqeta[in_well]

        # ── RK4 combination (vectorized) ──
        I_fj += h6 * (dI1 + 2.0*dI2 + 2.0*dI3 + dI4)
        H_fj += h6 * (dH1 + 2.0*dH2 + 2.0*dH3 + dH4)

        # ── Bounce settlement (vectorized) ──
        bounced = (isw == 2)
        valid = bounced & (I_fj > 1.0e-15)
        if np.any(valid):
            bigint += float(np.sum(H_fj[valid]**2 / I_fj[valid] * iswst[valid]))
        iswst[bounced] = 1
        I_fj[bounced] = 0.0
        H_fj[bounced] = 0.0
        isw[bounced] = 0
        icount[bounced] = 0
        ipa[bounced] = 0

        # y2, y3 (same for all particles)
        y2 += h6 * (invB2_a[k] + 2.0*invB2_m[k] + 2.0*invB2_m[k] + invB2_a[k+1])
        y3 += h6 * (gp_a[k]*invB2_a[k] + 2.0*gp_m[k]*invB2_m[k]
                    + 2.0*gp_m[k]*invB2_m[k] + gp_a[k+1]*invB2_a[k+1])

    eps_eff_32 = coeps * bigint * y2 / y3**2
    eps_eff = eps_eff_32 ** (2.0 / 3.0)
    return {
        "eps_eff_32": float(eps_eff_32),
        "eps_eff": float(eps_eff),
        "bigint": float(bigint),
        "y2": float(y2), "y3": float(y3),
        "heta": float(heta), "iota": iota,
        "b0": float(b0), "bmin": float(bmin),
        "rt0_squared": float(rt0_sq),
    }
