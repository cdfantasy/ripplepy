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
    nzeta: int = 1024,
    nturn: int = 64,
    npart: int = 100,
    multra: int = 1,
) -> Dict[str, Any]:
    """ε_eff^(3/2) matching pyneo's flint_bo + rhs_bo1 algorithm exactly."""
    fl = sample_fieldline_from_boozer(booz, surf_idx, theta0, nzeta, nturn)
    B = fl.B; gp = fl.gradpsi; kg = fl.kg_gradpsi; pard = fl.pard
    npts = len(B)
    dphi = 2.0 * np.pi / nzeta
    b0 = np.max(B); bmin = np.min(B)

    inv_B2 = 1.0 / B**2
    e2 = np.sum(inv_B2) * dphi
    e3 = np.sum(inv_B2 * gp) * dphi

    etamin = bmin / b0
    heta = (1.0 - etamin) / (npart - 1)
    eta_vals = etamin + heta / 2.0 + np.arange(npart) * heta

    xm = np.asarray(booz["ixm_b"], dtype=np.int32)
    xn = np.asarray(booz["ixn_b"], dtype=np.int32)
    rmnc = np.asarray(booz["rmnc_b"][surf_idx,:], dtype=np.float64)
    m0 = np.where((xm == 0) & (xn == 0))[0]
    rt0 = float(rmnc[m0[0]]) if len(m0) > 0 else 1.0
    rt0_sq = rt0**2
    coeps = np.pi * rt0_sq * heta / (8.0 * np.sqrt(2.0))

    bra_arr = B / b0; invB2_arr = inv_B2; kg_arr = kg
    sqrt_eta_cache = np.sqrt(eta_vals)
    bigint_total = 0.0

    for i_eta, eta in enumerate(eta_vals):
        sqeta = sqrt_eta_cache[i_eta]
        isw = 0; iswst = 0; icount = 0; ipa = 0
        H_acc = 0.0; I_acc = 0.0; pard0 = pard[0]

        for k in range(npts):
            bra = bra_arr[k]; subsq = 1.0 - bra / eta
            if pard0 <= 0.0 and pard[k] > 0.0: ipass = 1
            else: ipass = 0
            if subsq > 0.0:
                isw = 1; icount += 1; ipa += ipass
                sq = np.sqrt(subsq) * invB2_arr[k]
                I_acc += sq * dphi
                H_acc += sq * (4.0 / bra - 1.0 / eta) * kg_arr[k] / sqeta * dphi
            else:
                if isw == 1: isw = 2
            if isw == 2:
                if I_acc > 1.0e-15:
                    bigint_total += H_acc*H_acc/I_acc * iswst
                iswst = 1; H_acc = 0.0; I_acc = 0.0; icount = 0; ipa = 0; isw = 0
            pard0 = pard[k]

    eps_eff_32 = coeps * bigint_total * e2 / e3**2
    eps_eff = eps_eff_32 ** (2.0 / 3.0)
    return {
        "eps_eff_32": float(eps_eff_32), "eps_eff": float(eps_eff),
        "e2": float(e2), "e3": float(e3), "bigint_total": float(bigint_total),
        "heta": float(heta), "iota": fl.iota,
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
