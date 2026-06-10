"""
Boozer-coordinate ε_eff verification.

Uses the EXACT same integrands as pyneo (rhs_bo1.f90) evaluated analytically
from Boozer Fourier harmonics — zero interpolation error.

The bp-integration algorithm matches ripplepy's approach:
  - B-profile scan over bp ∈ [B_min/B₀, 1]
  - Local-minimum-based well detection  (find_local_minima)
  - Per-segment H/I accumulation        (integrate_bounce_segment)

This isolates the integration algorithm from magnetic-field representation.

Two bp-integration modes:
  - rectangular (n_b uniform points, same as active ripplepy)
  - Gauss-Legendre quadrature

Formula (matching pyneo's flint_bo):
  ε_{eff}^{3/2} = (π R₀² Δη) / (8√2) · Σ_{bp} H²/I  ·  (∫dζ/B²) / (∫dζ/B² |∇ψ|)²
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Dict, Optional, Tuple, Any
from numpy.typing import NDArray
import dataclasses


# ═══════════════════════════════════════════════════════════════════════
# 1.  Direct Fourier evaluation along analytic field line  θ(ζ)=θ₀+ιζ
# ═══════════════════════════════════════════════════════════════════════

def _fourier_sum_cos(
    coeff: NDArray[np.float64],
    xm: NDArray[np.int32],
    xn: NDArray[np.int32],
    theta: NDArray[np.float64],
    zeta: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Σ coeff[m] cos(xm[m]*θ - xn[m]*ζ)  at every (θ,ζ) point (same length)."""
    arg = np.outer(xm, theta) - np.outer(xn, zeta)          # (nmodes, npoints)
    return np.dot(coeff, np.cos(arg))


def _fourier_sum_deriv_theta_cos(
    coeff: NDArray[np.float64],
    xm: NDArray[np.int32],
    xn: NDArray[np.int32],
    theta: NDArray[np.float64],
    zeta: NDArray[np.float64],
) -> NDArray[np.float64]:
    """∂/∂θ: Σ -xm·coeff sin(xm θ - xn ζ)"""
    arg = np.outer(xm, theta) - np.outer(xn, zeta)
    return np.dot(-xm.astype(np.float64) * coeff, np.sin(arg))


def _fourier_sum_deriv_zeta_cos(
    coeff: NDArray[np.float64],
    xm: NDArray[np.int32],
    xn: NDArray[np.int32],
    theta: NDArray[np.float64],
    zeta: NDArray[np.float64],
) -> NDArray[np.float64]:
    """∂/∂ζ: Σ +xn·coeff sin(xm θ - xn ζ)"""
    arg = np.outer(xm, theta) - np.outer(xn, zeta)
    return np.dot(+xn.astype(np.float64) * coeff, np.sin(arg))


def _fourier_sum_sin(
    coeff: NDArray[np.float64],
    xm: NDArray[np.int32],
    xn: NDArray[np.int32],
    theta: NDArray[np.float64],
    zeta: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Σ coeff[m] sin(xm[m]*θ - xn[m]*ζ)"""
    arg = np.outer(xm, theta) - np.outer(xn, zeta)
    return np.dot(coeff, np.sin(arg))


def _fourier_sum_deriv_theta_sin(
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
    I_val: float                      # buco = curr_pol
    J_val: float                      # bvco = curr_tor
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


def sample_fieldline_from_boozer(
    booz: Dict[str, Any],
    surf_idx: int,
    theta0: Optional[float] = None,
    nzeta: int = 1024,
    nturn: int = 64,
) -> FieldLineData:
    """
    Sample a single field line θ(ζ)=θ₀+ιζ by direct Fourier summation
    at every ζ point.  No grid, no interpolation.

    Parameters
    ----------
    booz : dict
        Keys: bmnc_b, rmnc_b, zmns_b, ixm_b, ixn_b, iota_b, buco_b, bvco_b, nfp_b.
        Arrays are (ns, mnmax) or (ns,) for profiles.
    surf_idx : int
        Zero-based flux-surface index.
    theta0 : float
        Initial poloidal angle.
    nzeta : int
        Number of ζ (=φ) points per toroidal turn.
    nturn : int
        Number of field periods (2π in ζ) to trace.
    """
    # --- extract surface data ---
    xm = np.asarray(booz["ixm_b"], dtype=np.int32)
    xn = np.asarray(booz["ixn_b"], dtype=np.int32)

    bmnc = np.asarray(booz["bmnc_b"][surf_idx, :], dtype=np.float64)
    rmnc = np.asarray(booz["rmnc_b"][surf_idx, :], dtype=np.float64)
    zmns = np.asarray(booz["zmns_b"][surf_idx, :], dtype=np.float64)

    iota = float(np.asarray(booz["iota_b"]).flat[surf_idx])
    I_   = float(np.asarray(booz["buco_b"]).flat[surf_idx])
    J_   = float(np.asarray(booz["bvco_b"]).flat[surf_idx])
    nfp  = int(np.asarray(booz.get("nfp_b", booz.get("nfp", 1))).flat[0])

    # --- auto-detect B_max location as starting point (matching pyneo) ---
    if theta0 is None:
        theta0, phi0_bmax = _find_bmax_location(bmnc, xm, xn)

    # --- field-line sampling ---
    ntot = nzeta * nturn
    dphi = 2.0 * np.pi / nzeta
    zeta = np.arange(ntot, dtype=np.float64) * dphi
    theta = theta0 + iota * zeta

    # --- |B| and its derivatives ---
    B       = _fourier_sum_cos(bmnc, xm, xn, theta, zeta)
    dBdtheta = _fourier_sum_deriv_theta_cos(bmnc, xm, xn, theta, zeta)
    dBdzeta  = _fourier_sum_deriv_zeta_cos(bmnc, xm, xn, theta, zeta)

    # --- R, Z geometry ---
    R        = _fourier_sum_cos(rmnc, xm, xn, theta, zeta)
    dRdtheta = _fourier_sum_deriv_theta_cos(rmnc, xm, xn, theta, zeta)
    dRdzeta  = _fourier_sum_deriv_zeta_cos(rmnc, xm, xn, theta, zeta)
    dZdtheta = _fourier_sum_deriv_theta_sin(zmns, xm, xn, theta, zeta)
    dZdzeta  = _fourier_sum_deriv_zeta_sin(zmns, xm, xn, theta, zeta)

    # --- metric: g_ij on the flux surface (neo_fourier.f90:174-181) ---
    gtb  = dRdtheta**2 + dZdtheta**2                      # g_θθ
    gpb  = dRdzeta**2  + dZdzeta**2  + R**2               # g_ζζ
    gtbp = dRdtheta * dRdzeta + dZdtheta * dZdzeta        # g_θζ

    fac = I_ + iota * J_
    isqrg = B**2 / fac                                     # 1 / √g
    sqrg11 = np.sqrt(np.abs(gtb * gpb - gtbp**2)) * isqrg # |∇ψ|

    # --- |∇ψ| κ_G  (neo_fourier.f90:183) ---
    kg_gradpsi = (J_ * dBdzeta - I_ * dBdtheta) / fac

    # --- parallel derivative of |B| (for well boundary detection) ---
    pard = dBdzeta + iota * dBdtheta

    return FieldLineData(
        zeta=zeta, B=B, gradpsi=sqrg11, kg_gradpsi=kg_gradpsi,
        pard=pard, iota=iota, I_val=I_, J_val=J_, nfp=nfp,
    )


# ═══════════════════════════════════════════════════════════════════════
# 3.  Well detection & bounce-segment integration (ripplepy algorithm)
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
    bp: float,
    b0: float,
    i1: int,
    i2: int,
    B: NDArray[np.float64],
    gradpsi: NDArray[np.float64],
    kg_gradpsi: NDArray[np.float64],
    dmeasure: NDArray[np.float64],
) -> Tuple[float, float]:
    """
    Integrate H and I over one bounce well [i1, i2) for a given bp.
    Uses pyneo's EXACT integrands from rhs_bo1.f90.

    H integrand:  √(bp-B/B₀)/B² · (4B₀/B - 1/bp) · (|∇ψ|κ_G) / bp
    I integrand:  √(bp-B/B₀) / (B²·√bp)
    """
    n = len(B)
    H, I_val = 0.0, 0.0
    for k in range(i1, i2):
        idx = k % n
        b_loc = B[idx] / b0
        if bp <= b_loc:
            continue

        sqrt_term = np.sqrt(bp - b_loc)
        inv_B2 = 1.0 / B[idx]**2

        # I integrand  (rhs_bo1:  p_i  = sqrt(1-bra/eta) * 1/B²)
        #   sqrt(1-bra/eta) = sqrt((bp-b_loc)/bp) = sqrt(bp-b_loc)/√bp
        dI = sqrt_term * inv_B2 / np.sqrt(bp)
        I_val += dI * dmeasure[idx]

        # H integrand (rhs_bo1:  p_h = p_i * (4/bra-1/eta) * geodcu/√eta)
        #   = sqrt(bp-b_loc)/(B²√bp) * (4B₀/B-1/bp) * |∇ψ|κ_G / √bp
        #   = sqrt(bp-b_loc)/B² * (4B₀/B-1/bp) * |∇ψ|κ_G / bp
        #   ≠ dI * (4/b_loc-1/bp) * kg / bp   (which has an extra 1/√bp)
        dH = sqrt_term * inv_B2 * (4.0 / b_loc - 1.0 / bp) * kg_gradpsi[idx] / bp
        H += dH * dmeasure[idx]

    return H, I_val


def _compute_H2_over_I_for_bp(
    bp: float,
    b0: float,
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
            bp, b0, i1, i2, B, gradpsi, kg_gradpsi, dmeasure
        )
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
    """
    ε_eff^(3/2) using ripplepy's B-profile-scan algorithm, evaluated
    analytically from Boozer Fourier coefficients.

    Uses the SAME conventions as pyneo by default:
      - starts from B_max location (theta0=None → auto-detected)
      - returns ε^(3/2) = π·rt0²/(8√2) · e1 · e2/e3²
      where rt0 is the m=n=0 rmnc coefficient for the flux surface.

    Returns dict with keys:
      eps_eff_32, eps_eff, e1, e2, e3, rt0_squared, ...
    """
    # --- sample field line (auto-detect B_max start) ---
    fl = sample_fieldline_from_boozer(booz, surf_idx, theta0, nzeta, nturn)

    B = fl.B
    gp = fl.gradpsi
    kg = fl.kg_gradpsi
    npts = len(B)
    dphi = 2.0 * np.pi / nzeta
    dmeasure = np.full(npts, dphi)           # uniform dζ

    b0 = np.max(B)
    bmin, bmax = np.min(B), b0

    # --- denominator integrals (pyneo convention: y2=∫1/B² dζ, y3=∫|∇ψ|/B² dζ) ---
    e2 = np.sum(dmeasure / B**2)
    e3 = np.sum(dmeasure * gp / B**2)

    # --- bp integration ---
    if use_gauss:
        from numpy.polynomial.legendre import leggauss
        nodes, weights = leggauss(n_gauss)          # on [-1, 1]
        bp_min, bp_max = bmin / b0, 1.0
        e1 = 0.0
        for node, wgt in zip(nodes, weights):
            bp = 0.5 * (bp_max + bp_min) + 0.5 * (bp_max - bp_min) * node
            e1 += _compute_H2_over_I_for_bp(
                bp, b0, B, gp, kg, dmeasure
            ) * wgt * 0.5 * (bp_max - bp_min)
    else:
        dbp = (bmax - bmin) / (n_b - 1) / b0
        e1 = 0.0
        for j in range(n_b):
            bp = bmin / b0 + j * dbp
            e1 += _compute_H2_over_I_for_bp(
                bp, b0, B, gp, kg, dmeasure
            ) * dbp

    # --- rt0² for scaling (matching pyneo's coeps convention) ---
    xm = np.asarray(booz["ixm_b"], dtype=np.int32)
    xn = np.asarray(booz["ixn_b"], dtype=np.int32)
    rmnc = np.asarray(booz["rmnc_b"][surf_idx, :], dtype=np.float64)
    m0 = np.where((xm == 0) & (xn == 0))[0]
    rt0 = float(rmnc[m0[0]]) if len(m0) > 0 else 1.0
    rt0_sq = rt0**2

    # --- final formula with rt0² scaling (matching pyneo convention) ---
    eps_eff_32 = np.pi * rt0_sq / (8.0 * np.sqrt(2.0)) * e1 * e2 / e3**2
    eps_eff = eps_eff_32 ** (2.0 / 3.0)

    result: Dict[str, Any] = {
        "eps_eff_32": float(eps_eff_32),
        "eps_eff": float(eps_eff),
        "e1": float(e1), "e2": float(e2), "e3": float(e3),
        "iota": fl.iota, "b0": float(b0),
        "bmin": float(bmin), "bmax": float(bmax),
        "rt0_squared": float(rt0_sq),
    }

    if return_debug:
        result["B_along"] = B
        result["gradpsi_along"] = gp
        result["kg_gradpsi_along"] = kg
        result["zeta"] = fl.zeta

    return result


# ═══════════════════════════════════════════════════════════════════════
# 4b.  pyneo-style state-machine integration (η-particle parallel accumulation)
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
    """
    ε_eff^(3/2) using pyneo's EXACT algorithm: η-particle state machine
    accumulated along the analytic Boozer field line.

    Differs from `eps_eff_from_boozer` ONLY in the (η, well) traversal:
      - `eps_eff_from_boozer`: outer bp loop → find_local_minima → H²/I
      - THIS function:        outer η loop → subsq state machine → H²/I
        (exactly replicates pyneo's `rhs_bo1.f90` + `flint_bo.f90`)

    Parameters
    ----------
    npart : int
        Number of η values (particle classes).  Same as pyneo's npart.
    multra : int
        Maximum trapping class (1 = all bounces treated equally).

    Returns
    -------
    dict with: eps_eff_32, eps_eff, e1, e2, e3, rt0_squared, ...
    """
    # ── sample field line ──
    fl = sample_fieldline_from_boozer(booz, surf_idx, theta0, nzeta, nturn)

    B = fl.B; gp = fl.gradpsi; kg = fl.kg_gradpsi; pard = fl.pard
    npts = len(B)
    dphi = 2.0 * np.pi / nzeta
    dmeasure = np.full(npts, dphi)

    b0 = np.max(B)
    bmin, bmax = np.min(B), b0

    # ── denominator integrals ──
    e2 = np.sum(dmeasure / B**2)
    e3 = np.sum(dmeasure * gp / B**2)

    # ── η values (same as pyneo: etamin + heta/2  offset) ──
    etamin = bmin / b0
    etamax = 1.0
    heta   = (etamax - etamin) / (npart - 1)
    eta_vals = etamin + heta / 2.0 + np.arange(npart) * heta

    # ── rt0² ──
    xm = np.asarray(booz["ixm_b"], dtype=np.int32)
    xn = np.asarray(booz["ixn_b"], dtype=np.int32)
    rmnc = np.asarray(booz["rmnc_b"][surf_idx, :], dtype=np.float64)
    m0 = np.where((xm == 0) & (xn == 0))[0]
    rt0 = float(rmnc[m0[0]]) if len(m0) > 0 else 1.0
    rt0_sq = rt0**2

    # ── per-η state machine ──
    bigint_total = 0.0   # Σ_η Σ_bounce H²/I  (without Δη factor)

    for eta in eta_vals:
        # State variables (matching pyneo)
        isw = 0         # 0=free, 1=in_well, 2=just_exited
        iswst = 0       # skip-first-bounce sentinel
        H_acc = 0.0     # accumulated H for current bounce
        I_acc = 0.0     # accumulated I for current bounce
        pard0 = 0.0     # previous step's pard (for boundary detection)

        for k in range(npts):
            b_loc = B[k] / b0
            subsq = 1.0 - b_loc / eta       # >0 → trapped

            if subsq > 0.0:
                # ── INSIDE WELL ──
                isw = 1
                # Use sqrt(η - B/B₀) convention (same as bp-scan version)
                sqrt_eta_minus_b = np.sqrt(eta - b_loc)
                inv_B2 = 1.0 / B[k]**2

                dI = sqrt_eta_minus_b * inv_B2 / np.sqrt(eta)           # Eq. (31)
                dH = sqrt_eta_minus_b * inv_B2 * (4.0/b_loc - 1.0/eta) * kg[k] / eta  # Eq. (30)

                H_acc += dH * dmeasure[k]
                I_acc += dI * dmeasure[k]

            else:
                # ── OUTSIDE WELL ──
                if isw == 1:
                    # Just exited a well → settle
                    isw = 2
                # else: isw stays 0 (free) or 2 (waiting to be processed)

            # Process well exit (matching flint_bo:183-198)
            if isw == 2:
                if I_acc > 1e-15:
                    add_on = (H_acc * H_acc / I_acc) * iswst
                    bigint_total += add_on
                iswst = 1          # subsequent bounces count normally
                H_acc = 0.0; I_acc = 0.0
                isw = 0

    # ── final formula (matching pyneo's coeps convention) ──
    # pyneo: epstot = π·rt0²·heta/(8√2) · Σ bigint · y2/y3²
    # Here: bigint_total = Σ_η Σ_bounce H²/I
    # e1_equivalent = heta * bigint_total
    eps_eff_32 = np.pi * rt0_sq * heta / (8.0 * np.sqrt(2.0)) * bigint_total * e2 / e3**2
    eps_eff = eps_eff_32 ** (2.0 / 3.0)

    return {
        "eps_eff_32": float(eps_eff_32),
        "eps_eff": float(eps_eff),
        "e1_equivalent": float(heta * bigint_total),
        "e2": float(e2), "e3": float(e3),
        "bigint_total": float(bigint_total),
        "heta": float(heta),
        "iota": fl.iota, "b0": float(b0),
        "bmin": float(bmin), "bmax": float(bmax),
        "rt0_squared": float(rt0_sq),
    }


# ═══════════════════════════════════════════════════════════════════════
# 5.  pyneo-compatible wrappers
# ═══════════════════════════════════════════════════════════════════════

def _boozer_obj_to_dict(boozer) -> Dict[str, Any]:
    """Convert pyneo BoozerData or SIMSOPT Boozer to plain dict.

    Handles naming differences:
      - pyneo BoozerData:  ixm_b, ixn_b, iota_b, buco_b, bvco_b, nfp_b
      - SIMSOPT booz_xform: xm_b, xn_b, iota,   (buco/bvco from equil.wout)
    """
    booz = {}

    # --- resolve the inner object ---
    bx = getattr(boozer, "bx", boozer)

    # --- mode numbers (SIMSOPT: xm_b/xn_b,  pyneo: ixm_b/ixn_b) ---
    for sims_key, pyneo_key in [("xm_b", "ixm_b"), ("xn_b", "ixn_b")]:
        val = getattr(bx, sims_key, None)
        if val is None:
            val = getattr(bx, pyneo_key, None)
        if val is not None:
            booz[pyneo_key] = np.asarray(val, dtype=np.int32)

    # --- Fourier coefficients ---
    for key in ("bmnc_b", "rmnc_b", "zmns_b"):
        val = getattr(bx, key, None)
        if val is not None:
            booz[key] = np.asarray(val, dtype=np.float64)
            # SIMSOPT has shape (nmodes, nsurf) → transpose to (nsurf, nmodes)
            if booz[key].ndim == 2 and booz[key].shape[1] < booz[key].shape[0]:
                booz[key] = booz[key].T

    # --- iota ---
    iota = getattr(bx, "iota_b", None)
    if iota is None:
        iota = getattr(bx, "iota", None)
    if iota is not None:
        booz["iota_b"] = np.asarray(iota, dtype=np.float64)

    # --- buco / bvco (curr_pol / curr_tor) ---
    for sims_key, pyneo_key in [
        ("buco_b", "buco_b"), ("bvco_b", "bvco_b"),
        ("Boozer_I", "buco_b"), ("Boozer_G", "bvco_b"),
    ]:
        if pyneo_key in booz:
            continue
        val = getattr(bx, sims_key, None)
        if val is not None:
            booz[pyneo_key] = np.asarray(val, dtype=np.float64)

    # --- buco/bvco from equil.wout if still missing ---
    if "buco_b" not in booz or "bvco_b" not in booz:
        equil = getattr(boozer, "equil", None)
        wout = getattr(equil, "wout", None) if equil is not None else None
        if wout is not None:
            for wout_key, pyneo_key in [("buco", "buco_b"), ("bvco", "bvco_b")]:
                if pyneo_key not in booz and hasattr(wout, wout_key):
                    booz[pyneo_key] = np.asarray(
                        getattr(wout, wout_key), dtype=np.float64
                    )

    # --- nfp ---
    for key in ("nfp_b", "nfp"):
        val = getattr(bx, key, None)
        if val is not None:
            booz["nfp_b"] = np.asarray(val)
            break
    if "nfp_b" not in booz:
        equil = getattr(boozer, "equil", None)
        wout = getattr(equil, "wout", None) if equil is not None else None
        if wout is not None and hasattr(wout, "nfp"):
            booz["nfp_b"] = np.asarray(getattr(wout, "nfp"))

    # --- surface index mapping ---
    # SIMSOPT: bmnc_b has shape (nmodes, nsurf_booz) or (nsurf_booz, nmodes)
    # boozer.compute_surfs maps booz-surface-index → full-VMEC-surface-index
    compute_surfs = getattr(bx, "compute_surfs", None)
    if compute_surfs is not None:
        booz["_compute_surfs"] = np.asarray(compute_surfs, dtype=np.int32)

    # iota for the BOOZER surfaces: need to index into full iota profile
    if "iota_b" in booz and compute_surfs is not None and booz["iota_b"].size > 10:
        # iota is full-profile (ns_in,), extract for boozer surfaces
        cs = np.asarray(compute_surfs, dtype=np.int32) - 1  # 1-based → 0-based
        booz["iota_b"] = booz["iota_b"][cs]

    return booz


def eps_eff_boozer_pyneo_style(
    boozer,          # pyneo BoozerData or SIMSOPT Boozer
    surf_idx: int,
    **kwargs,
) -> Dict[str, Any]:
    """Convenience: auto-convert Boozer → dict → eps_eff_from_boozer."""
    return eps_eff_from_boozer(_boozer_obj_to_dict(boozer), surf_idx, **kwargs)


# ═══════════════════════════════════════════════════════════════════════
# 6.  Benchmark comparison helper
# ═══════════════════════════════════════════════════════════════════════

def compare_with_pyneo(
    boozer,
    surf_indices,
    pyneo_epstot: NDArray[np.float64],
    theta0: float = 0.0,
    nzeta: int = 512,
    nturn: int = 32,
    n_b: int = 5000,
    n_gauss: int = 64,
    verbose: bool = True,
) -> Dict[str, NDArray[np.float64]]:
    """
    Run both rectangular and Gauss integration for a set of flux surfaces,
    compare with pyneo epstot.

    Returns dict with keys:
      surf_idx, pyneo, booz_rect, booz_gauss
    """
    booz_dict = _boozer_obj_to_dict(boozer)

    results = {"surf_idx": [], "pyneo": [], "booz_rect": [], "booz_gauss": []}

    for i, s in enumerate(surf_indices):
        r_rect = eps_eff_from_boozer(
            booz_dict, s, theta0=theta0, nzeta=nzeta, nturn=nturn,
            n_b=n_b, use_gauss=False,
        )
        r_gauss = eps_eff_from_boozer(
            booz_dict, s, theta0=theta0, nzeta=nzeta, nturn=nturn,
            n_gauss=n_gauss, use_gauss=True,
        )
        results["surf_idx"].append(s)
        results["pyneo"].append(float(pyneo_epstot[i]))
        results["booz_rect"].append(r_rect["eps_eff_32"])
        results["booz_gauss"].append(r_gauss["eps_eff_32"])

        if verbose:
            ratio_rect = r_rect["eps_eff_32"] / pyneo_epstot[i]
            ratio_gauss = r_gauss["eps_eff_32"] / pyneo_epstot[i]
            print(
                f"  surf {s:3d}  pyneo={pyneo_epstot[i]:.4e}  "
                f"rect={r_rect['eps_eff_32']:.4e} (×{ratio_rect:.3f})  "
                f"gauss={r_gauss['eps_eff_32']:.4e} (×{ratio_gauss:.3f})"
            )

    return {k: np.array(v) for k, v in results.items()}
