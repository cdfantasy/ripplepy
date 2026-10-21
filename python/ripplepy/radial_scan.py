"""Radial scan: walk outward from the magnetic axis to the LCFS.

One configuration = one radial column of flux surfaces.  Every surface is
traced once and everything (iota, volume, minor radius a, <B>, ripple, section
smoothness) is derived from that same line.

The oracle has two cost tiers, because eps is the one expensive and fragile
part:

  profile / screening  trace only, no eps.  Gives the DEPTH of the column
                       (how far out nested surfaces survive) and its geometry,
                       at ~0.7 s per face (nturn=100) to ~1.0 s (nturn=150).
  full                 same trace + eps + the grad-psi QC column, measured on
                       ONE surface only (a = ``eps_at_a``), because eps is
                       reproducible in the inner 80% of the column (1e-5
                       relative jitter 0.2%) and not near the edge (up to 66%).

Pipeline per configuration (``scan_config``)
  1. axis search (find_axis_multi_guess) + branch ranking: a one-turn fixed
     point is not necessarily the main axis -- an island centre has one too --
     so candidate branches are probed to ``axis_probe_dr`` and the branch whose
     column survives furthest wins
  2. dr = dr0, dr0+step, ... outward: one screening measurement per face; stop
     at the first face that fails (trace / residual / angular gap)
  3. one bisection inside the last bracket -> LCFS located to dr_step/2
  4. optional: one full-tier eps measurement on the face whose a is closest to
     ``eps_at_a`` (the reference surface used for ranking)

``stage1_probe`` is the even cheaper stage-1 test ("is there a REAL axis
here?"), used to map the axis region before any depth work.

Gate metrics (both evaluated on the same traced line as eps)
  residual : normalised FFT residual of the phi=0 section.  Measured to be
             resolution independent (0.0445 -> 0.0450 over 5x nturn), so one
             threshold is valid at every tier.
  gap_norm : largest angular hole of the section divided by the mean point
             spacing, i.e. ``gap_rad * nturn / (2*pi)``.  The raw ``gap_rad``
             scales like 1/nturn for the same surface (0.30 rad at 80 turns
             vs 0.089 rad at 400), so a fixed radian threshold would reject
             healthy surfaces at low nturn.  Measured: healthy faces 3.2-6.2
             (10.6 on the tiny innermost face, where the polar origin is off
             centre), an iota~5/7 face 55.  Threshold 15 keeps both apart and
             still rejects the 55.
"""

from __future__ import annotations

import contextlib
import json
import multiprocessing
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
from scipy.stats.qmc import Sobol

from .ripple import (
    calculate_plasma_params,
    compute_epstot,
    compute_initial_gradpsi_nemov,
    fieldline_smoothness_poincare,
    find_axis_multi_guess,
    initialize_mgrid_field,
    set_extcur,
    set_trace_parameters,
    trace_fieldline,
)

# ---------------------------------------------------------------------------
# face status codes (per traced surface)
# ---------------------------------------------------------------------------
ST_OK = 0          # traces, smooth section, no large hole
ST_TRACE = 1       # field-line integration failed (negative LSODE istate)
ST_RESIDUAL = 2    # section is not a smooth closed curve
ST_GAP = 3         # section leaves a large angular hole (island chain)
ST_GEOM = 4        # section does not wind monotonically in theta (geometry
                   # from calculate_plasma_params would be garbage: measured
                   # iota=28 and a=0.34 on such faces)
STATUS_NAMES = {ST_OK: "ok", ST_TRACE: "trace_fail",
                ST_RESIDUAL: "residual", ST_GAP: "gap", ST_GEOM: "bad_geom"}

# configuration status codes (whole radial column)
CFG_OK = 0
CFG_NO_AXIS = 1
CFG_NO_SURFACE = 2     # no face passed, not even the innermost one
CFG_HIT_DRMAX = 3      # nested column still healthy at dr_max
STATUS_CFG_NAMES = {CFG_OK: "ok", CFG_NO_AXIS: "no_axis",
                    CFG_NO_SURFACE: "no_surface",
                    CFG_HIT_DRMAX: "hit_drmax"}

# tier tags stored per profile row
TIER_PROFILE = 0
TIER_FULL = 1

# Fields recorded for every traced face, in HDF5 column order.  ``eps`` is
# only filled for faces that passed every gate; eps of a rejected face goes to
# ``eps_defect`` so the two never mix in a distribution.
PROFILE_FIELDS = (
    "dr", "R_start", "Z_start", "tier", "nturn", "nphi", "npart",
    "status", "istate", "residual", "gap_rad", "gap_norm",
    "a", "volume", "iota", "aspect", "elongation",
    "Bmean", "Bmax", "Bmin", "ripple",
    "r_loc_mean", "r_loc_min", "r_loc_max", "drift", "rational_dist",
    "eps", "eps_defect", "gradpsi_drift", "geom_ok",
)


def fmt_duration(seconds: float) -> str:
    """Human-readable duration, e.g. ``2h 05m``, ``12m 30s``, ``8.4s``."""
    sec = float(max(seconds, 0.0))
    hours = int(sec // 3600)
    minutes = int((sec % 3600) // 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {sec % 60:04.1f}s"
    return f"{sec:.1f}s"


@dataclass
class ScanConfig:
    """All knobs of the radial scan oracle."""
    mgrid_path: str
    nfp: int = 3
    full_torus: bool = False
    # L1 axis scan
    rmin: float = 1.10
    rmax: float = 1.35
    rstep: float = 0.05
    axis_nphi: int = 180
    axis_z_tol: float = 1e-6
    n_axis_tries: int = 3      # candidate axes probed before committing
    axis_probe_dr: float = 0.06  # prober radius used to rank axis branches
    axis_probe_nturn: int = 40
    axis_probe_npart: int = 200
    # tiers: nphi is full for both (the ODE cost is independent of nphi, only
    # the eps quadrature pays for it)
    nphi: int = 360
    profile_nturn: int = 150
    profile_npart: int = 500
    full_nturn: int = 400
    full_npart: int = 2000
    # radial walk
    walk_eps: bool = False     # True = measure eps on EVERY walked surface
    dr0: float = 0.02
    dr_step: float = 0.02
    dr_max: float = 0.60
    # gates
    residual_tol: float = 0.05
    gap_norm_tol: float = 15.0
    smooth_n_harmonics: int = 4
    # low-order rational bookkeeping (diagnostic only, no filtering yet)
    rational_qmax: int = 8
    # diagnostic: when a sample has several candidate axes, probe ALL of them
    # (instead of stopping at the first one that reaches the probe radius) and
    # keep their individual reaches.  Off by default: it only costs time.
    axis_record_branches: bool = False

    @property
    def max_faces(self) -> int:
        """Worst-case number of faces stored per configuration."""
        return int(round(self.dr_max / self.dr_step)) + 4


# Worker globals (set once per process by _worker_init)
_worker_cfg: Optional[ScanConfig] = None
_worker_eps_at_a: Optional[float] = None


# ---------------------------------------------------------------------------
# per-surface measurement
# ---------------------------------------------------------------------------

class _AxisRef:
    """Magnetic axis reference used to build start points and polar angles."""

    __slots__ = ("rz", "R0", "Ra", "Za")

    def __init__(self, rz, R0, Ra, Za):
        self.rz = rz
        self.R0 = R0
        self.Ra = Ra
        self.Za = Za

    def axis_data(self):
        """(nphi, 2) axis (R, Z) per toroidal angle, for calculate_plasma_params."""
        return np.column_stack([self.Ra, self.Za])


def _axis_ref(axis_rz, nphi: int) -> Optional[_AxisRef]:
    nphi = int(nphi)
    rz = np.asarray(axis_rz, dtype=np.float64, order="F")
    fld, ist = trace_fieldline(initial_rz=rz, nturn=2, nphi=nphi, verbose=False)
    if ist != 0 or fld.shape[0] < 2 * nphi:
        return None
    Ra = np.asarray(fld[:nphi, 0], dtype=np.float64)
    Za = np.asarray(fld[:nphi, 1], dtype=np.float64)
    R0 = float(np.mean(np.sqrt(Ra ** 2 + Za ** 2)))
    return _AxisRef(rz, R0, Ra, Za)


def _nearest_rational_dist(iota: float, qmax: int) -> float:
    if not np.isfinite(iota):
        return np.nan
    qs = np.arange(1, int(qmax) + 1)
    cands = np.unique(np.concatenate(
        [np.arange(0, int(3 * q) + 1) / float(q) for q in qs]))
    return float(np.min(np.abs(iota - cands)))


def _probe_reach(extcur, ax: _AxisRef, cfg: "ScanConfig") -> float:
    """Deepest dr (<= axis_probe_dr) whose column still passes at probe tier."""
    reach = 0.0
    dr = float(cfg.dr0)
    while dr <= min(cfg.axis_probe_dr, cfg.dr_max) + 1e-12:
        row = measure_surface(extcur, ax, dr, cfg.axis_probe_nturn, cfg.nphi,
                              cfg.axis_probe_npart, TIER_PROFILE, cfg,
                              compute_eps=False)
        if row["status"] != ST_OK:
            break
        reach = dr
        dr += float(cfg.dr_step)
    return reach


def _pick_axis(extcur, axes, cfg: "ScanConfig", all_branches: bool = False):
    """Rank axis branches by how far their nested column survives.

    With ``all_branches`` every candidate is probed (no early exit) and the
    per-branch reaches are returned as well, so a configuration with two
    usable axes can be identified.  Default behaviour is unchanged.
    """
    best = None
    ranked = []
    for cand in axes[: int(cfg.n_axis_tries)]:
        axis_rz = np.asarray(cand, dtype=np.float64)
        ax = _axis_ref(axis_rz, cfg.nphi)
        if ax is None:
            continue
        reach = _probe_reach(extcur, ax, cfg)
        ranked.append({"reach": float(reach),
                       "axis_R": float(axis_rz[0]),
                       "axis_Z": float(axis_rz[1])})
        if best is None or reach > best[1]:
            best = (ax, reach, axis_rz)
        if not all_branches and reach >= min(cfg.axis_probe_dr, cfg.dr_max) - 1e-12:
            break          # this branch already reaches the probe radius
    if best is None:
        return None, np.nan, np.array([np.nan, np.nan]), ranked
    return best[0], best[1], best[2], ranked


def measure_surface(extcur: np.ndarray, ax: _AxisRef, dr: float,
                    nturn: int, nphi: int, npart: int, tier: int,
                    cfg: ScanConfig, compute_eps: bool = True) -> dict:
    """Trace one surface and measure everything on that single trace.

    ``compute_eps=False`` is the cheap screening tier: the same trace, gates and
    1D quantities, but no eps.  eps needs the gradient transport, which is the
    single expensive and fragile part of the oracle, so it is only evaluated
    where it is actually wanted (``tier=TIER_FULL``).
    """
    row = {k: np.nan for k in PROFILE_FIELDS}
    row.update(dr=float(dr), tier=int(tier), nturn=int(nturn),
               nphi=int(nphi), npart=int(npart), status=ST_TRACE,
               istate=-9999)

    start = np.array([ax.rz[0] + float(dr), ax.rz[1]],
                     dtype=np.float64, order="F")
    row["R_start"] = float(start[0])
    row["Z_start"] = float(start[1])

    if compute_eps:
        gradpsi = compute_initial_gradpsi_nemov(extcur, start[0], start[1],
                                                verbose=False)
        # compute_epstot inherits the resolution of the last
        # set_trace_parameters call, and any trace_fieldline(nturn=2, ...) in
        # between silently changes it.  Always (re)set it immediately before.
        set_trace_parameters(int(nturn), int(nphi), npart=int(npart),
                             verbose=False)
        eps, _, fld, ist = compute_epstot(start, initial_gradpsi=gradpsi,
                                          return_fieldline=True,
                                          verbose=False)
        eps_val = np.nan if eps is None else float(eps)
    else:
        set_trace_parameters(int(nturn), int(nphi), npart=int(npart),
                             verbose=False)
        fld, ist = trace_fieldline(initial_rz=start, nturn=int(nturn),
                                   nphi=int(nphi), verbose=False)
        eps_val = np.nan
    row["istate"] = int(ist)
    if ist != 0 or fld is None or fld.shape[0] < int(nturn) * int(nphi):
        return row
    fld = fld[: int(nturn) * int(nphi)]

    # --- section smoothness (same points as eps) ---
    _, metrics = fieldline_smoothness_poincare(
        fld, nturn, nphi, axis_rz=ax.rz,
        n_harmonics=cfg.smooth_n_harmonics, residual_rms_frac_tol=1e9,
        max_angular_gap=1e9, min_points=16)
    resid = float(metrics.get("residual_rms_frac", np.nan))
    gap = float(metrics.get("max_gap_rad", np.nan))
    gap_norm = gap * float(nturn) / (2.0 * np.pi)
    row["residual"] = resid
    row["gap_rad"] = gap
    row["gap_norm"] = gap_norm

    # --- 0D geometry: volume, volume-equivalent minor radius a, iota ---
    # calculate_plasma_params validates the winding itself and returns NaN plus
    # info["ok"]=False when the section is not a simple closed curve about the
    # axis, so there is no second check to keep in sync here.
    row["geom_ok"] = 0.0
    try:
        vol, aminor, iota, info = calculate_plasma_params(
            fld, ax.axis_data(), nturn, nphi, ax.R0, return_status=True)
        if info["ok"] and np.isfinite(aminor):
            row["geom_ok"] = 1.0
            row["volume"] = float(vol)
            row["a"] = float(aminor)
            row["iota"] = float(iota)
            if aminor > 0:
                row["aspect"] = float(ax.R0 / aminor)
            row["elongation"] = float(info["elongation"])
            row["rational_dist"] = _nearest_rational_dist(row["iota"],
                                                          cfg.rational_qmax)
    except Exception:
        pass

    # --- field strength along the surface, and the real poloidal radius ---
    B = np.asarray(fld[:, 6], dtype=np.float64)
    if B.size == int(nturn) * int(nphi) and np.isfinite(B).all():
        bmax, bmin = float(B.max()), float(B.min())
        row["Bmean"] = float(B.mean())
        row["Bmax"] = bmax
        row["Bmin"] = bmin
        if bmax + bmin > 0:
            row["ripple"] = (bmax - bmin) / (bmax + bmin)
    R = np.asarray(fld[:, 0], dtype=np.float64).reshape(nturn, nphi)
    Z = np.asarray(fld[:, 1], dtype=np.float64).reshape(nturn, nphi)
    r_loc = np.sqrt((R - ax.Ra[None, :]) ** 2 + (Z - ax.Za[None, :]) ** 2)
    row["r_loc_mean"] = float(r_loc.mean())
    row["r_loc_min"] = float(r_loc.min())
    row["r_loc_max"] = float(r_loc.max())
    k = max(1, int(nturn) // 10)
    row["drift"] = float(r_loc[-k:].mean() - r_loc[:k].mean())


    # --- QC probe for the gradient transport (only where eps was computed) ---
    # The transported |grad psi| must stay a BOUNDED function on the surface, so
    # its ratio to a bounded geometric quantity (R*B_pol, flat along the line to
    # 0.3%) may vary geometrically but must not drift secularly.  When the
    # non-physical mode of the transport takes over, the ratio climbs (or falls)
    # monotonically along the line.  Anchors measured on H1: 0.87 clean,
    # 0.73 borderline, 5.44 corrupted.
    gmag = np.asarray(fld[:, 10], dtype=np.float64)
    if np.any(gmag > 0):
        bpol = np.hypot(np.asarray(fld[:, 3], float),
                        np.asarray(fld[:, 4], float))
        ref = np.asarray(fld[:, 0], float) * bpol
        good = np.isfinite(ref) & (ref > 1e-12) & np.isfinite(gmag)
        if good.sum() > 100:
            q = np.full(gmag.size, np.nan)
            q[good] = gmag[good] / ref[good]
            qt = q.reshape(int(nturn), int(nphi)).mean(axis=1)
            kq = max(1, int(nturn) // 10)
            a_, b_ = qt[:kq].mean(), qt[-kq:].mean()
            if np.isfinite(a_) and np.isfinite(b_) and abs(a_) > 0:
                row["gradpsi_drift"] = float(b_ / a_)

    # --- gate + eps split ---
    if not (np.isfinite(resid) and np.isfinite(gap_norm)):
        row["status"] = ST_RESIDUAL
    elif resid > cfg.residual_tol:
        row["status"] = ST_RESIDUAL
    elif gap_norm > cfg.gap_norm_tol:
        row["status"] = ST_GAP
    elif row["geom_ok"] == 0.0:
        row["status"] = ST_GEOM
    else:
        row["status"] = ST_OK

    if compute_eps:
        if row["status"] == ST_OK:
            row["eps"] = eps_val
        else:
            # kept apart from eps on purpose: never mix into a distribution
            row["eps_defect"] = eps_val
    return row


# ---------------------------------------------------------------------------
# one configuration: axis -> walk -> bisection -> full LCFS run
# ---------------------------------------------------------------------------

def _new_cfg_result() -> dict:
    return {
        "status": CFG_NO_AXIS,
        "rows": [],
        "axis_RZ": np.array([np.nan, np.nan]),
        "axis_count": 0,
        "axis_reach": np.nan,
        "lcfs_dr": np.nan,
        "dr_ref": np.nan,
        "a_ref": np.nan,
        "eps_at_a": np.nan,
        "eps_status": -1,
        "t_total": 0.0,
    }


def _dr_for_a(rows: list[dict], a_target: float):
    """dr whose measured minor radius equals a_target (linear in a), or None."""
    pts = [(r["dr"], r["a"]) for r in rows
           if r["tier"] == TIER_PROFILE and r["status"] == ST_OK
           and np.isfinite(r["a"]) and r["a"] > 0]
    if len(pts) < 2:
        return None
    dr = np.array([q[0] for q in pts])
    a = np.array([q[1] for q in pts])
    order = np.argsort(a)
    a, dr = a[order], dr[order]
    if not (a[0] <= a_target <= a[-1]):
        return None
    return float(np.interp(a_target, a, dr))


def scan_config(extcur: np.ndarray, cfg: ScanConfig,
                eps_at_a: Optional[float] = None) -> dict:
    """Radial scan for one coil-current vector.

    The walk always runs at the cheap ``profile`` tier WITHOUT eps: depth and
    geometry come from the trace itself, and eps is the one expensive (and, in
    the outer 20% of the column, fragile) step.  Passing ``eps_at_a`` adds a
    single ``full``-tier measurement on the surface whose minor radius is
    closest to that value -- the only place eps is ever computed.
    """
    extcur = np.asarray(extcur, dtype=np.float64)
    out = _new_cfg_result()
    t0 = time.perf_counter()
    set_extcur(extcur)

    axes = find_axis_multi_guess(
        rmin=cfg.rmin, rmax=cfg.rmax, rstep=cfg.rstep, z0=0.0,
        xtol=1e-6, max_iter=100, delta_r=0.01,
        axis_z_tol=cfg.axis_z_tol, nphi=cfg.axis_nphi)
    out["axis_count"] = len(axes)
    if not axes:
        out["t_total"] = time.perf_counter() - t0
        return out

    # A one-turn fixed point is not necessarily the main axis: an island-centre
    # fixed point also has a nested neighbourhood, and it breaks a few
    # centimetres out.  Both pass |Z| <= tol, so the branch must be chosen by
    # how far its column survives, not by scan order.  This matters: the
    # early-exit `find_axis_any` returned the first R-scan hit, and for a
    # measured sample that was the wrong branch (R=1.267 instead of the healthy
    # R=1.125), which silently rejected a good current configuration.
    want_all = bool(getattr(cfg, "axis_record_branches", False)) and len(axes) > 1
    ax, reach, axis_rz, ranked = _pick_axis(extcur, axes, cfg,
                                            all_branches=want_all)
    if ax is None:
        out["t_total"] = time.perf_counter() - t0
        return out
    out["axis_RZ"] = axis_rz
    out["axis_reach"] = reach
    if want_all:
        out["branches"] = ranked

    rows = out["rows"]
    dr = float(cfg.dr0)
    last_ok = 0.0
    bracket = None
    while dr <= float(cfg.dr_max) + 1e-12:
        row = measure_surface(extcur, ax, dr, cfg.profile_nturn, cfg.nphi,
                              cfg.profile_npart, TIER_PROFILE, cfg,
                              compute_eps=bool(cfg.walk_eps))
        rows.append(row)
        if row["status"] != ST_OK:
            bracket = (last_ok, dr)
            break
        last_ok = dr
        dr += float(cfg.dr_step)

    if bracket is None:
        # healthy all the way to dr_max; the outermost measured face is the LCFS
        out["status"] = CFG_HIT_DRMAX
        lcfs_dr = last_ok
    elif bracket[0] <= 0.0:
        # not even the innermost face passed
        out["status"] = CFG_NO_SURFACE
        out["lcfs_dr"] = np.nan
        out["t_total"] = time.perf_counter() - t0
        return out
    else:
        # exactly one bisection: dr_step -> dr_step/2
        lo, hi = bracket
        mid = 0.5 * (lo + hi)
        row = measure_surface(extcur, ax, mid, cfg.profile_nturn, cfg.nphi,
                              cfg.profile_npart, TIER_PROFILE, cfg,
                              compute_eps=bool(cfg.walk_eps))
        rows.append(row)
        if row["status"] == ST_OK:
            lo = mid
        else:
            hi = mid
        lcfs_dr = lo
        out["status"] = CFG_OK

    out["lcfs_dr"] = float(lcfs_dr)

    # --- optional: the single full-tier eps measurement, at a = eps_at_a ---
    if eps_at_a is not None:
        dr_ref = _dr_for_a(rows, float(eps_at_a))
        if dr_ref is not None:
            row = measure_surface(extcur, ax, dr_ref, cfg.full_nturn, cfg.nphi,
                                  cfg.full_npart, TIER_FULL, cfg,
                                  compute_eps=True)
            rows.append(row)
            out["dr_ref"] = dr_ref
            out["a_ref"] = row["a"]
            # the MEASURED eps at that surface (not the requested a0!), plus the
            # face status so a failed measurement is not mistaken for a value
            out["eps_at_a"] = float(row["eps"]) if np.isfinite(row["eps"]) else np.nan
            out["eps_status"] = int(row["status"])
    out["t_total"] = time.perf_counter() - t0
    return out


def measure_reference(extcur: np.ndarray, cfg: ScanConfig, axis_rz,
                      dr_ref: float) -> Optional[dict]:
    """One full-tier measurement (with eps) at a known axis and dr.

    Stage 3 knows the axis and the surface it wants (a = a0, taken from the
    stage-2 depth map), so the walk does not have to be repeated: this measures
    exactly one face and returns its row.
    """
    extcur = np.asarray(extcur, dtype=np.float64)
    set_extcur(extcur)
    ax = _axis_ref(np.asarray(axis_rz, dtype=np.float64), cfg.nphi)
    if ax is None:
        return None
    return measure_surface(extcur, ax, float(dr_ref), cfg.full_nturn, cfg.nphi,
                           cfg.full_npart, TIER_FULL, cfg, compute_eps=True)


def _map_config_indexed(args):
    idx, point = args
    return idx, scan_config(point, _worker_cfg, _worker_eps_at_a)


def _worker_init(cfg: ScanConfig, eps_at_a: Optional[float] = None):
    global _worker_cfg, _worker_eps_at_a
    _worker_cfg = cfg
    _worker_eps_at_a = eps_at_a
    with open(os.devnull, "w") as sink:
        with contextlib.redirect_stdout(sink):
            initialize_mgrid_field(str(cfg.mgrid_path), nfp=cfg.nfp,
                                   full_torus=cfg.full_torus)


# ---------------------------------------------------------------------------
# sampling + batch scan
# ---------------------------------------------------------------------------

def sample_bounds(bounds: np.ndarray, n_samples: int, seed: int = 0) -> np.ndarray:
    """Sobol-sample absolute bounds; locked coils (lo == hi) stay fixed."""
    bounds = np.asarray(bounds, dtype=np.float64)
    lo, hi = bounds[:, 0], bounds[:, 1]
    u = Sobol(d=len(lo), scramble=True, seed=seed).random(n_samples)
    return lo + u * (hi - lo)


def _flat_results(samples: np.ndarray, results: list[dict], cfg: ScanConfig):
    """Pack per-config results into flat arrays (+ ragged profiles, NaN padded).

    Two groups of scalar columns:
      lcfs_* : geometry of the outermost certified face (from the walk)
      ref_*  : the single full-tier measurement where eps lives (a = eps_at_a)
    """
    n = len(results)
    nf = cfg.max_faces
    prof = {k: np.full((n, nf), np.nan) for k in PROFILE_FIELDS}
    keys = ("axis_R", "axis_Z", "axis_count", "axis_reach", "status",
            "n_faces", "lcfs_dr",
            "lcfs_a", "lcfs_volume", "lcfs_iota", "lcfs_aspect",
            "lcfs_elongation",
            "lcfs_residual", "lcfs_gap_norm", "lcfs_ripple",
            "ref_dr", "ref_a", "ref_eps", "ref_eps_defect", "ref_iota",
            "ref_volume", "ref_ripple", "ref_gradpsi_drift",
            "ref_residual", "ref_gap_norm",
            "ref_nturn", "ref_nphi", "ref_npart", "t_total")
    scal = {k: np.full(n, np.nan) for k in keys}
    for k in ("axis_count", "status", "n_faces", "ref_nturn", "ref_nphi",
              "ref_npart"):
        scal[k] = np.zeros(n, dtype=np.int32)
    scal["status"] = np.full(n, -1, dtype=np.int8)

    for i, r in enumerate(results):
        scal["axis_R"][i] = r["axis_RZ"][0]
        scal["axis_Z"][i] = r["axis_RZ"][1]
        scal["axis_count"][i] = r["axis_count"]
        scal["axis_reach"][i] = r["axis_reach"]
        scal["status"][i] = r["status"]
        scal["lcfs_dr"][i] = r["lcfs_dr"]
        scal["t_total"][i] = r["t_total"]
        rows = r["rows"]
        scal["n_faces"][i] = len(rows)
        for j, row in enumerate(rows[:nf]):
            for k in PROFILE_FIELDS:
                prof[k][i, j] = row[k]
        # geometry of the outermost certified face
        for row in reversed(rows):
            if row["tier"] == TIER_PROFILE and row["status"] == ST_OK:
                scal["lcfs_a"][i] = row["a"]
                scal["lcfs_volume"][i] = row["volume"]
                scal["lcfs_iota"][i] = row["iota"]
                scal["lcfs_aspect"][i] = row["aspect"]
                scal["lcfs_elongation"][i] = row["elongation"]
                scal["lcfs_residual"][i] = row["residual"]
                scal["lcfs_gap_norm"][i] = row["gap_norm"]
                scal["lcfs_ripple"][i] = row["ripple"]
                break
        # the one full-tier eps measurement
        for row in reversed(rows):
            if row["tier"] == TIER_FULL:
                scal["ref_dr"][i] = row["dr"]
                scal["ref_a"][i] = row["a"]
                scal["ref_iota"][i] = row["iota"]
                scal["ref_volume"][i] = row["volume"]
                scal["ref_ripple"][i] = row["ripple"]
                scal["ref_gradpsi_drift"][i] = row["gradpsi_drift"]
                scal["ref_residual"][i] = row["residual"]
                scal["ref_gap_norm"][i] = row["gap_norm"]
                scal["ref_eps"][i] = row["eps"]
                scal["ref_eps_defect"][i] = row["eps_defect"]
                scal["ref_nturn"][i] = row["nturn"]
                scal["ref_nphi"][i] = row["nphi"]
                scal["ref_npart"][i] = row["npart"]
                break
    return {"samples": samples, "profile": prof, "scalars": scal,
            "config": asdict(cfg)}


def save_scan_h5(path: Path, flat: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        g = f.create_group("profile")
        for k, v in flat["profile"].items():
            g.create_dataset(k, data=v, compression="gzip")
        s = f.create_group("scalars")
        for k, v in flat["scalars"].items():
            s.create_dataset(k, data=v, compression="gzip")
        f.create_dataset("samples", data=flat["samples"], compression="gzip")
        f.attrs["config"] = json.dumps(flat["config"], indent=1)
        f.attrs["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            import subprocess
            f.attrs["git_sha"] = subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                cwd=str(Path(__file__).resolve().parent)).stdout.strip()
        except Exception:
            pass


def load_scan_h5(path: Path) -> dict:
    path = Path(path)
    with h5py.File(path, "r") as f:
        flat = {
            "samples": f["samples"][:],
            "profile": {k: f["profile"][k][:] for k in f["profile"]},
            "scalars": {k: f["scalars"][k][:] for k in f["scalars"]},
            "config": json.loads(f.attrs["config"]),
        }
        flat["attrs"] = dict(f.attrs)
    return flat


def scan_box(cfg: ScanConfig, bounds: np.ndarray, n_samples: int = 256,
             seed: int = 0, processes: int = 0,
             samples: Optional[np.ndarray] = None,
             output_h5: Optional[Path] = None,
             progress_interval: int = 64,
             eps_at_a: Optional[float] = None) -> dict:
    """Sample the box (or take ``samples``) and run the radial scan on each."""
    bounds = np.asarray(bounds, dtype=np.float64)
    if samples is None:
        samples = sample_bounds(bounds, n_samples, seed=seed)
    else:
        samples = np.asarray(samples, dtype=np.float64)
    n = int(samples.shape[0])
    t0 = time.perf_counter()
    results: list[dict] = [None] * n

    if processes == 0:
        print(f"  scanning {n} samples serially", flush=True)
        _worker_init(cfg, eps_at_a)
        for k in range(n):
            results[k] = scan_config(samples[k], cfg, eps_at_a)
            if progress_interval > 0 and (k + 1) % progress_interval == 0:
                el = time.perf_counter() - t0
                print(f"    {k + 1}/{n}  {k / max(el, 1e-9):.1f} samples/s"
                      f"  elapsed {fmt_duration(el)}"
                      f"  ETA {fmt_duration(el / (k + 1) * (n - k - 1))}",
                      flush=True)
    else:
        tasks = [(k, samples[k]) for k in range(n)]
        print(f"  scanning {n} samples with {processes} workers", flush=True)
        with multiprocessing.Pool(processes=processes,
                                  initializer=_worker_init,
                                  initargs=(cfg, eps_at_a)) as pool:
            for done, (idx, res) in enumerate(
                    pool.imap_unordered(_map_config_indexed, tasks,
                                        chunksize=1), start=1):
                results[idx] = res
                if progress_interval > 0 and done % progress_interval == 0:
                    el = time.perf_counter() - t0
                    print(f"    {done}/{n}  {done / max(el, 1e-9):.1f} samples/s"
                          f"  elapsed {fmt_duration(el)}"
                          f"  ETA {fmt_duration(el / done * (n - done))}",
                          flush=True)

    flat = _flat_results(samples, results, cfg)
    if output_h5 is not None:
        save_scan_h5(Path(output_h5), flat)
        print(f"  saved -> {output_h5}", flush=True)
    _report(flat, time.perf_counter() - t0)
    return flat


def _report(flat: dict, elapsed: float):
    s = flat["scalars"]
    n = len(s["status"])
    codes = s["status"]
    n_axis = int(np.sum(s["axis_count"] > 0))
    n_ok = int(np.sum(codes == CFG_OK))
    n_max = int(np.sum(codes == CFG_HIT_DRMAX))
    has_eps = np.isfinite(s["ref_eps"])
    print(f"  {n} samples in {fmt_duration(elapsed)}"
          f" ({elapsed / max(n, 1):.2f} s/sample)"
          f": axis {n_axis}, LCFS {n_ok}, hit_drmax {n_max},"
          f" eps measured {int(has_eps.sum())}", flush=True)
    good = np.isfinite(s["lcfs_dr"]) & (s["lcfs_dr"] > 0)
    if good.any():
        for name in ("lcfs_dr", "lcfs_a", "lcfs_volume", "lcfs_iota",
                     "lcfs_gap_norm"):
            v = s[name][good]
            q = np.percentile(v, [0, 5, 50, 95, 100])
            print(f"    {name:16s} min {q[0]:.4f} p5 {q[1]:.4f}"
                  f" med {q[2]:.4f} p95 {q[3]:.4f} max {q[4]:.4f}", flush=True)
    if has_eps.any():
        for name in ("ref_a", "ref_eps", "ref_gradpsi_drift"):
            v = s[name][has_eps]
            v = v[np.isfinite(v)]
            if v.size == 0:
                continue
            q = np.percentile(v, [0, 5, 50, 95, 100])
            print(f"    {name:16s} min {q[0]:.4f} p5 {q[1]:.4f}"
                  f" med {q[2]:.4f} p95 {q[3]:.4f} max {q[4]:.4f}", flush=True)


# ---------------------------------------------------------------------------
# stage 1: cheap "does a REAL axis exist here" probe
# ---------------------------------------------------------------------------

def stage1_probe(extcur: np.ndarray, cfg: ScanConfig) -> dict:
    """Cheapest feasibility probe used to map the axis region.

    A one-turn fixed point alone is NOT evidence of a nested column: an
    island-centre fixed point has one too.  The probe therefore follows
    ``_pick_axis`` -- it walks each candidate branch out to ``axis_probe_dr``
    at the nturn=40 tier and reports how far the best branch survived.

    Cost is dominated by the axis enumeration (find_axis_multi_guess scans the
    whole R window): measured 4.4 s/sample on one core with up to 3 branches,
    i.e. ~1.5x cheaper than the screening walk per unit depth, and it needs no
    eps and no long traces.
    """
    t0 = time.perf_counter()
    extcur = np.asarray(extcur, dtype=np.float64)
    out = {"n_axes": 0, "axis_R": np.nan, "axis_Z": np.nan, "reach": np.nan,
           "branches": [], "t_total": 0.0}
    set_extcur(extcur)
    axes = find_axis_multi_guess(
        rmin=cfg.rmin, rmax=cfg.rmax, rstep=cfg.rstep, z0=0.0,
        xtol=1e-6, max_iter=100, delta_r=0.01,
        axis_z_tol=cfg.axis_z_tol, nphi=cfg.axis_nphi)
    out["n_axes"] = len(axes)
    if axes:
        # when asked, probe every branch so that two usable axes are not hidden
        want_all = bool(getattr(cfg, "axis_record_branches", False)) and len(axes) > 1
        ax, reach, axis_rz, ranked = _pick_axis(extcur, axes, cfg,
                                                all_branches=want_all)
        if ax is not None:
            out["axis_R"] = float(axis_rz[0])
            out["axis_Z"] = float(axis_rz[1])
            out["reach"] = float(reach)
        if want_all:
            out["branches"] = ranked
    out["t_total"] = time.perf_counter() - t0
    return out


def _probe_indexed(args):
    idx, point = args
    return idx, stage1_probe(point, _worker_cfg)


def _save_dual_axis(samples: np.ndarray, res: list, cfg: "ScanConfig", path: Path):
    """Collect the configurations where TWO axis branches pass the first probe.

    A configuration counts as dual-axis when at least two distinct candidate
    branches both survive the first probe step (dr0), i.e. both have a usable
    nested column.  This is a side product of stage 1: it is written to its own
    file so the main axis map keeps its schema.
    """
    first = float(cfg.dr0) - 1e-9
    rows = []
    for i, r in enumerate(res):
        br = r.get("branches") or []
        good = [b for b in br if b["reach"] >= first]
        if len(good) >= 2:
            rows.append((i, good))
    if not rows:
        print(f"  dual-axis: none found (wrote nothing to {path})", flush=True)
        return
    maxb = max(len(g) for _, g in rows)
    m = len(rows)
    samples = np.asarray(samples)[[i for i, _ in rows]]
    reach = np.full((m, maxb), np.nan)
    bR = np.full((m, maxb), np.nan)
    bZ = np.full((m, maxb), np.nan)
    for k, (_, g) in enumerate(rows):
        for j, b in enumerate(sorted(g, key=lambda q: -q["reach"])):
            reach[k, j] = b["reach"]
            bR[k, j] = b["axis_R"]
            bZ[k, j] = b["axis_Z"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("samples", data=samples)
        f.create_dataset("branch_reach", data=reach)
        f.create_dataset("branch_R", data=bR)
        f.create_dataset("branch_Z", data=bZ)
        f.create_dataset("n_branches", data=np.sum(np.isfinite(reach), axis=1))
        f.attrs["probe_dr"] = float(cfg.axis_probe_dr)
        f.attrs["first_step"] = float(cfg.dr0)
        f.attrs["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"  dual-axis: {m} configurations with >=2 usable axes -> {path}",
          flush=True)


def scan_all_box(cfg: ScanConfig, bounds: np.ndarray, n_samples: int = 5000,
                 seed: int = 0, samples: Optional[np.ndarray] = None,
                 processes: int = 0, out_dir: Optional[Path] = None,
                 chunk_size: int = 1000,
                 eps_at_a: Optional[float] = None,
                 dual_axis_h5: Optional[Path] = None,
                 prefix: str = "chunk") -> list:
    """One-pass scan: axis + outward walk (+ per-surface eps) for every sample.

    Everything happens in a single loop over the samples: the axis is found and
    ranked, then the column is walked outward to its LCFS and, when
    ``cfg.walk_eps`` is on, eps is measured on every surface on the way.  The
    per-sample axis position is stored in the chunk scalars, so a later re-walk
    (different eps resolution, dr step or gates) can skip the axis search,
    which dominates the cost of the whole scan.

    Chunks are written to ``out_dir/<prefix>_NNNN.h5`` as soon as they finish
    and existing chunks are skipped on restart, so the job is interruptible.
    """
    bounds = np.asarray(bounds, dtype=np.float64)
    if samples is None:
        samples = sample_bounds(bounds, n_samples, seed=seed)
    samples = np.asarray(samples, dtype=np.float64)
    n = int(samples.shape[0])
    out_dir = Path(out_dir) if out_dir is not None else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    nchunks = (n + chunk_size - 1) // chunk_size
    print(f"  one-pass scan: {n} samples in {nchunks} chunks of {chunk_size};"
          f" walk_eps={bool(cfg.walk_eps)}; output {out_dir}", flush=True)

    t0 = time.perf_counter()
    done_before = 0
    pool = None
    if processes > 0:
        pool = multiprocessing.Pool(processes=processes,
                                    initializer=_worker_init,
                                    initargs=(cfg, eps_at_a))
    try:
        for ci in range(nchunks):
            lo = ci * chunk_size
            sub = samples[lo:lo + chunk_size]
            path = out_dir / f"{prefix}_{ci:04d}.h5"
            if path.exists():
                done_before += len(sub)
                print(f"    chunk {ci + 1}/{nchunks}: exists, skipped", flush=True)
                continue
            res: list = [None] * len(sub)
            if pool is None:
                _worker_init(cfg, eps_at_a)
                for k, point in enumerate(sub):
                    res[k] = scan_config(point, cfg, eps_at_a)
            else:
                tasks = [(k, sub[k]) for k in range(len(sub))]
                for idx, r in pool.imap_unordered(_map_config_indexed, tasks,
                                                  chunksize=4):
                    res[idx] = r
            flat = _flat_results(sub, res, cfg)
            save_scan_h5(path, flat)
            if dual_axis_h5 is not None and cfg.axis_record_branches:
                _save_dual_axis(sub, res, cfg,
                                out_dir / f"dual_{prefix}_{ci:04d}.h5")
            done = lo + len(sub)
            el = time.perf_counter() - t0
            rate = (done - done_before) / max(el, 1e-9)
            eta = (n - done) / max(rate, 1e-9)
            print(f"    chunk {ci + 1}/{nchunks}  {done}/{n}  {rate:.2f}/s"
                  f"  elapsed {fmt_duration(el)}"
                  f"  ETA {fmt_duration(eta)}", flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    print(f"  all chunks in {fmt_duration(time.perf_counter() - t0)}",
          flush=True)
    return samples


def merge_chunks(out_dir, out_h5, prefix: str = "chunk"):
    """Concatenate the chunk files of a one-pass scan into a single file."""
    files = sorted(Path(out_dir).glob(f"{prefix}_*.h5"))
    if not files:
        raise FileNotFoundError(f"no {prefix}_*.h5 chunks under {out_dir}")
    print(f"  merging {len(files)} chunks", flush=True)
    parts = [load_scan_h5(f) for f in files]
    flat = {
        "samples": np.concatenate([p["samples"] for p in parts]),
        "profile": {k: np.concatenate([p["profile"][k] for p in parts])
                    for k in parts[0]["profile"]},
        "scalars": {k: np.concatenate([p["scalars"][k] for p in parts])
                    for k in parts[0]["scalars"]},
        "config": parts[0].get("config", {}),
    }
    save_scan_h5(Path(out_h5), flat)
    print(f"  merged -> {out_h5}", flush=True)
    return flat


def merge_dual_chunks(out_dir, out_h5, prefix: str = "chunk"):
    """Concatenate the dual-axis side files of a one-pass scan (if any)."""
    files = sorted(Path(out_dir).glob(f"dual_{prefix}_*.h5"))
    if not files:
        print("  no dual-axis side files", flush=True)
        return None
    parts = []
    for f in files:
        with h5py.File(f, "r") as h:
            parts.append({k: h[k][:] for k in
                          ("samples", "branch_reach", "branch_R", "branch_Z",
                           "n_branches")})
    n = sum(len(p["samples"]) for p in parts)
    maxb = max(p["branch_reach"].shape[1] for p in parts)
    out = {k: np.full((n, maxb), np.nan) for k in
           ("branch_reach", "branch_R", "branch_Z")}
    out["samples"] = np.concatenate([p["samples"] for p in parts])
    out["n_branches"] = np.concatenate([p["n_branches"] for p in parts])
    k = 0
    for p in parts:
        m = len(p["samples"])
        for key in ("branch_reach", "branch_R", "branch_Z"):
            out[key][k:k + m, :p[key].shape[1]] = p[key]
        k += m
    with h5py.File(out_h5, "w") as h:
        for key, val in out.items():
            h.create_dataset(key, data=val)
    print(f"  dual-axis: {n} configurations -> {out_h5}", flush=True)
    return out


def probe_box(cfg: ScanConfig, bounds: np.ndarray, n_samples: int = 5000,
              seed: int = 0, processes: int = 0,
              samples: Optional[np.ndarray] = None,
              output_h5: Optional[Path] = None,
              dual_axis_h5: Optional[Path] = None,
              progress_interval: int = 500) -> dict:
    """Run stage 1 over a Sobol set (or an explicit sample list) and save it."""
    bounds = np.asarray(bounds, dtype=np.float64)
    if samples is None:
        samples = sample_bounds(bounds, n_samples, seed=seed)
    else:
        samples = np.asarray(samples, dtype=np.float64)
    n = int(samples.shape[0])
    t0 = time.perf_counter()
    res: list[dict] = [None] * n

    if processes == 0:
        print(f"  stage1 probing {n} samples serially", flush=True)
        _worker_init(cfg)
        for k in range(n):
            res[k] = stage1_probe(samples[k], cfg)
            if progress_interval and (k + 1) % progress_interval == 0:
                el = time.perf_counter() - t0
                print(f"    {k + 1}/{n}  {k / max(el, 1e-9):.1f}/s"
                      f"  {fmt_duration(el)}",
                      flush=True)
    else:
        tasks = [(k, samples[k]) for k in range(n)]
        print(f"  stage1 probing {n} samples with {processes} workers",
              flush=True)
        with multiprocessing.Pool(processes=processes,
                                  initializer=_worker_init,
                                  initargs=(cfg,)) as pool:
            for done, (idx, r) in enumerate(
                    pool.imap_unordered(_probe_indexed, tasks, chunksize=8),
                    start=1):
                res[idx] = r
                if progress_interval and done % progress_interval == 0:
                    el = time.perf_counter() - t0
                    print(f"    {done}/{n}  {done / max(el, 1e-9):.1f}/s"
                          f"  {fmt_duration(el)}"
                          f"  ETA {fmt_duration(el / done * (n - done))}",
                          flush=True)

    scal = {k: np.full(n, np.nan) for k in
            ("n_axes", "axis_R", "axis_Z", "reach", "t_total")}
    for i, r in enumerate(res):
        for k in scal:
            scal[k][i] = r[k]
    elapsed = time.perf_counter() - t0
    n_axis = int(np.sum(scal["n_axes"] > 0))
    n_real = int(np.sum(scal["reach"] > 0))
    n_multi = int(np.sum(scal["n_axes"] > 1))
    print(f"  {n} samples in {fmt_duration(elapsed)}"
          f" ({elapsed / max(n, 1):.2f} s/sample)"
          f": any axis {n_axis} ({100.0 * n_axis / max(n, 1):.1f}%),"
          f" real axis (reach>0) {n_real} ({100.0 * n_real / max(n, 1):.1f}%),"
          f" multi-axis {n_multi} ({100.0 * n_multi / max(n, 1):.1f}%)",
          flush=True)
    if n_axis:
        v = scal["reach"][np.isfinite(scal["reach"])]
        print("    reach distribution:", {round(float(x), 2): int((np.abs(v - x) < 1e-9).sum())
                                          for x in np.unique(v)}, flush=True)
    if dual_axis_h5 is not None:
        _save_dual_axis(samples, res, cfg, Path(dual_axis_h5))
    if output_h5 is not None:
        path = Path(output_h5)
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as f:
            f.create_dataset("samples", data=samples, compression="gzip")
            for k, v in scal.items():
                f.create_dataset(k, data=v, compression="gzip")
            f.attrs["config"] = json.dumps(asdict(cfg), indent=1)
            f.attrs["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"  saved -> {path}", flush=True)
    return {"samples": samples, "scalars": scal, "config": asdict(cfg)}
