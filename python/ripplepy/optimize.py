"""
ripplepy.optimize - Stellarator coil-current optimization via Differential Evolution.

Provides:
  • OptimizationConfig     — typed parameter container (dataclass)
  • StellaratorObjective   — objective function wrapper with error classification
  • DifferentialEvolution  — main DE engine with parallel evaluation & logging
  • run()                  — convenience entry point
  • Legacy functional API  — backward-compatible wrappers
"""

from __future__ import annotations

import csv
import logging
import math
import multiprocessing
import os
import random
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from multiprocessing import Pool
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

from .ripple import (
    find_axis,
    compute_epstot,
    calculate_plasma_params,
    set_extcur,
    compute_initial_gradpsi_nemov,
    set_trace_parameters,
    trace_fieldline,
)

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

@contextmanager
def _silent():
    """Temporarily suppress stdout (for noisy Fortran calls during eval)."""
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")
    try:
        yield
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

logger = logging.getLogger("ripplepy.optimize")


def _setup_logging(log_file: Optional[Path] = None, level=logging.INFO):
    """Configure the module logger (console + optional file)."""
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(log_file), mode="a")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    # Suppress noisy external loggers in workers
    for name in ("h5py", "matplotlib", "PIL"):
        logging.getLogger(name).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Failure classification
# ---------------------------------------------------------------------------

class FailureType(Enum):
    NONE = "none"
    AXIS_NOT_FOUND = "axis_not_found"
    AXIS_OFF_SYMMETRY = "axis_off_symmetry"
    TRACING_FAILED = "tracing_failed"
    EPSILON_NAN = "epsilon_nan"
    CORE_COLLAPSED = "core_collapsed"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class OptimizationConfig:
    """All controllable parameters for a stellarator DE optimisation run.

    Parameters
    ----------
    mgrid_path : str or Path
        Path to the mgrid NetCDF file for the magnetic field.
    nfp : int
        Number of field periods.
    initial_rz : tuple[float, float]
        Starting (R, Z) guess for the magnetic axis search.
    initial_bounds : ndarray, shape (n_coils, 2)
        [nominal, fraction] for each coil.  fraction=0 locks the coil;
        fraction>0 gives [nominal×(1-f), nominal×(1+f)].
    full_torus : bool
        Whether to expand the mgrid to full torus (2π).
    nturn : int
        Toroidal turns for field-line tracing.
    nphi : int
        Poloidal points per turn.
    npart : int
        Number of η values (particle classes) for the ε_eff integrator.
    n_pop : int
        Population size.
    max_gen : int
        Maximum number of generations.
    F : float
        Differential weight (mutation scaling).
    CR : float
        Crossover probability (initial mu_CR for JADE).
    strategy : str
        DE strategy: "jade" (default; adaptive F/CR, current-to-pbest/1 with
        external archive) or "rand1bin" (classic DE/rand/1/bin, fixed F and CR).
    p_best : float
        JADE: fraction of the population defining the "pbest" pool (0 < p <= 1).
    adapt_rate : float
        JADE: adaptation rate c for the mu_F / mu_CR updates (default 0.1).
    archive_size : int
        JADE: maximum external-archive size; 0 = auto (= n_pop).
    delt_r : float
        Radial offset from axis for starting field lines.
    bounds_fraction : float
        Fraction used to build automatic absolute bounds when initial_bounds
        is given as a plain 1-D array of nominal coil currents (default 0.2,
        i.e. ±20% around each initial current).
    processes : int
        Number of parallel worker processes.
    output_dir : str or Path
        Directory for HDF5 / CSV / checkpoint output.
    csv_filename : str
        Name of the summary CSV file.
    device_name : str or None
        Optional device label used in output filenames.
    seed : int or None
        Random seed for reproducibility.
    log_file : str or Path or None
        Path to structured log file.
    log_level : int
        Logging level (e.g. logging.INFO, logging.DEBUG).
    ftol : float or None
        Early-stop tolerance on best-fitness change (disabled if None).
    ftol_relative : bool
        If True (default), `ftol` is interpreted as a RELATIVE change
        (fraction of the current best fitness); if False, as an absolute change.
    patience : int
        Generations to wait after ftol triggers before stopping.
    axis_z_tol : float
        Maximum allowed |Z_axis| (m) under the stellarator-symmetry assumption.
        Configurations whose magnetic axis lies further off the Z=0 symmetry
        plane are treated as invalid (default 0.02).
    min_minor_radius : float
        Absolute lower bound on the acceptable minor radius (m).  A nearly-zero
        minor radius means the magnetic surface has collapsed — the DE
        "improves" epsilon_eff by shrinking the configuration to a thin tube,
        which is a numerical artefact that fails at full resolution.  The
        effective threshold is max(min_minor_radius, min_minor_radius_frac x
        nominal_minor_radius), where nominal_minor_radius is measured when the
        DE evaluates its nominal baseline (default 0.02).
    min_minor_radius_frac : float
        Fraction of the nominal configuration's minor radius that the
        effective core-collapse threshold must reach (default 0.5: a surface
        may shrink to no less than half the nominal size).  The nominal minor
        radius is measured automatically on the first (nominal) evaluation.
    adapt_bounds : bool
        If True (default), each generation checks whether the current best
        presses a box edge; if so, a cheap local feasibility probe extends the
        bound into the feasible region before continuing.
    adapt_bounds_every : int
        Check/adapt bounds every N generations (default 1 = every generation).
    adapt_bounds_n_samples : int
        Number of equally-spaced probe points per pressed direction used in the
        1-D bound-adaptation scan (default 16).
    adapt_bounds_max_invalid : float
        Skip the bound-adaptation probe while more than this fraction of the
        population is invalid — widening into a region the DE cannot use would
        only add more failures (default 0.3).
    adapt_bounds_expand : float
        How far beyond the pressed edge the probe extends (multiple of the
        current coil range; default 1.5).
    adapt_bounds_margin : float
        A coil is "pressed" if the best value is within this fraction of the
        coil's current range of the box edge (default 0.02 = 2%).
    reinit_perturb_frac : float
        When an individual is re-initialised after repeated failures, it is
        placed near a random FEASIBLE population member by perturbing each
        coil by this fraction of the current box width (default 0.1 = ±10%).
        Whole-box random re-initialisation wastes evaluations once the
        feasible region is a small fraction of the box and collapses the
        population into clones.
    """
    mgrid_path: str
    nfp: int
    initial_rz: np.ndarray
    initial_bounds: np.ndarray
    full_torus: bool = False
    nturn: int = 400
    nphi: int = 360
    npart: int = 100
    n_pop: int = 100
    max_gen: int = 100
    F: float = 0.5
    CR: float = 0.7
    strategy: str = "jade"
    p_best: float = 0.1
    adapt_rate: float = 0.1
    archive_size: int = 0
    delt_r: float = 0.05
    bounds_fraction: float = 0.2
    processes: int = 8
    output_dir: Path = field(default_factory=lambda: Path("."))
    csv_filename: str = "Individual_info_list.csv"
    device_name: Optional[str] = None
    seed: Optional[int] = None
    log_file: Optional[Path] = None
    log_level: int = logging.INFO
    ftol: Optional[float] = None
    ftol_relative: bool = True
    patience: int = 10
    axis_z_tol: float = 1e-6
    min_minor_radius: float = 0.02
    min_minor_radius_frac: float = 0.5
    adapt_bounds: bool = True
    adapt_bounds_every: int = 1
    adapt_bounds_n_samples: int = 16
    adapt_bounds_max_invalid: float = 0.3
    adapt_bounds_expand: float = 1.5
    adapt_bounds_margin: float = 0.02
    reinit_perturb_frac: float = 0.1

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        if self.log_file is not None:
            self.log_file = Path(self.log_file)
        for arr_name in ("initial_rz", "initial_bounds"):
            val = getattr(self, arr_name)
            if val is not None:
                setattr(self, arr_name, np.asarray(val, dtype=np.float64))

        # ── Convert bounds → absolute [lo, hi] ─────────────────────────────
        #   initial_bounds may be:
        #     (n_coils, 2)  → [nominal, fraction] per coil
        #     (n_coils,)    → plain nominal currents; auto-bounds ±bounds_fraction
        #   fraction = 0  → locked at nominal (lo == hi)
        #   fraction > 0  → [nominal−|nominal|·fraction, nominal+|nominal|·fraction]
        #   (symmetric around nominal — also correct for NEGATIVE nominal currents)
        raw = np.asarray(self.initial_bounds, dtype=np.float64)
        auto = raw.ndim == 1
        if auto:
            raw = np.column_stack([raw, np.full(len(raw), self.bounds_fraction)])
        elif raw.ndim != 2 or raw.shape[1] != 2:
            raise ValueError(
                "initial_bounds must be shape (n_coils, 2) as [nominal, fraction] "
                "or a 1-D array of nominal coil currents (auto-bounds).")
        # Normalise so downstream code (e.g. the DE baseline) always sees 2-D.
        self.initial_bounds = raw
        n_coils = len(raw)
        abs_bounds = np.empty((n_coils, 2), dtype=np.float64)
        for i in range(n_coils):
            nominal, frac = float(raw[i, 0]), float(raw[i, 1])
            if nominal == 0.0:
                if auto:
                    frac = 0.0      # a coil initially off stays locked at 0
                else:
                    logger.warning(
                        "Coil %d has nominal=0; setting to 1.0 for fraction-based "
                        "bounds.  Consider using a non‑zero nominal value.", i)
                    nominal = 1.0
            d = abs(nominal) * frac
            abs_bounds[i, 0] = nominal - d
            abs_bounds[i, 1] = nominal + d
        self._abs_bounds = abs_bounds

        n_fixed = int(np.sum(abs_bounds[:, 0] == abs_bounds[:, 1]))
        n_free = n_coils - n_fixed
        if n_free == 0:
            logger.warning(
                "All %d coils are fixed — nothing to optimise.", n_coils)
        logger.info(
            "Bounds: %d coils total, %d fixed (fraction=0), %d free",
            n_coils, n_fixed, n_free,
        )


# ---------------------------------------------------------------------------
# Objective function wrapper
# ---------------------------------------------------------------------------

# Process-local cache: each worker (or the main process) initialises the mgrid
# field exactly once.  Without this, every Pool creation would reload the
# interpolation matrix from disk, which is extremely expensive.
_mgrid_initialized: bool = False


class StellaratorObjective:
    """Evaluate one set of coil currents and return fitness + diagnostics.

    Encapsulates the full chain:
    set_extcur → find_axis → compute_initial_gradpsi → trace_fieldline
    → compute_eps_tot → compute plasma params.
    """

    INVALID_FITNESS = 1e4

    def __init__(self, config: OptimizationConfig):
        self.cfg = config

    def evaluate(self, extcur_free: np.ndarray, gen: int, ind_idx: int,
                 initial_rz: Optional[np.ndarray] = None,
                 quiet: bool = False) -> tuple[float, dict]:
        """Run the full evaluation chain.

        Parameters
        ----------
        initial_rz : ndarray or None
            Optional first-choice (R, Z) guess for find_axis.  The DE passes
            the parent individual's recorded axis here; if that guess fails,
            the search falls back to the config-level nominal guess.  This
            keeps the warm-start benefit without locking every trial to the
            previous generation's best axis basin.
        quiet : bool
            If True, suppress per-evaluation warnings and the stdout summary
            line (used by the cheap in-loop bounds probe, which must run the
            same criterion without flooding the log).

        Returns
        -------
        fitness : float
            ε_eff (1e4 on failure).
        info : dict
            Metadata dict (Generation, Individual, extcur, epsilon_eff, …).
        """
        # Initialise mgrid once per process (critical for macOS spawn —
        # pickle does NOT re-run __init__ in the worker).
        global _mgrid_initialized
        if not _mgrid_initialized:
            with _silent():
                from .ripple import initialize_mgrid_field as _init_mgrid
                _init_mgrid(
                    str(self.cfg.mgrid_path),
                    nfp=self.cfg.nfp,
                    full_torus=self.cfg.full_torus,
                )
                _mgrid_initialized = True

        # The DE individual is the full extcur vector — no concatenation needed.
        extcur = np.asarray(extcur_free, dtype=np.float64)
        with _silent():
            extcur = set_extcur(extcur)

        info = {
            "Generation": gen,
            "Individual": ind_idx,
            "extcur": extcur,
            "epsilon_eff": self.INVALID_FITNESS,
            "iota": np.nan,
            "volume": np.nan,
            "average B": np.nan,
            "failure_flag": True,
            "failure_type": FailureType.UNKNOWN.value,
            "failure_message": "",
        }

        # --- find magnetic axis ---
        # Multi-guess: first try the per-individual warm-start guess (parent
        # axis for DE trials), then the config-level nominal guess.  A single
        # global warm-start (e.g. always the previous best's axis) biases the
        # search: trials whose real axis lives in another basin get killed by
        # AXIS_NOT_FOUND even though they form a valid configuration.
        guesses: list[np.ndarray] = []
        if initial_rz is not None:
            guesses.append(np.asarray(initial_rz, dtype=np.float64))
        cfg_guess = np.asarray(self.cfg.initial_rz, dtype=np.float64)
        if not any(np.allclose(g, cfg_guess) for g in guesses):
            guesses.append(cfg_guess)

        axis_result = None
        first_success = None
        for guess in guesses:
            try:
                with _silent():
                    axis_result = find_axis(
                        guess,
                        xtol=1e-5, max_iter=100, delta_r=0.01, verbose=False,
                    )
            except Exception:
                axis_result = (None, None, None, False)
            if axis_result[3]:
                if first_success is None:
                    first_success = axis_result
                if abs(axis_result[0][1]) <= self.cfg.axis_z_tol:
                    break
        if axis_result is None and first_success is not None:
            axis_result = first_success
        axis_rz, R0, axis_fieldline, success = axis_result
        if not success:
            info["failure_type"] = FailureType.AXIS_NOT_FOUND.value
            info["failure_message"] = "Magnetic axis not found"
            if not quiet:
                logger.warning("Gen %s, Ind %s: %s", gen, ind_idx,
                               info["failure_message"])
            return self.INVALID_FITNESS, info
        if abs(axis_rz[1]) > self.cfg.axis_z_tol:
            # Stellarator symmetry: the axis must lie on the Z=0 symmetry plane.
            # A large |Z_axis| means the configuration is degenerate / broken,
            # even though find_axis converged (early-warning validity check).
            info["failure_type"] = FailureType.AXIS_OFF_SYMMETRY.value
            info["failure_message"] = (f"Magnetic axis off symmetry plane "
                                       f"(Z={axis_rz[1]:.4f} > tol={self.cfg.axis_z_tol})")
            if not quiet:
                logger.warning("Gen %s, Ind %s: %s", gen, ind_idx,
                               info["failure_message"])
            return self.INVALID_FITNESS, info
        info["axis_rz"] = np.asarray(axis_rz, dtype=np.float64)

        # --- compute initial grad-psi ---
        RZ = np.array(
            [axis_rz[0] + self.cfg.delt_r, axis_rz[1]], dtype=np.float64, order="F"
        )
        initial_gradpsi = compute_initial_gradpsi_nemov(extcur, RZ[0], RZ[1], verbose=False)

        # --- trace field line ---
        set_trace_parameters(self.cfg.nturn, self.cfg.nphi, npart=self.cfg.npart, verbose=False)
        epstot_result = compute_epstot(
            [axis_rz[0] + self.cfg.delt_r, axis_rz[1]],
            initial_gradpsi=initial_gradpsi,
            return_fieldline=True,
            verbose=False,
        )
        epsilon_eff, bboundary, fieldline_data, trace_istate = epstot_result

        if trace_istate != 0 or epsilon_eff is None:
            info["failure_type"] = FailureType.TRACING_FAILED.value
            info["failure_message"] = f"Field-line tracing failed (istate={trace_istate})"
            if not quiet:
                logger.warning("Gen %s, Ind %s: %s", gen, ind_idx,
                               info["failure_message"])
            return self.INVALID_FITNESS, info

        if np.isnan(epsilon_eff) or np.isinf(epsilon_eff):
            info["failure_type"] = FailureType.EPSILON_NAN.value
            info["failure_message"] = f"epsilon_eff is {epsilon_eff}"
            if not quiet:
                logger.warning("Gen %s, Ind %s: %s", gen, ind_idx,
                               info["failure_message"])
            return self.INVALID_FITNESS, info

        # --- plasma parameters ---
        try:
            vol, minor_radius, iota = calculate_plasma_params(
                fieldline_data, axis_fieldline, self.cfg.nturn, self.cfg.nphi, R0
            )
            # Core-collapse guard: a nearly-zero minor radius means the surface
            # has collapsed into a thin tube — the epsilon_eff "improvement" is
            # numerical, not physical, and such configurations fail at full
            # resolution.  Reject them (checked before Aspect_ratio to avoid a
            # divide-by-zero on exactly zero radius).
            if minor_radius < self.cfg.min_minor_radius:
                info["failure_type"] = FailureType.CORE_COLLAPSED.value
                info["failure_message"] = (
                    f"minor radius {minor_radius:.6f} < "
                    f"min_minor_radius {self.cfg.min_minor_radius}")
                if not quiet:
                    logger.warning("Gen %s, Ind %d: %s", gen, ind_idx,
                                   info["failure_message"])
                return self.INVALID_FITNESS, info
            Aspect_ratio = R0 / minor_radius
        except Exception as exc:
            logger.warning("Gen %s, Ind %s: plasma param calc failed: %s", gen, ind_idx, exc)
            vol, minor_radius, iota = np.nan, np.nan, np.nan
        
        if not quiet:
            print(f"Gen {gen}, Ind {ind_idx}: Axis @ R = {axis_rz[0]:.3f}, "
                  f"epsilon_eff = {epsilon_eff:.3e}, iota = {iota:.3f} ,"
                  f"minor radius = {minor_radius:.3f}")

        info.update(
            epsilon_eff=float(epsilon_eff),
            iota=float(iota),
            volume=float(vol),
            failure_flag=False,
            failure_type=FailureType.NONE.value,
            failure_message="",
        )
        info["minor radius"] = float(minor_radius)
        info["Aspect ratio"] = float(Aspect_ratio)
        info["average B"] = float(
            bboundary[0] if hasattr(bboundary, "__len__") else bboundary
        )
        return float(epsilon_eff), info


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_hdf5(info: dict, fieldline_data: np.ndarray,
              output_dir: Path, device_name: Optional[str] = None,
              tag: Optional[str] = None):
    """Save one individual's results to HDF5."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gen = info.get('Generation', '?')
    ind = info.get('Individual', '?')

    if tag:
        parts = [tag]
    else:
        parts = [f"Gen{gen}", f"Ind{ind}"]
    if device_name:
        parts.insert(0, device_name)
    filename = output_dir / ("_".join(parts) + ".h5")

    base = filename
    suffix = 1
    while filename.exists():
        stem = base.stem
        filename = base.with_name(f"{stem}_{suffix}{base.suffix}")
        suffix += 1

    with h5py.File(filename, "w") as f:
        for key, value in info.items():
            if isinstance(value, np.ndarray):
                f.create_dataset(key, data=value)
            elif value is None:
                f.attrs[key] = np.nan
            elif isinstance(value, (list, tuple)):
                f.create_dataset(key, data=np.asarray(value))
            else:
                f.attrs[key] = value
        f.create_dataset("fieldline_data", data=fieldline_data)
    logger.debug("Saved %s", filename)


def _csv_value(value):
    if isinstance(value, np.ndarray):
        return np.array2string(value, separator=", ")
    if value is None:
        return ""
    return value


def save_csv(individual_infos: list[dict], filename: Path):
    """Save summary CSV of all evaluated individuals."""
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "Generation", "Individual", "extcur", "axis_R", "epsilon_eff",
        "iota", "volume", "Aspect ratio", "average B",
        "failure_flag", "failure_type", "failure_message",
    ]
    with open(filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for info in individual_infos:
            axis_rz = info.get("axis_rz")
            axis_r = float(axis_rz[0]) if isinstance(axis_rz, np.ndarray) \
                and axis_rz.size >= 1 else ""
            writer.writerow([
                info.get("Generation"),
                info.get("Individual"),
                _csv_value(info.get("extcur")),
                axis_r,
                info.get("epsilon_eff"),
                info.get("iota"),
                info.get("volume"),
                info.get("Aspect ratio"),
                info.get("average B"),
                info.get("failure_flag"),
                info.get("failure_type", ""),
                info.get("failure_message", ""),
            ])
    logger.info("CSV summary saved to %s", filename)


# ---------------------------------------------------------------------------
# Feasibility survey
# ---------------------------------------------------------------------------

def _n_workers(processes: int | None) -> int:
    """Resolve the worker count: explicit arg, else all available cores."""
    return int(processes) if processes and processes > 0 else multiprocessing.cpu_count()


_survey_cfg: OptimizationConfig | None = None
_survey_criterion: str = "axis"
_survey_short_nturn: int = 20
_survey_short_nphi: int = 72


def _survey_worker_init(config: OptimizationConfig, criterion: str = "axis",
                        short_nturn: int = 20, short_nphi: int = 72):
    """Initialise the survey worker (field once; needed for spawn platforms)."""
    global _survey_cfg, _survey_criterion, _survey_short_nturn, _survey_short_nphi
    _survey_cfg = config
    _survey_criterion = criterion
    _survey_short_nturn = int(short_nturn)
    _survey_short_nphi = int(short_nphi)
    global _mgrid_initialized
    if not _mgrid_initialized:
        with _silent():
            from .ripple import initialize_mgrid_field as _init_mgrid
            _init_mgrid(
                str(config.mgrid_path),
                nfp=config.nfp,
                full_torus=config.full_torus,
            )
            _mgrid_initialized = True


def _survey_point(point) -> bool:
    """Feasibility of one coil-current vector under the configured criterion.

    axis        : magnetic axis found AND on the Z=0 symmetry plane.
    short_trace : axis criterion + a short field-line trace from axis+delt_r.
    full        : the complete DE evaluation chain (find_axis + Z-symmetry +
                  full nturn trace + epsilon_eff checks).
    """
    cfg = _survey_cfg
    if _survey_criterion == "full":
        return _probe_point_full(cfg, point)
    if _survey_criterion == "short_trace":
        return _probe_point_short_trace(cfg, point,
                                        _survey_short_nturn, _survey_short_nphi)
    try:
        with _silent():
            set_extcur(np.asarray(point, dtype=np.float64))
            axis_result = find_axis(
                cfg.initial_rz, xtol=1e-6, max_iter=100,
                delta_r=0.01, verbose=False,
            )
        if not axis_result[3]:
            return False
        return abs(axis_result[0][1]) <= cfg.axis_z_tol
    except Exception:
        return False


def _probe_point_full(cfg, point, initial_rz=None) -> bool:
    """Feasibility probe used by the in-loop bounds adaptation.

    Runs the REAL evaluation chain (find_axis + Z-symmetry + nturn trace +
    NaN check) with logging suppressed, so a point marked feasible here is
    guaranteed to be usable by the DE — this is the criterion that was
    previously inconsistent with the DE evaluation (find_axis-only probes
    widened the box into regions where the nturn trace then failed, drowning
    the population in invalid trials).

    Runs inside the persistent DE process pool (workers already hold the mgrid
    field), so it must not create a new Pool.
    """
    try:
        fit, _ = StellaratorObjective(cfg).evaluate(
            np.asarray(point, dtype=np.float64), gen="probe", ind_idx=-1,
            initial_rz=initial_rz, quiet=True)
        return fit < StellaratorObjective.INVALID_FITNESS
    except Exception:
        return False


def _probe_point_short_trace(cfg, point, nturn: int = 20, nphi: int = 72
                             ) -> bool:
    """Cheap feasibility probe: find_axis + Z-symmetry + short trace.

    Traces only `nturn` x `nphi` points from axis+delt_r (instead of the full
    DE-resolution trace), which catches the dominant failure mode — the surface
    at the required DELT_R is missing / stochastic / escaping — at a fraction
    of the full evaluate cost.  Not identical to the DE criterion, but far
    tighter than axis-only.
    """
    try:
        with _silent():
            set_extcur(np.asarray(point, dtype=np.float64))
            axis_result = find_axis(
                cfg.initial_rz, xtol=1e-5, max_iter=100,
                delta_r=0.01, verbose=False,
            )
        axis_rz, _, _, success = axis_result
        if not success:
            return False
        if abs(axis_rz[1]) > cfg.axis_z_tol:
            return False
        with _silent():
            start_rz = np.array([axis_rz[0] + cfg.delt_r, axis_rz[1]],
                                dtype=np.float64, order="F")
            # Match the full DE chain: use the same non-zero initial grad-psi
            # (with zero grad-psi the v(3:5) equations stay identically zero,
            # which can make the short probe less strict than the DE ODE).
            gradpsi = compute_initial_gradpsi_nemov(
                np.asarray(point, dtype=np.float64),
                start_rz[0], start_rz[1], verbose=False)
            _, trace_istate = trace_fieldline(
                initial_rz=start_rz, initial_gradpsi=gradpsi,
                nturn=nturn, nphi=nphi, verbose=False)
        return trace_istate == 0
    except Exception:
        return False


def survey_feasibility(
    config: OptimizationConfig,
    n_samples: int = 256,
    seed: int = 0,
    expand_factor: float = 1.5,
    frac_max: float = 1.0,
    processes: int | None = None,
    criterion: str = "short_trace",
    short_nturn: int = 20,
    short_nphi: int = 72,
) -> dict:
    """Coarse sweep of the current search box to map feasible regions.

    A point is marked FEASIBLE according to `criterion`:
      - "axis"        : find_axis succeeds and the axis lies on Z=0 (fast,
                         but overestimates the box usable by the full DE chain);
      - "short_trace" : axis criterion + a short trace from axis+delt_r
                         (default; catches most collapsed/stochastic surfaces);
      - "full"        : the complete DE evaluation chain (find_axis +
                         Z-symmetry + full-resolution trace + epsilon checks).
                         Identical to the DE criterion, but expensive.

    Returns overall feasibility, per-coil statistics, a suggested tighter
    bounding box, and an ADAPTED set of initial_bounds (see below); the raw
    survey is written to <output_dir>/survey_points.csv.

    The per-point checks run IN PARALLEL over a process pool (processes
    workers, or all available cores by default).

    Adaptive-bounds policy (feed adjusted_bounds back as initial_bounds):
      - feasible fraction >= 0.9  → the box is mostly valid, so WIDEN it by
        expand_factor (a bigger search envelope = more population diversity);
      - feasible fraction <= 0.5  → the box is mostly dead, so SHRINK onto the
        feasible island (2nd-98th percentile of the feasible points);
      - otherwise                 → keep the current fractions.
    """
    from scipy.stats.qmc import Sobol

    if criterion not in ("axis", "short_trace", "full"):
        raise ValueError(
            f"Unknown feasibility criterion '{criterion}' — expected "
            "'axis', 'short_trace' or 'full'")

    # One-time field initialisation (same machinery as StellaratorObjective).
    global _mgrid_initialized
    if not _mgrid_initialized:
        with _silent():
            from .ripple import initialize_mgrid_field as _init_mgrid
            _init_mgrid(
                str(config.mgrid_path),
                nfp=config.nfp,
                full_torus=config.full_torus,
            )
            _mgrid_initialized = True

    lo = config._abs_bounds[:, 0]
    hi = config._abs_bounds[:, 1]
    n_dim = len(lo)

    u = Sobol(d=n_dim, scramble=True, seed=seed).random(n_samples)
    points = lo + u * (hi - lo)

    # Parallel feasibility check: one task per point.
    n_workers = _n_workers(processes)
    with Pool(processes=n_workers, initializer=_survey_worker_init,
              initargs=(config, criterion, short_nturn, short_nphi)) as pool:
        feasible = np.asarray(pool.map(_survey_point, points), dtype=bool)

    n_feasible = int(feasible.sum())

    # Suggested tighter bounds: [2nd, 98th] percentile of the feasible points,
    # clamped to the original box (percentiles guard against overfitting to
    # single outliers of the coarse sweep).
    suggested = np.empty_like(config._abs_bounds)
    for d in range(n_dim):
        vals = points[feasible, d]
        if n_feasible == 0:
            suggested[d] = config._abs_bounds[d]
        elif n_feasible == 1:
            suggested[d, 0] = suggested[d, 1] = float(vals[0])
        else:
            q = np.percentile(vals, [2.0, 98.0])
            suggested[d, 0] = max(lo[d], float(q[0]))
            suggested[d, 1] = min(hi[d], float(q[1]))

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "survey_points.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([f"coil_{d}" for d in range(n_dim)] + ["feasible"])
        for k in range(n_samples):
            writer.writerow(list(points[k]) + [int(feasible[k])])

    feasible_rate = n_feasible / n_samples

    # ── Adaptive bounds: feed back as new initial_bounds ──────────────────
    raw = np.asarray(config.initial_bounds, dtype=np.float64)
    if raw.ndim == 1:
        nominal_vals = raw
        old_fracs = np.full(n_dim, config.bounds_fraction)
    else:
        nominal_vals = raw[:, 0]
        old_fracs = raw[:, 1]
    new_fracs = old_fracs.copy()
    if feasible_rate >= 0.9:
        # Mostly feasible → widen the box to increase population diversity.
        new_fracs = np.minimum(frac_max, old_fracs * expand_factor)
        action = f"EXPAND x{expand_factor}"
    elif feasible_rate <= 0.5:
        # Mostly infeasible → shrink onto the feasible island.
        for d in range(n_dim):
            if abs(nominal_vals[d]) < 1e-12 or old_fracs[d] <= 0:
                continue   # keep nominal=0 and locked coils fixed
            vals = points[feasible, d]
            if n_feasible == 0:
                continue
            elif n_feasible == 1:
                new_fracs[d] = 0.01
            else:
                q = np.percentile(vals, [2.0, 98.0])
                need = (max(abs(q[1] - nominal_vals[d]),
                            abs(nominal_vals[d] - q[0]))
                        / max(abs(nominal_vals[d]), 1e-12))
                new_fracs[d] = min(frac_max, need * 1.05)
        action = "SHRINK to feasible island"
    else:
        action = "KEEP current fractions"
    adjusted_bounds = np.column_stack([nominal_vals, new_fracs])

    criterion_label = {"axis": "axis found",
                      "short_trace": "axis + short trace",
                      "full": "full DE chain"}[criterion]
    print(f"\n=== Feasibility survey ({n_samples} Sobol points) ===")
    print(f"  feasible ({criterion_label}): {n_feasible}/{n_samples} "
          f"({feasible_rate * 100:.1f}%)")
    for d in range(n_dim):
        print(f"  coil {d}: original [{lo[d]:10.1f}, {hi[d]:10.1f}]  "
              f"-> suggested [{suggested[d, 0]:10.1f}, {suggested[d, 1]:10.1f}]")
    print(f"  survey CSV -> {csv_path}")
    if feasible_rate < 0.5:
        print("  WARNING: less than half the box is feasible - narrow the "
              "bounds or move the search box before optimising.")
    print(f"  adaptive bounds: {action}")
    for d in range(n_dim):
        print(f"    coil {d}: fraction {old_fracs[d]:.3f} -> {new_fracs[d]:.3f}  "
              f"({nominal_vals[d]:10.1f} +/- {abs(nominal_vals[d]) * new_fracs[d]:10.1f})")

    return {
        "feasible_fraction": float(feasible_rate),
        "n_feasible": n_feasible,
        "n_samples": n_samples,
        "suggested_bounds": suggested,
        "adjusted_bounds": adjusted_bounds,
    }


def explore_feasible_region(
    config: OptimizationConfig,
    n_samples: int = 128,
    seed: int = 0,
    expand_factor: float = 1.5,
    frac_max: float = 1.0,
    processes: int | None = None,
    max_rounds: int = 6,
    criterion: str = "short_trace",
    short_nturn: int = 20,
    short_nphi: int = 72,
) -> dict:
    """Adaptively explore the feasible-region extent with Sobol + probe.

    Surveys the search box, widens it while it stays almost fully feasible
    (>= 90%), and falls back to the last fully-feasible box once feasibility
    drops — bracketing the boundary of the region that passes `criterion`
    (default "short_trace": axis + a short trace from axis+delt_r, so the
    DE is not handed a box full of points that only pass find_axis but then
    fail the real trace).  When the feasible fraction becomes small, the
    per-round sample count is doubled automatically: what matters for the
    boundary estimate is the number of points landing INSIDE the feasible
    island (low-discrepancy coverage itself depends on the dimension, not on
    the box volume).  A box with 50-90% feasibility is kept as-is instead of
    being shrunk back, so the DE can explore the low-density edge region.

    Each round's survey runs in parallel over a process pool
    (processes workers, or all available cores by default).

    Returns:
      - extent_bounds : [nominal, fraction] box covering the explored feasible
                        region — feed back as initial_bounds for the real run.
      - core_bounds   : [nominal, fraction] inner box (2nd-98th percentile of
                        the feasible points) where results should be reliable.
      - rounds        : list of (round, feasible_fraction).

    Edge regions of extent_bounds may give poorer epsilon_eff; that is expected
    — the optimiser (JADE) will simply prefer the better interior solutions.
    """
    raw = np.asarray(config.initial_bounds, dtype=np.float64)
    if raw.ndim == 1:
        nominal_vals = raw
        fracs = np.full(len(raw), config.bounds_fraction)
    else:
        nominal_vals = raw[:, 0]
        fracs = raw[:, 1].copy()
    n_dim = len(nominal_vals)
    n_workers = _n_workers(processes)

    rounds: list[tuple[int, float]] = []
    n_cur = n_samples
    core_abs = None
    prev_fracs = fracs.copy()
    doubled = False
    for rnd in range(max_rounds):
        sub = OptimizationConfig(
            mgrid_path=config.mgrid_path,
            nfp=config.nfp,
            full_torus=config.full_torus,
            initial_rz=config.initial_rz,
            initial_bounds=np.column_stack([nominal_vals, fracs]),
            delt_r=config.delt_r,
            processes=n_workers,
            output_dir=config.output_dir,
        )
        res = survey_feasibility(sub, n_samples=n_cur, seed=seed + rnd,
                                 criterion=criterion, short_nturn=short_nturn,
                                 short_nphi=short_nphi)
        rate = float(res["feasible_fraction"])
        # core = feasible-island of the last fully-feasible (>= 90%) round,
        # so that core_bounds always lies INSIDE extent_bounds.
        if rate >= 0.9 or core_abs is None:
            core_abs = res["suggested_bounds"]
        rounds.append((rnd + 1, rate))
        print(f"    round {rnd + 1}: feasible = {rate * 100:.1f}%  "
              f"fracs = {np.round(fracs, 3).tolist()}  (n={n_cur})")

        if rate >= 0.9:
            prev_fracs = fracs.copy()
            fracs = np.minimum(frac_max, fracs * expand_factor)
            if np.allclose(fracs, prev_fracs):
                break          # already at frac_max - cannot widen further
        elif rate < 0.3 and not doubled and n_cur < 1024:
            # Feasible island is small: sharpen the boundary with more points.
            n_cur *= 2
            doubled = True
            print(f"    -> doubling points to {n_cur} for a sharper boundary")
            continue
        elif rate >= 0.5:
            # 50-90% feasible: the edge is low-density but far from dead.
            # Keep the current box and let the DE handle the edge region;
            # reverting to the previous (smaller) box would hide the very
            # boundary region where the best solutions often lie.
            print("    -> keeping current box: feasible in [50%, 90%) — "
                  "let DE handle the low-density edge")
            break
        else:
            fracs = prev_fracs  # boundary bracketed: fall back to last good box
            break

    extent_bounds = np.column_stack([nominal_vals, fracs])

    # Inner "core" box: fraction needed to cover the [q2, q98] feasible extent.
    core_fracs = np.zeros(n_dim)
    for d in range(n_dim):
        if abs(nominal_vals[d]) < 1e-12:
            continue
        need = (max(abs(core_abs[d, 1] - nominal_vals[d]),
                    abs(nominal_vals[d] - core_abs[d, 0]))
                / abs(nominal_vals[d]))
        core_fracs[d] = min(frac_max, max(need, 1e-3))
    core_bounds = np.column_stack([nominal_vals, core_fracs])

    print(f"  explored extent: {np.round(extent_bounds[:, 1], 3).tolist()}  "
          f"(nominal {nominal_vals.tolist()})")
    print(f"  core (inner)   : {np.round(core_bounds[:, 1], 3).tolist()}")
    return {
        "extent_bounds": extent_bounds,
        "core_bounds": core_bounds,
        "rounds": rounds,
    }


# ---------------------------------------------------------------------------
# DE core
# ---------------------------------------------------------------------------

class DifferentialEvolution:
    """Differential Evolution optimiser for stellarator coil currents."""

    def __init__(self, config: OptimizationConfig):
        self.cfg = config
        _setup_logging(config.log_file, config.log_level)

        self.n_dim = len(config.initial_bounds)
        self.objective = StellaratorObjective(config)

        # Population state
        self.pop: list[np.ndarray] = []
        self.fitnesses: list[float] = []
        self.invalid_count: list[int] = []

        # History
        self.all_infos: list[dict] = []

        # Elitism: archive of the best-known solution (never lost by re-init).
        self._best_ever_ind: np.ndarray | None = None
        self._best_ever_fit: float = float("inf")

        # Per-individual magnetic axis of the last successful evaluation,
        # used to warm-start find_axis for the next generation.
        self._axes: list[np.ndarray | None] = []

        # Convergence tracking
        self._best_fitness_history: list[float] = []
        self._patience_counter = 0

        # Last probed best (bounds adaptation skips re-probing an unchanged best)
        self._last_probe_best: np.ndarray | None = None

        # Persistent process pool (created once in run(), reused across gens)
        self._pool: Pool | None = None

        # Random seed
        if config.seed is not None:
            random.seed(config.seed)
            np.random.seed(config.seed)

        # JADE adaptive parameters (initialised from F / CR) + external archive
        self.mu_F: float = float(config.F)
        self.mu_CR: float = float(config.CR)
        self.archive: list[np.ndarray] = []
        self._archive_max: int = (config.archive_size if config.archive_size > 0
                                  else config.n_pop)

    # ---- public entry point ------------------------------------------------

    def run(self) -> tuple[np.ndarray, float, list[dict]]:
        """Execute the full DE optimisation.

        Returns
        -------
        best_individual : ndarray
        best_fitness : float
        all_infos : list[dict]
        """
        # Create the process pool once and reuse across all generations.
        # On macOS spawn this is critical — every Pool() spawns N new
        # processes, which is extremely expensive if done per generation.
        self._pool = Pool(processes=self.cfg.processes)
        try:
            return self._run_with_pool()
        finally:
            self._pool.terminate()
            self._pool = None

    def _run_with_pool(self) -> tuple[np.ndarray, float, list[dict]]:
        """Inner optimisation loop (pool is guaranteed to exist)."""
        logger.info("Starting fresh optimisation")

        # ── Evaluate the nominal extcur as a "start" baseline ──────────
        nominal_extcur = self.cfg.initial_bounds[:, 0].copy()
        logger.info(
            "Evaluating nominal extcur baseline: %s", nominal_extcur)
        start_fit, start_info = self.objective.evaluate(
            nominal_extcur, gen="start", ind_idx=0)
        self.all_infos.append(start_info)
        if start_fit >= StellaratorObjective.INVALID_FITNESS:
            logger.warning(
                "Nominal extcur evaluation FAILED (fitness=%.1f).  "
                "Optimisation will proceed but the baseline is invalid.",
                start_fit)
        else:
            logger.info("Nominal baseline ε = %.6e", start_fit)
            self._best_ever_fit = start_fit
            self._best_ever_ind = nominal_extcur.copy()
            # Scale the core-collapse guard to the nominal configuration:
            # reject surfaces that shrink below min_minor_radius_frac of the
            # nominal minor radius, never below the absolute min_minor_radius.
            nom_minor = start_info.get("minor radius")
            if nom_minor is not None and np.isfinite(nom_minor):
                scaled = self.cfg.min_minor_radius_frac * float(nom_minor)
                if scaled > self.cfg.min_minor_radius:
                    self.cfg.min_minor_radius = scaled
                    logger.info(
                        "min_minor_radius set to %.4f m (%.0f%% of nominal "
                        "minor radius %.4f m)",
                        self.cfg.min_minor_radius,
                        self.cfg.min_minor_radius_frac * 100, nom_minor)

        self._init_population()
        self._evaluate_and_record(self.pop, gen=0)

        # Seed the elitism archive with the best of the initial population.
        gen0_best = int(np.argmin(self.fitnesses))
        if self.fitnesses[gen0_best] < self._best_ever_fit:
            self._best_ever_fit = self.fitnesses[gen0_best]
            self._best_ever_ind = self.pop[gen0_best].copy()

        # ---- main loop ----
        for gen in range(1, self.cfg.max_gen + 1):
            t0 = time.perf_counter()

            # 1. Generate trial vectors (JADE: adaptive F/CR + pbest mutation)
            F_vals = None
            CR_vals = None
            if self.cfg.strategy == "jade":
                trials, F_vals, CR_vals = self._generate_trials_jade(self.pop)
            else:
                trials = []
                for i in range(self.cfg.n_pop):
                    mutant = self._mutate(self.pop[i], self.pop)
                    trials.append(self._crossover(self.pop[i], mutant))

            # 2. Evaluate (each trial warm-starts from its own parent's axis,
            # NOT from the previous generation's best axis — the latter locks
            # the search to one magnetic-axis basin).
            trial_fitnesses, trial_infos = self._evaluate_batch(
                trials, gen, initial_rz_list=self._axes)

            # 3. Selection (elitism: the current best is never re-initialised)
            invalid_solutions = 0
            accepted: list[int] = []
            reinit_indices: list[int] = []
            reinit_individuals: list[np.ndarray] = []
            reinit_axes: list[np.ndarray | None] = []
            best_idx = int(np.argmin(self.fitnesses))
            for i in range(self.cfg.n_pop):
                tf = trial_fitnesses[i]
                cf = self.fitnesses[i]

                if tf >= StellaratorObjective.INVALID_FITNESS:
                    self.invalid_count[i] += 1
                else:
                    self.invalid_count[i] = 0

                if self.invalid_count[i] >= 3 and i != best_idx:
                    # Re-initialise this individual (never the current best —
                    # its trials failing says nothing about its own quality).
                    # Collected here and evaluated as one parallel batch below.
                    new_ind, base_idx = self._reinit_individual()
                    reinit_indices.append(i)
                    reinit_individuals.append(new_ind)
                    reinit_axes.append(self._axes[base_idx]
                                       if base_idx >= 0 else None)
                elif self.invalid_count[i] >= 3:
                    # Protected best: keep it, reset its failure counter.
                    self.invalid_count[i] = 0
                elif tf < StellaratorObjective.INVALID_FITNESS and tf <= cf:
                    # Accepted: the discarded parent goes into the JADE archive.
                    if self.cfg.strategy == "jade":
                        self.archive.append(self.pop[i])
                    self.pop[i] = trials[i]
                    self.fitnesses[i] = tf
                    self.all_infos.append(trial_infos[i])
                    self._axes[i] = trial_infos[i].get("axis_rz")
                    accepted.append(i)

                if tf >= StellaratorObjective.INVALID_FITNESS:
                    invalid_solutions += 1

            # 3b. Evaluate all re-initialised individuals in one pool batch
            # (previously each re-init ran serially in the main process, which
            # made high-invalid generations take minutes).
            if reinit_indices:
                reinit_fits, reinit_infos = self._evaluate_batch(
                    reinit_individuals, gen, indices=reinit_indices,
                    initial_rz_list=reinit_axes)
                for i, new_ind, new_fit, new_info in zip(
                        reinit_indices, reinit_individuals,
                        reinit_fits, reinit_infos):
                    self.pop[i] = new_ind
                    self.fitnesses[i] = new_fit
                    self.all_infos.append(new_info)
                    self._axes[i] = new_info.get("axis_rz")
                    self.invalid_count[i] = 0
                    logger.info(
                        "Gen %d, Ind %d: re-initialised (fitness=%.6e)",
                        gen, i, new_fit,
                    )

            # JADE: adapt mu_F / mu_CR from the successful trials.
            if self.cfg.strategy == "jade":
                self._update_jade(F_vals, CR_vals, accepted)

            # 4. Update best / elitism archive / axis warm-start, then log
            t_elapsed = time.perf_counter() - t0
            best_idx = int(np.argmin(self.fitnesses))
            best_fit = self.fitnesses[best_idx]
            if best_fit < self._best_ever_fit:
                self._best_ever_fit = best_fit
                self._best_ever_ind = self.pop[best_idx].copy()
            self._best_fitness_history.append(self._best_ever_fit)

            # No global axis warm-start here: cfg.initial_rz stays at the
            # nominal guess for all evaluations.  Per-individual warm-start is
            # handled in step 2 via each trial's own parent axis.

            pct_invalid = invalid_solutions / self.cfg.n_pop * 100
            best_extcur = self.pop[best_idx]
            best_axis = self._axes[best_idx]
            axis_r_str = f"{best_axis[0]:.4f}" if best_axis is not None else "n/a"
            logger.info(
                "Gen %3d/%d  |  best ε=%.6e  |  best-ever ε=%.6e  |  "
                "invalid=%3d/%d (%.1f%%)  |  time=%6.2fs  |  "
                "best ind=%d | axis R=%s | extcur=%s",
                gen, self.cfg.max_gen, best_fit, self._best_ever_fit,
                invalid_solutions, self.cfg.n_pop, pct_invalid,
                t_elapsed, best_idx, axis_r_str,
                np.array2string(best_extcur, separator=', ',
                                formatter={'float_kind': lambda x: '%.1f' % x}),
            )

            # 4.5 Adaptive bounds: probe + widen if the best presses a box edge.
            if gen % self.cfg.adapt_bounds_every == 0:
                self._adapt_bounds_if_pressed(
                    gen, invalid_ratio=invalid_solutions / self.cfg.n_pop)

            # 5. Early stop via ftol
            if self._check_convergence(gen):
                logger.info(
                    "Convergence reached: fitness change < %.1e for %d generations",
                    self.cfg.ftol, self.cfg.patience,
                )
                break

        # ---- finalise ----
        if self._best_ever_ind is not None:
            best_ind = self._best_ever_ind
            best_fitness = self._best_ever_fit
            logger.info(
                "Optimisation finished. Best (historical) fitness = %.6e "
                "— archived, immune to re-initialisation.",
                best_fitness,
            )
        else:
            # No valid solution was ever found; fall back to current argmin.
            best_idx = int(np.argmin(self.fitnesses))
            best_ind = self.pop[best_idx]
            best_fitness = self.fitnesses[best_idx]
            logger.info(
                "Optimisation finished. Best fitness = %.6e at index %d",
                best_fitness, best_idx,
            )

        save_csv(self.all_infos, self.cfg.output_dir / self.cfg.csv_filename)
        return best_ind, best_fitness, self.all_infos

    # ---- population init ---------------------------------------------------

    def _init_individual(self) -> np.ndarray:
        return np.array([
            random.uniform(self.cfg._abs_bounds[i, 0],
                           self.cfg._abs_bounds[i, 1])
            for i in range(self.n_dim)
        ], dtype=np.float64)

    def _reinit_individual(self) -> tuple[np.ndarray, int]:
        """Re-initialise near a random FEASIBLE population member.

        Each coil is perturbed by +- reinit_perturb_frac of the current box
        width (default 0.1) and clamped back into the box.  Whole-box random
        re-initialisation wastes evaluations once the feasible region is a
        small fraction of the box, and re-seeding far from any known-feasible
        point is what collapses the population into a few clones.  Falls back
        to whole-box sampling when no feasible member exists yet.

        Returns (new_individual, base_index); base_index is -1 for the
        whole-box fallback so the caller can choose the axis warm-start.
        """
        feasible = [k for k, f in enumerate(self.fitnesses)
                    if f < StellaratorObjective.INVALID_FITNESS]
        if not feasible:
            return self._init_individual(), -1
        base_idx = random.choice(feasible)
        base = self.pop[base_idx]
        lo = self.cfg._abs_bounds[:, 0]
        hi = self.cfg._abs_bounds[:, 1]
        pert = self.cfg.reinit_perturb_frac * (hi - lo)
        new = base + np.array(
            [random.uniform(-p, p) for p in pert], dtype=np.float64)
        return np.clip(new, lo, hi), base_idx

    def _init_population(self):
        self.pop = [self._init_individual() for _ in range(self.cfg.n_pop)]
        self.fitnesses = []
        self.invalid_count = [0] * self.cfg.n_pop
        self._axes = [None] * self.cfg.n_pop

    # ---- DE operators ------------------------------------------------------

    def _mutate(self, individual: np.ndarray, population: list[np.ndarray]
                ) -> np.ndarray:
        """DE/rand/1 mutation with bounds clamping."""
        size = len(individual)
        idxs = [i for i in range(len(population))
                if not np.array_equal(population[i], individual)]
        r1, r2, r3 = random.sample(idxs, 3)
        mutant = np.empty(size, dtype=np.float64)
        for i in range(size):
            mutant[i] = (population[r1][i]
                         + self.cfg.F * (population[r2][i] - population[r3][i]))
            lo, hi = self.cfg._abs_bounds[i]
            mutant[i] = max(lo, min(hi, mutant[i]))
        return mutant

    def _crossover(self, individual: np.ndarray, mutant: np.ndarray,
                   CR: float | None = None) -> np.ndarray:
        """Binomial crossover (per-trial CR for JADE, cfg.CR otherwise)."""
        cr = self.cfg.CR if CR is None else CR
        size = len(individual)
        trial = individual.copy()
        j_rand = random.randint(0, size - 1)
        for i in range(size):
            if random.random() < cr or i == j_rand:
                trial[i] = mutant[i]
        return trial

    # ---- JADE: adaptive DE/current-to-pbest/1 ------------------------------

    def _sample_F(self) -> float:
        """Sample F from a Cauchy distribution around mu_F (ensure F > 0)."""
        while True:
            F = self.mu_F + 0.1 * math.tan(math.pi * (random.random() - 0.5))
            if F > 0.0:
                return F

    def _sample_CR(self) -> float:
        """Sample CR from a Normal distribution around mu_CR, clamped to [0, 1]."""
        return min(1.0, max(0.0, random.gauss(self.mu_CR, 0.1)))

    def _pick_jade_partners(self, i: int, population: list[np.ndarray],
                            pool: list[np.ndarray]) -> tuple[int, int]:
        """Distinct partner indices for JADE mutation.

        r1 is a population index, r2 an index into pop ∪ archive.  Both are
        chosen to be value-DIFFERENT from the target individual and from each
        other — clones (identical coil vectors) are never used as mutation
        partners, otherwise the differential x_r1 − x_r2 vanishes and the
        population stagnates on a single point.
        """
        n_pop = len(population)
        x_i = population[i]

        pop_cands = [k for k in range(n_pop)
                     if not np.array_equal(population[k], x_i)]
        if not pop_cands:                       # fully collapsed — safety fallback
            pop_cands = list(range(n_pop))
        r1 = random.choice(pop_cands)
        x_r1 = population[r1]

        n_pool = len(pool)
        pool_cands = [k for k in range(n_pool)
                      if not np.array_equal(pool[k], x_i)]
        if not pool_cands:
            pool_cands = list(range(n_pool))
        r2_cands = [k for k in pool_cands
                    if not np.array_equal(pool[k], x_r1)]
        if not r2_cands:
            r2_cands = list(pool_cands)
        r2 = random.choice(r2_cands)
        return r1, r2

    def _generate_trials_jade(self, population: list[np.ndarray]
                              ) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
        """Generate trials with DE/current-to-pbest/1 + per-individual F, CR."""
        n = len(population)
        order = np.argsort(self.fitnesses)
        n_pbest = max(1, int(round(self.cfg.p_best * n)))
        pbest_pool = order[:n_pbest].tolist()
        pool = population + self.archive

        trials: list[np.ndarray] = []
        F_vals = np.empty(n, dtype=np.float64)
        CR_vals = np.empty(n, dtype=np.float64)
        for i in range(n):
            F = self._sample_F()
            CR = self._sample_CR()
            F_vals[i], CR_vals[i] = F, CR
            x_i = population[i]
            x_pbest = population[random.choice(pbest_pool)]
            r1, r2 = self._pick_jade_partners(i, population, pool)
            mutant = np.empty_like(x_i)
            for d in range(len(x_i)):
                mutant[d] = (x_i[d]
                             + F * (x_pbest[d] - x_i[d])
                             + F * (population[r1][d] - pool[r2][d]))
                lo, hi = self.cfg._abs_bounds[d]
                mutant[d] = max(lo, min(hi, mutant[d]))
            trials.append(self._crossover(x_i, mutant, CR))
        return trials, F_vals, CR_vals

    def _update_jade(self, F_vals, CR_vals, accepted: list[int]):
        """Adapt mu_F / mu_CR from successful trials; trim the archive."""
        if not accepted:
            return
        S_F = F_vals[accepted]
        S_CR = CR_vals[accepted]
        sum_F = float(S_F.sum())
        if sum_F > 0.0:
            self.mu_F = ((1.0 - self.cfg.adapt_rate) * self.mu_F
                         + self.cfg.adapt_rate * float(S_F @ S_F) / sum_F)
        self.mu_CR = ((1.0 - self.cfg.adapt_rate) * self.mu_CR
                      + self.cfg.adapt_rate * float(S_CR.mean()))
        while len(self.archive) > self._archive_max:
            self.archive.pop(random.randrange(len(self.archive)))

    # ---- evaluation --------------------------------------------------------

    def _evaluate_batch(self, population: list[np.ndarray], gen: int,
                        indices: list[int] | None = None,
                        initial_rz_list: list[np.ndarray | None] | None = None
                        ) -> tuple[list[float], list[dict]]:
        """Evaluate a population using the persistent process pool.

        `indices` optionally supplies the individual indices to pass to the
        evaluator (used when re-initialising a subset of the population so the
        per-evaluation logs still carry the correct individual number).
        `initial_rz_list` optionally supplies one axis warm-start guess per
        evaluated point (the parent's recorded axis for DE trials); the
        evaluator falls back to the config-level nominal guess.
        """
        if indices is None:
            indices = list(range(len(population)))
        evaluate_func = self.objective.evaluate
        if initial_rz_list is None:
            args = [(ind, gen, i) for i, ind in zip(indices, population)]
        else:
            args = [(ind, gen, i, irz)
                    for i, ind, irz in zip(indices, population,
                                           initial_rz_list)]
        results = self._pool.starmap(evaluate_func, args)

        fitnesses, infos = [], []
        for fit, info in results:
            fitnesses.append(float(fit))
            infos.append(info)
        return fitnesses, infos

    def _evaluate_and_record(self, population: list[np.ndarray], gen: int):
        """Evaluate a population and record fitnesses + metadata."""
        fitnesses, infos = self._evaluate_batch(population, gen)
        for k, (fit, info) in enumerate(zip(fitnesses, infos)):
            self.fitnesses.append(float(fit))
            self.all_infos.append(info)
            self._axes[k] = info.get("axis_rz")

    # ---- adaptive bounds ---------------------------------------------------

    def _coil_pressed(self, best: np.ndarray) -> list[int]:
        """Indices of coils whose best value sits near a box edge."""
        lo = self.cfg._abs_bounds[:, 0]
        hi = self.cfg._abs_bounds[:, 1]
        span = hi - lo
        margin = self.cfg.adapt_bounds_margin * span
        pressed = []
        for d in range(len(best)):
            if span[d] <= 1e-12:               # locked coil — never "pressed"
                continue
            if best[d] <= lo[d] + margin[d] or best[d] >= hi[d] - margin[d]:
                pressed.append(d)
        return pressed

    def _adapt_bounds_if_pressed(self, gen: int,
                                 invalid_ratio: float | None = None) -> bool:
        """If the current best presses a box edge, scan along the pressed
        direction(s) in the DE pool and widen the bound to the deepest point
        that still passes the FULL evaluation criterion (identical to the DE
        fitness: find_axis + Z-symmetry + nturn trace + NaN check).

        Returns True if the bounds changed.  The probe is skipped while the
        best has not moved, so a wall is not re-surveyed every generation.

        Hard constraint: coil currents must never change sign.  Widened bounds
        are clamped to [0, +inf) for positive-nominal coils and to (-inf, 0]
        for negative-nominal coils.
        """
        if not self.cfg.adapt_bounds or self._pool is None:
            return False

        # Do not widen while the current generation is already drowning in
        # invalid trials — the new region would only add more failures.
        # Use the just-evaluated generation's invalid-trial ratio (not the
        # stored population fitnesses, which mostly hold the last valid parent
        # and therefore under-report how bad the current box edge is).
        if invalid_ratio is None:
            invalid_ratio = np.mean(
                [f >= StellaratorObjective.INVALID_FITNESS
                 for f in self.fitnesses])
        if invalid_ratio > self.cfg.adapt_bounds_max_invalid:
            logger.info(
                "Gen %d: best at bound but invalid=%.0f%% > %.0f%% — "
                "skipping bounds probe", gen, invalid_ratio * 100,
                self.cfg.adapt_bounds_max_invalid * 100)
            return False

        best_idx = int(np.argmin(self.fitnesses))
        best = self.pop[best_idx]
        pressed = self._coil_pressed(best)
        if not pressed:
            self._last_probe_best = None
            return False
        if (self._last_probe_best is not None
                and np.allclose(self._last_probe_best, best)):
            return False
        self._last_probe_best = best.copy()

        lo = self.cfg._abs_bounds[:, 0]
        hi = self.cfg._abs_bounds[:, 1]
        K = self.cfg.adapt_bounds_n_samples
        nominal = self.cfg.initial_bounds[:, 0]

        # Outward direction for each pressed coil: +1 = beyond hi, -1 = beyond lo.
        dirs = [1 if best[d] >= hi[d] - 1e-12 else -1 for d in pressed]

        if len(pressed) == 1:
            # Single pressed coil: deterministic 1-D scan from the box edge
            # outward to edge +- expand x range; other coils locked at best.
            d = pressed[0]
            span = hi[d] - lo[d]
            if dirs[0] > 0:
                xs = np.linspace(
                    hi[d], hi[d] + self.cfg.adapt_bounds_expand * span, K)
            else:
                xs = np.linspace(
                    lo[d], lo[d] - self.cfg.adapt_bounds_expand * span, K)
            points = np.tile(best, (K, 1)).astype(np.float64, copy=True)
            points[:, d] = xs
        else:
            # Multiple pressed coils: 1-D scans are blind to diagonal feasible
            # extensions (e.g. coil0 must go up while coil3 goes down
            # simultaneously).  Sample the outward hyper-rectangle of all
            # pressed coils with a scrambled Sobol set and widen to the
            # feasible point deepest in its weakest outward direction.
            n_samples = max(K, K * len(pressed))
            from scipy.stats.qmc import Sobol
            u = Sobol(d=len(pressed), scramble=True,
                      seed=(self.cfg.seed or 0) + gen).random(n_samples)
            points = np.tile(best, (n_samples, 1)).astype(np.float64, copy=True)
            for k, d in enumerate(pressed):
                span = hi[d] - lo[d]
                if dirs[k] > 0:
                    points[:, d] = (hi[d]
                                    + u[:, k] * self.cfg.adapt_bounds_expand * span)
                else:
                    points[:, d] = (lo[d]
                                    - u[:, k] * self.cfg.adapt_bounds_expand * span)

        from functools import partial
        best_axis = self._axes[best_idx]
        try:
            feasible = np.asarray(
                self._pool.map(partial(_probe_point_full, self.cfg,
                                       initial_rz=best_axis), points),
                dtype=bool)
        except Exception as exc:
            logger.warning("Gen %d: bounds probe failed: %s", gen, exc)
            return False
        if not feasible.any():
            logger.info("Gen %d: best at bound on coils %s, probe found no "
                        "feasible extension — keeping bounds", gen, pressed)
            return False

        # Widen the pressed bounds, clamped so the coil current never changes
        # sign.  For a single pressed coil this is the deepest feasible 1-D
        # point; for multiple pressed coils it is the feasible Sobol point
        # deepest in its weakest outward direction (the "diagonal" case).
        changed = False
        if len(pressed) == 1:
            d = pressed[0]
            vals = points[:, d]
            if dirs[0] > 0:
                new_hi = float(vals[feasible].max())
                if nominal[d] < 0.0:
                    new_hi = min(new_hi, 0.0)   # never flip sign
                if new_hi > hi[d]:
                    self.cfg._abs_bounds[d, 1] = new_hi
                    changed = True
            else:
                new_lo = float(vals[feasible].min())
                if nominal[d] > 0.0:
                    new_lo = max(new_lo, 0.0)   # never flip sign
                if new_lo < lo[d]:
                    self.cfg._abs_bounds[d, 0] = new_lo
                    changed = True
        else:
            spans = np.array([hi[d] - lo[d] for d in pressed], dtype=np.float64)
            feasible_points = points[feasible]
            out = np.empty((len(feasible_points), len(pressed)), dtype=np.float64)
            for k, d in enumerate(pressed):
                denom = max(self.cfg.adapt_bounds_expand * spans[k], 1e-12)
                if dirs[k] > 0:
                    out[:, k] = (feasible_points[:, d] - hi[d]) / denom
                else:
                    out[:, k] = (lo[d] - feasible_points[:, d]) / denom
            combo_idx = int(np.argmax(np.min(out, axis=1)))
            combo = feasible_points[combo_idx]
            for k, d in enumerate(pressed):
                val = float(combo[d])
                if dirs[k] > 0:
                    if nominal[d] < 0.0:
                        val = min(val, 0.0)     # never flip sign
                    if val > hi[d]:
                        self.cfg._abs_bounds[d, 1] = val
                        changed = True
                else:
                    if nominal[d] > 0.0:
                        val = max(val, 0.0)     # never flip sign
                    if val < lo[d]:
                        self.cfg._abs_bounds[d, 0] = val
                        changed = True
        if changed:
            logger.info("Gen %d: best at bound on coils %s — widened bounds "
                        "after feasibility probe: %s", gen, pressed,
                        np.round(self.cfg._abs_bounds, 1).tolist())
        else:
            logger.info("Gen %d: best at bound on coils %s, probe found no "
                        "extension beyond the edge — keeping bounds", gen, pressed)
        return changed

    # ---- convergence -------------------------------------------------------

    def _check_convergence(self, gen: int) -> bool:
        """Return True if ftol-based early-stop should trigger."""
        if self.cfg.ftol is None or len(self._best_fitness_history) < 2:
            return False

        # Check if best fitness has stabilised (relative or absolute tolerance)
        window = min(self.cfg.patience, len(self._best_fitness_history))
        recent = self._best_fitness_history[-window:]
        delta = abs(recent[-1] - recent[0])
        if self.cfg.ftol_relative:
            scale = max(abs(recent[-1]), abs(recent[0]), 1e-30)
            improvement = delta / scale
        else:
            improvement = delta
        if improvement < self.cfg.ftol:
            self._patience_counter += 1
        else:
            self._patience_counter = 0

        return self._patience_counter >= self.cfg.patience


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

def run(config: OptimizationConfig) -> tuple[np.ndarray, float, list[dict]]:
    """Convenience wrapper around DifferentialEvolution."""
    optimizer = DifferentialEvolution(config)
    return optimizer.run()

