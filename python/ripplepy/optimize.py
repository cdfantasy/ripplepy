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
    TRACING_FAILED = "tracing_failed"
    EPSILON_NAN = "epsilon_nan"
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
        Crossover probability.
    delt_r : float
        Radial offset from axis for starting field lines.
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
    patience : int
        Generations to wait after ftol triggers before stopping.
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
    delt_r: float = 0.05
    processes: int = 8
    output_dir: Path = field(default_factory=lambda: Path("."))
    csv_filename: str = "Individual_info_list.csv"
    device_name: Optional[str] = None
    seed: Optional[int] = None
    log_file: Optional[Path] = None
    log_level: int = logging.INFO
    ftol: Optional[float] = None
    patience: int = 10

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        if self.log_file is not None:
            self.log_file = Path(self.log_file)
        for arr_name in ("initial_rz", "initial_bounds"):
            val = getattr(self, arr_name)
            if val is not None:
                setattr(self, arr_name, np.asarray(val, dtype=np.float64))

        # ── Convert relative bounds [nominal, fraction] → absolute [lo, hi] ─
        #   fraction = 0  → locked at nominal (lo == hi)
        #   fraction > 0  → [nominal×(1-fraction), nominal×(1+fraction)]
        #   nominal  = 0  → warning, set to 1.0
        raw = self.initial_bounds
        n_coils = len(raw)
        abs_bounds = np.empty((n_coils, 2), dtype=np.float64)
        for i in range(n_coils):
            nominal, frac = float(raw[i, 0]), float(raw[i, 1])
            if nominal == 0.0:
                logger.warning(
                    "Coil %d has nominal=0; setting to 1.0 for fraction-based "
                    "bounds.  Consider using a non‑zero nominal value.", i)
                nominal = 1.0
            abs_bounds[i, 0] = nominal * (1.0 - frac)
            abs_bounds[i, 1] = nominal * (1.0 + frac)
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

    def evaluate(self, extcur_free: np.ndarray, gen: int, ind_idx: int
                 ) -> tuple[float, dict]:
        """Run the full evaluation chain.

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
            "major radius": np.nan,
            "average B": np.nan,
            "failure_flag": True,
            "failure_type": FailureType.UNKNOWN.value,
            "failure_message": "",
        }

        # --- find magnetic axis ---
        with _silent():
            axis_result = find_axis(
                self.cfg.initial_rz,
                xtol=1e-5, max_iter=100, delta_r=0.01, verbose=False,
            )
        axis_rz, R0, axis_fieldline, success = axis_result
        if not success:
            info["failure_type"] = FailureType.AXIS_NOT_FOUND.value
            info["failure_message"] = "Magnetic axis not found"
            logger.warning("Gen %d, Ind %d: %s", gen, ind_idx, info["failure_message"])
            return self.INVALID_FITNESS, info

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
            logger.warning("Gen %d, Ind %d: %s", gen, ind_idx, info["failure_message"])
            return self.INVALID_FITNESS, info

        if np.isnan(epsilon_eff) or np.isinf(epsilon_eff):
            info["failure_type"] = FailureType.EPSILON_NAN.value
            info["failure_message"] = f"epsilon_eff is {epsilon_eff}"
            logger.warning("Gen %d, Ind %d: %s", gen, ind_idx, info["failure_message"])
            return self.INVALID_FITNESS, info

        # --- plasma parameters ---
        try:
            vol, minor_radius, iota = calculate_plasma_params(
                fieldline_data, axis_fieldline, self.cfg.nturn, self.cfg.nphi, R0
            )
            Aspect_ratio = R0 / minor_radius
        except Exception as exc:
            logger.warning("Gen %d, Ind %d: plasma param calc failed: %s", gen, ind_idx, exc)
            vol, minor_radius, iota = np.nan, np.nan, np.nan
        
        print(f"Gen {gen}, Ind {ind_idx}: Axis @ R = {axis_rz[0]:.3f}, epsilon_eff = {epsilon_eff:.3e},iota = {iota:.3f} ,minor radius = {minor_radius:.3f}")

        info.update(
            epsilon_eff=float(epsilon_eff),
            iota=float(iota),
            volume=float(vol),
            failure_flag=False,
            failure_type=FailureType.NONE.value,
            failure_message="",
        )
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
        "Generation", "Individual", "extcur", "epsilon_eff",
        "iota", "volume", "Aspect ratio", "average B",
        "failure_flag", "failure_type", "failure_message",
    ]
    with open(filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for info in individual_infos:
            writer.writerow([
                info.get("Generation"),
                info.get("Individual"),
                _csv_value(info.get("extcur")),
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

        # Convergence tracking
        self._best_fitness_history: list[float] = []
        self._patience_counter = 0

        # Persistent process pool (created once in run(), reused across gens)
        self._pool: Pool | None = None

        # Random seed
        if config.seed is not None:
            random.seed(config.seed)
            np.random.seed(config.seed)

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

        self._init_population()
        self._evaluate_and_record(self.pop, gen=0)

        # ---- main loop ----
        for gen in range(1, self.cfg.max_gen + 1):
            t0 = time.perf_counter()

            # 1. Generate trial vectors
            trials = []
            for i in range(self.cfg.n_pop):
                mutant = self._mutate(self.pop[i], self.pop)
                trial = self._crossover(self.pop[i], mutant)
                trials.append(trial)

            # 2. Evaluate
            trial_fitnesses, trial_infos = self._evaluate_batch(trials, gen)

            # 3. Selection
            invalid_solutions = 0
            for i in range(self.cfg.n_pop):
                tf = trial_fitnesses[i]
                cf = self.fitnesses[i]

                if tf >= StellaratorObjective.INVALID_FITNESS:
                    self.invalid_count[i] += 1
                else:
                    self.invalid_count[i] = 0

                if self.invalid_count[i] >= 3:
                    # Re-initialise this individual
                    new_ind = self._init_individual()
                    new_fit, new_info = self.objective.evaluate(new_ind, gen, i)
                    self.pop[i] = new_ind
                    self.fitnesses[i] = new_fit
                    self.all_infos.append(new_info)
                    self.invalid_count[i] = 0
                    logger.info(
                        "Gen %d, Ind %d: re-initialised (fitness=%.6e)",
                        gen, i, new_fit,
                    )
                elif tf < StellaratorObjective.INVALID_FITNESS and tf <= cf:
                    self.pop[i] = trials[i]
                    self.fitnesses[i] = tf
                    self.all_infos.append(trial_infos[i])

                if tf >= StellaratorObjective.INVALID_FITNESS:
                    invalid_solutions += 1

            # 4. Log generation summary
            t_elapsed = time.perf_counter() - t0
            best_idx = int(np.argmin(self.fitnesses))
            best_fit = self.fitnesses[best_idx]
            self._best_fitness_history.append(best_fit)

            pct_invalid = invalid_solutions / self.cfg.n_pop * 100
            best_extcur = self.pop[best_idx]
            logger.info(
                "Gen %3d/%d  |  best ε=%.6e  |  invalid=%3d/%d (%.1f%%)  |  "
                "time=%6.2fs  |  extcur=%s",
                gen, self.cfg.max_gen, best_fit,
                invalid_solutions, self.cfg.n_pop, pct_invalid,
                t_elapsed,
                np.array2string(best_extcur, separator=', ',
                                formatter={'float_kind': lambda x: '%.1f' % x}),
            )

            # 5. Early stop via ftol
            if self._check_convergence(gen):
                logger.info(
                    "Convergence reached: fitness change < %.1e for %d generations",
                    self.cfg.ftol, self.cfg.patience,
                )
                break

        # ---- finalise ----
        best_idx = int(np.argmin(self.fitnesses))
        best_ind = self.pop[best_idx]
        best_fitness = self.fitnesses[best_idx]

        save_csv(self.all_infos, self.cfg.output_dir / self.cfg.csv_filename)

        logger.info(
            "Optimisation finished. Best fitness = %.6e at index %d",
            best_fitness, best_idx,
        )
        return best_ind, best_fitness, self.all_infos

    # ---- population init ---------------------------------------------------

    def _init_individual(self) -> np.ndarray:
        return np.array([
            random.uniform(self.cfg._abs_bounds[i, 0],
                           self.cfg._abs_bounds[i, 1])
            for i in range(self.n_dim)
        ], dtype=np.float64)

    def _init_population(self):
        self.pop = [self._init_individual() for _ in range(self.cfg.n_pop)]
        self.fitnesses = []
        self.invalid_count = [0] * self.cfg.n_pop

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

    def _crossover(self, individual: np.ndarray, mutant: np.ndarray) -> np.ndarray:
        """Binomial crossover."""
        size = len(individual)
        trial = individual.copy()
        j_rand = random.randint(0, size - 1)
        for i in range(size):
            if random.random() < self.cfg.CR or i == j_rand:
                trial[i] = mutant[i]
        return trial

    # ---- evaluation --------------------------------------------------------

    def _evaluate_batch(self, population: list[np.ndarray], gen: int
                        ) -> tuple[list[float], list[dict]]:
        """Evaluate a population using the persistent process pool."""
        evaluate_func = self.objective.evaluate
        args = [(ind, gen, i) for i, ind in enumerate(population)]
        results = self._pool.starmap(evaluate_func, args)

        fitnesses, infos = [], []
        for fit, info in results:
            fitnesses.append(float(fit))
            infos.append(info)
        return fitnesses, infos

    def _evaluate_and_record(self, population: list[np.ndarray], gen: int):
        """Evaluate a population and record fitnesses + metadata."""
        fitnesses, infos = self._evaluate_batch(population, gen)
        for fit, info in zip(fitnesses, infos):
            self.fitnesses.append(float(fit))
            self.all_infos.append(info)

    # ---- convergence -------------------------------------------------------

    def _check_convergence(self, gen: int) -> bool:
        """Return True if ftol-based early-stop should trigger."""
        if self.cfg.ftol is None or len(self._best_fitness_history) < 2:
            return False

        # Check if best fitness has stabilised
        window = min(self.cfg.patience, len(self._best_fitness_history))
        recent = self._best_fitness_history[-window:]
        improvement = abs(recent[-1] - recent[0])
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

