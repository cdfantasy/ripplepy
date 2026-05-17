"""
Differential Evolution (DE) optimization module for ripple minimization.

This module provides DE-based optimization for minimizing effective ripple (epsilon_eff)
by varying external coil currents (extcur).

Reference: test.py optimization workflow
"""

import numpy as np
import random
import os
from pathlib import Path
from multiprocessing import Pool, get_context
from typing import Callable, Tuple, Dict, List, Optional
from dataclasses import dataclass
import pandas as pd
from func_timeout import func_timeout, FunctionTimedOut

from .ripple import (
    find_axis,
    set_extcur,
    set_trace_parameters,
    compute_epstot,
    calculate_plasma_params,
)


_WORKER_TRACE_NTURN = None
_WORKER_TRACE_NPHI = None
_WORKER_SEED_EXTCUR = None


def _init_worker(nturn: int, nphi: int, seed_extcur=None):
    """Initialize backend state inside each worker process."""
    global _WORKER_TRACE_NTURN, _WORKER_TRACE_NPHI, _WORKER_SEED_EXTCUR
    _WORKER_TRACE_NTURN = int(nturn)
    _WORKER_TRACE_NPHI = int(nphi)
    _WORKER_SEED_EXTCUR = None if seed_extcur is None else np.asarray(seed_extcur, dtype=np.float64)

    # Re-apply trace parameters in the child process so tracing does not depend on fork inheritance.
    set_trace_parameters(_WORKER_TRACE_NTURN, _WORKER_TRACE_NPHI, verbose=False)

    # Prime the magnetic field state if a seed current set is provided.
    if _WORKER_SEED_EXTCUR is not None:
        set_extcur(_WORKER_SEED_EXTCUR)


@dataclass
class OptimizationConfig:
    """Configuration for DE optimization."""
    n_dim: int = 4
    n_pop: int = 16
    max_gen: int = 10
    F: float = 0.5  # Mutation scaling factor
    CR: float = 0.7  # Crossover probability
    processes: Optional[int] = None
    timeout_find_axis: float = 0.2  # seconds
    timeout_ripple: float = 10.0  # seconds
    nturn: int = 100
    nphi: int = 360


class ObjectiveFunction:
    """
    Wrapper for ripple minimization objective function.
    
    Computes epsilon_eff given coil currents (extcur).
    """
    
    def __init__(
        self,
        nfp: int,
        initial_rz: Tuple[float, float],
        deltaR: float = 0.08,
        mpol: int = 100,
        axis0: Optional[List[float]] = None,
        config: Optional[OptimizationConfig] = None,
        seed_extcur: Optional[List[float]] = None,
        output_dir: Optional[Path] = None,
    ):
        """
        Parameters
        ----------
        nfp : int
            Number of field periods.
        initial_rz : tuple
            Initial (R, Z) guess for magnetic axis search.
        deltaR : float
            Shift in R for axis calculation.
        mpol : int
            Poloidal grid resolution.
        axis0 : list, optional
            Initial axis position [R, Z]. Defaults to initial_rz if None.
        config : OptimizationConfig, optional
            Optimization configuration.
        output_dir : Path, optional
            Directory for output files (grid_qa_*.nc).
        """
        self.nfp = nfp
        self.initial_rz = np.array(initial_rz, dtype=np.float64)
        self.deltaR = deltaR
        self.mpol = mpol
        self.axis0 = axis0 if axis0 is not None else list(initial_rz)
        self.config = config or OptimizationConfig()
        self.seed_extcur = None if seed_extcur is None else np.asarray(seed_extcur, dtype=np.float64)
        self.output_dir = output_dir or Path("./outputfiles")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize trace parameters
        set_trace_parameters(self.config.nturn, self.config.nphi, verbose=False)
    
    def __call__(
        self,
        extcur: np.ndarray,
        gen: int = 0,
        ind_idx: int = 0,
    ) -> Tuple[float, Dict, Dict]:
        """
        Compute objective (epsilon_eff) for given coil currents.
        
        Parameters
        ----------
        extcur : np.ndarray
            External coil currents.
        gen : int
            Generation index (for logging).
        ind_idx : int
            Individual index (for logging).
        
        Returns
        -------
        epsilon_eff : float
            Effective ripple (objective to minimize).
        info_dict : dict
            Physical quantities (epsilon, iota, aspect ratio, etc.).
        x_values_dict : dict
            Parameter values used.
        """
        print(f"Gen {gen}, Ind {ind_idx}")
        
        # Initialize info dictionaries
        info_dict = {
            'gen': gen,
            'ind_idx': ind_idx,
            'epsilon': np.nan,
            'iota': np.nan,
            'asp': np.nan,
            'rm': np.nan,
            'am': np.nan,
            'volume': np.nan,
            'Baxis': np.nan,
            'Bboundary': np.nan
        }
        
        x_values_dict = {
            'gen': gen,
            'ind_idx': ind_idx,
            'x_values': str(extcur[:])
        }
        
        try:
            # Set external currents and compute field
            extcur=set_extcur(extcur)

            
            # Find magnetic axis with timeout (find_axis has internal timeout)
            axis_rz, R0, axis_fieldline, trace_error = find_axis(
                self.initial_rz.copy(),
                timeout=self.config.timeout_find_axis,  # add buffer
                xtol=1e-5,
                max_iter=100
            )
            
            if trace_error or np.isnan(R0):
                return 1e3, info_dict, x_values_dict
            
            # Set measurement point: shift axis position by deltaR in R direction
            initial_rz_measurement = np.array([axis_rz[0] + self.deltaR, axis_rz[1]], dtype=np.float64)
            
            # Compute effective ripple (with timeout protection via func_timeout)
            try:
                # Pre-allocate fieldline data buffer
                fieldline_data = np.zeros(
                    (self.config.nturn * self.config.nphi, 20),
                    dtype=np.float64,
                    order='F'
                )
                
                # Use func_timeout to protect compute_epstot
                epsilon_eff, bboundary, fieldline_data = func_timeout(
                    self.config.timeout_ripple,
                    compute_epstot,
                    kwargs={
                        'R0': R0,
                        'extcur': extcur,
                        'initial_rz': initial_rz_measurement,
                        'initial_gradpsi': None,
                        'fieldline_data': fieldline_data,
                        'return_fieldline': True,
                    }
                )
            except FunctionTimedOut:
                print(f"compute_epstot timed out after {self.config.timeout_ripple}s")
                return 1e3, info_dict, x_values_dict
            
            if np.isnan(epsilon_eff) or epsilon_eff >= 1e3:
                return 1e3, info_dict, x_values_dict
            
            # Compute plasma parameters
            vol, am, iota = calculate_plasma_params(
                fieldline_data,
                axis_fieldline,
                self.config.nturn,
                self.config.nphi,
                R0
            )
            
            # Update info dictionary
            info_dict.update({
                'epsilon': epsilon_eff,
                'iota': iota,
                'asp': am / R0 if R0 > 0 else np.nan,
                'rm': R0,
                'am': am,
                'volume': vol,
                'Baxis': np.nan,  # Could extract from axis_fieldline if needed
                'Bboundary': bboundary,
            })
            
            return epsilon_eff, info_dict, x_values_dict
            
        except Exception as e:
            print(f"Error in objective function: {e}")
            return 1e3, info_dict, x_values_dict


def mutate_de(
    individual: List[float],
    population: List[List[float]],
    bounds: np.ndarray,
    F: float = 0.5,
) -> List[float]:
    """
    DE/rand/1 mutation: v = x_r1 + F * (x_r2 - x_r3)
    
    Parameters
    ----------
    individual : list
        Current individual.
    population : list
        Current population.
    bounds : ndarray, shape (n_dim, 2)
        Search bounds [min, max] for each dimension.
    F : float
        Mutation scaling factor.
    
    Returns
    -------
    mutant : list
        Mutated vector (within bounds).
    """
    n_dim = len(individual)
    idxs = [i for i in range(len(population)) if population[i] != individual]
    
    if len(idxs) < 3:
        raise ValueError("Population too small for mutation")
    
    r1, r2, r3 = random.sample(idxs, 3)
    
    mutant = []
    for i in range(n_dim):
        v_i = population[r1][i] + F * (population[r2][i] - population[r3][i])
        # Enforce bounds
        v_i = np.clip(v_i, bounds[i, 0], bounds[i, 1])
        mutant.append(v_i)
    
    return mutant


def crossover_de(
    individual: List[float],
    mutant: List[float],
    CR: float = 0.7,
) -> List[float]:
    """
    Uniform crossover: trial_i = mutant_i if rand < CR else individual_i
    
    Parameters
    ----------
    individual : list
        Original individual.
    mutant : list
        Mutant vector.
    CR : float
        Crossover probability [0, 1].
    
    Returns
    -------
    trial : list
        Trial vector.
    """
    n_dim = len(individual)
    trial = individual.copy()
    j_rand = random.randint(0, n_dim - 1)
    
    for i in range(n_dim):
        if random.random() < CR or i == j_rand:
            trial[i] = mutant[i]
    
    return trial


def init_individual(bounds: np.ndarray, n_dim: int) -> List[float]:
    """Initialize a random individual within bounds."""
    return [
        random.uniform(bounds[i, 0], bounds[i, 1])
        for i in range(n_dim)
    ]


def init_population(
    n_pop: int,
    bounds: np.ndarray,
    n_dim: int,
    seed_individual: Optional[List[float]] = None,
) -> List[List[float]]:
    """Initialize population with an optional seeded first individual."""
    population = []
    if seed_individual is not None:
        seed = list(seed_individual)
        if len(seed) != n_dim:
            raise ValueError("seed_individual must match n_dim")
        population.append(seed)
    
    while len(population) < n_pop:
        population.append(init_individual(bounds, n_dim))
    
    return population


def evaluate_population(
    population: List[List[float]],
    evaluate_func: Callable,
    gen: int,
    processes: Optional[int] = None,
    worker_nturn: Optional[int] = None,
    worker_nphi: Optional[int] = None,
    worker_seed_extcur=None,
) -> Tuple[List[float], List[Dict], List[Dict]]:
    """
    Parallel evaluation of population.
    
    Parameters
    ----------
    population : list
        List of individuals (parameter vectors).
    evaluate_func : callable
        Objective function: evaluate_func(ind, gen, idx) -> (fit, info, x_vals).
    gen : int
        Generation number.
    processes : int, optional
        Number of parallel processes.
    
    Returns
    -------
    fitnesses : list
        Objective values.
    infos : list
        Info dictionaries.
    x_values_infos : list
        Parameter value dictionaries.
    """
    initializer = None
    initargs = ()
    if worker_nturn is not None and worker_nphi is not None:
        initializer = _init_worker
        initargs = (worker_nturn, worker_nphi, worker_seed_extcur)

    pool_kwargs = {
        "processes": processes,
    }
    if initializer is not None:
        pool_kwargs["initializer"] = initializer
        pool_kwargs["initargs"] = initargs

    try:
        pool_context = get_context("fork")
    except ValueError:
        # Fallback to the platform default when fork is unavailable.
        pool_context = get_context()

    with pool_context.Pool(**pool_kwargs) as pool:
        args = [(ind, gen, i) for i, ind in enumerate(population)]
        results = pool.starmap(evaluate_func, args)
    
    fitnesses = [result[0] for result in results]
    infos = [result[1] for result in results]
    x_values_infos = [result[2] for result in results]
    
    return fitnesses, infos, x_values_infos


def save_log(
    infos: List[Dict],
    output_dir: Path,
    log_suffix: str = "log",
) -> None:
    """Save physical quantities log."""
    log_file = output_dir / f"temp_{log_suffix}_{os.getpid()}.csv"
    df = pd.DataFrame(infos)
    with open(log_file, 'a', newline='') as f:
        df.to_csv(f, index=False, header=not log_file.exists())


def save_x_values_log(
    x_values_infos: List[Dict],
    output_dir: Path,
    log_suffix: str = "x_log",
) -> None:
    """Save parameter values log."""
    log_file = output_dir / f"temp_{log_suffix}_{os.getpid()}.csv"
    df = pd.DataFrame(x_values_infos)
    with open(log_file, 'a', newline='') as f:
        df.to_csv(f, index=False, header=not log_file.exists())


def differential_evolution(
    objective_func: Callable,
    bounds: np.ndarray,
    config: OptimizationConfig,
    seed_individual: Optional[List[float]] = None,
) -> Tuple[np.ndarray, float]:
    """
    Differential Evolution algorithm.
    
    Parameters
    ----------
    objective_func : callable
        Objective function: f(individual, gen, ind_idx) -> (fitness, info, x_vals).
    bounds : ndarray, shape (n_dim, 2)
        Search bounds for each parameter.
    config : OptimizationConfig
        DE configuration.
    seed_individual : list, optional
        Initial individual to place at population index 0.
    
    Returns
    -------
    best_individual : ndarray
        Best solution found.
    best_fitness : float
        Best objective value.
    """
    n_dim = config.n_dim
    n_pop = config.n_pop
    max_gen = config.max_gen
    F = config.F
    CR = config.CR
    
    if n_pop < 4:
        raise ValueError("n_pop must be at least 4 for DE mutation")
    
    # Initialize population
    population = init_population(n_pop, bounds, n_dim, seed_individual=seed_individual)
    
    # Evaluate initial population
    fitnesses, infos, x_values_infos = evaluate_population(
        population,
        objective_func,
        gen=0,
        processes=config.processes,
        worker_nturn=config.nturn,
        worker_nphi=config.nphi,
        worker_seed_extcur=getattr(objective_func, "seed_extcur", None),
    )
    
    # Assign fitness to individuals
    pop_with_fitness = list(zip(population, fitnesses))
    
    # Save initial logs
    if hasattr(objective_func, 'output_dir'):
        save_log(infos, objective_func.output_dir)
        save_x_values_log(x_values_infos, objective_func.output_dir)
    
    invalid_count = {i: 0 for i in range(n_pop)}
    
    # Main DE loop
    for gen in range(max_gen):
        best_ind, best_fit = min(pop_with_fitness, key=lambda x: x[1])
        
        # Mutation and crossover
        trials = []
        trial_fitnesses_list = []
        
        for i in range(n_pop):
            mutant = mutate_de(population[i], population, bounds, F=F)
            trial = crossover_de(population[i], mutant, CR=CR)
            trials.append(trial)
        
        # Evaluate trials
        trial_fitnesses, trial_infos, trial_x_infos = evaluate_population(
            trials,
            objective_func,
            gen=gen + 1,
            processes=config.processes,
            worker_nturn=config.nturn,
            worker_nphi=config.nphi,
            worker_seed_extcur=getattr(objective_func, "seed_extcur", None),
        )
        
        # Selection
        invalid_solutions = 0
        for i in range(n_pop):
            trial_fit = trial_fitnesses[i]
            current_fit = fitnesses[i]
            
            print(
                f"Gen {gen+1}, Ind {i}, "
                f"Current fitness = {current_fit:.6e}, "
                f"Trial fitness = {trial_fit:.6e}"
            )
            
            if trial_fit >= 1e3:
                invalid_solutions += 1
                invalid_count[i] += 1
            else:
                invalid_count[i] = 0
            
            # Replace if trial is better, or reset if too many invalids
            if invalid_count[i] >= 3:
                population[i] = init_individual(bounds, n_dim)
                fitness_new = objective_func(
                    np.array(population[i]),
                    gen=gen + 1,
                    ind_idx=i
                )[0]
                fitnesses[i] = fitness_new
                invalid_count[i] = 0
            elif trial_fit < 1e3 and trial_fit <= current_fit:
                population[i] = trials[i]
                fitnesses[i] = trial_fit
        
        # Update population with fitness
        pop_with_fitness = list(zip(population, fitnesses))
        
        # Save logs
        if hasattr(objective_func, 'output_dir'):
            save_log(trial_infos, objective_func.output_dir)
            save_x_values_log(trial_x_infos, objective_func.output_dir)
        
        best_ind, best_fit = min(pop_with_fitness, key=lambda x: x[1])
        
        print(
            f"Generation {gen+1}, "
            f"Invalid solutions: {invalid_solutions}/{n_pop} "
            f"({invalid_solutions/n_pop*100:.2f}%)"
        )
        print(f"Generation {gen+1}, Best Fitness: {best_fit:.6e}")
    
    best_ind, best_fit = min(pop_with_fitness, key=lambda x: x[1])
    
    return np.array(best_ind), best_fit
