# Ripple Optimization Guide

This guide explains how to use the **Differential Evolution (DE)** optimization module in ripplepy to minimize effective ripple (`epsilon_eff`) by varying external coil currents (`extcur`).

## Overview

The optimization module provides a parallel DE algorithm that:
- Minimizes `epsilon_eff` (effective ripple)
- Varies `extcur` (external coil currents)
- Uses timeout protection for long-running computations
- Logs detailed optimization progress and results
- Supports parallel evaluation via multiprocessing

## Core Components

### 1. OptimizationConfig

Configuration dataclass for DE algorithm parameters:

```python
from ripplepy import OptimizationConfig

config = OptimizationConfig(
    n_dim=4,                      # Number of parameters to optimize
    n_pop=16,                     # Population size
    max_gen=100,                  # Maximum generations
    F=0.5,                        # Mutation scaling factor
    CR=0.7,                       # Crossover probability
    processes=8,                  # Number of parallel processes
    timeout_find_axis=0.2,        # Timeout for axis finding (seconds)
    timeout_ripple=10.0,          # Timeout for ripple computation (seconds)
    nturn=100,                    # Toroidal windings for tracing
    nphi=360,                     # Poloidal grid points
)
```

### 2. ObjectiveFunction

Wrapper class that computes `epsilon_eff` for given `extcur` values:

```python
from ripplepy import ObjectiveFunction
from pathlib import Path

obj_func = ObjectiveFunction(
    nfp=3,                           # Number of field periods
    initial_rz=(1.3, 0.0),           # Initial (R, Z) for axis search
    deltaR=0.08,                     # R offset for measurement point
    mpol=100,                        # Poloidal grid resolution
    axis0=[1.23, 0.0],              # Expected axis position
    config=config,
    output_dir=Path("./results"),
)
```

The objective function can be called as:
```python
epsilon_eff, info_dict, x_values_dict = obj_func(
    extcur=np.array([5000, 1, -80000, -40000]),
    gen=0,  # Generation number (for logging)
    ind_idx=0,  # Individual index (for logging)
)
```

Returns:
- `epsilon_eff`: Objective value (float)
- `info_dict`: Physical quantities (dict with epsilon, iota, aspect ratio, etc.)
- `x_values_dict`: Parameter values used (dict)

### 3. differential_evolution

Main DE algorithm:

```python
from ripplepy import differential_evolution
import numpy as np

bounds = np.array([
    [4500, 5500],      # extcur[1] bounds
    [-100, 100],       # extcur[2] bounds
    [-90000, -70000],  # extcur[3] bounds
    [-50000, -30000],  # extcur[4] bounds
])

best_individual, best_fitness = differential_evolution(
    objective_func=obj_func,
    bounds=bounds,
    config=config,
)

print(f"Best epsilon_eff: {best_fitness:.6e}")
print(f"Best extcur: {best_individual}")
```

## Basic Usage Example

```python
import numpy as np
from pathlib import Path
from ripplepy import (
    initialize_mgrid_field,
    ObjectiveFunction,
    OptimizationConfig,
    differential_evolution,
)

# 1. Initialize field
mgrid_file = Path("./tests/test_file/mgrid_c09r00.nc")
initialize_mgrid_field(str(mgrid_file), nfp=3, full_torus=False)

# 2. Set up configuration
config = OptimizationConfig(
    n_dim=4,
    n_pop=16,
    max_gen=50,
    processes=8,
)

# 3. Create objective function
obj_func = ObjectiveFunction(
    nfp=3,
    initial_rz=(1.3, 0.0),
    config=config,
    output_dir=Path("./opt_results"),
)

# 4. Define search bounds (±10% of initial values)
extcur_init = np.array([50000, 5000, 1, -80000, -40000])
n_dim = 4
bounds = np.zeros((n_dim, 2))
for i in range(n_dim):
    x = extcur_init[i + 1]
    bounds[i] = [x - 0.1*abs(x), x + 0.1*abs(x)]

# 5. Run optimization
best_ind, best_fit = differential_evolution(obj_func, bounds, config)

print(f"Optimal epsilon_eff: {best_fit:.6e}")
print(f"Optimal extcur[1:5]: {best_ind}")
```

## Running the Example

A complete example is provided in `optimize_example.py`:

```bash
cd /home/zkg/ripplepy
python -m ripplepy.optimize_example
```

This example demonstrates:
- Loading mgrid file
- Configuring DE optimization
- Running the algorithm
- Saving results to CSV logs

## Output Files

The optimization produces the following outputs in `output_dir`:

### Log Files
- `temp_log_*.csv`: Physical quantities (epsilon, iota, aspect ratio, etc.)
- `temp_x_log_*.csv`: Parameter values used in evaluations
- `optimization_result.txt`: Summary of best solution

### Grid Output Files
- `grid_qa_gen{gen}_ind{idx}.nc`: VMEC-compatible output for each evaluation

## Algorithm Details

### DE Mutation (DE/rand/1)
$$v_i = x_{r1} + F \cdot (x_{r2} - x_{r3})$$

Where $r1, r2, r3$ are random indices distinct from $i$, and $F$ is the scaling factor.

### Uniform Crossover
$$u_{i,j} = \begin{cases} v_{i,j} & \text{if } \text{rand} < CR \\ x_{i,j} & \text{otherwise} \end{cases}$$

Where $CR$ is the crossover probability.

### Selection
Replace current individual if trial has better or equal fitness:
$$x_i^{g+1} = \begin{cases} u_i & \text{if } f(u_i) \le f(x_i^g) \\ x_i^g & \text{otherwise} \end{cases}$$

## Timeout Protection

The module includes timeout protection for potentially long-running computations:

```python
timeout_find_axis=0.2   # If axis finding > 0.2s, returns invalid (1e3)
timeout_ripple=10.0     # If ripple computation > 10s, returns invalid
```

Invalid solutions are marked with `epsilon_eff = 1e3` and:
- Reset if appearing 3+ times consecutively for same individual
- Not selected unless population becomes all invalid

## Parallel Evaluation

Evaluation uses Python's `multiprocessing.Pool`:

```python
processes=8  # Number of parallel workers
```

- Each generation evaluates population in parallel
- Trial solutions also evaluated in parallel
- Total evaluations per generation: `2 * n_pop` (assuming all replace)

## Performance Tips

1. **Reduce Population**: Start with `n_pop=8` for testing, increase to 16-32 for production
2. **Limit Generations**: Use `max_gen=10-20` initially, increase as needed
3. **Parallel Processes**: Set `processes = min(n_pop, CPU_count - 2)`
4. **Timeout Values**: Adjust based on your hardware (larger values = longer per evaluation)
5. **Bounds**: Use tight bounds (±5-10%) for faster convergence

## Integration with ripplepy.py

The optimization module uses these core functions from `ripple.py`:

- `initialize_mgrid_field()`: Set up magnetic field
- `set_extcur()`: Update coil currents
- `find_axis()`: Locate magnetic axis
- `compute_epstot()`: Compute effective ripple and fieldline data
- `calculate_plasma_params()`: Compute volume, iota, aspect ratio

No modifications needed to `ripple.py` - optimization uses existing public API.

## Troubleshooting

### All evaluations return epsilon_eff = 1e3
- Check if mgrid file is loaded correctly
- Verify `initial_rz` is near actual magnetic axis
- Increase `timeout_find_axis` if hardware is slow

### Slow convergence
- Use wider bounds to allow more exploration
- Increase mutation factor `F` (e.g., 0.8-1.0)
- Increase crossover probability `CR` (e.g., 0.8-0.9)

### Memory issues with large populations
- Reduce `n_pop`
- Reduce `nturn` or `nphi` in config
- Use fewer parallel processes to reduce memory per worker

## References

- DE Algorithm: Price, K. V., Storn, R. M., & Lampinen, J. A. (2005). "Differential Evolution: A Practical Approach to Global Optimization"
- Nemov et al. (1999) for effective ripple calculation (see paper in tests/test_file)
