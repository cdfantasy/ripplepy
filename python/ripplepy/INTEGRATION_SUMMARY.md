# Ripplepy Optimization Module Integration Summary

## Changes Made

This document summarizes the addition of a **Differential Evolution (DE) optimization module** to ripplepy for minimizing effective ripple (`epsilon_eff`) via external coil current (`extcur`) variation.

## Files Modified/Created

### New Files

1. **optimize.py** (NEW)
   - Core DE implementation
   - Classes: `OptimizationConfig`, `ObjectiveFunction`
   - Functions: `differential_evolution`, `mutate_de`, `crossover_de`, etc.
   - ~400 lines, fully documented with docstrings

2. **optimize_example.py** (NEW)
   - Complete usage example
   - Demonstrates workflow from field initialization to result saving
   - ~200 lines with detailed comments

3. **OPTIMIZE_GUIDE.md** (NEW)
   - Comprehensive user guide
   - API reference
   - Algorithm details and mathematical formulas
   - Troubleshooting section

### Modified Files

1. **__init__.py**
   - Added imports: `ObjectiveFunction`, `OptimizationConfig`, `differential_evolution`, `mutate_de`, `crossover_de`
   - Updated `__all__` list with optimization exports

## Module Architecture

```
ripplepy/
├── __init__.py              (MODIFIED - add optimization exports)
├── ripple.py                (NO CHANGE - uses existing functions)
├── mgrid.py                 (NO CHANGE)
├── optimize.py              (NEW - DE implementation)
├── optimize_example.py      (NEW - example usage)
├── OPTIMIZE_GUIDE.md        (NEW - documentation)
└── INTEGRATION_SUMMARY.md   (THIS FILE)
```

## Key Components

### 1. OptimizationConfig (Dataclass)
- Encapsulates DE parameters (population size, generations, F, CR)
- Includes compute settings (timeouts, trace parameters)
- Easy to modify and experiment with

### 2. ObjectiveFunction (Class)
- Wraps physical computation pipeline
- Handles timeout protection with `func_timeout`
- Returns: fitness (epsilon_eff) + info dicts for logging
- Supports parallel evaluation via multiprocessing

### 3. DE Algorithm Functions
- `differential_evolution()`: Main loop with population management
- `mutate_de()`: DE/rand/1 mutation
- `crossover_de()`: Uniform crossover
- Supporting functions: `init_population()`, `evaluate_population()`, etc.

### 4. Logging & I/O
- `save_log()`: Save physical quantities to CSV
- `save_x_values_log()`: Save parameter values to CSV
- Integration with `pathlib.Path` for portability

## Integration Points with ripple.py

The optimization module **reuses existing ripplepy functions without modification**:

```python
from .ripple import (
    find_axis,
    set_extcur,
    compute_epstot,
    calculate_plasma_params,
)
```

**Data Flow:**
```
extcur (array)
  ↓
set_extcur() → update field
  ↓
find_axis() → locate magnetic axis (with timeout)
  ↓
compute_epstot() → calculate epsilon_eff (with timeout)
  ↓
calculate_plasma_params() → get volume, iota, aspect ratio
  ↓
epsilon_eff (scalar) → objective value
```

## Physical Computation Pipeline

The ObjectiveFunction implements this workflow (matching ripple_test.ipynb):

1. **Axis Finding**
   ```python
   axis_rz, R0, axis_fieldline, error = find_axis(
       initial_rz, timeout=10.0, xtol=1e-5, max_iter=100
   )
   ```

2. **Ripple Computation**
   ```python
   epsilon_eff, bboundary, fieldline_data = compute_epstot(
       R0=R0, extcur=extcur, initial_rz=axis_rz,
       initial_gradpsi=None, return_fieldline=True
   )
   ```

3. **Plasma Parameters**
   ```python
   volume, am, iota = calculate_plasma_params(
       fieldline_data, axis_fieldline, nturn, nphi, R0
   )
   ```

## Usage Pattern

**Minimal example:**

```python
from ripplepy import (
    initialize_mgrid_field,
    ObjectiveFunction,
    OptimizationConfig,
    differential_evolution,
)
import numpy as np
from pathlib import Path

# 1. Setup field
initialize_mgrid_field("mgrid.nc", nfp=3)

# 2. Create objective function
obj = ObjectiveFunction(
    nfp=3, initial_rz=(1.3, 0.0),
    output_dir=Path("./results")
)

# 3. Define bounds and run
bounds = np.array([[4500, 5500], [-100, 100], [-90000, -70000], [-50000, -30000]])
best_ind, best_fit = differential_evolution(
    obj, bounds, OptimizationConfig(n_dim=4, n_pop=16, max_gen=50)
)

print(f"Optimal epsilon_eff: {best_fit}")
```

## Timeout Protection

Two-level timeout protection prevents hangs:

```python
ObjectiveFunction(..., config=OptimizationConfig(
    timeout_find_axis=0.2,   # Axis search timeout
    timeout_ripple=10.0,     # Ripple computation timeout
))
```

- Timeouts return invalid objective value (1e3)
- Prevents deadlock on difficult or singular cases
- Configurable per physical computation

## Parallel Evaluation

Population and trials evaluated in parallel via `multiprocessing.Pool`:

```python
n_pop = 16
processes = 8
# Per generation: evaluate 16 individuals + 16 trials in parallel = 32 calls
```

**Initialization pattern:**
```python
def init_process():
    # Called once per worker process
    # Can set up process-local state if needed
    pass
```

## Logging & Results

Each generation saves:

1. **Physical Quantities** (`temp_log_*.csv`)
   - Generation, individual index, epsilon, iota, aspect ratio, etc.
   - Used for analyzing convergence and final solution quality

2. **Parameter Values** (`temp_x_log_*.csv`)
   - Generation, individual index, extcur array used
   - Trace optimization path through parameter space

3. **VMEC Output** (`grid_qa_gen{g}_ind{i}.nc`)
   - Optional detailed output files for each evaluation

**Post-optimization:** Merge logs from all workers:
```python
pd.concat([pd.read_csv(f) for f in glob.glob("temp_log_*.csv")])
```

## Algorithm Details

### DE/rand/1 Mutation
$$v_i = x_{r1} + F \cdot (x_{r2} - x_{r3})$$

- F ∈ (0, 2), typically 0.5-0.8
- Bounds enforced after mutation

### Uniform Crossover
$$u_{i,j} = \begin{cases} v_{i,j} & \text{if } \text{rand}() < CR \\ x_{i,j} & \text{otherwise} \end{cases}$$

- CR ∈ (0, 1), typically 0.7-0.9
- At least one dimension from mutant (j_rand)

### Greedy Selection
$$x_i^{g+1} = \begin{cases} u_i^g & \text{if } f(u_i^g) \leq f(x_i^g) \\ x_i^g & \text{otherwise} \end{cases}$$

### Invalid Solution Handling
- Mark as invalid: epsilon_eff ≥ 1e3
- Reset individual after 3 consecutive invalids
- Population maintains diversity through restarts

## Performance Characteristics

### Time per Evaluation
- `find_axis()`: ~0.1-0.5 seconds
- `compute_epstot()`: ~1-10 seconds (depending on trace resolution)
- `calculate_plasma_params()`: ~0.1 seconds
- **Total per individual**: ~2-15 seconds

### Typical Convergence
- Population size 16, Generations 100
- Total evaluations: ~3200 (2 per generation due to trials)
- **Estimated runtime**: 2-4 hours on 8 parallel processes

## Testing

Run the provided example:
```bash
cd /home/zkg/ripplepy
python -m ripplepy.optimize_example
```

Expected output:
- Console progress (generation, fitness improvements)
- CSV logs in `./optimization_results/`
- Best solution summary to `optimization_result.txt`

## Dependencies

Required packages (already in ripplepy environment):
- `numpy`
- `scipy`
- `pandas`
- `func_timeout`
- `deap` (already used in test.py)

## Future Enhancements

Potential improvements:
1. Constraint handling (equality/inequality constraints on plasma params)
2. Multi-objective optimization (balance epsilon_eff with other quantities)
3. Adaptive F and CR during evolution
4. Alternative selection strategies (tournament, rank-based)
5. Integration with VMEC/STELLOPT for full equilibrium optimization

## Notes for Maintenance

1. **No changes to ripple.py needed** - optimization uses public API only
2. **Timeout values are hardware-dependent** - adjust in OptimizationConfig
3. **Log merging** requires manual post-processing or use of provided save_log functions
4. **Memory scaling** is linear in population size and nturn*nphi

## References

- DE Algorithm: Price, Storn, Lampinen (2005)
- Nemov et al. (1999) - effective ripple calculation
- test.py - original DE implementation (basis for this module)
