# Quick Start Guide for Ripple Optimization

## Installation

Make sure ripplepy is installed with the optimization module:

```bash
cd /home/zkg/ripplepy
pip install -e .
```

## Basic Example

```python
import numpy as np
from pathlib import Path
from ripplepy import (
    initialize_mgrid_field,
    ObjectiveFunction,
    OptimizationConfig,
    differential_evolution,
)

# 1. Initialize magnetic field from mgrid file
mgrid_file = Path("tests/test_file/mgrid_c09r00.nc")
initialize_mgrid_field(str(mgrid_file), nfp=3, full_torus=False)

# 2. Create configuration
config = OptimizationConfig(
    n_dim=4,           # Optimize 4 currents
    n_pop=16,          # Population size
    max_gen=50,        # Generations
    processes=8,       # Parallel workers
)

# 3. Create objective function
obj_func = ObjectiveFunction(
    nfp=3,
    initial_rz=(1.3, 0.0),
    config=config,
    output_dir=Path("./my_results"),
)

# 4. Define search bounds (±10% around initial values)
extcur_init = np.array([50000, 5000, 1, -80000, -40000])
bounds = np.zeros((4, 2))
for i in range(4):
    x = extcur_init[i + 1]
    bounds[i] = [x - 0.1*abs(x), x + 0.1*abs(x)]

# 5. Run optimization
best_ind, best_fit = differential_evolution(obj_func, bounds, config)

print(f"Best epsilon_eff: {best_fit:.6e}")
print(f"Optimal extcur[1:5]: {best_ind}")
```

## Running the Complete Example

```bash
cd /home/zkg/ripplepy
python -m ripplepy.optimize_example
```

## Expected Output

- Console logs showing generation progress and fitness values
- CSV files in `./optimization_results/`:
  - `temp_log_*.csv`: Physical quantities per evaluation
  - `temp_x_log_*.csv`: Parameter values per evaluation
  - `optimization_result.txt`: Summary of best solution

## Next Steps

1. **Read OPTIMIZE_GUIDE.md** for detailed API reference and algorithm explanation
2. **Read INTEGRATION_SUMMARY.md** for technical architecture details
3. **Modify OptimizationConfig** parameters for your problem:
   - Reduce `n_pop` and `max_gen` for faster testing
   - Adjust `F` (mutation) and `CR` (crossover) for exploration/exploitation balance
   - Change bounds for different search regions

## Troubleshooting

- **All evaluations return epsilon_eff = 1e3**: Check mgrid file path and initial_rz guess
- **Slow convergence**: Use wider bounds or increase `F`
- **Memory issues**: Reduce `n_pop` or `nturn`

See OPTIMIZE_GUIDE.md for more troubleshooting tips.
