"""
Example usage of the DE optimization module for epsilon_eff minimization.

This script demonstrates how to use the ripplepy optimization module to
minimize effective ripple by varying external coil currents (extcur).

Reference: tests/test.py DE algorithm
"""

import numpy as np
from pathlib import Path
import sys

# Import ripplepy components
from ripplepy import (
    initialize_mgrid_field,
    ObjectiveFunction,
    OptimizationConfig,
    differential_evolution,
)


def run_optimization_example():
    """
    Run a simple DE optimization example.
    
    This example:
    1. Initializes the magnetic field from mgrid file
    2. Sets up objective function with physical constraints
    3. Runs DE algorithm to minimize epsilon_eff
    4. Reports best solution and saves logs
    """
    
    # ============ Configuration ============
    
    # File paths
    # input_dir = Path("./tests/test_file")
    mgrid_filename = Path("/home/zkg/CN_H1_scan_fieldlines/H1_design/coils/mgrid_h1_design.nc")
    
    # Create output directory
    output_dir = Path("./optimization_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Physical parameters
    nfp = 3  # Field periods
    full_torus = False
    
    # Initial condition for axis search
    initial_rz = (1.3, 0.0)
    axis0 = [1.26, 0.0]
    
    # Coil current parameters
    extcur_initial = np.array([50000, 5000, 1, -80000, -40000], dtype=np.float64)
    n_dim = 4  # Optimize first 4 currents (indices 1:5)
    
    # Define search bounds (±10% of initial value)
    bounds = np.zeros((n_dim, 2))
    for i in range(n_dim):
        x_i = extcur_initial[i + 1]
        bounds[i, 0] = x_i - 0.1 * abs(x_i)
        bounds[i, 1] = x_i + 0.1 * abs(x_i)
    
    # ============ Setup ============
    
    print(f"Loading mgrid from: {mgrid_filename}")
    if not mgrid_filename.exists():
        print(f"ERROR: mgrid file not found at {mgrid_filename}")
        # print(f"Searched in: {input_dir}")
        return
    
    # Initialize magnetic field
    try:
        initialize_mgrid_field(mgrid_filename, nfp, full_torus=full_torus)
        print(f"✓ Magnetic field initialized (nfp={nfp}, full_torus={full_torus})")
    except Exception as e:
        print(f"ERROR initializing field: {e}")
        return
    
    # ============ Optimization Config ============
    
    config = OptimizationConfig(
        n_dim=n_dim,
        n_pop=4,  # Minimum population size for DE/rand/1
        max_gen=1,  # Generations (reduced for demo)
        F=0.5,
        CR=0.7,
        processes=4,
        timeout_find_axis=0.5,
        timeout_ripple=10.0,
        nturn=100,
        nphi=360,
    )
    
    # ============ Setup Objective Function ============
    
    objective_func = ObjectiveFunction(
        nfp=nfp,
        initial_rz=initial_rz,
        deltaR=0.08,
        mpol=100,
        axis0=axis0,
        config=config,
        seed_extcur=extcur_initial,
        output_dir=output_dir,
    )
    
    print(f"✓ Objective function configured")
    print(f"  - Search dimensions: {n_dim}")
    print(f"  - Bounds: {bounds}")
    print(f"  - Population: {config.n_pop}")
    print(f"  - Generations: {config.max_gen}")
    
    # ============ Run DE Algorithm ============
    
    print("\n" + "=" * 60)
    print("Running Differential Evolution optimization...")
    print("=" * 60)
    
    try:
        best_individual, best_fitness = differential_evolution(
            objective_func=objective_func,
            bounds=bounds,
            config=config,
            seed_individual=extcur_initial[1:],
        )
        
        # ============ Results ============
        
        print("\n" + "=" * 60)
        print("OPTIMIZATION COMPLETED")
        print("=" * 60)
        
        print(f"\n✓ Best Solution Found:")
        print(f"  Objective (epsilon_eff): {best_fitness:.6e}")
        print(f"  Optimized extcur[1:5]: {best_individual}")
        
        # Reconstruct full extcur
        extcur_optimal = extcur_initial.copy()
        extcur_optimal[1:5] = best_individual
        print(f"  Full extcur: {extcur_optimal}")
        
        # Save results
        result_file = output_dir / "optimization_result.txt"
        with open(result_file, 'w') as f:
            f.write("Differential Evolution Optimization Results\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Best Objective (epsilon_eff): {best_fitness:.6e}\n")
            f.write(f"Optimized extcur[1:5]:\n")
            for i, val in enumerate(best_individual):
                f.write(f"  extcur[{i+1}] = {val:.6e}\n")
            f.write(f"\nFull extcur:\n")
            for i, val in enumerate(extcur_optimal):
                f.write(f"  extcur[{i}] = {val:.6e}\n")
        
        print(f"\n✓ Results saved to: {result_file}")
        print(f"✓ Detailed logs in: {output_dir}")
        
    except Exception as e:
        print(f"\nERROR during optimization: {e}")
        import traceback
        traceback.print_exc()
        return
    
    return best_individual, best_fitness, extcur_optimal


if __name__ == "__main__":
    result = run_optimization_example()
    
    if result is None:
        sys.exit(1)
    else:
        best_ind, best_fit, extcur_opt = result
        print(f"\nFinal result: epsilon_eff = {best_fit:.6e}")
