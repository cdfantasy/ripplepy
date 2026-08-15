# ripplepy

Python package for computing the **effective ripple** ε_eff^(3/2) — the `1/ν` neoclassical-transport proxy — of a stellarator magnetic field, backed by a Fortran 90 core.

- Real-space field-line tracing with the Netlib [DLSODE](https://computing.llnl.gov/projects/odepack) stiff-ODE integrator
- Cubic Hermite (PPLIB) interpolation of VMEC `mgrid` coil fields
- Geodesic curvature + a pyneo-style η "state-machine" integrator for ε_eff^(3/2)
- Coil-current optimisation via Differential Evolution

## Requirements

**Build time**

- a Fortran compiler (gfortran recommended)
- CMake ≥ 3.18, Ninja, Meson
- [scikit-build-core](https://scikit-build-core.readthedocs.io/) ≥ 0.10
- [f90wrap](https://github.com/jameskermode/f90wrap) ≥ 0.2
- NumPy

**Runtime**

- Python ≥ 3.8
- NumPy ≥ 1.19, SciPy ≥ 1.5, f90wrap ≥ 0.2
- h5py (used by `ripplepy.optimize`)
- optional: `plotly` for 3-D field-line plots, `matplotlib` for 2-D mgrid plots

## Installation

The easiest path uses the provided Conda environment, which pins the Fortran/build toolchain:

~~~bash
conda env create -f environment.yml
conda activate ripplepy
pip install -e .
~~~

Or, with an existing toolchain, install directly:

~~~bash
pip install -e .
~~~

This compiles the Fortran backend (f90wrap + f2py) into the `ripplepy._effective_ripple` extension. Importing `ripplepy` without a built extension raises an `ImportError`; `pip install -e .` is the documented way to build it.

## Quick start

~~~python
from ripplepy import (
    initialize_mgrid_field,
    set_extcur,
    set_trace_parameters,
    find_axis,
    compute_epstot,
)

# 1. Load and initialise the magnetic field from a VMEC mgrid NetCDF file.
initialize_mgrid_field("mgrid.nc", nfp=2, full_torus=False)

# 2. Set the external coil currents.
set_extcur([50000.0, 5000.0, 2000.0, -80000.0, -40000.0])

# 3. Locate the magnetic axis.
axis_rz, R0, axis_fieldline, ok = find_axis((1.2, 0.0))
print(f"axis: R={axis_rz[0]:.4f}, Z={axis_rz[1]:.4f}, R0={R0:.4f}")

# 4. Configure tracing and evaluate ε_eff^(3/2) on a surface offset from the axis.
set_trace_parameters(nturn=200, nphi=360, npart=5000)
eps_eff_32, Bboundary, fieldline_data, istate = compute_epstot(
    [axis_rz[0] + 0.05, axis_rz[1]], return_fieldline=True
)
print(f"epsilon_eff^(3/2) = {eps_eff_32:.6e}")
~~~

## API overview

The public API is exported from `ripplepy`.

### Field setup and tracing

| Function | Purpose |
| --- | --- |
| `initialize_mgrid_field(mgrid, nfp, full_torus=True)` | Load an mgrid NetCDF file and initialise the Fortran field |
| `set_extcur(extcur)` | Set the external coil currents |
| `get_bfield_matrix(extcur, r, z, phi)` | Interpolate B (and derivatives) at points |
| `set_trace_parameters(nturn, nphi, npart=5000)` | Set field-line tracing resolution |
| `trace_fieldline(initial_rz, ...)` | Trace a single field line |
| `find_axis(initial_rz, ...)` | Locate the magnetic axis by root-finding |
| `compute_epstot(initial_rz, ...)` | Compute ε_eff^(3/2) |
| `compute_initial_gradpsi_nemov(...)` | Estimate the initial ∇ψ direction (Nemov convention) |
| `calculate_plasma_params(...)` | Volume, minor radius, and |ι| from a trace |
| `plot_fieldline_3d(fieldline_data, ...)` | 3-D field-line plot (plotly) |

### mgrid I/O

`ripplepy.MGrid` reads/writes VMEC `mgrid` NetCDF files (cylindrical B components on a tensor-product grid), with coil-group handling, `extcur` scaling, and full-torus expansion:

~~~python
from ripplepy import MGrid

mgrid = MGrid.from_file("mgrid.nc")
mgrid.expand_to_full_torus(nfp=2, full_torus=False)
~~~

### Optimisation

`ripplepy.optimize` provides a Differential-Evolution optimiser for coil currents that minimises ε_eff^(3/2). The full evaluation chain — `set_extcur → find_axis → trace → compute_epstot → plasma params` — is wrapped in `StellaratorObjective`, with multiprocessing and CSV summary output. Field-line data is not persisted during the run; selected individuals can be re-evaluated afterwards (e.g. with the `save_hdf5` utility) for HDF5 output.

~~~python
import numpy as np
from ripplepy import OptimizationConfig, run_optimization

config = OptimizationConfig(
    mgrid_path="mgrid.nc",
    nfp=3,
    initial_rz=(1.26, 0.0),
    # Each row is [nominal_current, fraction]; fraction=0 locks that coil.
    initial_bounds=np.array([
        [ 50000.0, 0.0 ],   # fixed
        [  5000.0, 0.5 ],
        [  2000.0, 0.5 ],
        [-80000.0, 0.5 ],
        [-40000.0, 0.5 ],
    ]),
    full_torus=False,
    n_pop=50,
    max_gen=100,
    processes=8,
    output_dir="results",
)

best_extcur, best_eps, all_infos = run_optimization(config)
print("Best extcur:", best_extcur)
print("Best eps_eff^(3/2):", best_eps)
~~~

## Project layout

~~~
ripplepy/
├── fortran/                 # Fortran 90 backend
│   ├── ripple.f90           #   effective_ripple module (trace + ε_eff^(3/2))
│   ├── DLSODE.f             #   vendored Netlib ODE integrator
│   ├── hybrd.f              #   vendored Netlib root-finder
│   └── pspline/             #   vendored PPLIB spline library
├── python/ripplepy/         # Python package
│   ├── __init__.py          #   public API
│   ├── ripple.py            #   thin wrapper around the Fortran backend
│   ├── mgrid.py             #   mgrid NetCDF I/O (MGrid)
│   ├── optimize.py          #   differential-evolution coil optimisation
│   └── boozer_eps_verify.py #   pure-Python ε_eff cross-check (dev only)
├── tests/                   # tests, benchmarks, and input data
├── pyproject.toml           # scikit-build-core build definition
├── CMakeLists.txt           # top-level CMake project
└── environment.yml          # Conda environment
~~~

## Development notes

- The core numerical chain is `compute_ripple → trace_gradpsi_internal → geodesic_curvature_internal → effective_ripple_pyneo` in `fortran/ripple.f90`.
- `python/ripplepy/boozer_eps_verify.py` is a development/verification module (pure-Python reimplementations used to cross-check against [pyneo](https://github.com/landreman/pyneo)); it is **not** part of the public API.
- Several scripts under `tests/` (the `bench_*.py`, `boozerxform*.py`, `h1_optimise.py`, `plot_opt_result.py`) are development/benchmark helpers rather than automated tests, and require `simsopt`/`neo`.

## License

MIT
