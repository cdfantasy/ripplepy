# ripplepy

Python package for evaluating the **effective ripple** $\varepsilon_{\mathrm{eff}}^{3/2}$ — the
$1/\nu$ neoclassical-transport proxy of a stellarator — directly in a three-dimensional
vacuum coil field, with a Fortran 90 backend.

The field is taken from a VMEC `mgrid` NetCDF file and interpolated with cubic Hermite
(PPLIB) splines, so no equilibrium reconstruction and no Boozer transformation are needed.
Field lines are traced in real space with the Netlib DLSODE stiff-ODE integrator, with
$\nabla\psi$ transported along the line; the effective ripple follows from the Nemov
integral, and the geodesic curvature entering it is evaluated in cylindrical coordinates.

Built on that evaluation chain, the package also provides tools for magnetic-axis searches,
radial surface scans (geometry, rotational transform and $\varepsilon_{\mathrm{eff}}^{3/2}$
on every surface), and coil-current optimisation.

> **Status:** research software under active development; the API may change between
> releases.

## Features

- **Real-space evaluation of $\varepsilon_{\mathrm{eff}}^{3/2}$** in the vacuum coil field
  (`mgrid`), without VMEC or a Boozer-coordinate transformation.
- **Cubic Hermite interpolation** of the coil field (`(B_R, B_\phi, B_Z)` on a cylindrical
  grid), with analytic derivatives supplied to the integrator.
- **Magnetic-axis search** by one-turn return-map root finding, with residual-ranked
  multi-guess and multi-branch support (island-centre fixed points are not mistaken for
  the main axis).
- **Radial scans**: outward walk from the axis to the last closed flux surface, with
  per-surface geometry ($a$, $V$, $\iota$, elongation, ripple) and optional
  $\varepsilon_{\mathrm{eff}}^{3/2}$ on every surface. Two resolution tiers (a cheap
  walk tier and a full-resolution tier), configurable regularity gates, multiprocessing
  and HDF5 output.
- **Coil-current optimisation** via Differential Evolution, with multiprocessing, a
  feasibility survey and CSV/HDF5 summaries.
- **`mgrid` I/O** (read/write, coil-group handling, `extcur` scaling, full-torus
  expansion) and publication-style plotting helpers.

## Requirements

**Build time**

- a Fortran compiler (gfortran recommended)
- CMake ≥ 3.18, Ninja, Meson
- [scikit-build-core](https://scikit-build-core.readthedocs.io/) ≥ 0.10
- [f90wrap](https://github.com/jameskermode/f90wrap) ≥ 0.2
- NumPy

**Runtime**

- Python ≥ 3.8 (tested with 3.10)
- NumPy ≥ 1.19, SciPy ≥ 1.5, f90wrap ≥ 0.2, h5py ≥ 3.0
- optional: `matplotlib` for the plotting helpers, `plotly` for 3-D field-line plots

## Installation

The easiest path uses the provided Conda environment, which pins the Fortran/build
toolchain:

```bash
conda env create -f environment.yml
conda activate ripplepy
pip install -e .
```

With an existing toolchain, an editable install into any environment is enough:

```bash
pip install -e .
```

The editable install is configured to build **in place** (`tool.scikit-build.editable`
in `pyproject.toml`): imports read the Python sources in `python/ripplepy/` directly, so
edits to Python code take effect immediately. The compiled Fortran extension is copied
into `python/ripplepy/` as well. After changing Fortran sources, re-run the install to
rebuild:

```bash
pip install -e .
```

Check which copy of the package is in use:

```bash
python -c "import ripplepy; print(ripplepy.__file__)"
# editable install -> .../ripplepy/python/ripplepy/__init__.py
```

A non-editable install is also supported:

```bash
pip install .
```

## Quick start

### Effective ripple on one surface

```python
from ripplepy import (
    initialize_mgrid_field,
    set_extcur,
    find_axis,
    set_trace_parameters,
    compute_initial_gradpsi_nemov,
    compute_epstot,
)

# 1. Load the coil field from a VMEC mgrid file.
initialize_mgrid_field("mgrid.nc", nfp=3, full_torus=False)

# 2. Set the coil currents (one value per coil group).
extcur = [50000.0, 5000.0, 2000.0, -80000.0, -40000.0]
set_extcur(extcur)

# 3. Locate the magnetic axis.
axis_rz, R0, axis_fieldline, ok = find_axis((1.2, 0.0))

# 4. Trace a surface offset radially from the axis and evaluate eps_eff^(3/2).
set_trace_parameters(nturn=400, nphi=360, npart=2000)
start = [axis_rz[0] + 0.05, axis_rz[1]]
gradpsi = compute_initial_gradpsi_nemov(extcur, start[0], start[1])
eps_eff_32, _, fieldline, istate = compute_epstot(
    start, initial_gradpsi=gradpsi, return_fieldline=True
)
print(f"epsilon_eff^(3/2) = {eps_eff_32:.6e}")
```

`nturn` is the number of toroidal turns traced (convergence of the field-line integral),
`nphi` the number of points per turn, and `npart` the number of pitch-angle grid points in
the bounce averages of the Nemov integral.

### Radial scan of one configuration

```python
from ripplepy.radial_scan import ScanConfig, scan_config

cfg = ScanConfig(
    mgrid_path="mgrid.nc", nfp=3, nphi=360,
    walk_eps=True,              # evaluate eps on every walked surface
    profile_nturn=200, profile_npart=500,   # walk tier
    full_nturn=400, full_npart=2000,        # full tier
    dr0=0.02, dr_step=0.02, dr_max=0.60,
)
result = scan_config(extcur, cfg)          # one configuration
for row in result["rows"]:
    print(row["dr"], row["a"], row["iota"], row["eps"], row["status"])
```

To scan a box of coil currents (or an explicit list of current vectors), use
`scan_box` / `scan_all_box` from `ripplepy.radial_scan`, which also write the results to an
HDF5 file. The scripts under `tests/` are complete drivers for such scans.

### Coil-current optimisation

```python
import numpy as np
from ripplepy import OptimizationConfig, run_optimization

config = OptimizationConfig(
    mgrid_path="mgrid.nc",
    nfp=3,
    initial_rz=(1.26, 0.0),
    # rows are [nominal current, fraction]; fraction=0 locks the coil group
    initial_bounds=np.array([
        [ 50000.0, 0.0],
        [  5000.0, 0.5],
        [  2000.0, 0.5],
        [-80000.0, 0.5],
        [-40000.0, 0.5],
    ]),
    full_torus=False,
    n_pop=64,
    max_gen=100,
    processes=8,
    output_dir="results",
)

best_extcur, best_eps, all_infos = run_optimization(config)
```

## API overview

### Field setup and tracing (`ripplepy`)

| Function | Purpose |
| --- | --- |
| `initialize_mgrid_field(mgrid, nfp, full_torus=True)` | Load an `mgrid` file and initialise the Fortran field |
| `set_extcur(extcur)` | Set the external coil currents |
| `get_bfield_matrix(extcur, r, z, phi)` | Interpolate $\bm{B}$ and its derivatives at points |
| `set_trace_parameters(nturn, nphi, npart=5000)` | Set the field-line/quadrature resolution |
| `trace_fieldline(initial_rz, ...)` | Trace a single field line |
| `find_axis(initial_rz, ...)` | Locate a magnetic axis by root finding |
| `find_axis_any(...)`, `find_axis_multi_guess(...)` | Scan a radial window for one / all axes |
| `compute_initial_gradpsi_nemov(...)` | Initial $\nabla\psi$ direction (Nemov convention) |
| `compute_epstot(initial_rz, ...)` | Compute $\varepsilon_{\mathrm{eff}}^{3/2}$ |
| `calculate_plasma_params(...)` | Volume, volume-equivalent minor radius, $\iota$ from a trace |
| `fieldline_smoothness_poincare(...)` | Regularity diagnostics of a Poincaré section |
| `plot_fieldline_3d(...)` | 3-D field-line plot (plotly) |

### Radial scans (`ripplepy.radial_scan`)

| Object | Purpose |
| --- | --- |
| `ScanConfig` | Resolution tiers, radial walk, gates and axis-search settings |
| `measure_surface(...)` | Trace one surface and measure geometry and (optionally) $\varepsilon_{\mathrm{eff}}^{3/2}$ |
| `scan_config(extcur, cfg)` | Axis search + outward walk for one current vector |
| `scan_box(...)` | Sample a current box (or take explicit samples) and scan each |
| `scan_all_box(...)` | One-pass survey: axis + walk + per-surface $\varepsilon$ for every sample |
| `save_scan_h5(...)`, `load_scan_h5(...)` | HDF5 I/O for scan results |
| `sample_bounds(...)` | Sobol sampling inside a box |

### `mgrid` I/O

```python
from ripplepy import MGrid

mgrid = MGrid.from_file("mgrid.nc")   # read
mgrid.apply_extcur(extcur)            # scale coil fields by the currents
mgrid.expand_to_full_torus(nfp=3, full_torus=False)
mgrid.write("mgrid_out.nc")           # write
```

### Optimisation (`ripplepy.optimize`)

| Object | Purpose |
| --- | --- |
| `OptimizationConfig` | Coil bounds, resolution, population, outputs |
| `StellaratorObjective` | Full evaluation chain wrapped as the objective |
| `DifferentialEvolution` | Differential-Evolution optimiser |
| `run_optimization` | Convenience wrapper (alias of `optimize.run`) |

### Plotting (`ripplepy.plotting`)

`setup_publication_style()`, `save_figure(fig, path_stem)` and the Okabe–Ito palette
`PUB_COLORS` provide a headless-safe matplotlib style and PDF/PNG export. Requires the
optional `matplotlib` dependency.

## Project layout

```text
ripplepy/
├── fortran/                 # Fortran 90 backend
│   ├── ripple.f90           #   effective_ripple module (tracing + eps_eff^(3/2))
│   ├── DLSODE.f             #   vendored Netlib ODE integrator
│   ├── hybrd.f              #   vendored Netlib root finder
│   └── pspline/             #   vendored PPLIB spline library
├── python/
│   ├── CMakeLists.txt       #   f90wrap + f2py build of the extension
│   └── ripplepy/            # Python package
│       ├── __init__.py      #   public API
│       ├── ripple.py        #   wrapper around the Fortran backend
│       ├── radial_scan.py   #   axis search, radial walk, per-surface measurements
│       ├── mgrid.py         #   mgrid NetCDF I/O (MGrid)
│       ├── optimize.py      #   differential-evolution coil optimisation
│       ├── plotting.py      #   publication-style matplotlib helpers
│       └── boozer_eps_verify.py  # dev-only Eulerian cross-check
├── tests/                   # development drivers, benchmarks and figures scripts
├── pyproject.toml           # scikit-build-core build definition
├── CMakeLists.txt           # top-level CMake project
└── environment.yml          # Conda environment
```

## Development notes

- **Build chain:** f90wrap generates a Fortran interface, f2py compiles it against the
  static Fortran library, and the result is installed as `ripplepy._effective_ripple`.
  The top-level CMake project keeps all generated files out of the wheel except the
  extension and the generated wrapper module.
- **Rebuilding:** for an editable install, `pip install -e .` rebuilds the extension after
  Fortran changes and refreshes the in-tree artefacts. Python-only edits need no rebuild.
- **Input data:** the repository does not ship `mgrid` files or VMEC `wout` files. The
  scripts under `tests/` expect them locally; most are development drivers or benchmarks
  (the `bench_*.py` scripts additionally require `simsopt` and `neo`).
- **Numerical conventions:** $\varepsilon_{\mathrm{eff}}^{3/2}$ is evaluated only on
  certified regular surfaces; the walk stops at the first surface that fails the
  smoothness/gap/geometry gates, followed by one bisection to locate the last closed
  surface. The cheap walk tier is meant for trends and ranking; absolute values should be
  taken from the full tier.
- **Cross-checks:** `boozer_eps_verify.py` is a development-only pure-Python
  reimplementation used to cross-check the backend against an independent integrator
  (including comparisons with the `pyneo`/NEO implementation); it is not part of the
  public API.

## License

MIT
