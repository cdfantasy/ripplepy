# ripplepy

## Test Version Notice

This repository is currently a **test version** intended for method development,
numerical experimentation, and workflow validation.
It is **not production-ready** and should not be used as a validated engineering tool
without independent verification.

## Abstract

ripplepy provides a Python-Fortran workflow for computing effective ripple metrics
from magnetic-field-line data. The package combines:

- A compiled Fortran backend for field interpolation and field-line tracing
- A Python post-processing layer for geodesic-curvature and ripple diagnostics
- Hybrid numerical quadrature tailored to turning-point singular behavior

The current implementation follows a Nemov-style formulation, including
the geodesic curvature expression and the nested integrals for effective ripple.

## Scientific Scope

The present code is designed for exploratory studies of non-axisymmetric field effects,
with emphasis on the computation of an effective ripple proxy from traced trajectories.

Implemented model pathway:

1. Trace field lines on a precomputed magnetic grid.
2. Evaluate local geometric quantities and field derivatives.
3. Compute geodesic curvature term $k_g$.
4. Assemble well-wise integrals and outer $b'$ integration.
5. Return an effective ripple quantity scaled by major radius.

## Numerical Methodology

The integration strategy in the Python post-processing layer uses a hybrid approach:

1. Core smooth intervals: composite Newton-Cotes (Simpson rule).
2. Near turning points: Gauss-Legendre quadrature.
3. Endpoint correction: linearized asymptotic analytic compensation.

This design targets improved robustness for integrands containing square-root
endpoint behavior near trapped-particle turning points.

## Equation-to-Code Mapping

The current implementation maps literature equations to code-level operations as follows:

- Geodesic curvature term: Eq. (17)
- Effective ripple assembly: Eq. (29)
- Intermediate integral $H$: Eq. (30)
- Intermediate integral $I$: Eq. (31)

Core implementation file:

- [python/ripplepy/ripple.py](python/ripplepy/ripple.py)

## Software Status

- Version: 0.1.0
- API is unstable and may change without deprecation guarantees.
- Numerical defaults and parameterization are still being tuned.
- Validation coverage is incomplete.

## Requirements

- Python >= 3.8
- CMake >= 3.18
- Ninja >= 1.10
- Meson >= 1.0
- NumPy >= 1.19
- SciPy >= 1.5
- f90wrap >= 0.2
- A Fortran compiler toolchain

## Installation (Research/Development)

From the repository root:

```bash
pip install -e .
```

If the extension build is missing, import will fail by design to prevent
silent fallback to incomplete functionality.

## Minimal Reproducible Workflow

```python
from ripplepy import MagneticField, FieldLineTracer, epsilon_eff

field = MagneticField.from_mgrid_file(
	mgrid_filename="path/to/mgrid.nc",
	extcur=[1.0],
	nfp=3,
	full_torus=True,
)

tracer = FieldLineTracer(field, nturn=2, nphi=360)
processor = epsilon_eff(field, nturn=2, nphi=360)

epsilon_value, bboundary, fieldline_data, geocur = processor.compute_ripple_python(
	initial_rz=[1.0, 0.0],
	n_b=1200,
)

print("epsilon_eff =", epsilon_value)
```

## Repository Structure

- [python/ripplepy](python/ripplepy): Python API and post-processing implementation
- [fortran](fortran): Fortran backend sources
- [tests](tests): notebooks and sample test inputs
- [examples](examples): auxiliary example resources

## Reproducibility Notes

For reproducible numerical comparisons, record at minimum:

- Grid source and resolution
- External current vector
- Trace parameters ($nturn$, $nphi$)
- Integration controls ($n_b$, Gauss order, Newton-Cotes points, endpoint fraction)
- Python/NumPy/SciPy/compiler versions

## Validation and Limitations

- This implementation is under active development and should be treated as provisional.
- Numerical agreement may depend on resolution and turning-point handling settings.
- Independent cross-checking against trusted reference workflows is recommended.

## Citation

If you use this repository for research prototypes, please cite:

1. The relevant effective-ripple literature you follow for the equations.
2. This repository (commit hash and access date) as software used in analysis.

## License

See project metadata for licensing details.
