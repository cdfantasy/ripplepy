"""
ripplepy - Python interface for effective ripple calculation
"""

# Import from Fortran extension
try:
    from . import _effective_ripple
except ImportError:
    raise ImportError("Fortran extension not built. Run: pip install -e .")

# Import Python modules
from .mgrid import MGrid
from .ripple import (
    compute_epstot,
    fieldline_smoothness_poincare,
    compute_initial_gradpsi_nemov,
    find_axis,
    find_axis_multi_guess,
    get_bfield_matrix,
    initialize_mgrid_field,
    plot_fieldline_3d,
    set_extcur,
    set_trace_parameters,
    trace_fieldline,
    calculate_plasma_params,
    set_trace_verbose,
    get_trace_verbose,
)
from .optimize import (
    OptimizationConfig,
    StellaratorObjective,
    DifferentialEvolution,
    run as run_optimization,
)

__version__ = "0.1.0"
__all__ = [
    "MGrid",
    "compute_epstot",
    "fieldline_smoothness_poincare",
    "initialize_mgrid_field",
    "set_extcur",
    "get_bfield_matrix",
    "trace_fieldline",
    "plot_fieldline_3d",
    "compute_initial_gradpsi_nemov",
    "find_axis",
    "find_axis_multi_guess",
    "set_trace_parameters",
    "calculate_plasma_params",
    "OptimizationConfig",
    "StellaratorObjective",
    "DifferentialEvolution",
    "run_optimization",
    "set_trace_verbose",
    "get_trace_verbose",
]
