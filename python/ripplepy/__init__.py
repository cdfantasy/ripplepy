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
    compute_initial_gradpsi_nemov,
    find_axis,
    get_bfield_matrix,
    initialize_mgrid_field,
    plot_fieldline_3d,
    set_extcur,
    set_trace_parameters,
    trace_fieldline,
    compute_kg_cylindrical,
    compute_effective_ripple
)

__version__ = "0.1.0"
__all__ = [
    "MGrid",
    "initialize_mgrid_field",
    "set_extcur",
    "get_bfield_matrix",
    "trace_fieldline",
    "plot_fieldline_3d",
    "compute_initial_gradpsi_nemov",
    "find_axis",
    "set_trace_parameters",
    "compute_kg_cylindrical",
    "compute_effective_ripple"
]
