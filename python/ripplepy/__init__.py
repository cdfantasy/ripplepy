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
from .ripple import get_bfield_matrix, initialize_mgrid_field

__version__ = "0.1.0"
__all__ = ["MGrid", "initialize_mgrid_field", "get_bfield_matrix"]
