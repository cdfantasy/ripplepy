
#!/usr/bin/env python3
"""Test: Boozer analytic field line → Fortran compute_r0 + effective_ripple_pyneo.

Computes ε_eff from Boozer Fourier harmonics using the same Fortran
η-state-machine as the mgrid pipeline.  Compares with pyneo's native result.

The field line is θ(φ)=θ₀+ι·φ, sampled analytically via Fourier summation.
|B|, |∇ψ|, κ_G, R, Z, Bφ are all evaluated from the Boozer harmonics
without any grid interpolation or field-line tracing.

Set CACHE_FIELDLINE = True to skip Fourier summation on reruns.
"""

import numpy as np
from pathlib import Path
from simsopt.mhd import Boozer, Vmec
import neo
from neo import NeoContext, neo_surfaces_from_simsopt_boozer
from ripplepy.boozer_eps_verify import (
    _boozer_obj_to_dict, _find_bmax_location, _sample_fieldline_fourier,
    eps_eff_pyneo_ode_fast,
)
from ripplepy.ripple import Effective_Ripple, set_trace_parameters

BASE = str(Path(__file__).resolve().parent.parent)

# ═══════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════

DEVICE = "CFQS"
VMEC_PATH = f"{BASE}/tests/test_file/wout_cfqs_test_m10_n5_fixed.nc"

# DEVICE = "H1"
# VMEC_PATH = f"{BASE}/tests/test_file/wout_h1_design.nc"


SURF_IDX_LIST = np.linspace(0.1, 1.0, 10)
NTURN = 100
NPHI = 200    # Fortran trapezoidal: grid pts per turn (match 4 × nstep_per)
NPART = 500
COMPARE_PYTHON = True            # True → also run Python η-state-machine + diagnostics
CACHE_FIELDLINE = False
CACHE_DIR = Path(__file__).resolve().parent / "fieldline_cache"



VMEC_PATH = f"{BASE}/tests/test_file/wout_cfqs_test_m10_n5_fixed.nc"

print("\n[1] Loading VMEC + Boozer …")
vmec = Vmec(str(VMEC_PATH))
boozer = Boozer(vmec)
boozer.mpol = 72; boozer.ntor = 36
boozer.register(SURF_IDX_LIST)
boozer.run()
booz_dict = _boozer_obj_to_dict(boozer)