import os
import numpy as np

# COESA 1976 needs no Earth-orientation data; skip pyatmos's optional download.
os.environ.setdefault("ENABLE_IERS_LOAD", "false")

from pyatmos import coesa76
from constants import r_earth

def get_rho(r: float | np.ndarray) -> float | np.ndarray:
    """Return density in kg/m^3 for Earth-centered radii in km.

    Array inputs retain their shape, including batches passed by jacobian.
    """
    alt = np.asarray(r) - r_earth  # geometric altitude in km
    rho = coesa76(alt.reshape(-1)).rho.reshape(alt.shape)
    return rho.item() if rho.ndim == 0 else rho
