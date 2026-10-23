import os

# COESA 1976 needs no Earth-orientation data; skip pyatmos's optional download.
os.environ.setdefault("ENABLE_IERS_LOAD", "false")

from pyatmos import coesa76
from constants import r_earth

def get_rho(r: float) -> float:
    alt = r - r_earth  # geometric altitude in km
    return coesa76(alt).rho.item()  # density in kg/m^3
