from pyatmos import coesa76
from constants import r_earth

def rho(r: float) -> float:
    alt = r-r_earth
    return coesa76(alt).rho
