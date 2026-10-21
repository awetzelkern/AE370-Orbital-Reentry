import numpy as np
from constants import G, m_earth

""" 
r = Distance from Earth center
V = Vehicle velocity
gamma = Flight path angle
psi = Azimuth angle
phi = Latitude
theta = Longitude
F_D = Aerodynamic drag force
F_L = Aerodynamic lift force
"""

def F_L(rho: float, V: float, C_L: float, A: float) -> float:
    return 0.5 * rho * V**2 * C_L * A

def F_D(rho: float, V: float, C_D: float, A: float) -> float:
    return 0.5 * rho * V**2 * C_D * A

def A(r_b: float) -> float:
    return np.pi * r_b**2

def beta(m: float, C_D: float, A: float) -> float:
    return m/(C_D*A)

def r_dot(V: float, gamma: float) -> float:
    return V * np.sin(gamma)


def V_dot(
        F_d: float, 
        m: float, 
        g: float, 
        gamma: float, 
        omega: float, 
        r: float, 
        phi: float, 
        psi: float
    ) -> float:
    return -(F_d/m)-g*np.sin(gamma)+(omega**2)*r*np.cos(phi)*(np.sin(gamma)*np.cos(phi)-np.cos(gamma)*np.sin(phi)*np.sin(psi))
    

def gamma_dot(
        F_l: float, 
        m: float, 
        V: float, 
        g: float, 
        gamma: float, 
        omega: float, 
        r: float, 
        phi: float, 
        psi: float, 
        nu: float
    ) -> float:
    return (1/V)(F_l*np.cos(nu)/m - g*np.cos(gamma) + omega**2 * r * np.cos(phi)*(np.cos(gamma)*np.cos(phi) + np.sin(gamma)*np.sin(phi)*np.sin(psi)) + (V**2 * np.cos(gamma)/r) + 2* omega*V*np.cos(phi)*np.cos(psi))

def psi_dot(
        V: float, 
        F_L: float, 
        m: float, 
        gamma: float, 
        omega: float, 
        r: float, 
        phi: float, 
        psi: float,
        nu: float
    ) -> float:
    return (1/V) * (F_L * np.sin(nu) / (m*np.cos(gamma)) - (V**2/r) * (np.cos(gamma)*np.cos(psi)*np.tan(phi)) + 2*omega*V*(np.tan(gamma)*np.cos(phi)*np.sin(psi) - np.sin(phi)) - (omega**2*r)/(np.cos(gamma))*np.sin(phi)*np.cos(phi)*np.cos(psi))

def phi_dot (V: float, gamma: float, psi: float, r: float) -> float:
    return V*np.cos(gamma)*np.sin(psi)/r

def theta_dot (V: float, gamma: float, psi: float, r: float, phi: float) -> float:
    return V*np.cos(gamma)*np.cos(psi)/(r*np.cos(phi))

def g(r: float) -> float:
    return G*m_earth/(r**2)

