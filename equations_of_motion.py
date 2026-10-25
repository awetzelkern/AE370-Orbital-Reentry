import numpy as np
from constants import G, m_earth, r_earth, nu, cd, cl, m_capsule, parachute_deployment_altitude, omega_earth, A_parachutes, A
from atmospheric_model import get_rho
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

def F_L(rho: float, V: float) -> float:
    return 0.5 * rho * V**2 * cl * A

def F_D(rho: float, V: float) -> float:
    return 0.5 * rho * V**2 * cd * A

def cross_sectional_area(r_b: float) -> float:
    return np.pi * r_b**2

def beta(m: float, C_D: float) -> float:
    return m/(C_D*A)

def r_dot(V: float, gamma: float) -> float:
    return V * np.sin(gamma)

def V_dot(
        F_D: float,
        m: float, 
        g: float, 
        gamma: float, 
        omega: float, 
        r: float, 
        phi: float, 
        psi: float
    ) -> float:
    return -(F_D/m)-g*np.sin(gamma)+(omega**2)*r*np.cos(phi)*(np.sin(gamma)*np.cos(phi)-np.cos(gamma)*np.sin(phi)*np.sin(psi))
    

def gamma_dot(
        F_L: float,
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
    return (1/V) * (F_L*np.cos(nu)/m - g*np.cos(gamma) + omega**2 * r * np.cos(phi)*(np.cos(gamma)*np.cos(phi) + np.sin(gamma)*np.sin(phi)*np.sin(psi)) + (V**2 * np.cos(gamma)/r) + 2* omega*V*np.cos(phi)*np.cos(psi))

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

def f(state, t) -> np.ndarray:
    r, V, gamma, psi, phi, theta = state
    gravity = g(r)
    rho = get_rho(r)
    drag_area = np.where(r - r_earth < parachute_deployment_altitude, A_parachutes, A)
    drag_force = 0.5 * rho * V**2 * drag_area * cd
    lift_force = 0.5 * rho * V**2 * A * cl
    m = m_capsule
    omega = omega_earth
    return np.array([
        r_dot(V, gamma),
        V_dot(drag_force, m, gravity, gamma, omega, r, phi, psi),
        gamma_dot(lift_force, m, V, gravity, gamma, omega, r, phi, psi, nu),
        psi_dot(V, lift_force, m, gamma, omega, r, phi, psi, nu),
        phi_dot(V, gamma, psi, r),
        theta_dot(V, gamma, psi, r, phi)
    ])

