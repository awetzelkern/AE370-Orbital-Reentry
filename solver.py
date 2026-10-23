"""Contains functions to solve our equations of motion"""
import constants as const
import equations_of_motion as eom
from scipy.differentiate import jacobian
import initial_conditions as init
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, brentq

from atmospheric_model import get_rho

drag_force = eom.F_D(rho=get_rho(init.r), V=init.v)
lift_force = eom.F_L(rho=get_rho(init.r), V=init.v)

J = jacobian(
    lambda state: eom.f(state, drag_force, lift_force, const.m_capsule, const.omega_earth, const.nu),
    init.state,
).df
print(J)


eigvals, eigvecs = np.linalg.eig(J)
stiffness_ratio = np.max(np.abs(eigvals)) / np.min(np.abs(eigvals))

R_s = np.max(np.abs(eigvals))
dt_limit = 2.0 / R_s

print("Eigenvalues:", eigvals)
print("Min magnitude:", np.min(np.abs(eigvals)))
print("Max magnitude:", np.max(np.abs(eigvals)))

print(stiffness_ratio, R_s, dt_limit)

def forward_euler(f, u_start, t_end, dt):
    """March u' = f(u, t) from t = 0 to t_end with forward Euler (Lecture 10).

    Returns
    -------
    u : ndarray, shape (n + 1, len(u_start)); row k is the state at t_k
    times : ndarray, the n + 1 times t_k
    """
    n = int(round(t_end / dt))
    u = np.empty((n + 1, np.size(u_start)))
    u[0] = u_start
    for k in range(n):
        u[k + 1] = u[k] + dt * f(u[k], k * dt)
    return u, np.arange(n + 1) * dt

def bdf2(f, u_start, t_end, dt):
    '''March u' = f(u, t) from t = 0 to t_end with BDF2, started with one trapezoid step.

    Returns
    -------
    u : ndarray, shape (n + 1, len(u_start)); row k is the state at t_k
    times : ndarray, the n + 1 times t_k
    '''
    ### YOUR CODE HERE ###
    n = int(round(t_end/dt))
    u = np.empty((n+1, np.size(u_start)))
    u[0] = u_start
    trap_residual = lambda u1: u1 - u[0] - (dt / 2) * (f(u[0], 0.0) + f(u1, dt))
    u[1] = fsolve(trap_residual, u[0])

    for k in range(1,n):
        bdf2_residual = lambda z: 3 * z - 4*u[k] + u[k-1] - 2*dt * f(z, (k+1)*dt)
        guess = u[k]
        u[k+1] = fsolve(bdf2_residual, guess, xtol=1e-13)

    times = np.arange(n+1) * dt
    return u, times


def heun(f, w_start, t_end, dt):
    """March w' = f(w, t) from t = 0 to t_end with Heun's method.

    Parameters
    ----------
    f : callable
        The dynamics, f(w, t) -> ndarray
    w_start : ndarray
        The state at t = 0
    t_end : float
        Final time
    dt : float
        Time step

    Returns
    -------
    w : ndarray
        Shape (n + 1, len(w_start)); row k is the state at t_k
    times : ndarray
        The n + 1 times t_k
    """
    ### YOUR CODE HERE ###
    n = int(round(t_end / dt))
    w = np.empty((n + 1, np.size(w_start)))
    w[0] = w_start
    for k in range(n):
        w[k+1] = w[k] + (dt/2) * (f(w[k], k*dt) + f(w[k] + dt*f(w[k], k*dt), (k+1)*dt))
    return w, np.arange(n + 1) * dt

def ab3(f, w_start, t_end, dt, starter=heun):
    """March w' = f(w, t) from t = 0 to t_end with the AB3 method.

    Parameters
    ----------
    f : callable
        The dynamics, f(w, t) -> ndarray
    w_start : ndarray
        The state at t = 0
    t_end : float
        Final time
    dt : float
        Time step
    starter : callable
        A marcher with this same signature, used to produce the values AB3
        needs before its first step

    Returns
    -------
    w : ndarray
        Shape (n + 1, len(w_start)); row k is the state at t_k
    times : ndarray
        The n + 1 times t_k
    """

    ### YOUR CODE HERE ###
    n = int(round(t_end / dt))
    w = np.empty((n + 1, np.size(w_start)))
    w[:3] = starter(f, w_start, 2*dt, dt)[0]

    f_k_2 = f(w[0],0)
    f_k_1 = f(w[1],dt)
    for k in range(2, n):
        f_new = f(w[k], k*dt)
        w[k+1] = w[k] + dt*((5/12)*f_k_2 - (4/3)*f_k_1 + 23/12*f_new)

        f_k_2 = f_k_1
        f_k_1 = f_new

    return w, np.arange(n + 1) * dt