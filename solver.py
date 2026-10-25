"""Contains functions to solve our equations of motion"""
import constants as const
import equations_of_motion as eom
from scipy.differentiate import jacobian
import initial_conditions as init
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, brentq

# J = jacobian(
#     lambda state: eom.f(state, 0.0),
#     init.state,
# ).df
# print(J)

def jac_scipy(f, state, t=0.0):
    res = jacobian(lambda x: f(x, t), np.asarray(state, dtype=float))
    return res.df   # shape (6, 6)

J_init = jac_scipy(eom.f, init.state, 0)

pre_deployment_state = np.array([6368001,  1.5490282085199362e+02, -1.2283937305619541e+00,  1.2218222139501493e+00,
  5.9195644814515047e-01, -2.6818091457096522e+00])
J_threshold = jac_scipy(eom.f, pre_deployment_state, 0)

eigvals_init, eigvecs_init = np.linalg.eig(J_init)
stiffness_ratio_init = np.max(np.abs(eigvals_init)) / np.min(np.abs(eigvals_init))

eigvals_thresh, eigvecs_thresh = np.linalg.eig(J_threshold)
stiffness_ratio_thresh = np.max(np.abs(eigvals_thresh)) / np.min(np.abs(eigvals_thresh))

R_s_init = np.max(np.abs(eigvals_init))
decaying_init = eigvals_init[eigvals_init.real < -1e-12]

R_s_thresh = np.max(np.abs(eigvals_thresh))
decaying_thresh = eigvals_thresh[eigvals_thresh.real < 1e-12]

dt_limit_init = (
    np.min(-2 * decaying_init.real / np.abs(decaying_init)**2)
    if decaying_init.size else np.inf
)

dt_limit_thresh = (
    np.min(-2 * decaying_thresh.real / np.abs(decaying_thresh)**2)
    if decaying_init.size else np.inf
)

print("Initial State Stuff")
print("Eigenvalues:", eigvals_init)
print("Min magnitude:", np.min(np.abs(eigvals_init)))
print("Max magnitude:", np.max(np.abs(eigvals_init)))
print(stiffness_ratio_init, R_s_init, dt_limit_init)

print("Threshold State Stuff")
print("Eigenvalues:", eigvals_thresh)
print("Min magnitude:", np.min(np.abs(eigvals_thresh)))
print("Max magnitude:", np.max(np.abs(eigvals_thresh)))
print(stiffness_ratio_thresh, R_s_thresh, dt_limit_thresh)

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
