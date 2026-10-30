"""Contains functions to solve our equations of motion"""
import equations_of_motion as eom
from scipy.differentiate import jacobian
import initial_conditions as init
import numpy as np

def jac_scipy(f, state, t=0.0):
    res = jacobian(lambda x: f(x, t), np.asarray(state, dtype=float))
    return res.df

J_init = jac_scipy(eom.f, init.state, 0)

eigvals_init, eigvecs_init = np.linalg.eig(J_init)
stiffness_ratio_init = np.max(np.abs(eigvals_init)) / np.min(np.abs(eigvals_init))

R_s_init = np.max(np.abs(eigvals_init))
decaying_init = eigvals_init[eigvals_init.real < -1e-12]

dt_limit_init = (
    np.min(-2 * decaying_init.real / np.abs(decaying_init)**2)
    if decaying_init.size else np.inf
)

print("Initial State Stuff")
print("Eigenvalues:", eigvals_init)
print("Min magnitude:", np.min(np.abs(np.real(eigvals_init))))
print("Max magnitude:", np.max(np.abs(np.real(eigvals_init))))
print(stiffness_ratio_init, R_s_init, dt_limit_init)

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
    n = int(round(t_end / dt))
    w = np.empty((n + 1, np.size(w_start)))
    w[0] = w_start
    for k in range(n):
        w[k + 1] = w[k] + (dt / 2) * (
            f(w[k], k * dt) + f(w[k] + dt * f(w[k], k * dt), (k + 1) * dt)
        )
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

    n = int(round(t_end / dt))
    w = np.empty((n + 1, np.size(w_start)))
    w[:3] = starter(f, w_start, 2 * dt, dt)[0]

    f_k_2 = f(w[0], 0)
    f_k_1 = f(w[1], dt)
    for k in range(2, n):
        f_new = f(w[k], k * dt)
        w[k + 1] = w[k] + dt * ((5 / 12) * f_k_2 - (4 / 3) * f_k_1 + 23 / 12 * f_new)

        f_k_2 = f_k_1
        f_k_1 = f_new

    return w, np.arange(n + 1) * dt
