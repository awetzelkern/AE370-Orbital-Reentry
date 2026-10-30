"""Save the five AB3 checkpoint stability plots as a single PNG.

Run ``python3 stability_region.py`` to create
``figures/stability/ab3_jacobian_checkpoints.png``. Checkpoints come from a
BDF reference trajectory sampled every second. The plotted limits describe
local AB3 decay stability. Each region shows eigenvalues scaled by the
smallest checkpoint timestep limit.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.differentiate import jacobian
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks

BETA_AB3 = np.array([5 / 12, -4 / 3, 23 / 12])


def ab_coeffs(w, beta=BETA_AB3):
    """Coefficients of zeta^r-zeta^(r-1)-w*sum(beta_i*zeta^i)."""
    r = len(beta)
    coeffs = np.zeros(r + 1, dtype=complex)
    coeffs[0], coeffs[1] = 1, -1
    for i, b in enumerate(beta):
        coeffs[r - i] -= w * b
    return coeffs


def is_stable(w, beta=BETA_AB3, tol=1e-10):
    """Root condition: roots inside/on unit circle, unit roots simple.

    Includes the neutral origin and stable boundary, with floating-point tolerance.
    """
    roots = np.roots(ab_coeffs(w, beta))
    if np.any(np.abs(roots) > 1 + tol):
        return False
    unit = roots[np.abs(np.abs(roots) - 1) <= tol]
    return all(abs(a - b) > 1e-7 for i, a in enumerate(unit) for b in unit[i + 1:])


def stability_region(re_lim=(-0.8, 0.3), im_lim=(-0.9, 0.9), n=301, beta=BETA_AB3):
    """Interior of the stability region, evaluated with batched companion matrices."""
    x, y = np.meshgrid(np.linspace(*re_lim, n), np.linspace(*im_lim, n))
    w = x + 1j*y
    r = len(beta)
    matrices = np.zeros(w.shape + (r, r), dtype=complex)
    matrices[..., 0, 0] = 1
    for i, b in enumerate(beta):
        matrices[..., 0, r - 1 - i] += b * w
    for i in range(1, r):
        matrices[..., i, i - 1] = 1
    return x, y, np.max(np.abs(np.linalg.eigvals(matrices)), axis=-1) < 1


def mode_kind(lam, tol=1e-12):
    if abs(lam) <= tol:
        return "neutral"
    if lam.real > tol:
        return "growing"
    return "decaying" if lam.real < -tol else "oscillatory"


def dt_limit(lams, tol=1e-12, s_max=20, ds=0.005, beta=BETA_AB3):
    """First AB3 stability exit along each decaying/oscillatory eigenvalue ray.

    Returns (limit, number of neutral/growing modes excluded). Infinity means
    no constraining modes, not a claim that arbitrary steps are accurate.
    """
    lams = np.atleast_1d(lams).astype(complex)
    if not np.all(np.isfinite(lams)) or ds <= 0 or s_max <= ds:
        raise ValueError("Finite eigenvalues and 0 < ds < s_max are required")
    usable = [lam for lam in lams if mode_kind(lam, tol) in ("decaying", "oscillatory")]
    best = np.inf
    for lam in usable:
        direction = lam / abs(lam)
        lo, hi = 0.0, ds
        while hi <= s_max and is_stable(hi * direction, beta):
            lo, hi = hi, hi + ds
        if hi > s_max:
            raise ValueError("No stability exit found; increase s_max")
        for _ in range(50):
            mid = (lo + hi) / 2
            if is_stable(mid * direction, beta):
                lo = mid
            else:
                hi = mid
        best = min(best, lo / abs(lam))
    return float(best), len(lams) - len(usable)


def checked_jacobian(f, state, time):
    """Validate adaptive derivatives against independent central differences.

    State scales are meters, m/s, then radians. Fail explicitly if an atmosphere
    segment boundary or an angular singularity prevents a trustworthy derivative.
    """
    state = np.asarray(state, dtype=float)
    result = jacobian(
        lambda x: f(x, time), state,
        initial_step=np.array([10., .1, .001, .001, .001, .001]),
        tolerances={"atol": 1e-12, "rtol": 1e-7},
    )
    if not np.all(result.success) or not np.all(np.isfinite(result.df)):
        raise RuntimeError(f"Jacobian did not converge at t={time:g} s; "
                           f"failed entries: {np.argwhere(~result.success).tolist()}")

    def central(steps):
        offsets = np.diag(steps)
        return np.column_stack([
            (f(state + offset, time) - f(state - offset, time)) / (2 * step)
            for offset, step in zip(offsets, steps)
        ])

    steps = np.array([1., .01, 1e-4, 1e-4, 1e-4, 1e-4])
    coarse, fine = central(steps), central(steps / 2)
    allowance = 2e-10 + 1e-4 * np.abs(result.df)
    error = max(float(np.max(np.abs(fine - result.df) / allowance)),
                float(np.max(np.abs(coarse - fine) / allowance)))
    if error > 1:
        raise RuntimeError(f"Independent Jacobian checks disagree at t={time:g} s "
                           f"(scaled discrepancy {error:.3g}); inspect atmosphere boundaries.")
    return result.df, result.error, error


def choose_checkpoints(states, times, earth_radius):
    """Five trajectory samples: entry, two load phases, 10 km, and above 3 km."""
    from atmospheric_model import get_rho

    if len(times) < 6:
        raise ValueError("Need at least six trajectory samples")
    h = states[:, 0] - earth_radius
    q = .5 * get_rho(states[:, 0]) * states[:, 1]**2
    peaks, _ = find_peaks(q, prominence=.01 * np.max(q))
    first = int(peaks[0]) if len(peaks) else len(times) // 4
    second = (int(peaks[1:][np.argmax(q[peaks[1:]])])
              if len(peaks) > 1 else len(times) // 2)
    below = np.flatnonzero(h <= 10000)
    low = int(below[0]) if len(below) else 3 * len(times) // 4
    below_3km = np.flatnonzero(h <= 3000)
    if not len(below_3km):
        raise ValueError("Trajectory does not reach 3 km for checkpoint 5")
    if below_3km[0] == 0:
        raise ValueError("Trajectory starts at or below 3 km")
    selected = [
        ("Entry", 0),
        ("First load peak" if len(peaks) else "Early flight", first),
        ("Second load peak" if len(peaks) > 1 else "Midflight", second),
        ("Below 10 km" if len(below) else "Late descent", low),
        ("Above 3 km", int(below_3km[0]) - 1),
    ]
    indices = [k for _, k in selected]
    if len(set(indices)) != 5 or indices != sorted(indices):
        raise ValueError("Trajectory phases overlap; choose distinct checkpoint times for this trajectory")
    return selected


def draw_region(ax, grid):
    x, y, stable = grid
    ax.contourf(x, y, stable.astype(float), levels=[.5, 1.5], colors=["#dcebf5"])
    ax.contour(x, y, stable.astype(float), levels=[.5], colors=["#53748b"], linewidths=.8)
    ax.axhline(0, color=".65", linewidth=.6)
    ax.axvline(0, color=".65", linewidth=.6)
    ax.set(xlabel=r"Re$(\Delta t\lambda)$", ylabel=r"Im$(\Delta t\lambda)$")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=.15)


def run_analysis(output_dir=None):
    """Generate only the checkpoint stability PNG and return its path."""
    import constants as c
    import equations_of_motion as eom
    import initial_conditions as init

    def ground_contact(time, state):
        return state[0] - c.r_earth

    ground_contact.terminal = True
    ground_contact.direction = -1
    trajectory = solve_ivp(
        lambda time, state: eom.f(state, time),
        (0, 1500), init.state, method="BDF", dense_output=True,
        events=ground_contact, max_step=1, rtol=1e-8,
        atol=[1e-3, 1e-5, 1e-10, 1e-10, 1e-10, 1e-10],
    )
    if not trajectory.success:
        raise RuntimeError(f"Trajectory integration failed: {trajectory.message}")
    times = np.append(np.arange(0, trajectory.t[-1], 1), trajectory.t[-1])
    states = trajectory.sol(times).T
    if not np.all(np.isfinite(states)) or np.any(states[:, 1] <= 0):
        raise RuntimeError("Invalid trajectory; cannot analyze its Jacobians")

    grid = stability_region()
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), layout="constrained")
    fig.suptitle("AB3 stability at five trajectory states", fontsize=15)
    checkpoints = []
    for label, k in choose_checkpoints(states, times, c.r_earth):
        J, _, _ = checked_jacobian(eom.f, states[k], times[k])
        eigenvalues = np.linalg.eigvals(J)
        limit, _ = dt_limit(eigenvalues)
        if not np.isfinite(limit) or limit <= 0:
            raise RuntimeError(f"No finite positive AB3 decay limit at {label}")
        checkpoints.append((label, k, eigenvalues, limit))

    minimum_limit = min(checkpoint[3] for checkpoint in checkpoints)
    for number, (ax, (label, k, eigenvalues, limit)) in enumerate(zip(axes.flat, checkpoints), 1):
        draw_region(ax, grid)
        w = minimum_limit * eigenvalues
        ax.scatter(w.real, w.imag, color="#0072B2", s=36, zorder=4)
        ax.set_xlim(min(-.8, w.real.min() * 1.1), max(.3, w.real.max() * 1.1))
        ax.set_ylim(min(-.9, w.imag.min() * 1.1), max(.9, w.imag.max() * 1.1))
        magnitudes = np.abs(eigenvalues)
        nonzero = magnitudes[magnitudes > 1e-12]
        stiffness = nonzero.max() / nonzero.min() if len(nonzero) else np.nan
        altitude_km = (states[k, 0] - c.r_earth) / 1000
        ax.set_title(f"Checkpoint {number} · {times[k]:g} s\n"
                     f"h = {altitude_km:.2f} km; stiffness ratio = {stiffness:.2e}\n"
                     f"Δt = {minimum_limit:.3g} s (minimum limit)", fontsize=10)

    ax = axes.flat[-1]
    ax.semilogy([times[k] for _, k, _, _ in checkpoints],
                [limit for _, _, _, limit in checkpoints], "o-", label="Local AB3 limit")
    ax.axhline(1, color=".4", linestyle="--", label="Reference timestep: 1 s")
    ax.set(xlabel="Time since entry (s)", ylabel="Timestep (s)",
           title="Local limits at selected states")
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    output_dir = (Path(output_dir) if output_dir is not None else
                  Path(__file__).resolve().parent / "figures" / "stability")
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = output_dir / "ab3_jacobian_checkpoints.png"
    fig.savefig(filename, dpi=200)
    plt.close(fig)
    return filename


if __name__ == "__main__":
    run_analysis()
