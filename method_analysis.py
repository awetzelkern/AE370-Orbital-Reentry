"""Compare AB3, started with Heun, with analytical falling-motion solutions.

Run ``python3 method_analysis.py`` to save PNGs in ``figures/method_analysis``.
These reduced models use altitude and downward speed, with no lift or rotation.
Constant-gravity free fall is quadratic and is integrated exactly in exact
arithmetic by this method; use the other cases to measure convergence order.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from constants import G, m_earth, r_earth

MU = G * m_earth
GRAVITY = MU / r_earth**2


@dataclass
class Benchmark:
    name: str
    title: str
    assumptions: str
    rhs: Callable
    exact: Callable
    duration: float
    timesteps: tuple


@dataclass
class Result:
    method: str
    dt: float
    times: np.ndarray
    states: np.ndarray
    errors: np.ndarray

    @property
    def max_errors(self):
        """Maximum absolute altitude and speed errors over all computed times."""
        return np.max(self.errors, axis=0)


def constant_gravity_exact(times, height=10000., speed=50., gravity=GRAVITY):
    """h = h0 - v0*t - g*t²/2; v = v0 + g*t (downward positive)."""
    times = np.asarray(times, dtype=float)
    return np.stack((height - speed * times - .5 * gravity * times**2,
                     speed + gravity * times), axis=-1)


def inverse_square_exact(times, height=500000.):
    """Exact radial fall from rest, evaluated through its analytical time law.

    With r = r0*cos(eta)², energy conservation gives
    t = sqrt(r0³/(2*mu))*(eta + sin(eta)*cos(eta)) and
    v = sqrt(2*mu/r0)*tan(eta). A scalar root solve inverts this time law;
    the reference does not use an ODE integrator.
    """
    times = np.asarray(times, dtype=float)
    radius = r_earth + height
    scale = np.sqrt(radius**3 / (2 * MU))
    if np.any(~np.isfinite(times)) or np.any(times < 0) or np.any(times >= scale * np.pi / 2):
        raise ValueError("Times must be finite, nonnegative, and before the central collision")
    angles = np.array([
        brentq(lambda eta: eta + np.sin(eta) * np.cos(eta) - time / scale,
               0, np.pi / 2, xtol=5e-15)
        for time in times.ravel()
    ]).reshape(times.shape)
    # This form avoids subtracting two Earth-sized radii near t = 0.
    altitude = height - radius * np.sin(angles)**2
    speed = np.sqrt(2 * MU / radius) * np.tan(angles)
    return np.stack((altitude, speed), axis=-1)


def linear_drag_exact(times, height=10000., speed=50., rate=.1, gravity=GRAVITY):
    """Exact constant-gravity fall with linear drag: v' = g - rate*v."""
    if rate <= 0:
        raise ValueError("The linear drag rate must be positive")
    times = np.asarray(times, dtype=float)
    terminal_speed = gravity / rate
    transient = speed - terminal_speed
    altitude = height - terminal_speed * times + transient * np.expm1(-rate * times) / rate
    velocity = terminal_speed + transient * np.exp(-rate * times)
    return np.stack((altitude, velocity), axis=-1)


def benchmarks():
    """Independent reduced equations; the reentry model's settings are untouched."""
    return [
        Benchmark(
            "constant_gravity", "Constant-gravity free fall",
            "No lift, drag, or rotation; constant g",
            lambda state, time: np.array([-state[1], GRAVITY]),
            constant_gravity_exact, 20., (2., 1., .5, .25, .125),
        ),
        Benchmark(
            "inverse_square", "Inverse-square gravity free fall",
            "No lift, drag, or rotation; g = GM/(R + h)²; starts from rest",
            lambda state, time: np.array([-state[1], MU / (r_earth + state[0])**2]),
            inverse_square_exact, 240., (16., 8., 4., 2., 1.),
        ),
        Benchmark(
            "linear_drag", "Falling with linear drag",
            "No lift or rotation; constant g; drag acceleration = −0.1v",
            lambda state, time: np.array([-state[1], GRAVITY - .1 * state[1]]),
            linear_drag_exact, 20., (1., .5, .25, .125, .0625),
        ),
    ]


def analyze_case(case):
    """Run AB3 with Heun startup and compare at exactly its returned times."""
    from solver import ab3, heun

    initial_state = case.exact(0.)
    results = []
    for dt in case.timesteps:
        steps = case.duration / dt
        if not np.isclose(steps, round(steps), rtol=0, atol=1e-10) or steps < 2:
            raise ValueError("Each timestep must divide the duration into at least two steps")
        states, times = ab3(case.rhs, initial_state.copy(), case.duration, dt, starter=heun)
        if not np.all(np.isfinite(states)):
            raise RuntimeError(f"AB3 produced a nonfinite solution for {case.name}, dt={dt}")
        errors = np.abs(states - case.exact(times))
        results.append(Result("AB3", dt, times, states, errors))
    return results


def observed_order(coarse, fine, component):
    """Order from errors on two grids: log(Ec/Ef) / log(dtc/dtf)."""
    errors = [coarse.max_errors[component], fine.max_errors[component]]
    # A relative roundoff floor prevents assigning orders to exact polynomials.
    scale = max(1., np.max(np.abs(coarse.states[:, component])),
                np.max(np.abs(fine.states[:, component])))
    if min(errors) <= 100 * np.finfo(float).eps * scale:
        return np.nan
    return np.log(errors[0] / errors[1]) / np.log(coarse.dt / fine.dt)


def plot_case(case, results, filename):
    """AB3 trajectory and absolute errors at the coarsest timestep."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    fig.suptitle(f"{case.title} · AB3 with Heun starter\n{case.assumptions}", fontsize=13)
    exact_times = np.linspace(0, case.duration, 501)
    exact = case.exact(exact_times)
    for component, quantity, unit in ((0, "Altitude", "m"), (1, "Downward speed", "m/s")):
        trajectory_ax, error_ax = axes[0, component], axes[1, component]
        trajectory_ax.plot(exact_times, exact[:, component], "k-", label="Analytical solution")
        result = results[0]
        trajectory_ax.plot(result.times, result.states[:, component], "o--",
                           color="C0", markersize=3, label=f"AB3, Δt = {result.dt:g} s")
        error_ax.plot(result.times, result.errors[:, component], color="C0", label="AB3")
        trajectory_ax.set(ylabel=f"{quantity} ({unit})", title="Trajectory")
        error_ax.set(ylabel=f"Absolute error ({unit})", title=f"{quantity} error vs analytical solution")
        for ax in (trajectory_ax, error_ax):
            ax.set_xlabel("Time (s)")
            ax.grid(alpha=.25)
            ax.legend(fontsize=8)
    fig.savefig(filename, dpi=180)
    plt.close(fig)


def plot_convergence(analyses, filename):
    """Maximum errors across the trajectory versus timestep, in physical units."""
    fig, axes = plt.subplots(len(analyses), 2, figsize=(12, 11), layout="constrained")
    fig.suptitle("AB3 with Heun starter: error under timestep refinement", fontsize=14)
    for row, (case, results) in zip(axes, analyses):
        for component, unit in ((0, "m"), (1, "m/s")):
            ax = row[component]
            ax.set_xscale("log")
            if case.name != "constant_gravity":
                ax.set_yscale("log")
            else:
                ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            steps = np.array([result.dt for result in results])
            errors = np.array([result.max_errors[component] for result in results])
            order = observed_order(results[-2], results[-1], component)
            label = f"AB3: p ≈ {order:.2f}" if np.isfinite(order) else "AB3: roundoff level"
            line, = ax.plot(steps, errors, "o-", label=label)
            if np.isfinite(order):
                reference = errors[0] * (steps / steps[0])**3
                ax.plot(steps, reference, ":", color=line.get_color(), alpha=.6,
                        label="Δt³ reference")
            quantity = "Altitude" if component == 0 else "Speed"
            ax.set(title=f"{case.title}: {quantity.lower()}", xlabel="Timestep Δt (s)",
                   ylabel=f"Max absolute error ({unit})")
            ax.grid(alpha=.25, which="both")
            ax.set_xticks(case.timesteps, labels=[f"{dt:g}" for dt in case.timesteps])
            ax.tick_params(axis="x", which="minor", labelbottom=False)
            ax.legend(fontsize=8)
    fig.savefig(filename, dpi=180)
    plt.close(fig)


def run_analysis(output_dir=None):
    """Save three trajectory/error PNGs and one convergence PNG."""
    output_dir = (Path(output_dir) if output_dir is not None else
                  Path(__file__).resolve().parent / "figures" / "method_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    analyses = [(case, analyze_case(case)) for case in benchmarks()]
    print("\nAB3 with Heun starter: maximum errors (speed is downward positive):")
    print(f"{'Case':20s} {'Method':6s} {'Δt (s)':>8s} {'Altitude (m)':>14s} {'Speed (m/s)':>14s}")
    for case, results in analyses:
        plot_case(case, results, output_dir / f"{case.name}.png")
        for result in results:
            height_error, speed_error = result.max_errors
            print(f"{case.name:20s} {result.method:6s} {result.dt:8g} "
                  f"{height_error:14.6g} {speed_error:14.6g}")
    plot_convergence(analyses, output_dir / "convergence.png")
    print(f"Saved method-analysis plots to {output_dir}")
    return analyses


if __name__ == "__main__":
    run_analysis()
