# AE370-Orbital-Reentry

This project simulates the atmospheric reentry of an Orion capsule using a
six-state, rotating-Earth model. It plots the trajectory and ground track and
includes separate analyses of the numerical method and its local stability.
The capsule-only model does not include parachutes.

## Setup and simulation

Install dependencies using the same Python interpreter that runs the simulation:

```sh
python3 -m pip install -r requirements.txt
python3 main.py
```

`pyatmos` requires `pkg_resources`, supplied by `setuptools<82`. The version
constraint in `requirements.txt` is necessary because setuptools 82 removed it.
If an existing environment reports `No module named 'pkg_resources'`, repair it
with `python3 -m pip install 'setuptools<82'`.

The simulation uses an axisymmetric blunt body and calorically perfect air.
Initial conditions and physical parameters are in `initial_conditions.py` and
`constants.py`. Atmospheric density comes from the COESA 1976 model via
`pyatmos`; the ground track uses the local Natural Earth data in `assets/`.
The state consists of Earth-centered radius, speed, flight path angle, azimuth,
latitude, and longitude, in SI units and radians.

`main.py` advances the equations with third-order Adams–Bashforth (AB3), using
Heun's method for the first two steps, and writes five PNGs to `figures/`.
It uses a nominal 700 s duration and a 4.49 s timestep; the fixed-step solver
rounds the number of steps and does not stop or interpolate at ground contact.
The plots label the last sample as ground contact only if it is at ground level.

The main files are:

- `equations_of_motion.py` and `atmospheric_model.py`: dynamics and density.
- `solver.py`: Heun and AB3 time integration and initial-state diagnostics.
- `main.py` and `ground_track.py`: trajectory and map plots.
- `method_analysis.py` and `stability_region.py`: numerical analyses.

Run all analysis tests with:

```sh
python3 -m unittest discover -s tests -v
```

## Jacobian and AB3 stability checkpoints

Run `python3 stability_region.py` to analyze five states from a BDF
reference trajectory: entry, two aerodynamic-loading phases, descent through
10 km, and the last sample before ground contact. The selection adapts to the
trajectory and falls back to other phases if peaks or ground contact are absent.

The script writes `figures/stability/ab3_jacobian_checkpoints.png`. Each
Jacobian must pass adaptive differentiation and independent central-difference
checks. Neutral and locally growing modes are reported separately from the
decaying modes. These are local **AB3** limits, not BDF2 limits or a global
accuracy guarantee.

## Analytical method benchmarks

Run `python3 method_analysis.py` to compare AB3, using Heun for its first two
steps (`starter=heun`), with three analytical solutions:

- Constant-gravity free fall: `h = h0 - v0*t - g*t²/2`, `v = v0 + g*t`.
  AB3 with Heun startup reproduces this quadratic solution to roundoff; it
  checks correctness but cannot measure convergence order.
- Radial free fall from rest under `g = GM/(R+h)²`, with lift, drag, and
  rotation absent. The exact solution uses `r = r0*cos(eta)²` and
  `t = sqrt(r0³/(2GM)) * (eta + sin(eta)*cos(eta))`; a scalar root solve
  inverts this analytical time law without integrating a reference ODE.
- Constant-gravity falling with linear drag, `v' = g - 0.1*v`, whose
  exponential solution approaches terminal speed `g/0.1`. This is a
  controlled benchmark, distinct from the capsule's quadratic drag model.

Four PNGs are saved in `figures/method_analysis/`: one trajectory/error figure
per case and `convergence.png`. Each case figure compares altitude and
downward speed against the exact solution at the coarsest timestep.
Convergence plots and the console table use the maximum absolute error over
all computed times, in meters and m/s, for five successively halved timesteps.
Observed orders use the two finest grids and are omitted at roundoff level.
All trajectories end above ground, avoiding impact/interpolation error.

These benchmarks test AB3 with Heun startup on reduced equations without
changing the simulation's lift, drag, rotation, or initial conditions. A test
also verifies that the inverse-square model matches the six-state reentry
equations for vertical flight with lift, drag, and rotation turned off.
The other tests check the analytical identities, Heun startup values,
roundoff accuracy for constant gravity, and third-order AB3 convergence:

```sh
python3 -m unittest discover -s tests -p 'test_method_analysis.py' -v
```
