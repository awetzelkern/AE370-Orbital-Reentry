"""Initial six-state entry conditions in SI units and radians."""

import numpy as np

from constants import r_earth

r = 99900 + r_earth  # Earth-centered radius, m
v = 10700  # speed, m/s
gamma = np.deg2rad(-6.2)  # flight path angle
psi = np.deg2rad(40)  # azimuth, measured from east toward north
phi = np.deg2rad(19)  # latitude
theta = np.deg2rad(-139)  # longitude

state = np.array([r, v, gamma, psi, phi, theta])