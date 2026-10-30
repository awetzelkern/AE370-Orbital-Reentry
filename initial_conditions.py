"""Initial six-state entry conditions in SI units and radians."""

import numpy as np

from constants import r_earth

r = 99900 + r_earth  # Earth-centered radius, m
v = 10700  # speed, m/s
gamma = np.deg2rad(-6.5)  # flight path angle
psi = np.deg2rad(20)  # azimuth, measured from east toward north
phi = np.deg2rad(34)  # latitude
theta = np.deg2rad(-118)  # longitude

state = np.array([r, v, gamma, psi, phi, theta])
