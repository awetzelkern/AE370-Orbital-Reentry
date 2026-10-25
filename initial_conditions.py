from constants import r_earth
import numpy as np

r = 99900 + r_earth # m

v = 7800 #m/s

gamma = np.deg2rad(-3)#radians - flight path angle

psi = np.deg2rad(15) #radians - orbital inclination from equator

phi = np.deg2rad(21.248373) #radians - latitude starting hawaii

theta = np.deg2rad(-157.874719) #radians - longitude starting hawaii

state = np.array([r, v, gamma, psi, phi, theta])

