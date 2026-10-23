from constants import r_earth
import numpy as np

r = 100 + r_earth # km

v = 7.8 #km/s

gamma = np.deg2rad(-5)#radians

psi = np.deg2rad(75) #radians

phi = np.deg2rad(21.248373) #radians

theta = np.deg2rad(-157.874719) #radians

state = np.array([r, v, gamma, psi, phi, theta])

