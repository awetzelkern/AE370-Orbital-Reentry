import numpy as np
import matplotlib.pyplot as plt
from solver import heun, ab3
from scipy.differentiate import jacobian
import initial_conditions as init
import equations_of_motion as eom
import constants as c


if __name__ == "__main__":
    dynamics = lambda state, t: eom.f(state, t, c.m_capsule, c.omega_earth, c.nu, c.A)
    us, times = ab3(dynamics, init.state, 200, 1, starter=heun)
    plt.plot(times, us[:, 0]-c.r_earth)
    plt.title("Altitude vs time")
    plt.axhline(0, linestyle = ':', color='red')    
    plt.xlabel("Time (s)")
    plt.ylabel("Altitude (km)")
    # plt.xlim(0, 150)
    plt.savefig("figures/altitude_vs_time.png")

    plt.plot(times, us[:, 1])
    plt.title("Velocity vs time")
    plt.axhline(0, linestyle = ':', color='red')    
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.savefig("figures/velocity_vs_time.png")

    plt.plot(us[:, 0], us[:, 1])
    plt.title("Velocity vs Altitude")
    plt.gca().invert_xaxis()
    plt.xlabel("Altitude (km)")
    plt.ylabel("Velocity (m/s)")
    plt.savefig("figures/velocity_vs_altitude.png")




