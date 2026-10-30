from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from solver import ab3, heun
from ground_track import plot_ground_track
import initial_conditions as init
import equations_of_motion as eom
import constants as c


if __name__ == "__main__":
    ground_event = lambda state, time: state[0] - c.r_earth
    us, times = ab3(eom.f, init.state, 700, 4.49, starter=heun)
    if not np.all(np.isfinite(us)) or np.any(us[:, 1] < 0):
        raise RuntimeError("Invalid trajectory: nonfinite state or negative speed; plots not generated.")
    reached_ground = abs(ground_event(us[-1], times[-1])) < 1e-6
    endpoint_label = "Ground contact" if reached_ground else "End of simulation"
    # The state uses SI units; display height above the surface in km.
    altitude_km = (us[:, 0] - c.r_earth) / 1000
    speed_m_s = us[:, 1]
    flight_path_angle_deg = np.rad2deg(us[:, 2])
    output_dir = Path(__file__).resolve().parent / "figures"
    output_dir.mkdir(exist_ok=True)

    if reached_ground:
        print(f"Ground contact: t = {times[-1]:.2f} s, speed = {speed_m_s[-1]:.2f} m/s "
              "(capsule-only model, no parachutes)")

    plots = [
        (times, altitude_km, "Altitude vs time", "Time (s)", "Altitude (km)",
         "altitude_vs_time.png"),
        (times, speed_m_s, "Speed vs time", "Time (s)", "Speed (m/s)",
         "velocity_vs_time.png"),
        (altitude_km, speed_m_s, "Speed vs altitude", "Altitude (km)", "Speed (m/s)",
         "velocity_vs_altitude.png"),
        (altitude_km, flight_path_angle_deg, "Flight path angle vs altitude",
         "Altitude (km)", "Flight path angle γ (deg)",
         "flight_path_angle_vs_altitude.png"),
    ]
    for x, y, title, xlabel, ylabel, filename in plots:
        fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
        fig.suptitle("Capsule-only reentry · No parachutes", fontsize=10)
        ax.set_xlim(0, times[-1] / 60)
        ax.plot(x, y, label="Simulated trajectory")
        ax.scatter(x[-1], y[-1], marker="x", color="black", label=endpoint_label,
                   zorder=3, clip_on=False)
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.grid(True, alpha=0.3)

        if filename == "altitude_vs_time.png":
            ax.axhline(0, linestyle=":", color="red", label="Ground")
            ax.set_ylim(bottom=0)

        if filename in ("velocity_vs_time.png", "velocity_vs_altitude.png"):
            ax.set_ylim(bottom=0)

        if filename == "flight_path_angle_vs_altitude.png":
            ax.axhline(0, linestyle="--", color="gray", label="Horizontal flight (γ = 0°)")

        if filename in ("velocity_vs_altitude.png", "flight_path_angle_vs_altitude.png"):
            ax.scatter(x[0], y[0], marker="o", color="green", label="Start", zorder=3)
            ax.axvline(0, linestyle=":", color="red", label="Ground")
            ax.set_xlim(max(altitude_km.max() * 1.02, 1), 0)
        else:
            ax.set_xlim(times[0], times[-1])
            if reached_ground:
                ax.axvline(times[-1], linestyle=":", color="red",
                           label=f"Ground contact: {times[-1]:.1f} s")

        ax.legend()
        fig.savefig(output_dir / filename, dpi=150)
        plt.close(fig)

    plot_ground_track(us, times, output_dir / "ground_track.png", endpoint_label=endpoint_label)
