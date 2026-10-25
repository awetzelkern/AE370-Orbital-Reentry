from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from solver import heun, ab3, bdf2, forward_euler
from ground_track import plot_ground_track
import initial_conditions as init
import equations_of_motion as eom
import constants as c


if __name__ == "__main__":
    # Include the first sample below the deployment threshold.
    us, times = bdf2(eom.f, init.state, 1500, 1)
    # The state uses SI units; display height above the surface in km.
    altitude_km = (us[:, 0] - c.r_earth) / 1000
    speed_m_s = us[:, 1]
    flight_path_angle_deg = np.rad2deg(us[:, 2])
    deployment_altitude_km = c.parachute_deployment_altitude / 1000
    deployment_label = f"Parachute region (below {deployment_altitude_km:g} km)"
    output_dir = Path(__file__).resolve().parent / "figures"
    output_dir.mkdir(exist_ok=True)

    # Identify the first ground crossing without changing the solver output.
    below_ground = np.flatnonzero(altitude_km <= 0)
    ground_time = None
    if below_ground.size:
        k = below_ground[0]
        ground_time = times[k]
        if k > 0:
            fraction = altitude_km[k - 1] / (altitude_km[k - 1] - altitude_km[k])
            ground_time = times[k - 1] + fraction * (times[k] - times[k - 1])

    # Interpolate the threshold crossing for the speed-versus-time shading.
    deployed = np.flatnonzero(altitude_km < deployment_altitude_km)
    deployment_time = None
    pre_deployment_time = None
    pre_deployment_state = None
    if deployed.size:
        k = deployed[0]
        deployment_time = times[k]
        if k > 0:
            fraction = ((altitude_km[k - 1] - deployment_altitude_km)
                        / (altitude_km[k - 1] - altitude_km[k]))
            deployment_time = times[k - 1] + fraction * (times[k] - times[k - 1])
            # Keep the last actual AB3 sample before the drag-area change.
            # These can be used as the initial time/state for the next method.
            pre_deployment_time = float(times[k - 1])
            pre_deployment_state = us[k - 1].copy()
            print("\nLast sampled state before parachute deployment:")
            print(f"pre_deployment_time = {pre_deployment_time:.12g}  # s")
            print("# State order: [r (m), V (m/s), gamma, psi, phi, theta (all angles in rad)]")
            print("pre_deployment_state = np.array(" + np.array2string(
                pre_deployment_state, precision=12, separator=", ",
                floatmode="unique", max_line_width=100,
            ) + ")")
            print(f"Altitude above surface: {altitude_km[k - 1] * 1000:.6f} m")
            print(f"Estimated threshold crossing: {deployment_time:.6f} s "
                  "(linearly interpolated)")
        else:
            print("No pre-deployment state: the initial state is already below "
                  "the parachute deployment altitude.")
    else:
        print(f"Parachute deployment not reached by t = {times[-1]:g} s; "
              f"final altitude = {altitude_km[-1]:.6f} km.")

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
        ax.plot(x, y, label="Simulated trajectory")
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.grid(True, alpha=0.3)

        if filename == "altitude_vs_time.png":
            ax.axhspan(0, deployment_altitude_km, color="orange", alpha=0.2,
                       label=deployment_label)
            ax.axhline(0, linestyle=":", color="red", label="Ground")

        if filename == "flight_path_angle_vs_altitude.png":
            ax.axhline(0, linestyle="--", color="gray", label="Horizontal flight (γ = 0°)")

        if filename in ("velocity_vs_altitude.png", "flight_path_angle_vs_altitude.png"):
            ax.axvspan(0, deployment_altitude_km, color="orange", alpha=0.2,
                       label=deployment_label)
            ax.scatter(x[0], y[0], marker="o", color="green", label="Start", zorder=3)
            ax.scatter(x[-1], y[-1], marker="x", color="black", label="End", zorder=3)
            ax.axvline(0, linestyle=":", color="red", label="Ground")
            ax.invert_xaxis()
            if ground_time is not None:
                ax.axvspan(min(altitude_km.min(), 0), 0, color="red", alpha=0.1,
                           label="Below ground (unphysical)")
        else:
            ax.set_xlim(times[0], times[-1])
            if filename == "velocity_vs_time.png":
                if deployment_time is not None:
                    deployment_end = ground_time if ground_time is not None else times[-1]
                    ax.axvspan(deployment_time, deployment_end, color="orange", alpha=0.2,
                               label=deployment_label)
                    ax.axvline(deployment_time, linestyle="--", color="darkorange",
                               label=f"Deployment: {deployment_time:.1f} s")
                else:
                    ax.text(0.02, 0.03,
                            f"Deployment altitude ({deployment_altitude_km:g} km) not reached",
                            transform=ax.transAxes, fontsize=9)
            if ground_time is not None:
                ax.axvline(ground_time, linestyle=":", color="red",
                           label=f"Ground crossing: {ground_time:.1f} s")
                ax.axvspan(ground_time, times[-1], color="red", alpha=0.1,
                           label="After ground crossing (unphysical)")

        ax.legend()
        fig.savefig(output_dir / filename, dpi=150)
        plt.close(fig)

    plot_ground_track(us, times, output_dir / "ground_track.png", deployment_time)
