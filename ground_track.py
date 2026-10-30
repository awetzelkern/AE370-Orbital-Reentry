"""Ground-track plotting with a local Natural Earth background."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Polygon, Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np


def plot_ground_track(states, times, output_path, endpoint_label="End of simulation"):
    """Plot Earth-fixed longitude theta versus latitude phi (state angles in rad)."""
    latitude = np.rad2deg(states[:, 4])
    longitude = np.rad2deg(np.unwrap(states[:, 5]))
    # Keep a continuous track across the date line, near the usual longitude range.
    longitude -= 360 * np.floor((longitude[0] + 180) / 360)
    center_lon = (longitude.min() + longitude.max()) / 2
    center_lat = (latitude.min() + latitude.max()) / 2
    lon_margin = max(5, np.ptp(longitude) * 0.3)
    lat_margin = max(3, np.ptp(latitude) * 0.2)
    detail = (longitude.min() - lon_margin, longitude.max() + lon_margin,
              max(-90, latitude.min() - lat_margin), min(90, latitude.max() + lat_margin))
    # Leave geographic context around a mostly east-west track rather than
    # squeezing the detailed map into a thin strip.
    half_height = max((detail[3] - detail[2]) / 2,
                      (detail[1] - detail[0]) * np.cos(np.deg2rad(center_lat)) * 0.35)
    detail = (detail[0], detail[1], max(-90, center_lat - half_height),
              min(90, center_lat + half_height))
    overview = (min(center_lon - 55, detail[0]), max(center_lon + 55, detail[1]),
                max(-90, min(center_lat - 40, detail[2])),
                min(90, max(center_lat + 40, detail[3])))

    map_path = Path(__file__).resolve().parent / "assets" / "ne_50m_land.geojson"
    with map_path.open() as source:
        features = json.load(source)["features"]
    land = []
    for feature in features:
        geometry = feature["geometry"]
        polygons = (geometry["coordinates"] if geometry["type"] == "MultiPolygon"
                    else [geometry["coordinates"]])
        land.extend(np.asarray(polygon[0]) for polygon in polygons)

    def lon_label(value, _):
        value = (value + 180) % 360 - 180
        return f"{abs(value):g}°{'W' if value < 0 else 'E' if value > 0 else ''}"

    def lat_label(value, _):
        return f"{abs(value):g}°{'S' if value < 0 else 'N' if value > 0 else ''}"

    points = np.column_stack((longitude, latitude))
    segments = np.stack((points[:-1], points[1:]), axis=1)
    norm = Normalize(times[0], times[-1])
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), layout="constrained")
    fig.suptitle("Reentry ground track", fontsize=17)
    for ax, bounds, title in zip(axes, (overview, detail), ("Geographic context", "Trajectory detail")):
        ax.set_facecolor("#e3f1f7")
        patches = []
        # Repeat land across the date line to match unwrapped track longitudes.
        shifts = range(int(np.floor((bounds[0] + 180) / 360)) - 1,
                       int(np.ceil((bounds[1] + 180) / 360)) + 1)
        for ring in land:
            for shift in shifts:
                vertices = ring + [360 * shift, 0]
                if (vertices[:, 0].max() >= bounds[0] and vertices[:, 0].min() <= bounds[1]
                        and vertices[:, 1].max() >= bounds[2] and vertices[:, 1].min() <= bounds[3]):
                    patches.append(Polygon(vertices, closed=True))
        ax.add_collection(PatchCollection(patches, facecolor="#e0dfc8",
                                          edgecolor="#7d887a", linewidth=0.5, zorder=1))
        track = LineCollection(segments, cmap="viridis", norm=norm, linewidth=3, zorder=3)
        track.set_array((times[:-1] + times[1:]) / 2)
        ax.add_collection(track)
        ax.scatter(longitude[0], latitude[0], s=65, color="limegreen", edgecolor="black",
                   label="Start", zorder=5)
        ax.scatter(longitude[-1], latitude[-1], marker="x", s=65, color="black",
                   linewidth=2, label=endpoint_label, zorder=6)
        ax.set(xlim=bounds[:2], ylim=bounds[2:], title=title,
               xlabel="Longitude θ", ylabel="Latitude φ")
        # Equirectangular view with longitude scaled at the central latitude.
        ax.set_aspect(1 / max(np.cos(np.deg2rad(center_lat)), 0.1))
        ax.xaxis.set_major_formatter(FuncFormatter(lon_label))
        ax.yaxis.set_major_formatter(FuncFormatter(lat_label))
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(6))
        ax.grid(alpha=0.3, linewidth=0.6)

    axes[0].add_patch(Rectangle((detail[0], detail[2]), detail[1] - detail[0],
                               detail[3] - detail[2], fill=False, linestyle="--",
                               edgecolor="#485c69", linewidth=1.2, zorder=4))
    axes[1].annotate(f"Start · {times[0]:g} s", points[0], xytext=(-55, 15),
                     textcoords="offset points", fontsize=9)
    axes[1].annotate(f"{endpoint_label}\n{times[-1]:.1f} s", points[-1], xytext=(-65, 28),
                     textcoords="offset points", fontsize=9,
                     arrowprops={"arrowstyle": "-", "color": "black"})
    axes[1].legend(loc="lower right", fontsize=8)
    fig.colorbar(track, ax=list(axes), location="bottom", shrink=0.75, pad=0.06,
                 label="Elapsed time (s)")
    fig.supxlabel("Earth-fixed coordinates · Equirectangular map · Made with Natural Earth",
                  fontsize=9, color="#55616a")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
