"""
A2D2 pose reconstruction from the original A2D2 source files.

BASE-PRINCIPLES VERSION
-----------------------
This script does NOT use:
    - MapAnything
    - lidar_to_camera.txt
    - the old pose/*.txt files
    - any manually invented coordinate conversion

It uses:
    1. A2D2 camera JSON files for camera timestamps.
    2. cams_lidars.json for the fixed camera-to-vehicle mounting transform.
    3. bus_signals.json for GPS trajectory.
    4. GPS trajectory to estimate vehicle heading.
    5. The fixed camera mounting transform to obtain camera poses.

The script produces:
    generated_poses/*.txt
    camera_trajectory.ply
    camera_trajectory_xy.png
    camera_heading.png
    pose_debug.npz

IMPORTANT:
The first objective is to verify the A2D2 trajectory.
Do not feed these poses into MapAnything until the trajectory PLY
looks physically correct.
"""

import json
import math
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path("a2d2")

SEQUENCE = "20180810_150607"

BUS_FILE = (
    ROOT
    / "camera_lidar"
    / SEQUENCE
    / "bus"
    / "20180810150607_bus_signals.json"
)

CAMERA_DIR = (
    ROOT
    / "camera_lidar"
    / SEQUENCE
    / "camera"
    / "cam_front_center"
)

CALIB_FILE = ROOT / "cams_lidars.json"

OUTPUT_DIR = ROOT / "generated_poses"

TRAJECTORY_PLY = ROOT / "camera_trajectory.ply"
TRAJECTORY_PNG = ROOT / "camera_trajectory_xy.png"
HEADING_PNG = ROOT / "camera_heading.png"

CAMERA_NAME = "front_center"

# Camera files may be very numerous.
# Set to None to process all of them.
MAX_FRAMES = 2500

# Set to 1 for every camera frame.
# For example 10 means every 10th camera frame.
STRIDE = 1

# Number of GPS samples used on each side when estimating heading.
HEADING_WINDOW = 5


# ============================================================
# HELPERS
# ============================================================

def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


def rotation_angle_deg(R):
    """
    Rotation angle of a proper rotation matrix.
    """
    trace = np.trace(R)
    c = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(c))


def make_transform(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def inverse_transform(T):
    R = T[:3, :3]
    t = T[:3, 3]

    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t

    return T_inv


# ============================================================
# LOAD A2D2 CALIBRATION
# ============================================================

print_section("1. LOAD A2D2 CAMERA CALIBRATION")

print("Calibration:")
print(CALIB_FILE)

calib = load_json(CALIB_FILE)

if CAMERA_NAME not in calib["cameras"]:
    raise RuntimeError(
        f"Camera '{CAMERA_NAME}' not found in cams_lidars.json"
    )

camera_calib = calib["cameras"][CAMERA_NAME]

camera_origin_vehicle = np.array(
    camera_calib["view"]["origin"],
    dtype=float
)

camera_x_vehicle = normalize(
    np.array(
        camera_calib["view"]["x-axis"],
        dtype=float
    )
)

camera_y_vehicle = normalize(
    np.array(
        camera_calib["view"]["y-axis"],
        dtype=float
    )
)

# A2D2 gives two camera axes.
# Recover the third axis with a cross product.
camera_z_vehicle = normalize(
    np.cross(camera_x_vehicle, camera_y_vehicle)
)

# Keep the supplied axes as the primary information.
R_vehicle_camera = np.column_stack(
    [
        camera_x_vehicle,
        camera_y_vehicle,
        camera_z_vehicle,
    ]
)

print("\nCamera:", CAMERA_NAME)

print("Camera origin in vehicle frame:")
print(camera_origin_vehicle)

print("\nCamera X axis in vehicle frame:")
print(camera_x_vehicle)

print("\nCamera Y axis in vehicle frame:")
print(camera_y_vehicle)

print("\nDerived camera Z axis:")
print(camera_z_vehicle)

print("\nCamera rotation matrix:")
print(R_vehicle_camera)

print("\nCalibration checks:")

print(
    "det(R) =",
    np.linalg.det(R_vehicle_camera)
)

print(
    "orthogonality error =",
    np.linalg.norm(
        R_vehicle_camera.T @ R_vehicle_camera
        - np.eye(3)
    )
)

if np.linalg.det(R_vehicle_camera) < 0:
    print(
        "\nWARNING: derived camera basis is left-handed."
    )

# This transform maps camera coordinates into vehicle coordinates.
T_vehicle_camera = make_transform(
    R_vehicle_camera,
    camera_origin_vehicle
)

print("\nT_vehicle_camera:")
print(T_vehicle_camera)


# ============================================================
# LOAD BUS SIGNALS
# ============================================================

print_section("2. LOAD A2D2 BUS / GPS DATA")

print("Bus file:")
print(BUS_FILE)

bus = load_json(BUS_FILE)

required = [
    "latitude_degree",
    "longitude_degree",
]

for key in required:
    if key not in bus:
        raise RuntimeError(
            f"Missing required bus signal: {key}"
        )


def signal_values(name):
    """
    Return timestamps and values from an A2D2 bus signal.
    """
    values = np.asarray(
        bus[name]["values"],
        dtype=float
    )

    timestamps = values[:, 0]
    measurements = values[:, 1]

    return timestamps, measurements


gps_t, lat = signal_values("latitude_degree")
lon_t, lon = signal_values("longitude_degree")

print("GPS samples:", len(gps_t))

print(
    "GPS time range:",
    gps_t[0],
    "->",
    gps_t[-1]
)

print(
    "Latitude range:",
    lat.min(),
    "->",
    lat.max()
)

print(
    "Longitude range:",
    lon.min(),
    "->",
    lon.max()
)


# ============================================================
# GPS -> LOCAL METRIC COORDINATES
# ============================================================

print_section("3. CONVERT GPS TO LOCAL METRIC TRAJECTORY")

# Equirectangular approximation is sufficient over this
# relatively small A2D2 sequence.

lat0_rad = np.radians(lat[0])
lon0_rad = np.radians(lon[0])

EARTH_RADIUS = 6371000.0

lat_rad = np.radians(lat)
lon_rad = np.radians(lon)

gps_x = (
    (lon_rad - lon0_rad)
    * EARTH_RADIUS
    * np.cos(lat0_rad)
)

gps_y = (
    (lat_rad - lat0_rad)
    * EARTH_RADIUS
)

gps_positions = np.column_stack(
    [
        gps_x,
        gps_y,
    ]
)

print("Local GPS origin:")
print(
    f"lat={lat[0]:.9f}, "
    f"lon={lon[0]:.9f}"
)

print("\nFirst GPS position:")
print(gps_positions[0])

print("\nLast GPS position:")
print(gps_positions[-1])

print(
    "\nStart -> end displacement:",
    gps_positions[-1] - gps_positions[0]
)

print(
    "Start -> end distance:",
    np.linalg.norm(
        gps_positions[-1] - gps_positions[0]
    ),
    "m"
)


# ============================================================
# GPS HEADING
# ============================================================

print_section("4. ESTIMATE VEHICLE HEADING FROM GPS")

"""
Heading convention used internally:

    vehicle forward = +X vehicle axis

GPS local frame:
    +X = east
    +Y = north

Therefore:

    yaw = atan2(dY, dX)

This gives the direction of travel in the local GPS XY plane.

We deliberately do NOT force this into an OpenCV convention.
"""

gps_heading = np.zeros(len(gps_positions))

for i in range(len(gps_positions)):

    i0 = max(0, i - HEADING_WINDOW)
    i1 = min(
        len(gps_positions) - 1,
        i + HEADING_WINDOW
    )

    delta = (
        gps_positions[i1]
        - gps_positions[i0]
    )

    distance = np.linalg.norm(delta)

    if distance < 0.05:

        if i > 0:
            gps_heading[i] = gps_heading[i - 1]
        else:
            gps_heading[i] = 0.0

    else:

        gps_heading[i] = math.atan2(
            delta[1],
            delta[0]
        )

# unwrap for plotting
gps_heading_unwrapped = np.unwrap(
    gps_heading
)

print(
    "Heading range:",
    np.degrees(gps_heading_unwrapped.min()),
    "->",
    np.degrees(gps_heading_unwrapped.max()),
    "degrees"
)


# ============================================================
# CAMERA FRAME DISCOVERY
# ============================================================

print_section("5. FIND CAMERA FRAMES")

camera_jsons = sorted(
    CAMERA_DIR.glob("*.json")
)

print("Camera JSON files:", len(camera_jsons))

if len(camera_jsons) == 0:
    raise RuntimeError(
        f"No JSON files found in {CAMERA_DIR}"
    )

selected_files = camera_jsons[::STRIDE]

if MAX_FRAMES is not None:
    selected_files = selected_files[:MAX_FRAMES]

print("Stride:", STRIDE)
print("Maximum selected:", MAX_FRAMES)
print("Selected frames:", len(selected_files))


# ============================================================
# CAMERA TIMESTAMP EXTRACTION
# ============================================================

print_section("6. READ CAMERA TIMESTAMPS")

camera_records = []

for path in selected_files:

    data = load_json(path)

    timestamp = float(
        data["cam_tstamp"]
    )

    image_name = data.get(
        "image_png",
        ""
    )

    camera_records.append(
        {
            "path": path,
            "timestamp": timestamp,
            "image": image_name,
        }
    )

print(
    "First camera timestamp:",
    camera_records[0]["timestamp"]
)

print(
    "Last camera timestamp:",
    camera_records[-1]["timestamp"]
)


# ============================================================
# INTERPOLATE GPS POSITION
# ============================================================

def interpolate_gps(timestamp):

    x = np.interp(
        timestamp,
        gps_t,
        gps_x
    )

    y = np.interp(
        timestamp,
        gps_t,
        gps_y
    )

    return np.array([x, y])


def interpolate_heading(timestamp):

    """
    Interpolate the unwrapped heading, then return radians.
    """

    return np.interp(
        timestamp,
        gps_t,
        gps_heading_unwrapped
    )


# ============================================================
# BUILD CAMERA POSES
# ============================================================

print_section("7. BUILD CAMERA POSES")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

camera_positions = []
camera_headings = []
frame_ids = []
poses = []

for index, record in enumerate(camera_records):

    timestamp = record["timestamp"]

    # --------------------------------------------
    # Vehicle position from GPS
    # --------------------------------------------

    vehicle_xy = interpolate_gps(
        timestamp
    )

    # --------------------------------------------
    # Vehicle heading
    # --------------------------------------------

    yaw = interpolate_heading(
        timestamp
    )

    # --------------------------------------------
    # Vehicle rotation
    #
    # Vehicle frame:
    #   +X = forward
    #   +Y = left/right according to A2D2
    #   +Z = up
    #
    # We rotate the vehicle's horizontal
    # frame according to GPS heading.
    # --------------------------------------------

    cy = math.cos(yaw)
    sy = math.sin(yaw)

    R_world_vehicle = np.array(
        [
            [cy, -sy, 0.0],
            [sy,  cy, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    vehicle_position = np.array(
        [
            vehicle_xy[0],
            vehicle_xy[1],
            0.0,
        ]
    )

    # --------------------------------------------
    # Camera pose in world
    #
    # T_world_camera =
    #     T_world_vehicle @ T_vehicle_camera
    # --------------------------------------------

    T_world_vehicle = make_transform(
        R_world_vehicle,
        vehicle_position
    )

    T_world_camera = (
        T_world_vehicle
        @ T_vehicle_camera
    )

    # --------------------------------------------
    # Save
    # --------------------------------------------

    frame_stem = record["path"].stem

    pose_file = (
        OUTPUT_DIR
        / f"{frame_stem}.txt"
    )

    np.savetxt(
        pose_file,
        T_world_camera,
        fmt="%.12f"
    )

    camera_positions.append(
        T_world_camera[:3, 3]
    )

    camera_headings.append(yaw)

    frame_ids.append(frame_stem)
    poses.append(T_world_camera)

    if index < 10:

        print(
            f"\nFrame {frame_stem}"
        )

        print(
            "timestamp:",
            timestamp
        )

        print(
            "vehicle XY:",
            vehicle_xy
        )

        print(
            "yaw:",
            np.degrees(yaw),
            "deg"
        )

        print(
            "camera position:",
            T_world_camera[:3, 3]
        )

        print(
            "camera forward:",
            T_world_camera[:3, 2]
        )


camera_positions = np.asarray(
    camera_positions
)

camera_headings = np.asarray(
    camera_headings
)

poses = np.asarray(poses)


# ============================================================
# POSE MOTION CHECK
# ============================================================

print_section("8. CAMERA TRAJECTORY CHECK")

if len(camera_positions) > 1:

    deltas = (
        camera_positions[1:]
        - camera_positions[:-1]
    )

    distances = np.linalg.norm(
        deltas,
        axis=1
    )

    print(
        "Mean frame displacement:",
        distances.mean(),
        "m"
    )

    print(
        "Median frame displacement:",
        np.median(distances),
        "m"
    )

    print(
        "Maximum frame displacement:",
        distances.max(),
        "m"
    )

print("\nCamera position range:")
print(
    "X:",
    camera_positions[:, 0].min(),
    "->",
    camera_positions[:, 0].max()
)
print(
    "Y:",
    camera_positions[:, 1].min(),
    "->",
    camera_positions[:, 1].max()
)
print(
    "Z:",
    camera_positions[:, 2].min(),
    "->",
    camera_positions[:, 2].max()
)


# ============================================================
# SAVE DEBUG NPZ
# ============================================================

np.savez(
    ROOT / "pose_debug.npz",
    frame_ids=np.asarray(frame_ids),
    timestamps=np.asarray(
        [r["timestamp"] for r in camera_records]
    ),
    camera_positions=camera_positions,
    camera_headings=camera_headings,
    poses=poses,
    gps_t=gps_t,
    gps_x=gps_x,
    gps_y=gps_y,
)

print(
    "\nSaved:",
    ROOT / "pose_debug.npz"
)


# ============================================================
# XY TRAJECTORY PLOT
# ============================================================

print_section("9. CREATE TRAJECTORY PLOT")

plt.figure(figsize=(10, 8))

plt.plot(
    gps_x,
    gps_y,
    label="GPS vehicle trajectory"
)

plt.plot(
    camera_positions[:, 0],
    camera_positions[:, 1],
    "o-",
    markersize=3,
    label="Generated camera trajectory"
)

plt.scatter(
    camera_positions[0, 0],
    camera_positions[0, 1],
    s=80,
    label="Start"
)

plt.scatter(
    camera_positions[-1, 0],
    camera_positions[-1, 1],
    s=80,
    label="End"
)

plt.axis("equal")
plt.xlabel("Local X (m)")
plt.ylabel("Local Y (m)")
plt.title("A2D2 GPS and Generated Camera Trajectory")
plt.legend()
plt.grid(True)

plt.savefig(
    TRAJECTORY_PNG,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print("Saved:", TRAJECTORY_PNG)


# ============================================================
# HEADING PLOT
# ============================================================

plt.figure(figsize=(12, 5))

plt.plot(
    np.asarray(
        [r["timestamp"] for r in camera_records]
    ),
    np.degrees(camera_headings),
)

plt.xlabel("Camera timestamp")
plt.ylabel("Heading (degrees)")
plt.title("Generated Camera Heading")
plt.grid(True)

plt.savefig(
    HEADING_PNG,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print("Saved:", HEADING_PNG)


# ============================================================
# PLY WRITER
# ============================================================

def save_trajectory_ply(
    filename,
    positions,
    poses,
    axis_length=1.5,
):
    """
    Save camera trajectory as an ASCII PLY.

    Each camera gets:
        - one white vertex at its position
        - one red-ish X-axis endpoint
        - one green-ish Y-axis endpoint
        - one blue-ish Z-axis endpoint

    Lines connect:
        camera -> X endpoint
        camera -> Y endpoint
        camera -> Z endpoint

    This is purely for visualization/debugging.
    """

    vertices = []
    colors = []
    edges = []

    for i, T in enumerate(poses):

        p = T[:3, 3]
        R = T[:3, :3]

        x = p + axis_length * R[:, 0]
        y = p + axis_length * R[:, 1]
        z = p + axis_length * R[:, 2]

        base = len(vertices)

        vertices.extend(
            [
                p,
                x,
                y,
                z,
            ]
        )

        colors.extend(
            [
                [255, 255, 255],
                [255, 0, 0],
                [0, 255, 0],
                [0, 0, 255],
            ]
        )

        edges.extend(
            [
                [base, base + 1],
                [base, base + 2],
                [base, base + 3],
            ]
        )

    # Add trajectory vertices as an additional polyline.
    trajectory_start = len(vertices)

    for p in positions:
        vertices.append(p)
        colors.append([255, 255, 255])

    for i in range(len(positions) - 1):
        edges.append(
            [
                trajectory_start + i,
                trajectory_start + i + 1,
            ]
        )

    vertices = np.asarray(vertices)
    colors = np.asarray(colors, dtype=np.uint8)
    edges = np.asarray(edges, dtype=np.int32)

    with open(filename, "w") as f:

        f.write("ply\n")
        f.write("format ascii 1.0\n")

        f.write(
            f"element vertex {len(vertices)}\n"
        )

        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")

        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")

        f.write(
            f"element edge {len(edges)}\n"
        )

        f.write(
            "property int vertex1\n"
        )
        f.write(
            "property int vertex2\n"
        )

        f.write("end_header\n")

        for p, c in zip(vertices, colors):

            f.write(
                f"{p[0]:.6f} "
                f"{p[1]:.6f} "
                f"{p[2]:.6f} "
                f"{int(c[0])} "
                f"{int(c[1])} "
                f"{int(c[2])}\n"
            )

        for e in edges:

            f.write(
                f"{e[0]} {e[1]}\n"
            )


# ============================================================
# SAVE PLY
# ============================================================

print_section("10. SAVE CAMERA TRAJECTORY PLY")

save_trajectory_ply(
    TRAJECTORY_PLY,
    camera_positions,
    poses,
)

print("Saved:", TRAJECTORY_PLY)


# ============================================================
# FINAL SUMMARY
# ============================================================

print_section("A2D2 POSE GENERATION COMPLETE")

print("Camera:", CAMERA_NAME)
print("Frames:", len(frame_ids))
print("Stride:", STRIDE)

print("\nGenerated files:")

print(
    "  Pose directory:",
    OUTPUT_DIR
)

print(
    "  Trajectory PLY:",
    TRAJECTORY_PLY
)

print(
    "  XY plot:",
    TRAJECTORY_PNG
)

print(
    "  Heading plot:",
    HEADING_PNG
)

print(
    "  Debug data:",
    ROOT / "pose_debug.npz"
)

print("\nIMPORTANT:")
print(
    "Do NOT use these poses with MapAnything yet."
)
print(
    "First inspect camera_trajectory.ply."
)
print(
    "The PLY is the primary coordinate-frame sanity check."
)