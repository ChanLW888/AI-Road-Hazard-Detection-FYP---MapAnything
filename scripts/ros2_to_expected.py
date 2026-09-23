import csv
import ast
import re
from pathlib import Path

import numpy as np


# ============================================================================
# CONFIG
# ============================================================================

EXTRACTED = Path(
    "/home/lcha0115/bt60_scratch/lcha_data/map-anything/"
    "personal_data/camera_lidar_imu_test_04/extracted"
)

OUTPUT = EXTRACTED


# Camera streams to process.
CAMERAS = ["CAM1", "CAM2", "CAM6"]


# ============================================================================
# HELPERS
# ============================================================================

def parse_list(value):
    """
    Parse calibration arrays from camera_info.csv.

    Handles formats such as:

        [1.0, 2.0, 3.0]

        [1.0 2.0 3.0]

        np.float64(1.0), np.float64(2.0), ...
    """

    if value is None:
        return None

    value = str(value).strip()

    if not value:
        return None

    # ------------------------------------------------------------
    # Remove numpy scalar wrappers
    # ------------------------------------------------------------

    value = value.replace("np.float64(", "")
    value = value.replace("np.float32(", "")
    value = value.replace("np.int64(", "")
    value = value.replace("np.int32(", "")

    # Remove closing parentheses left by np.float64(...)
    value = value.replace(")", "")

    # Remove brackets
    value = value.replace("[", "")
    value = value.replace("]", "")

    # ------------------------------------------------------------
    # Convert commas to spaces
    # ------------------------------------------------------------

    value = value.replace(",", " ")

    # ------------------------------------------------------------
    # Parse numeric values
    # ------------------------------------------------------------

    values = np.fromstring(
        value,
        sep=" ",
        dtype=np.float64,
    )

    if values.size == 0:

        raise ValueError(
            "Could not parse calibration array:\n"
            + str(value)
        )

    return values


def load_csv(path):
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def write_matrix(path, matrix):
    """
    Write a 4x4 pose matrix in plain text.
    """

    with open(path, "w") as f:

        for row in matrix:

            f.write(
                " ".join(
                    f"{x:.12f}"
                    for x in row
                )
                + "\n"
            )


def normalize(v):
    n = np.linalg.norm(v)

    if n < 1e-12:
        return v

    return v / n


# ============================================================================
# GPS -> LOCAL METRIC COORDINATES
# ============================================================================

def gps_to_local(gps_rows):
    """
    Convert latitude/longitude to a local tangent-plane approximation.

    Origin = first GPS sample.

    X = East
    Y = North
    Z = altitude difference
    """

    lat0 = float(gps_rows[0]["latitude_deg"])
    lon0 = float(gps_rows[0]["longitude_deg"])
    alt0 = float(gps_rows[0]["altitude_m"])

    R_EARTH = 6378137.0

    lat0_rad = np.deg2rad(lat0)

    positions = []

    for row in gps_rows:

        lat = float(row["latitude_deg"])
        lon = float(row["longitude_deg"])
        alt = float(row["altitude_m"])

        dlat = np.deg2rad(lat - lat0)
        dlon = np.deg2rad(lon - lon0)

        east = (
            dlon
            * R_EARTH
            * np.cos(lat0_rad)
        )

        north = (
            dlat
            * R_EARTH
        )

        up = alt - alt0

        positions.append(
            [
                east,
                north,
                up,
            ]
        )

    return np.asarray(positions)


# ============================================================================
# GPS VELOCITY -> YAW
# ============================================================================

def velocity_yaw(velocity_rows):
    """
    Estimate heading from GPS velocity.

    Assumes ENU velocity:

        vx = East
        vy = North
        vz = Up

    yaw is measured from +X (East) toward +Y (North).
    """

    result = {}

    for row in velocity_rows:

        t = int(row["ros_timestamp_ns"])

        vx = float(row["velocity_x"])
        vy = float(row["velocity_y"])

        speed = np.hypot(vx, vy)

        if speed < 0.5:

            result[t] = None

        else:

            result[t] = np.arctan2(
                vy,
                vx,
            )

    return result


# ============================================================================
# NEAREST TIMESTAMP
# ============================================================================

def nearest_timestamp(t, timestamps):

    if len(timestamps) == 0:
        return None, None

    timestamps = np.asarray(
        timestamps,
        dtype=np.int64,
    )

    index = np.searchsorted(
        timestamps,
        t,
    )

    candidates = []

    if index > 0:
        candidates.append(
            timestamps[index - 1]
        )

    if index < len(timestamps):
        candidates.append(
            timestamps[index]
        )

    best = min(
        candidates,
        key=lambda x: abs(int(x) - t),
    )

    return int(best), abs(int(best) - t)


# ============================================================================
# ROTATION
# ============================================================================

def rotation_z(yaw):

    c = np.cos(yaw)
    s = np.sin(yaw)

    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1],
    ])


# ============================================================================
# QUATERNION -> ROTATION
# ============================================================================

def quaternion_to_rotation(x, y, z, w):

    return np.array([
        [
            1 - 2*y*y - 2*z*z,
            2*x*y - 2*z*w,
            2*x*z + 2*y*w,
        ],
        [
            2*x*y + 2*z*w,
            1 - 2*x*x - 2*z*z,
            2*y*z - 2*x*w,
        ],
        [
            2*x*z - 2*y*w,
            2*y*z + 2*x*w,
            1 - 2*x*x - 2*y*y,
        ],
    ])


# ============================================================================
# TF PARSER
# ============================================================================

def parse_tf_static(path):

    if not path.exists():

        print(
            "WARNING: tf_static.txt not found:"
        )

        print(path)

        return []

    transforms = []

    with open(path) as f:

        lines = [
            line.rstrip()
            for line in f
        ]

    current_time = None
    current_parent = None
    current_child = None
    current_translation = None
    current_quaternion = None

    for line in lines:

        if line.startswith("ROS timestamp:"):

            current_time = int(
                line.split(":")[1].strip()
            )

        elif " -> " in line:

            current_parent, current_child = (
                line.strip().split(" -> ")
            )

        elif "translation:" in line:

            values = line.split(
                "translation:"
            )[1].strip().split()

            current_translation = np.array(
                [float(v) for v in values]
            )

        elif "quaternion:" in line:

            values = line.split(
                "quaternion:"
            )[1].strip().split()

            current_quaternion = np.array(
                [float(v) for v in values]
            )

            if (
                current_parent is not None
                and current_child is not None
                and current_translation is not None
                and current_quaternion is not None
            ):

                transforms.append({
                    "time": current_time,
                    "parent": current_parent,
                    "child": current_child,
                    "translation": current_translation.copy(),
                    "quaternion": current_quaternion.copy(),
                })

                current_translation = None
                current_quaternion = None

    return transforms


# ============================================================================
# EXTRINSIC LOOKUP
# ============================================================================

def find_camera_extrinsic(transforms, camera_name):

    """
    Try to find a TF involving the camera.

    We deliberately print all candidates rather than silently guessing.
    """

    candidates = []

    camera_tokens = [
        camera_name.lower(),
        camera_name.replace("CAM", "cam").lower(),
    ]

    for tr in transforms:

        parent = tr["parent"].lower()
        child = tr["child"].lower()

        if any(
            token in parent or token in child
            for token in camera_tokens
        ):

            candidates.append(tr)

    return candidates


# ============================================================================
# CAMERA CALIBRATION
# ============================================================================

def build_camera_calibration(camera):

    camera_info_file = (
        EXTRACTED
        / camera
        / "camera_info.csv"
    )

    if not camera_info_file.exists():

        raise FileNotFoundError(
            camera_info_file
        )

    rows = load_csv(
        camera_info_file
    )

    if len(rows) == 0:

        raise RuntimeError(
            f"No camera info for {camera}"
        )

    row = rows[0]

    K = parse_list(row["K"])
    D = parse_list(row["D"])
    R = parse_list(row["R"])
    P = parse_list(row["P"])

    K = K.reshape(3, 3)
    R = R.reshape(3, 3)
    P = P.reshape(3, 4)

    width = int(row["width"])
    height = int(row["height"])

    output_dir = (
        OUTPUT
        / "calibration"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ------------------------------------------------------------------------
    # NumPy calibration file
    # ------------------------------------------------------------------------

    np.savez(
        output_dir
        / f"{camera}_intrinsics.npz",

        K=K,
        D=D,
        R=R,
        P=P,

        width=width,
        height=height,
    )

    # ------------------------------------------------------------------------
    # Human-readable calibration
    # ------------------------------------------------------------------------

    calib_file = (
        output_dir
        / f"{camera}_calib.txt"
    )

    with open(calib_file, "w") as f:

        f.write(
            f"# {camera} calibration\n"
        )

        f.write(
            f"width: {width}\n"
        )

        f.write(
            f"height: {height}\n"
        )

        f.write(
            "\nK:\n"
        )

        for row_K in K:

            f.write(
                " ".join(
                    f"{x:.12f}"
                    for x in row_K
                )
                + "\n"
            )

        f.write(
            "\nD:\n"
        )

        f.write(
            " ".join(
                f"{x:.12f}"
                for x in D
            )
            + "\n"
        )

        f.write(
            "\nR:\n"
        )

        for row_R in R:

            f.write(
                " ".join(
                    f"{x:.12f}"
                    for x in row_R
                )
                + "\n"
            )

        f.write(
            "\nP:\n"
        )

        for row_P in P:

            f.write(
                " ".join(
                    f"{x:.12f}"
                    for x in row_P
                )
                + "\n"
            )

    print()
    print(camera)
    print("  Resolution:", width, "x", height)
    print("  fx:", K[0, 0])
    print("  fy:", K[1, 1])
    print("  cx:", K[0, 2])
    print("  cy:", K[1, 2])

    return K, D, R, P


# ============================================================================
# BUILD CAMERA POSES
# ============================================================================

def build_poses(
    camera,
    gps_rows,
    velocity_rows,
    transforms,
):

    image_timestamp_file = (
        EXTRACTED
        / camera
        / "timestamps.csv"
    )

    if not image_timestamp_file.exists():

        raise FileNotFoundError(
            image_timestamp_file
        )

    image_rows = load_csv(
        image_timestamp_file
    )

    image_rows = [
        r for r in image_rows
        if r["stream"] == "raw"
    ]

    if len(image_rows) == 0:

        raise RuntimeError(
            f"No raw images found for {camera}"
        )

    # ------------------------------------------------------------------------
    # GPS trajectory
    # ------------------------------------------------------------------------

    gps_positions = gps_to_local(
        gps_rows
    )

    gps_times = np.asarray(
        [
            int(r["ros_timestamp_ns"])
            for r in gps_rows
        ],
        dtype=np.int64,
    )

    # ------------------------------------------------------------------------
    # GPS velocity heading
    # ------------------------------------------------------------------------

    velocity_headings = velocity_yaw(
        velocity_rows
    )

    velocity_times = sorted(
        velocity_headings.keys()
    )

    # ------------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------------

    pose_dir = (
        OUTPUT
        / "poses"
        / camera
    )

    pose_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    trajectory_rows = []

    previous_yaw = 0.0

    # ------------------------------------------------------------------------
    # Process every image
    # ------------------------------------------------------------------------

    for index, image_row in enumerate(image_rows):

        t = int(
            image_row["ros_timestamp_ns"]
        )

        # ---------------------------------------------------------------
        # Position
        # ---------------------------------------------------------------

        gps_t, gps_error = nearest_timestamp(
            t,
            gps_times,
        )

        if gps_t is None:

            print(
                f"WARNING: no GPS for {camera} "
                f"image {index}"
            )

            continue

        gps_index = np.searchsorted(
            gps_times,
            gps_t,
        )

        gps_index = min(
            gps_index,
            len(gps_positions) - 1,
        )

        position = (
            gps_positions[gps_index]
        )

        # ---------------------------------------------------------------
        # Heading
        # ---------------------------------------------------------------

        vel_t, vel_error = nearest_timestamp(
            t,
            velocity_times,
        )

        yaw = None

        if vel_t is not None:

            yaw = velocity_headings[
                vel_t
            ]

        if yaw is None:

            yaw = previous_yaw

        else:

            previous_yaw = yaw

        # ---------------------------------------------------------------
        # World -> body rotation
        #
        # We construct an ENU world:
        #
        # X = East
        # Y = North
        # Z = Up
        #
        # Camera pose will initially use:
        #
        # roll  = 0
        # pitch = 0
        # yaw   = GPS heading
        # ---------------------------------------------------------------

        R_world_body = rotation_z(
            yaw
        )

        # Camera pose in world.
        #
        # For now this is the vehicle/body trajectory.
        # Camera TF is handled separately below.
        #

        T_world_body = np.eye(4)

        T_world_body[:3, :3] = (
            R_world_body
        )

        T_world_body[:3, 3] = position

        # ---------------------------------------------------------------
        # Save
        # ---------------------------------------------------------------

        pose_file = (
            pose_dir
            / f"{index:06d}.txt"
        )

        write_matrix(
            pose_file,
            T_world_body,
        )

        trajectory_rows.append(
            (
                index,
                t,
                position[0],
                position[1],
                position[2],
                np.rad2deg(yaw),
                gps_error,
                vel_error if vel_t is not None else "",
            )
        )

    # ------------------------------------------------------------------------
    # Save trajectory CSV
    # ------------------------------------------------------------------------

    trajectory_file = (
        OUTPUT
        / "poses"
        / "trajectory.csv"
    )

    write_csv(
        trajectory_file,
        [
            "frame",
            "ros_timestamp_ns",
            "x_east_m",
            "y_north_m",
            "z_up_m",
            "yaw_deg",
            "gps_delta_ns",
            "velocity_delta_ns",
        ],
        trajectory_rows,
    )

    print(
        f"{camera}: generated "
        f"{len(trajectory_rows)} poses"
    )


# ============================================================================
# CSV WRITER
# ============================================================================

def write_csv(path, header, rows):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        path,
        "w",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(header)

        writer.writerows(rows)


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 80)
    print("A2D2 EXTRACTED DATA -> CALIBRATION + POSES")
    print("=" * 80)

    print()
    print("Input:")
    print(EXTRACTED)

    if not EXTRACTED.exists():

        raise FileNotFoundError(
            EXTRACTED
        )

    # =========================================================================
    # GPS
    # =========================================================================

    gps_file = (
        EXTRACTED
        / "gps"
        / "position.csv"
    )

    velocity_file = (
        EXTRACTED
        / "gps"
        / "velocity.csv"
    )

    if not gps_file.exists():

        raise FileNotFoundError(
            gps_file
        )

    if not velocity_file.exists():

        raise FileNotFoundError(
            velocity_file
        )

    gps_rows = load_csv(
        gps_file
    )

    velocity_rows = load_csv(
        velocity_file
    )

    print()
    print("GPS samples:")
    print(len(gps_rows))

    print("GPS velocity samples:")
    print(len(velocity_rows))

    # =========================================================================
    # TF
    # =========================================================================

    tf_file = (
        EXTRACTED
        / "tf"
        / "tf_static.txt"
    )

    transforms = parse_tf_static(
        tf_file
    )

    print()
    print("TF static transforms:")
    print(len(transforms))

    for tr in transforms:

        print(
            f"  {tr['parent']} "
            f"-> "
            f"{tr['child']}"
        )

    # =========================================================================
    # CAMERA CALIBRATION
    # =========================================================================

    print()
    print("=" * 80)
    print("CAMERA CALIBRATION")
    print("=" * 80)

    for camera in CAMERAS:

        build_camera_calibration(
            camera
        )

    # =========================================================================
    # EXTRINSICS
    # =========================================================================

    print()
    print("=" * 80)
    print("CAMERA EXTRINSICS")
    print("=" * 80)

    extrinsics_file = (
        OUTPUT
        / "calibration"
        / "extrinsics.txt"
    )

    with open(
        extrinsics_file,
        "w"
    ) as f:

        for camera in CAMERAS:

            candidates = find_camera_extrinsic(
                transforms,
                camera
            )

            f.write(
                f"\n{'=' * 70}\n"
            )

            f.write(
                f"{camera}\n"
            )

            f.write(
                f"{'=' * 70}\n"
            )

            if not candidates:

                f.write(
                    "NO CAMERA TF FOUND\n"
                )

                print(
                    camera,
                    ": NO TF FOUND"
                )

                continue

            print()
            print(
                camera,
                "TF candidates:"
            )

            for tr in candidates:

                print(
                    f"  {tr['parent']} "
                    f"-> "
                    f"{tr['child']}"
                )

                q = tr["quaternion"]

                R = quaternion_to_rotation(
                    q[0],
                    q[1],
                    q[2],
                    q[3],
                )

                T = np.eye(4)

                T[:3, :3] = R
                T[:3, 3] = tr[
                    "translation"
                ]

                f.write(
                    f"\n{tr['parent']} "
                    f"-> "
                    f"{tr['child']}\n"
                )

                for row in T:

                    f.write(
                        " ".join(
                            f"{x:.12f}"
                            for x in row
                        )
                        + "\n"
                    )

    # =========================================================================
    # POSES
    # =========================================================================

    print()
    print("=" * 80)
    print("GENERATING CAMERA POSES")
    print("=" * 80)

    for camera in CAMERAS:

        build_poses(
            camera,
            gps_rows,
            velocity_rows,
            transforms,
        )

    # =========================================================================
    # DONE
    # =========================================================================

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)

    print()
    print("Calibration:")
    print(
        OUTPUT / "calibration"
    )

    print()
    print("Poses:")
    print(
        OUTPUT / "poses"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "The initial poses use GPS position "
        "+ GPS velocity heading."
    )

    print(
        "Roll and pitch are currently set to zero."
    )

    print(
        "Camera TF extrinsics are extracted separately "
        "from tf_static."
    )


if __name__ == "__main__":
    main()