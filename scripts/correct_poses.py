#!/usr/bin/env python3

from pathlib import Path
import csv
import numpy as np


# ============================================================
# CONFIG
# ============================================================

EXTRACTED = Path(
    "/home/lcha0115/bt60_scratch/lcha_data/map-anything/"
    "personal_data/camera_lidar_gps_test_101/extracted"
)

CAMERA = "CAM2"

OUTPUT = (
    EXTRACTED.parent
    / "corrected_poses"
    / CAMERA
)

IMAGE_TIMESTAMP_FILE = (
    EXTRACTED / CAMERA / "timestamps.csv"
)

GPS_FILE = (
    EXTRACTED / "gps" / "position.csv"
)

VELOCITY_FILE = (
    EXTRACTED / "gps" / "velocity.csv"
)

# Maximum allowed timestamp mismatch at the beginning/end
SYNC_TOLERANCE_NS = 200_000_000  # 200 ms


# ============================================================
# YOUR ACTUAL VELODYNE -> CAM2 EXTRINSIC
# ============================================================

# From:
#
# velodyne -> cam2_optical_frame
#
# translation:
# -0.35812314453904737
#  0.7418722578139009
# -0.9622312541336254
#
# quaternion:
# -0.696591590508534
# -0.30318163432606277
#  0.27662795664738776
#  0.5884879151191293


VELODYNE_CAM2_TRANSLATION = np.array(
    [
        -0.35812314453904737,
         0.7418722578139009,
        -0.9622312541336254,
    ],
    dtype=np.float64,
)


VELODYNE_CAM2_QUATERNION = np.array(
    [
        -0.696591590508534,
        -0.30318163432606277,
         0.27662795664738776,
         0.5884879151191293,
    ],
    dtype=np.float64,
)


# ============================================================
# QUATERNION -> ROTATION
# ============================================================

def quaternion_to_rotation(q):

    qx, qy, qz, qw = q

    norm = np.linalg.norm(q)

    if norm < 1e-12:
        raise ValueError(
            "Invalid quaternion."
        )

    qx /= norm
    qy /= norm
    qz /= norm
    qw /= norm

    return np.array(
        [
            [
                1 - 2 * (qy*qy + qz*qz),
                2 * (qx*qy - qz*qw),
                2 * (qx*qz + qy*qw),
            ],
            [
                2 * (qx*qy + qy*qw),
                1 - 2 * (qx*qx + qz*qz),
                2 * (qy*qz - qx*qw),
            ],
            [
                2 * (qx*qz - qy*qw),
                2 * (qy*qz + qx*qw),
                1 - 2 * (qx*qx + qy*qy),
            ],
        ],
        dtype=np.float64,
    )


# ============================================================
# BUILD EXTRINSIC
# ============================================================

R_velodyne_cam2 = quaternion_to_rotation(
    VELODYNE_CAM2_QUATERNION
)

T_velodyne_cam2 = np.eye(
    4,
    dtype=np.float64,
)

T_velodyne_cam2[:3, :3] = (
    R_velodyne_cam2
)

T_velodyne_cam2[:3, 3] = (
    VELODYNE_CAM2_TRANSLATION
)


# ============================================================
# CSV LOADER
# ============================================================

def load_csv(path):

    if not path.exists():

        raise FileNotFoundError(
            f"\nFile not found:\n{path}"
        )

    with open(
        path,
        "r",
        newline="",
    ) as f:

        rows = list(
            csv.DictReader(f)
        )

    if not rows:

        raise RuntimeError(
            f"\nCSV is empty:\n{path}"
        )

    return rows


def find_column(
    rows,
    candidates,
):

    fields = list(
        rows[0].keys()
    )

    # Exact
    for candidate in candidates:

        if candidate in fields:
            return candidate

    # Case insensitive
    lower = {
        field.lower(): field
        for field in fields
    }

    for candidate in candidates:

        if candidate.lower() in lower:

            return lower[
                candidate.lower()
            ]

    raise RuntimeError(
        f"\nCould not find any of:\n"
        f"{candidates}\n\n"
        f"Available columns:\n"
        f"{fields}"
    )


# ============================================================
# IMAGE TIMESTAMPS
# ============================================================

def load_image_timestamps():

    rows = load_csv(
        IMAGE_TIMESTAMP_FILE
    )

    timestamp_col = find_column(
        rows,
        [
            "ros_timestamp_ns",
            "timestamp_ns",
            "timestamp",
        ],
    )

    timestamps = []
    filenames = []

    for row in rows:

        # Your original extraction contains
        # raw/non-raw streams.
        if "stream" in row:

            if (
                row["stream"]
                .strip()
                .lower()
                != "raw"
            ):
                continue

        try:

            timestamp = int(
                float(
                    row[timestamp_col]
                )
            )

        except Exception:

            continue

        timestamps.append(
            timestamp
        )

        filename = ""

        for key in [
            "filename",
            "file",
            "image",
            "name",
            "path",
        ]:

            if key in row:

                filename = row[key]
                break

        filenames.append(
            filename
        )

    timestamps = np.asarray(
        timestamps,
        dtype=np.int64,
    )

    order = np.argsort(
        timestamps
    )

    timestamps = timestamps[order]

    filenames = [
        filenames[i]
        for i in order
    ]

    return (
        timestamps,
        filenames,
    )


# ============================================================
# GPS
# ============================================================

def load_gps():

    rows = load_csv(
        GPS_FILE
    )

    timestamp_col = find_column(
        rows,
        [
            "ros_timestamp_ns",
            "timestamp_ns",
            "timestamp",
        ],
    )

    lat_col = find_column(
        rows,
        [
            "latitude_deg",
            "latitude",
            "lat",
        ],
    )

    lon_col = find_column(
        rows,
        [
            "longitude_deg",
            "longitude",
            "lon",
        ],
    )

    alt_col = find_column(
        rows,
        [
            "altitude_m",
            "altitude",
            "alt",
        ],
    )

    timestamps = []
    latitudes = []
    longitudes = []
    altitudes = []

    for row in rows:

        try:

            timestamps.append(
                int(
                    float(
                        row[timestamp_col]
                    )
                )
            )

            latitudes.append(
                float(
                    row[lat_col]
                )
            )

            longitudes.append(
                float(
                    row[lon_col]
                )
            )

            altitudes.append(
                float(
                    row[alt_col]
                )
            )

        except Exception:

            continue

    timestamps = np.asarray(
        timestamps,
        dtype=np.int64,
    )

    latitudes = np.asarray(
        latitudes,
        dtype=np.float64,
    )

    longitudes = np.asarray(
        longitudes,
        dtype=np.float64,
    )

    altitudes = np.asarray(
        altitudes,
        dtype=np.float64,
    )

    order = np.argsort(
        timestamps
    )

    return (
        timestamps[order],
        latitudes[order],
        longitudes[order],
        altitudes[order],
    )


# ============================================================
# GPS -> LOCAL ENU
# ============================================================

def gps_to_enu(
    latitudes,
    longitudes,
    altitudes,
):

    """
    Local ENU frame:

        X = East
        Y = North
        Z = Up

    Origin = first GPS sample.
    """

    lat0 = latitudes[0]
    lon0 = longitudes[0]
    alt0 = altitudes[0]

    R_EARTH = 6378137.0

    lat0_rad = np.deg2rad(
        lat0
    )

    east = (
        np.deg2rad(
            longitudes - lon0
        )
        * R_EARTH
        * np.cos(lat0_rad)
    )

    north = (
        np.deg2rad(
            latitudes - lat0
        )
        * R_EARTH
    )

    up = (
        altitudes - alt0
    )

    return np.column_stack(
        [
            east,
            north,
            up,
        ]
    )


# ============================================================
# COURSE
# ============================================================

def load_course():

    rows = load_csv(
        VELOCITY_FILE
    )

    timestamp_col = find_column(
        rows,
        [
            "ros_timestamp_ns",
            "timestamp_ns",
            "timestamp",
        ],
    )

    course_col = find_column(
        rows,
        [
            "course_deg",
            "course",
        ],
    )

    timestamps = []
    courses = []

    for row in rows:

        try:

            timestamps.append(
                int(
                    float(
                        row[timestamp_col]
                    )
                )
            )

            courses.append(
                float(
                    row[course_col]
                )
            )

        except Exception:

            continue

    timestamps = np.asarray(
        timestamps,
        dtype=np.int64,
    )

    courses = np.asarray(
        courses,
        dtype=np.float64,
    )

    order = np.argsort(
        timestamps
    )

    return (
        timestamps[order],
        courses[order],
    )


# ============================================================
# COURSE -> ENU YAW
# ============================================================

def course_to_enu_yaw(
    course_deg
):

    """
    SBG course:

        0°   = North
        90°  = East
        180° = South
        270° = West

    ENU yaw:

        0°   = East
        90°  = North
        180° = West
        270° = South

    Therefore:

        yaw = 90° - course
    """

    return np.deg2rad(
        90.0 - course_deg
    )


# ============================================================
# INTERPOLATE GPS WITH TOLERANCE
# ============================================================

def interpolate_gps(
    image_times,
    gps_times,
    gps_positions,
):

    result = np.zeros(
        (
            len(image_times),
            3,
        ),
        dtype=np.float64,
    )

    for i, t in enumerate(
        image_times
    ):

        # ----------------------------------------------------
        # Before GPS
        # ----------------------------------------------------

        if t < gps_times[0]:

            delta = (
                gps_times[0] - t
            )

            if delta > SYNC_TOLERANCE_NS:

                raise RuntimeError(
                    f"Camera frame {i} is "
                    f"{delta / 1e6:.3f} ms "
                    f"before GPS data. "
                    f"Tolerance = "
                    f"{SYNC_TOLERANCE_NS / 1e6:.0f} ms."
                )

            # Small synchronization gap:
            # use first GPS position.
            result[i] = (
                gps_positions[0]
            )

            continue

        # ----------------------------------------------------
        # After GPS
        # ----------------------------------------------------

        if t > gps_times[-1]:

            delta = (
                t - gps_times[-1]
            )

            if delta > SYNC_TOLERANCE_NS:

                raise RuntimeError(
                    f"Camera frame {i} is "
                    f"{delta / 1e6:.3f} ms "
                    f"after GPS data. "
                    f"Tolerance = "
                    f"{SYNC_TOLERANCE_NS / 1e6:.0f} ms."
                )

            result[i] = (
                gps_positions[-1]
            )

            continue

        # ----------------------------------------------------
        # Normal interpolation
        # ----------------------------------------------------

        idx = np.searchsorted(
            gps_times,
            t,
            side="right",
        ) - 1

        idx = max(
            0,
            min(
                idx,
                len(gps_times) - 2,
            ),
        )

        t0 = gps_times[idx]
        t1 = gps_times[idx + 1]

        p0 = gps_positions[idx]
        p1 = gps_positions[idx + 1]

        if t1 == t0:

            result[i] = p0

        else:

            alpha = (
                (t - t0)
                /
                float(t1 - t0)
            )

            result[i] = (
                (1.0 - alpha) * p0
                +
                alpha * p1
            )

    return result


# ============================================================
# INTERPOLATE COURSE WITH TOLERANCE
# ============================================================

def interpolate_course(
    image_times,
    course_times,
    course_deg,
):

    yaw = course_to_enu_yaw(
        course_deg
    )

    yaw_unwrapped = np.unwrap(
        yaw
    )

    result = np.zeros(
        len(image_times),
        dtype=np.float64,
    )

    for i, t in enumerate(
        image_times
    ):

        # ----------------------------------------------------
        # Before course
        # ----------------------------------------------------

        if t < course_times[0]:

            delta = (
                course_times[0] - t
            )

            if delta > SYNC_TOLERANCE_NS:

                raise RuntimeError(
                    f"Camera frame {i} is "
                    f"{delta / 1e6:.3f} ms "
                    f"before course data."
                )

            result[i] = (
                yaw_unwrapped[0]
            )

            continue

        # ----------------------------------------------------
        # After course
        # ----------------------------------------------------

        if t > course_times[-1]:

            delta = (
                t - course_times[-1]
            )

            if delta > SYNC_TOLERANCE_NS:

                raise RuntimeError(
                    f"Camera frame {i} is "
                    f"{delta / 1e6:.3f} ms "
                    f"after course data."
                )

            result[i] = (
                yaw_unwrapped[-1]
            )

            continue

        # ----------------------------------------------------
        # Normal interpolation
        # ----------------------------------------------------

        result[i] = np.interp(
            t,
            course_times,
            yaw_unwrapped,
        )

    # Wrap to [-pi, pi]
    result = np.arctan2(
        np.sin(result),
        np.cos(result),
    )

    return result


# ============================================================
# WORLD -> VELODYNE
# ============================================================

def make_world_velodyne(
    position,
    yaw,
):

    """
    Velodyne is the BASE FRAME.

    World:
        ENU

    Velodyne:
        X = forward
        Y = left
        Z = up

    yaw is ENU heading measured from East.
    """

    c = np.cos(yaw)
    s = np.sin(yaw)

    R_world_velodyne = np.array(
        [
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    T = np.eye(
        4,
        dtype=np.float64,
    )

    T[:3, :3] = (
        R_world_velodyne
    )

    T[:3, 3] = position

    return T


# ============================================================
# PLY
# ============================================================

def save_trajectory_ply(
    path,
    points,
):

    with open(
        path,
        "w",
    ) as f:

        f.write(
            "ply\n"
        )

        f.write(
            "format ascii 1.0\n"
        )

        f.write(
            f"element vertex {len(points)}\n"
        )

        f.write(
            "property float x\n"
        )

        f.write(
            "property float y\n"
        )

        f.write(
            "property float z\n"
        )

        f.write(
            "end_header\n"
        )

        for p in points:

            f.write(
                f"{p[0]:.6f} "
                f"{p[1]:.6f} "
                f"{p[2]:.6f}\n"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print("MAPANYTHING CAM2 POSE GENERATOR")
    print("=" * 80)

    print()
    print("Base frame: VELODYNE")
    print("Camera frame: CAM2 optical")
    print(
        f"Sync tolerance: "
        f"{SYNC_TOLERANCE_NS / 1e6:.0f} ms"
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    pose_dir = (
        OUTPUT / "poses"
    )

    pose_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load image timestamps
    # --------------------------------------------------------

    print()
    print("Loading camera timestamps...")

    (
        image_times,
        image_files,
    ) = load_image_timestamps()

    print(
        f"  Camera frames: "
        f"{len(image_times)}"
    )

    # --------------------------------------------------------
    # Load GPS
    # --------------------------------------------------------

    print()
    print("Loading GPS...")

    (
        gps_times,
        latitudes,
        longitudes,
        altitudes,
    ) = load_gps()

    print(
        f"  GPS samples: "
        f"{len(gps_times)}"
    )

    gps_positions = gps_to_enu(
        latitudes,
        longitudes,
        altitudes,
    )

    # --------------------------------------------------------
    # Load course
    # --------------------------------------------------------

    print()
    print("Loading SBG course...")

    (
        course_times,
        course_deg,
    ) = load_course()

    print(
        f"  Course samples: "
        f"{len(course_times)}"
    )

    # --------------------------------------------------------
    # Timestamp diagnostics
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("TIMESTAMP SYNCHRONISATION")
    print("=" * 80)

    cam_start = int(
        image_times[0]
    )

    cam_end = int(
        image_times[-1]
    )

    gps_start = int(
        gps_times[0]
    )

    gps_end = int(
        gps_times[-1]
    )

    course_start = int(
        course_times[0]
    )

    course_end = int(
        course_times[-1]
    )

    print(
        f"\nCamera:"
        f"\n  {cam_start}"
        f"\n  {cam_end}"
    )

    print(
        f"\nGPS:"
        f"\n  {gps_start}"
        f"\n  {gps_end}"
    )

    print(
        f"\nCourse:"
        f"\n  {course_start}"
        f"\n  {course_end}"
    )

    if cam_start < gps_start:

        print(
            f"\nCamera starts "
            f"{(gps_start - cam_start)/1e6:.3f} ms "
            f"before GPS."
        )

    if cam_start < course_start:

        print(
            f"Camera starts "
            f"{(course_start - cam_start)/1e6:.3f} ms "
            f"before course."
        )

    if cam_end > gps_end:

        print(
            f"Camera ends "
            f"{(cam_end - gps_end)/1e6:.3f} ms "
            f"after GPS."
        )

    if cam_end > course_end:

        print(
            f"Camera ends "
            f"{(cam_end - course_end)/1e6:.3f} ms "
            f"after course."
        )

    # --------------------------------------------------------
    # Interpolate
    # --------------------------------------------------------

    print()
    print(
        "Interpolating GPS..."
    )

    camera_positions = interpolate_gps(
        image_times,
        gps_times,
        gps_positions,
    )

    print(
        "Interpolating SBG course..."
    )

    camera_yaws = interpolate_course(
        image_times,
        course_times,
        course_deg,
    )

    # --------------------------------------------------------
    # Print extrinsic
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("VELODYNE -> CAM2")
    print("=" * 80)

    print(
        T_velodyne_cam2
    )

    # --------------------------------------------------------
    # Generate poses
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("GENERATING POSES")
    print("=" * 80)

    camera_centres = []

    trajectory_rows = []

    previous_camera = None

    for i in range(
        len(image_times)
    ):

        timestamp = int(
            image_times[i]
        )

        position = (
            camera_positions[i]
        )

        yaw = (
            camera_yaws[i]
        )

        # ----------------------------------------------------
        # WORLD -> VELODYNE
        # ----------------------------------------------------

        T_world_velodyne = (
            make_world_velodyne(
                position,
                yaw,
            )
        )

        # ----------------------------------------------------
        # WORLD -> CAM2
        # ----------------------------------------------------

        T_world_cam2 = (
            T_world_velodyne
            @
            T_velodyne_cam2
        )

        # ----------------------------------------------------
        # Camera centre
        # ----------------------------------------------------

        camera_position = (
            T_world_cam2[:3, 3]
        )

        camera_centres.append(
            camera_position.copy()
        )

        # ----------------------------------------------------
        # Step
        # ----------------------------------------------------

        if previous_camera is None:

            step = 0.0

        else:

            step = np.linalg.norm(
                camera_position
                - previous_camera
            )

        previous_camera = (
            camera_position.copy()
        )

        # ----------------------------------------------------
        # Save pose
        # ----------------------------------------------------

        pose_file = (
            pose_dir
            / f"{i:06d}.txt"
        )

        np.savetxt(
            pose_file,
            T_world_cam2,
            fmt="%.12f",
        )

        trajectory_rows.append(
            [
                i,
                timestamp,
                (
                    image_files[i]
                    if i < len(image_files)
                    else ""
                ),
                position[0],
                position[1],
                position[2],
                np.rad2deg(yaw),
                camera_position[0],
                camera_position[1],
                camera_position[2],
                step,
            ]
        )

        if (
            i < 10
            or i % 1000 == 0
        ):

            course_display = (
                90.0
                - np.rad2deg(yaw)
            ) % 360.0

            print(
                f"{i:6d} | "
                f"course="
                f"{course_display:8.3f}° | "
                f"yaw="
                f"{np.rad2deg(yaw):8.3f}° | "
                f"cam=("
                f"{camera_position[0]:9.3f}, "
                f"{camera_position[1]:9.3f}, "
                f"{camera_position[2]:9.3f}) | "
                f"step="
                f"{step:.4f} m"
            )

    camera_centres = np.asarray(
        camera_centres
    )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    trajectory_csv = (
        OUTPUT
        / "trajectory.csv"
    )

    with open(
        trajectory_csv,
        "w",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "frame",
                "ros_timestamp_ns",
                "image",
                "velodyne_x",
                "velodyne_y",
                "velodyne_z",
                "yaw_deg",
                "camera_x",
                "camera_y",
                "camera_z",
                "step_distance_m",
            ]
        )

        writer.writerows(
            trajectory_rows
        )

    # --------------------------------------------------------
    # Save PLY
    # --------------------------------------------------------

    trajectory_ply = (
        OUTPUT
        / "camera_trajectory.ply"
    )

    save_trajectory_ply(
        trajectory_ply,
        camera_centres,
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    steps = np.linalg.norm(
        np.diff(
            camera_centres,
            axis=0,
        ),
        axis=1,
    )

    print()
    print("=" * 80)
    print("TRAJECTORY STATISTICS")
    print("=" * 80)

    print(
        f"Mean step:     "
        f"{np.mean(steps):.4f} m"
    )

    print(
        f"Median step:   "
        f"{np.median(steps):.4f} m"
    )

    print(
        f"Min step:      "
        f"{np.min(steps):.4f} m"
    )

    print(
        f"Max step:      "
        f"{np.max(steps):.4f} m"
    )

    print(
        f"Total distance:"
        f" {np.sum(steps):.3f} m"
    )

    # --------------------------------------------------------
    # First / second / last pose
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("FIRST POSE")
    print("=" * 80)

    print(
        np.loadtxt(
            pose_dir / "000000.txt"
        )
    )

    if len(image_times) > 1:

        print()
        print("=" * 80)
        print("SECOND POSE")
        print("=" * 80)

        print(
            np.loadtxt(
                pose_dir / "000001.txt"
            )
        )

    print()
    print("=" * 80)
    print("LAST POSE")
    print("=" * 80)

    print(
        np.loadtxt(
            pose_dir
            / f"{len(image_times)-1:06d}.txt"
        )
    )

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)

    print()
    print(
        f"Pose files:\n{pose_dir}"
    )

    print()
    print(
        f"Trajectory CSV:\n"
        f"{trajectory_csv}"
    )

    print()
    print(
        f"Trajectory PLY:\n"
        f"{trajectory_ply}"
    )


if __name__ == "__main__":
    main()