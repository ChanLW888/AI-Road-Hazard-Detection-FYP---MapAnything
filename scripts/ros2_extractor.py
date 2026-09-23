import csv
import struct
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
from rosbags.typesys import register_types, types
from rosbags.typesys.msg import get_types_from_msg


# ============================================================================
# CONFIG
# ============================================================================

BAG_PATH = Path(
    "/home/lcha0115/bt60_scratch/lcha_data/map-anything/"
    "personal_data/camera_lidar_imu_test_04"
)

OUT = BAG_PATH / "extracted_new"

SBG_MSG_DIR = Path(
    "/home/lcha0115/bt60_scratch/lcha_data/sbg_ros2_driver/msg"
)

# Nearest-neighbour synchronisation tolerance.
# 20 ms = 20,000,000 ns.
TOL_NS = 20_000_000


CAMERAS = {
    "CAM1": (
        "/CAM1/CAM1_node/image_raw",
        "/CAM1/CAM1_node/image_rect",
        "/CAM1/CAM1_node/camera_info",
    ),
    "CAM2": (
        "/CAM2/CAM2_node/image_raw",
        "/CAM2/CAM2_node/image_rect",
        "/CAM2/CAM2_node/camera_info",
    ),
    "CAM6": (
        "/CAM6/CAM6_node/image_raw",
        "/CAM6/CAM6_node/image_rect",
        "/CAM6/CAM6_node/camera_info",
    ),
}

LIDAR = "/velodyne_points"
IMU = "/sbg/imu_data"
GPSPOS = "/sbg/gps_pos"
GPSVEL = "/sbg/gps_vel"
TF = "/tf_static"


# ============================================================================
# SBG MESSAGE REGISTRATION
# ============================================================================

def register_sbg_messages():
    """
    Register all SBG .msg definitions into the rosbags type system.

    The bag uses:
        sbg_driver/msg/...

    but the source directory is:
        sbg_ros2_driver/msg/...

    Therefore we explicitly register the definitions under sbg_driver/msg.
    """

    print("\n" + "=" * 80)
    print("REGISTERING SBG MESSAGE DEFINITIONS")
    print("=" * 80)

    if not SBG_MSG_DIR.exists():
        raise FileNotFoundError(
            f"SBG message directory not found:\n{SBG_MSG_DIR}"
        )

    msg_files = sorted(SBG_MSG_DIR.glob("*.msg"))

    if not msg_files:
        raise RuntimeError(
            f"No .msg files found in:\n{SBG_MSG_DIR}"
        )

    print("SBG message directory:")
    print(SBG_MSG_DIR)
    print()
    print(f"Found {len(msg_files)} message definitions.")

    all_types = {}

    for msg_file in msg_files:
        msg_name = msg_file.stem

        # Read message definition.
        text = msg_file.read_text()

        # Parse it using rosbags' message parser.
        parsed = get_types_from_msg(
            text,
            f"sbg_driver/msg/{msg_name}",
        )

        all_types.update(parsed)

    print(f"Parsed {len(all_types)} registered message types.")

    # Register everything into the module-level type store.
    register_types(all_types, typestore=types)

    print("SBG message registration complete.")

    required = [
        "sbg_driver/msg/SbgImuData",
        "sbg_driver/msg/SbgGpsPos",
        "sbg_driver/msg/SbgGpsVel",
    ]

    print("\nRequired types:")

    for name in required:
        if name in types.FIELDDEFS:
            print(f"  OK   {name}")
        else:
            print(f"  FAIL {name}")

    print()


# ============================================================================
# ROS TIMESTAMP
# ============================================================================

def ros_timestamp_ns(msg, fallback):
    """
    Get ROS header timestamp in nanoseconds.

    Falls back to rosbag timestamp if the message does not contain a header.
    """

    try:
        return (
            int(msg.header.stamp.sec) * 1_000_000_000
            + int(msg.header.stamp.nanosec)
        )
    except Exception:
        return int(fallback)


# ============================================================================
# NEAREST TIMESTAMP
# ============================================================================

def nearest_timestamp(t, timestamps):

    if not timestamps:
        return None, None

    arr = np.asarray(timestamps, dtype=np.int64)

    i = np.searchsorted(arr, t)

    candidates = []

    if i > 0:
        candidates.append(int(arr[i - 1]))

    if i < len(arr):
        candidates.append(int(arr[i]))

    best = min(
        candidates,
        key=lambda x: abs(x - t)
    )

    return best, abs(best - t)


# ============================================================================
# IMAGE
# ============================================================================

def image_array(msg):

    encoding = str(msg.encoding).lower()

    height = int(msg.height)
    width = int(msg.width)

    data = bytes(msg.data)

    # ------------------------------------------------------------
    # Standard RGB/BGR
    # ------------------------------------------------------------

    if encoding in ("rgb8", "bgr8"):

        arr = np.frombuffer(
            data,
            dtype=np.uint8
        ).reshape(height, width, 3)

        if encoding == "bgr8":
            return cv2.cvtColor(
                arr,
                cv2.COLOR_BGR2RGB
            )

        return arr

    # ------------------------------------------------------------
    # RGBA/BGRA
    # ------------------------------------------------------------

    if encoding in ("rgba8", "bgra8"):

        arr = np.frombuffer(
            data,
            dtype=np.uint8
        ).reshape(height, width, 4)

        if encoding == "bgra8":
            return cv2.cvtColor(
                arr,
                cv2.COLOR_BGRA2RGBA
            )

        return arr

    # ------------------------------------------------------------
    # MONO
    # ------------------------------------------------------------

    if encoding in ("mono8", "8uc1"):

        return np.frombuffer(
            data,
            dtype=np.uint8
        ).reshape(height, width)

    if encoding in ("mono16", "16uc1"):

        return np.frombuffer(
            data,
            dtype=np.uint16
        ).reshape(height, width)

    # ------------------------------------------------------------
    # BAYER RGGB 8-bit
    # ------------------------------------------------------------

    if encoding == "bayer_rggb8":

        raw = np.frombuffer(
            data,
            dtype=np.uint8
        ).reshape(height, width)

        # RGGB Bayer -> RGB
        rgb = cv2.cvtColor(
            raw,
            cv2.COLOR_BAYER_RG2RGB
        )

        return rgb

    # ------------------------------------------------------------
    # Other Bayer patterns
    # ------------------------------------------------------------

    if encoding == "bayer_bggr8":

        raw = np.frombuffer(
            data,
            dtype=np.uint8
        ).reshape(height, width)

        return cv2.cvtColor(
            raw,
            cv2.COLOR_BAYER_BG2RGB
        )

    if encoding == "bayer_gbrg8":

        raw = np.frombuffer(
            data,
            dtype=np.uint8
        ).reshape(height, width)

        return cv2.cvtColor(
            raw,
            cv2.COLOR_BAYER_GB2RGB
        )

    if encoding == "bayer_grbg8":

        raw = np.frombuffer(
            data,
            dtype=np.uint8
        ).reshape(height, width)

        return cv2.cvtColor(
            raw,
            cv2.COLOR_BAYER_GR2RGB
        )

    raise ValueError(
        f"Unsupported image encoding: {msg.encoding}"
    )

# ============================================================================
# PLY
# ============================================================================

def save_ply(points, path):

    points = np.asarray(points, dtype=np.float32)

    points = points[
        np.isfinite(points).all(axis=1)
    ]

    with open(path, "w") as f:

        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")

        for x, y, z in points:
            f.write(
                f"{x:.6f} {y:.6f} {z:.6f}\n"
            )


# ============================================================================
# POINTCLOUD2
# ============================================================================

def xyz_from_pc2(msg):

    fields = {
        str(field.name): field
        for field in msg.fields
    }

    for name in ("x", "y", "z"):

        if name not in fields:

            raise ValueError(
                f"PointCloud2 missing {name}. "
                f"Fields = {list(fields)}"
            )

    raw = bytes(msg.data)

    points = []

    endian = ">" if msg.is_bigendian else "<"

    formats = {
        7: "f",  # FLOAT32
        8: "d",  # FLOAT64
    }

    height = int(msg.height)
    width = int(msg.width)

    for row in range(height):

        for col in range(width):

            base = (
                row * int(msg.row_step)
                + col * int(msg.point_step)
            )

            xyz = []

            for name in ("x", "y", "z"):

                field = fields[name]

                datatype = int(field.datatype)

                if datatype not in formats:

                    raise ValueError(
                        f"Unsupported PointCloud2 datatype "
                        f"{datatype} for {name}"
                    )

                value = struct.unpack_from(
                    endian + formats[datatype],
                    raw,
                    base + int(field.offset)
                )[0]

                xyz.append(value)

            points.append(xyz)

    return np.asarray(
        points,
        dtype=np.float32
    )


# ============================================================================
# CSV
# ============================================================================

def write_csv(path, header, rows):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        path,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow(header)

        writer.writerows(rows)


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 80)
    print("ROS2 BAG COMPLETE EXTRACTION")
    print("=" * 80)

    print("Bag:")
    print(BAG_PATH)

    print("\nOutput:")
    print(OUT)

    if not BAG_PATH.exists():

        raise FileNotFoundError(
            f"Bag not found:\n{BAG_PATH}"
        )

    # ------------------------------------------------------------------------
    # Register SBG messages BEFORE opening/deserializing the bag.
    # ------------------------------------------------------------------------

    register_sbg_messages()

    # ------------------------------------------------------------------------
    # Create output directories.
    # ------------------------------------------------------------------------

    for camera in CAMERAS:

        (
            OUT / camera / "images_raw"
        ).mkdir(
            parents=True,
            exist_ok=True
        )

        (
            OUT / camera / "images_rect"
        ).mkdir(
            parents=True,
            exist_ok=True
        )

    for directory in (
        "lidar",
        "imu",
        "gps",
        "tf",
        "sync",
    ):

        (
            OUT / directory
        ).mkdir(
            parents=True,
            exist_ok=True
        )

    # ------------------------------------------------------------------------
    # Topic lookup.
    # ------------------------------------------------------------------------

    camera_topics = {}

    for camera, topics in CAMERAS.items():

        raw_topic, rect_topic, info_topic = topics

        camera_topics[raw_topic] = (
            camera,
            "raw"
        )

        camera_topics[rect_topic] = (
            camera,
            "rect"
        )

        camera_topics[info_topic] = (
            camera,
            "info"
        )

    # ------------------------------------------------------------------------
    # Storage.
    # ------------------------------------------------------------------------

    records = {
        camera: {
            "raw": [],
            "rect": [],
            "info": [],
        }
        for camera in CAMERAS
    }

    lidar_records = []
    imu_records = []
    gps_position_records = []
    gps_velocity_records = []
    tf_records = []

    # ------------------------------------------------------------------------
    # Read bag.
    # ------------------------------------------------------------------------

    with Reader(BAG_PATH) as reader:

        print("\n" + "=" * 80)
        print("BAG CONNECTIONS")
        print("=" * 80)

        for connection in reader.connections:

            print(
                f"{connection.topic:48s} "
                f"{connection.msgtype:40s} "
                f"{connection.msgcount}"
            )

        print("\n" + "=" * 80)
        print("EXTRACTING")
        print("=" * 80)

        for connection, bag_timestamp, raw in reader.messages():

            topic = connection.topic

            interesting = (
                topic in camera_topics
                or topic == LIDAR
                or topic == IMU
                or topic == GPSPOS
                or topic == GPSVEL
                or topic == TF
            )

            if not interesting:
                continue

            # ---------------------------------------------------------------
            # Deserialize using installed rosbags type store.
            # ---------------------------------------------------------------

            msg = deserialize_cdr(
                raw,
                connection.msgtype,
            )

            t = ros_timestamp_ns(
                msg,
                bag_timestamp
            )

            # ---------------------------------------------------------------
            # CAMERA
            # ---------------------------------------------------------------

            if topic in camera_topics:

                camera, kind = camera_topics[topic]

                records[camera][kind].append(t)

                if kind in ("raw", "rect"):

                    array = image_array(msg)

                    if kind == "raw":

                        output_dir = (
                            OUT
                            / camera
                            / "images_raw"
                        )

                    else:

                        output_dir = (
                            OUT
                            / camera
                            / "images_rect"
                        )

                    filename = (
                        f"{t:019d}.png"
                    )

                    Image.fromarray(
                        array
                    ).save(
                        output_dir / filename
                    )

                elif kind == "info":

                    camera_info_file = (
                        OUT
                        / camera
                        / "camera_info.csv"
                    )

                    if not camera_info_file.exists():

                        write_csv(
                            camera_info_file,
                            [
                                "ros_timestamp_ns",
                                "width",
                                "height",
                                "distortion_model",
                                "K",
                                "D",
                                "R",
                                "P",
                            ],
                            [
                                (
                                    t,
                                    msg.width,
                                    msg.height,
                                    msg.distortion_model,
                                    list(msg.k),
                                    list(msg.d),
                                    list(msg.r),
                                    list(msg.p),
                                )
                            ],
                        )

            # ---------------------------------------------------------------
            # LIDAR
            # ---------------------------------------------------------------

            elif topic == LIDAR:

                points = xyz_from_pc2(msg)

                output_file = (
                    OUT
                    / "lidar"
                    / f"{t:019d}.ply"
                )

                save_ply(
                    points,
                    output_file
                )

                frame_id = ""

                try:
                    frame_id = str(
                        msg.header.frame_id
                    )
                except Exception:
                    pass

                lidar_records.append(
                    (
                        t,
                        str(output_file),
                        len(points),
                        frame_id,
                    )
                )

            # ---------------------------------------------------------------
            # IMU
            # ---------------------------------------------------------------

            elif topic == IMU:

                accel = msg.accel
                gyro = msg.gyro

                delta_vel = msg.delta_vel
                delta_angle = msg.delta_angle

                imu_status = msg.imu_status

                imu_records.append(
                    (
                        t,

                        int(msg.time_stamp),

                        accel.x,
                        accel.y,
                        accel.z,

                        gyro.x,
                        gyro.y,
                        gyro.z,

                        msg.temp,

                        delta_vel.x,
                        delta_vel.y,
                        delta_vel.z,

                        delta_angle.x,
                        delta_angle.y,
                        delta_angle.z,

                        str(imu_status),
                    )
                )

            # ---------------------------------------------------------------
            # GPS POSITION
            # ---------------------------------------------------------------

            elif topic == GPSPOS:

                accuracy = msg.position_accuracy

                gps_position_records.append(
                    (
                        t,

                        int(msg.time_stamp),

                        int(msg.gps_tow),

                        float(msg.latitude),
                        float(msg.longitude),
                        float(msg.altitude),

                        float(msg.undulation),

                        float(accuracy.x),
                        float(accuracy.y),
                        float(accuracy.z),

                        int(msg.num_sv_tracked),
                        int(msg.num_sv_used),

                        int(msg.base_station_id),
                        int(msg.diff_age),

                        str(msg.status),
                    )
                )

            # ---------------------------------------------------------------
            # GPS VELOCITY
            # ---------------------------------------------------------------

            elif topic == GPSVEL:

                velocity = msg.velocity
                accuracy = msg.velocity_accuracy

                gps_velocity_records.append(
                    (
                        t,

                        int(msg.time_stamp),

                        int(msg.gps_tow),

                        float(velocity.x),
                        float(velocity.y),
                        float(velocity.z),

                        float(accuracy.x),
                        float(accuracy.y),
                        float(accuracy.z),

                        float(msg.course),
                        float(msg.course_acc),

                        str(msg.status),
                    )
                )

            # ---------------------------------------------------------------
            # STATIC TF
            # ---------------------------------------------------------------

            elif topic == TF:

                tf_records.append(
                    (
                        t,
                        msg,
                    )
                )

    # =========================================================================
    # SAVE CAMERA TIMESTAMPS
    # =========================================================================

    print("\n" + "=" * 80)
    print("SAVING CAMERA METADATA")
    print("=" * 80)

    for camera in CAMERAS:

        rows = []

        for kind in (
            "raw",
            "rect",
            "info",
        ):

            for t in records[camera][kind]:

                rows.append(
                    (
                        t,
                        kind,
                    )
                )

        write_csv(
            OUT
            / camera
            / "timestamps.csv",
            [
                "ros_timestamp_ns",
                "stream",
            ],
            sorted(rows),
        )

    # =========================================================================
    # SAVE LIDAR
    # =========================================================================

    write_csv(
        OUT
        / "lidar"
        / "timestamps.csv",
        [
            "ros_timestamp_ns",
            "ply_path",
            "point_count",
            "frame_id",
        ],
        lidar_records,
    )

    # =========================================================================
    # SAVE IMU
    # =========================================================================

    write_csv(
        OUT
        / "imu"
        / "imu.csv",
        [
            "ros_timestamp_ns",
            "sbg_time_stamp_us",

            "accel_x",
            "accel_y",
            "accel_z",

            "gyro_x",
            "gyro_y",
            "gyro_z",

            "temperature",

            "delta_vel_x",
            "delta_vel_y",
            "delta_vel_z",

            "delta_angle_x",
            "delta_angle_y",
            "delta_angle_z",

            "imu_status",
        ],
        imu_records,
    )

    # =========================================================================
    # SAVE GPS POSITION
    # =========================================================================

    write_csv(
        OUT
        / "gps"
        / "position.csv",
        [
            "ros_timestamp_ns",
            "sbg_time_stamp_us",
            "gps_tow_ms",

            "latitude_deg",
            "longitude_deg",
            "altitude_m",

            "undulation_m",

            "accuracy_x",
            "accuracy_y",
            "accuracy_z",

            "num_sv_tracked",
            "num_sv_used",

            "base_station_id",
            "diff_age",

            "status",
        ],
        gps_position_records,
    )

    # =========================================================================
    # SAVE GPS VELOCITY
    # =========================================================================

    write_csv(
        OUT
        / "gps"
        / "velocity.csv",
        [
            "ros_timestamp_ns",
            "sbg_time_stamp_us",
            "gps_tow_ms",

            "velocity_x",
            "velocity_y",
            "velocity_z",

            "accuracy_x",
            "accuracy_y",
            "accuracy_z",

            "course_deg",
            "course_accuracy_deg",

            "status",
        ],
        gps_velocity_records,
    )

    # =========================================================================
    # SAVE TF
    # =========================================================================

    tf_file = (
        OUT
        / "tf"
        / "tf_static.txt"
    )

    with open(tf_file, "w") as f:

        for t, msg in tf_records:

            f.write(
                f"ROS timestamp: {t}\n"
            )

            for transform in msg.transforms:

                translation = (
                    transform.transform.translation
                )

                rotation = (
                    transform.transform.rotation
                )

                f.write(
                    f"{transform.header.frame_id} "
                    f"-> "
                    f"{transform.child_frame_id}\n"
                )

                f.write(
                    "  translation: "
                    f"{translation.x} "
                    f"{translation.y} "
                    f"{translation.z}\n"
                )

                f.write(
                    "  quaternion: "
                    f"{rotation.x} "
                    f"{rotation.y} "
                    f"{rotation.z} "
                    f"{rotation.w}\n"
                )

    # =========================================================================
    # SYNCHRONISATION
    # =========================================================================

    print("\n" + "=" * 80)
    print("BUILDING TIMESTAMP SYNCHRONISATION")
    print("=" * 80)

    camera2_times = sorted(
        records["CAM2"]["raw"]
    )

    lidar_times = sorted(
        x[0]
        for x in lidar_records
    )

    imu_times = sorted(
        x[0]
        for x in imu_records
    )

    gps_position_times = sorted(
        x[0]
        for x in gps_position_records
    )

    gps_velocity_times = sorted(
        x[0]
        for x in gps_velocity_records
    )

    # -------------------------------------------------------------------------
    # CAM2 -> LiDAR + IMU + GPS
    # -------------------------------------------------------------------------

    rows = []

    for t in camera2_times:

        lidar_t, lidar_error = nearest_timestamp(
            t,
            lidar_times,
        )

        imu_t, imu_error = nearest_timestamp(
            t,
            imu_times,
        )

        gps_t, gps_error = nearest_timestamp(
            t,
            gps_position_times,
        )

        gps_vel_t, gps_vel_error = nearest_timestamp(
            t,
            gps_velocity_times,
        )

        lidar_ok = (
            lidar_error is not None
            and lidar_error <= TOL_NS
        )

        imu_ok = (
            imu_error is not None
            and imu_error <= TOL_NS
        )

        gps_ok = (
            gps_error is not None
            and gps_error <= TOL_NS
        )

        gps_vel_ok = (
            gps_vel_error is not None
            and gps_vel_error <= TOL_NS
        )

        rows.append(
            (
                t,

                lidar_t if lidar_ok else "",
                lidar_error if lidar_ok else "",

                imu_t if imu_ok else "",
                imu_error if imu_ok else "",

                gps_t if gps_ok else "",
                gps_error if gps_ok else "",

                gps_vel_t if gps_vel_ok else "",
                gps_vel_error if gps_vel_ok else "",

                int(lidar_ok),
                int(imu_ok),
                int(gps_ok),
                int(gps_vel_ok),
            )
        )

    write_csv(
        OUT
        / "sync"
        / "CAM2_sync.csv",
        [
            "camera_timestamp_ns",

            "lidar_timestamp_ns",
            "lidar_delta_ns",

            "imu_timestamp_ns",
            "imu_delta_ns",

            "gps_position_timestamp_ns",
            "gps_position_delta_ns",

            "gps_velocity_timestamp_ns",
            "gps_velocity_delta_ns",

            "lidar_synced",
            "imu_synced",
            "gps_position_synced",
            "gps_velocity_synced",
        ],
        rows,
    )

    # -------------------------------------------------------------------------
    # CAM2 -> CAM1 / CAM6
    # -------------------------------------------------------------------------

    for other_camera in (
        "CAM1",
        "CAM6",
    ):

        other_times = sorted(
            records[other_camera]["raw"]
        )

        rows = []

        for t in camera2_times:

            matched, error = nearest_timestamp(
                t,
                other_times,
            )

            good = (
                error is not None
                and error <= TOL_NS
            )

            rows.append(
                (
                    t,
                    matched if good else "",
                    error if good else "",
                    int(good),
                )
            )

        write_csv(
            OUT
            / "sync"
            / f"CAM2_{other_camera}_sync.csv",
            [
                "CAM2_timestamp_ns",
                f"{other_camera}_timestamp_ns",
                "delta_ns",
                "synced",
            ],
            rows,
        )

    # =========================================================================
    # SUMMARY
    # =========================================================================

    print("\n" + "=" * 80)
    print("EXTRACTION COMPLETE")
    print("=" * 80)

    for camera in CAMERAS:

        print(
            f"{camera}: "
            f"raw={len(records[camera]['raw'])}, "
            f"rect={len(records[camera]['rect'])}, "
            f"info={len(records[camera]['info'])}"
        )

    print()
    print("LiDAR:", len(lidar_records))
    print("IMU:", len(imu_records))
    print("GPS position:", len(gps_position_records))
    print("GPS velocity:", len(gps_velocity_records))
    print("TF messages:", len(tf_records))

    print()
    print(
        "Synchronisation tolerance:",
        TOL_NS / 1e6,
        "ms"
    )

    print()
    print("Output:")
    print(OUT)


if __name__ == "__main__":
    main()