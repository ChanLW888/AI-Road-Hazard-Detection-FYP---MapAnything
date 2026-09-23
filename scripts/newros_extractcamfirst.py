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
    "personal_data/camera_lidar_test_11"
)

OUT = BAG_PATH / "extracted_new_new"

SBG_MSG_DIR = Path(
    "/home/lcha0115/bt60_scratch/lcha_data/"
    "sbg_ros2_driver/msg"
)

TOL_NS = 20_000_000  # 20 ms


# ============================================================================
# CAMERA TOPICS
# ============================================================================

CAMERAS = {
    "CAM1": (
        "/CAM1/CAM1_node/image_raw/compressed",
        "/CAM1/CAM1_node/camera_info",
    ),
    "CAM2": (
        "/CAM2/CAM2_node/image_raw/compressed",
        "/CAM2/CAM2_node/camera_info",
    ),
    "CAM6": (
        "/CAM6/CAM6_node/image_raw/compressed",
        "/CAM6/CAM6_node/camera_info",
    ),
}


# ============================================================================
# OTHER TOPICS
# ============================================================================

LIDAR = "/velodyne_points"
IMU = "/sbg/imu_data"
GPSPOS = "/sbg/gps_pos"
GPSVEL = "/sbg/gps_vel"
TF = "/tf_static"


# ============================================================================
# CAMERA PROCESSING SETTINGS
# ============================================================================

# ---------------------------------------------------------------------------
# IMPORTANT
#
# If the compressed image says something like:
#
#     bayer_rggb8; jpeg compressed mono8
#
# we will:
#
#     JPEG -> mono8 -> Bayer RGGB -> RGB
#
# before rectification.
#
# ---------------------------------------------------------------------------

TRY_BAYER_DEBAYER = True


# ---------------------------------------------------------------------------
# Brightness normalization
#
# OFF for now.
#
# We first want to determine whether Bayer reconstruction works correctly.
# ---------------------------------------------------------------------------

ENABLE_BAYER_NORMALISATION = False

TARGET_PERCENTILE = 99.0
TARGET_BRIGHTNESS = 235.0
MIN_GAIN = 0.65
MAX_GAIN = 1.0


# ============================================================================
# BAYER CONVERSION TABLE
# ============================================================================

BAYER_CODES = {
    "bayer_rggb8": cv2.COLOR_BAYER_RG2RGB,
    "bayer_bggr8": cv2.COLOR_BAYER_BG2RGB,
    "bayer_gbrg8": cv2.COLOR_BAYER_GB2RGB,
    "bayer_grbg8": cv2.COLOR_BAYER_GR2RGB,
}


# ============================================================================
# SBG MESSAGE REGISTRATION
# ============================================================================

def register_sbg_messages():

    print("\n" + "=" * 80)
    print("REGISTERING SBG MESSAGE DEFINITIONS")
    print("=" * 80)

    if not SBG_MSG_DIR.exists():

        raise FileNotFoundError(
            f"SBG message directory not found:\n"
            f"{SBG_MSG_DIR}"
        )

    msg_files = sorted(
        SBG_MSG_DIR.glob("*.msg")
    )

    if not msg_files:

        raise RuntimeError(
            f"No .msg files found in:\n"
            f"{SBG_MSG_DIR}"
        )

    print(
        "SBG message directory:"
    )

    print(
        SBG_MSG_DIR
    )

    print(
        f"Found {len(msg_files)} message definitions."
    )

    all_types = {}

    for msg_file in msg_files:

        msg_name = msg_file.stem

        text = msg_file.read_text()

        parsed = get_types_from_msg(
            text,
            f"sbg_driver/msg/{msg_name}",
        )

        all_types.update(
            parsed
        )

    register_types(
        all_types,
        typestore=types,
    )

    print(
        f"Parsed {len(all_types)} registered message types."
    )

    required = [
        "sbg_driver/msg/SbgImuData",
        "sbg_driver/msg/SbgGpsPos",
        "sbg_driver/msg/SbgGpsVel",
    ]

    print("\nRequired types:")

    for name in required:

        if name in types.FIELDDEFS:

            print(
                f"  OK   {name}"
            )

        else:

            print(
                f"  FAIL {name}"
            )

    print()


# ============================================================================
# CSV HELPERS
# ============================================================================

def load_csv(path):

    with open(
        path,
        "r",
        newline=""
    ) as f:

        return list(
            csv.DictReader(f)
        )


def write_csv(
    path,
    header,
    rows,
):

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

        writer.writerow(
            header
        )

        writer.writerows(
            rows
        )


# ============================================================================
# ROS TIMESTAMP
# ============================================================================

def ros_timestamp_ns(
    msg,
    fallback,
):

    try:

        return (
            int(
                msg.header.stamp.sec
            )
            *
            1_000_000_000
            +
            int(
                msg.header.stamp.nanosec
            )
        )

    except Exception:

        return int(
            fallback
        )


# ============================================================================
# NEAREST TIMESTAMP
# ============================================================================

def nearest_timestamp(
    t,
    timestamps,
):

    if timestamps is None:

        return None, None

    arr = np.asarray(
        timestamps,
        dtype=np.int64
    ).reshape(-1)

    if arr.size == 0:

        return None, None

    t = int(t)

    if not np.all(
        arr[:-1] <= arr[1:]
    ):

        arr = np.sort(
            arr
        )

    i = np.searchsorted(
        arr,
        t,
        side="left",
    )

    candidates = []

    if i > 0:

        candidates.append(
            int(
                arr[i - 1]
            )
        )

    if i < len(arr):

        candidates.append(
            int(
                arr[i]
            )
        )

    if not candidates:

        return None, None

    best = min(
        candidates,
        key=lambda x: abs(
            x - t
        )
    )

    return (
        best,
        abs(
            best - t
        ),
    )


# ============================================================================
# TF STATIC
# ============================================================================

def parse_tf_static(path):

    if not path.exists():

        print(
            "WARNING: tf_static.txt not found:"
        )

        print(
            path
        )

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

        if line.startswith(
            "ROS timestamp:"
        ):

            current_time = int(
                line.split(
                    ":"
                )[1].strip()
            )

        elif " -> " in line:

            (
                current_parent,
                current_child
            ) = line.strip().split(
                " -> "
            )

        elif "translation:" in line:

            values = (
                line
                .split(
                    "translation:"
                )[1]
                .strip()
                .split()
            )

            current_translation = np.array(
                [
                    float(v)
                    for v in values
                ]
            )

        elif "quaternion:" in line:

            values = (
                line
                .split(
                    "quaternion:"
                )[1]
                .strip()
                .split()
            )

            current_quaternion = np.array(
                [
                    float(v)
                    for v in values
                ]
            )

            if (
                current_parent is not None
                and
                current_child is not None
                and
                current_translation is not None
                and
                current_quaternion is not None
            ):

                transforms.append(
                    {
                        "time": current_time,
                        "parent": current_parent,
                        "child": current_child,
                        "translation": current_translation.copy(),
                        "quaternion": current_quaternion.copy(),
                    }
                )

                current_translation = None
                current_quaternion = None

    return transforms


def write_matrix(
    path,
    matrix
):

    with open(
        path,
        "w"
    ) as f:

        for row in matrix:

            f.write(
                " ".join(
                    f"{x:.12f}"
                    for x in row
                )
                +
                "\n"
            )


# ============================================================================
# CAMERA CALIBRATION
# ============================================================================

def parse_calibration_list(value):

    value = str(
        value
    ).strip()

    for wrapper in [
        "np.float64(",
        "np.float32(",
        "np.int64(",
        "np.int32(",
    ]:

        value = value.replace(
            wrapper,
            ""
        )

    value = (
        value
        .replace(")", "")
        .replace("[", "")
        .replace("]", "")
        .replace(",", " ")
    )

    values = np.fromstring(
        value,
        sep=" ",
        dtype=np.float64
    )

    if values.size == 0:

        raise ValueError(
            f"Could not parse calibration array:\n"
            f"{value}"
        )

    return values


def load_camera_info_csv(
    path
):

    if not path.exists():

        raise FileNotFoundError(
            f"Camera info file not found:\n"
            f"{path}"
        )

    rows = load_csv(
        path
    )

    if not rows:

        raise RuntimeError(
            f"No rows found in:\n"
            f"{path}"
        )

    row = rows[0]

    width = int(
        row["width"]
    )

    height = int(
        row["height"]
    )

    K_values = parse_calibration_list(
        row["K"]
    )

    if K_values.size != 9:

        raise ValueError(
            f"Expected 9 K values in:\n"
            f"{path}\n"
            f"Got {K_values.size}"
        )

    K = K_values.reshape(
        3,
        3
    )

    D = parse_calibration_list(
        row["D"]
    ).reshape(-1)

    R_values = parse_calibration_list(
        row["R"]
    )

    if R_values.size != 9:

        raise ValueError(
            f"Expected 9 R values in:\n"
            f"{path}\n"
            f"Got {R_values.size}"
        )

    R = R_values.reshape(
        3,
        3
    )

    P_values = parse_calibration_list(
        row["P"]
    )

    if P_values.size != 12:

        raise ValueError(
            f"Expected 12 P values in:\n"
            f"{path}\n"
            f"Got {P_values.size}"
        )

    P = P_values.reshape(
        3,
        4
    )

    return {
        "width": width,
        "height": height,
        "K": K,
        "D": D,
        "R": R,
        "P": P,
    }


def save_calibration_files(
    output_dir,
    camera,
    width,
    height,
    K,
    D,
    R,
    P,
):

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    np.savez(
        output_dir
        /
        f"{camera}_intrinsics.npz",
        K=K,
        D=D,
        R=R,
        P=P,
        width=width,
        height=height,
    )

    calib_file = (
        output_dir
        /
        f"{camera}_calib.txt"
    )

    with open(
        calib_file,
        "w"
    ) as f:

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

        for row in K:

            f.write(
                " ".join(
                    f"{x:.12f}"
                    for x in row
                )
                +
                "\n"
            )

        f.write(
            "\nD:\n"
        )

        f.write(
            " ".join(
                f"{x:.12f}"
                for x in D
            )
            +
            "\n"
        )

        f.write(
            "\nR:\n"
        )

        for row in R:

            f.write(
                " ".join(
                    f"{x:.12f}"
                    for x in row
                )
                +
                "\n"
            )

        f.write(
            "\nP:\n"
        )

        for row in P:

            f.write(
                " ".join(
                    f"{x:.12f}"
                    for x in row
                )
                +
                "\n"
            )


def build_camera_calibration(
    camera
):

    camera_info_file = (
        OUT
        /
        camera
        /
        "camera_info.csv"
    )

    calibration = (
        load_camera_info_csv(
            camera_info_file
        )
    )

    width = calibration[
        "width"
    ]

    height = calibration[
        "height"
    ]

    K = calibration[
        "K"
    ]

    D = calibration[
        "D"
    ]

    R = calibration[
        "R"
    ]

    P = calibration[
        "P"
    ]

    map_x, map_y = (
        cv2.fisheye.initUndistortRectifyMap(
            K,
            D,
            R,
            P[:3, :3],
            (
                width,
                height
            ),
            cv2.CV_32FC1,
        )
    )

    calibration[
        "map_x"
    ] = map_x

    calibration[
        "map_y"
    ] = map_y

    calibration_file = (
        OUT
        /
        "calibration"
        /
        f"{camera}_calib.txt"
    )

    npz_file = (
        OUT
        /
        "calibration"
        /
        f"{camera}_intrinsics.npz"
    )

    if not calibration_file.exists():

        save_calibration_files(
            OUT / "calibration",
            camera,
            width,
            height,
            K,
            D,
            R,
            P,
        )

    if not npz_file.exists():

        np.savez(
            npz_file,
            K=K,
            D=D,
            R=R,
            P=P,
            width=width,
            height=height,
        )

    return calibration


# ============================================================================
# BAYER BRIGHTNESS NORMALISATION
# ============================================================================

def normalise_bayer_brightness(
    raw
):

    raw = np.asarray(
        raw,
        dtype=np.uint8
    )

    percentile_value = float(
        np.percentile(
            raw,
            TARGET_PERCENTILE
        )
    )

    if not ENABLE_BAYER_NORMALISATION:

        return raw, {
            "percentile": percentile_value,
            "gain": 1.0,
            "normalised": 0,
        }

    if percentile_value <= TARGET_BRIGHTNESS:

        return raw, {
            "percentile": percentile_value,
            "gain": 1.0,
            "normalised": 0,
        }

    gain = (
        TARGET_BRIGHTNESS
        /
        max(
            percentile_value,
            1.0
        )
    )

    gain = min(
        gain,
        MAX_GAIN
    )

    gain = max(
        gain,
        MIN_GAIN
    )

    corrected = (
        raw.astype(
            np.float32
        )
        *
        gain
    )

    corrected = np.clip(
        corrected,
        0,
        255
    ).astype(
        np.uint8
    )

    return corrected, {
        "percentile": percentile_value,
        "gain": gain,
        "normalised": 1,
    }


# ============================================================================
# DECODE COMPRESSED IMAGE
# ============================================================================

def decode_compressed_image(
    image_msg
):
    """
    Decode sensor_msgs/CompressedImage.

    Example:

        format =
        bayer_rggb8; jpeg compressed mono8

    The JPEG is deliberately decoded as a single-channel image.
    """

    if not hasattr(
        image_msg,
        "format"
    ):

        raise ValueError(
            "CompressedImage does not contain "
            "a format field."
        )

    if not hasattr(
        image_msg,
        "data"
    ):

        raise ValueError(
            "CompressedImage does not contain "
            "a data field."
        )

    format_string = str(
        image_msg.format
    ).strip()

    format_lower = (
        format_string.lower()
    )

    # ------------------------------------------------------------------------
    # Extract Bayer / image encoding.
    #
    # Example:
    #
    # bayer_rggb8; jpeg compressed mono8
    #
    # becomes:
    #
    # bayer_rggb8
    # ------------------------------------------------------------------------

    encoding = (
        format_lower
        .split(
            ";",
            1
        )[0]
        .strip()
    )

    compressed_bytes = np.frombuffer(
        bytes(
            image_msg.data
        ),
        dtype=np.uint8
    )

    # ------------------------------------------------------------------------
    # JPEG
    # ------------------------------------------------------------------------

    if (
        "jpeg" in format_lower
        or
        "jpg" in format_lower
    ):

        decoded = cv2.imdecode(
            compressed_bytes,
            cv2.IMREAD_GRAYSCALE
        )

        if decoded is None:

            raise RuntimeError(
                "OpenCV failed to decode JPEG."
            )

        return (
            decoded,
            encoding,
            format_string
        )

    # ------------------------------------------------------------------------
    # PNG
    # ------------------------------------------------------------------------

    if "png" in format_lower:

        decoded = cv2.imdecode(
            compressed_bytes,
            cv2.IMREAD_UNCHANGED
        )

        if decoded is None:

            raise RuntimeError(
                "OpenCV failed to decode PNG."
            )

        return (
            decoded,
            encoding,
            format_string
        )

    raise ValueError(
        f"Unsupported compressed format:\n"
        f"{format_string}"
    )


# ============================================================================
# DEBAYER + RECTIFY
# ============================================================================

def debayer_and_undistort(
    image_msg,
    calibration,
):
    """
    Camera processing pipeline:

        CompressedImage
              |
              v
        JPEG decompression
              |
              v
          mono8 image
              |
              v
       Bayer interpretation
              |
              v
          RGB image
              |
              v
       fisheye rectification
              |
              v
          RGB output
    """

    (
        decoded,
        encoding,
        format_string
    ) = decode_compressed_image(
        image_msg
    )

    brightness_stats = {
        "percentile": "",
        "gain": "",
        "normalised": 0,
    }

    # ========================================================================
    # BAYER IMAGE
    # ========================================================================

    if (
        TRY_BAYER_DEBAYER
        and
        encoding in BAYER_CODES
    ):

        # JPEG mono8 should be single channel.
        if decoded.ndim != 2:

            raise ValueError(
                f"Expected single-channel Bayer JPEG, "
                f"but received shape {decoded.shape}\n"
                f"Format: {format_string}"
            )

        bayer = np.asarray(
            decoded,
            dtype=np.uint8
        )

        height, width = (
            bayer.shape
        )

        # Check against calibration.
        if (
            width != calibration["width"]
            or
            height != calibration["height"]
        ):

            raise ValueError(
                "Image resolution does not match calibration: "
                f"image={width}x{height}, "
                f"calibration={calibration['width']}x"
                f"{calibration['height']}"
            )

        # ---------------------------------------------------------------
        # Optional brightness normalization.
        # ---------------------------------------------------------------

        (
            bayer,
            brightness_stats
        ) = normalise_bayer_brightness(
            bayer
        )

        # ---------------------------------------------------------------
        # ACTUAL DEBAYER
        # ---------------------------------------------------------------

        print(
            f"    Bayer: {encoding}"
        )

        rgb = cv2.cvtColor(
            bayer,
            BAYER_CODES[
                encoding
            ]
        )

        bgr = cv2.cvtColor(
            rgb,
            cv2.COLOR_RGB2BGR
        )

    # ========================================================================
    # MONO IMAGE WITHOUT BAYER
    # ========================================================================

    elif decoded.ndim == 2:

        height, width = (
            decoded.shape
        )

        if (
            width != calibration["width"]
            or
            height != calibration["height"]
        ):

            raise ValueError(
                "Image resolution does not match calibration: "
                f"image={width}x{height}, "
                f"calibration={calibration['width']}x"
                f"{calibration['height']}"
            )

        print(
            "    No Bayer encoding detected; "
            "using grayscale."
        )

        bgr = cv2.cvtColor(
            decoded,
            cv2.COLOR_GRAY2BGR
        )

    # ========================================================================
    # ALREADY-COLOR IMAGE
    # ========================================================================

    else:

        height, width = (
            decoded.shape[:2]
        )

        if (
            width != calibration["width"]
            or
            height != calibration["height"]
        ):

            raise ValueError(
                "Image resolution does not match calibration: "
                f"image={width}x{height}, "
                f"calibration={calibration['width']}x"
                f"{calibration['height']}"
            )

        bgr = decoded

    # ========================================================================
    # FISHEYE RECTIFICATION
    # ========================================================================

    rectified_bgr = cv2.remap(
        bgr,
        calibration["map_x"],
        calibration["map_y"],
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )

    rectified_rgb = cv2.cvtColor(
        rectified_bgr,
        cv2.COLOR_BGR2RGB
    )

    return (
        rectified_rgb,
        encoding,
        format_string,
        brightness_stats
    )


# ============================================================================
# GPS
# ============================================================================

def gps_to_local(
    gps_rows
):

    lat0 = float(
        gps_rows[0]["latitude_deg"]
    )

    lon0 = float(
        gps_rows[0]["longitude_deg"]
    )

    alt0 = float(
        gps_rows[0]["altitude_m"]
    )

    R_EARTH = 6378137.0

    lat0_rad = np.deg2rad(
        lat0
    )

    positions = []

    for row in gps_rows:

        lat = float(
            row["latitude_deg"]
        )

        lon = float(
            row["longitude_deg"]
        )

        alt = float(
            row["altitude_m"]
        )

        dlat = np.deg2rad(
            lat - lat0
        )

        dlon = np.deg2rad(
            lon - lon0
        )

        east = (
            dlon
            *
            R_EARTH
            *
            np.cos(
                lat0_rad
            )
        )

        north = (
            dlat
            *
            R_EARTH
        )

        up = (
            alt
            -
            alt0
        )

        positions.append(
            [
                east,
                north,
                up
            ]
        )

    return np.asarray(
        positions
    )


def velocity_yaw(
    velocity_rows
):

    result = {}

    for row in velocity_rows:

        t = int(
            row["ros_timestamp_ns"]
        )

        vx = float(
            row["velocity_x"]
        )

        vy = float(
            row["velocity_y"]
        )

        speed = np.hypot(
            vx,
            vy
        )

        if speed < 0.5:

            result[t] = None

        else:

            result[t] = np.arctan2(
                vy,
                vx
            )

    return result


# ============================================================================
# ROTATION
# ============================================================================

def rotation_z(
    yaw
):

    c = np.cos(
        yaw
    )

    s = np.sin(
        yaw
    )

    return np.array(
        [
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1],
        ]
    )


# ============================================================================
# QUATERNION
# ============================================================================

def quaternion_to_rotation(
    x,
    y,
    z,
    w
):

    return np.array(
        [
            [
                1 - 2*y*y - 2*z*z,
                2*x*y - 2*z*w,
                2*x*z + 2*y*w,
            ],
            [
                2*x*y + 2*y*w,
                1 - 2*x*x - 2*z*z,
                2*y*z - 2*x*w,
            ],
            [
                2*x*z - 2*y*w,
                2*y*z + 2*x*w,
                1 - 2*x*x - 2*y*y,
            ],
        ]
    )


# ============================================================================
# CAMERA EXTRINSICS
# ============================================================================

def find_camera_extrinsic(
    transforms,
    camera_name
):

    candidates = []

    camera_tokens = [
        camera_name.lower(),
        camera_name.replace(
            "CAM",
            "cam"
        ).lower(),
    ]

    for tr in transforms:

        parent = (
            tr["parent"]
            .lower()
        )

        child = (
            tr["child"]
            .lower()
        )

        if any(
            token in parent
            or
            token in child
            for token in camera_tokens
        ):

            candidates.append(
                tr
            )

    return candidates


# ============================================================================
# CAMERA POSES
# ============================================================================

def build_poses(
    camera,
    gps_rows,
    velocity_rows,
    transforms,
):

    image_timestamp_file = (
        OUT
        /
        camera
        /
        "timestamps.csv"
    )

    if not image_timestamp_file.exists():

        raise FileNotFoundError(
            image_timestamp_file
        )

    image_rows = load_csv(
        image_timestamp_file
    )

    image_rows = [
        r
        for r in image_rows
        if r["stream"] == "raw"
    ]

    if len(image_rows) == 0:

        raise RuntimeError(
            f"No raw images found for {camera}"
        )

    gps_positions = gps_to_local(
        gps_rows
    )

    gps_times = np.asarray(
        [
            int(
                r["ros_timestamp_ns"]
            )
            for r in gps_rows
        ],
        dtype=np.int64
    )

    velocity_headings = (
        velocity_yaw(
            velocity_rows
        )
    )

    velocity_times = sorted(
        velocity_headings.keys()
    )

    pose_dir = (
        OUT
        /
        "poses"
        /
        camera
    )

    pose_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    trajectory_rows = []

    previous_yaw = 0.0

    for index, image_row in enumerate(
        image_rows
    ):

        t = int(
            image_row["ros_timestamp_ns"]
        )

        gps_t, gps_error = (
            nearest_timestamp(
                t,
                gps_times
            )
        )

        if gps_t is None:

            print(
                f"WARNING: no GPS for "
                f"{camera} image {index}"
            )

            continue

        gps_index = np.searchsorted(
            gps_times,
            gps_t
        )

        gps_index = min(
            gps_index,
            len(gps_positions) - 1
        )

        position = (
            gps_positions[
                gps_index
            ]
        )

        vel_t, vel_error = (
            nearest_timestamp(
                t,
                velocity_times
            )
        )

        yaw = None

        if vel_t is not None:

            yaw = (
                velocity_headings[
                    vel_t
                ]
            )

        if yaw is None:

            yaw = previous_yaw

        else:

            previous_yaw = yaw

        T_world_body = np.eye(
            4
        )

        T_world_body[
            :3,
            :3
        ] = rotation_z(
            yaw
        )

        T_world_body[
            :3,
            3
        ] = position

        pose_file = (
            pose_dir
            /
            f"{index:06d}.txt"
        )

        write_matrix(
            pose_file,
            T_world_body
        )

        trajectory_rows.append(
            (
                index,
                t,
                position[0],
                position[1],
                position[2],
                np.rad2deg(
                    yaw
                ),
                gps_error,
                vel_error
                if vel_t is not None
                else "",
            )
        )

    trajectory_file = (
        OUT
        /
        "poses"
        /
        "trajectory.csv"
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
# POINTCLOUD2 -> XYZ + INTENSITY
# ============================================================================

def xyz_intensity_from_pc2(
    msg
):

    fields = {
        str(field.name): field
        for field in msg.fields
    }

    required = (
        "x",
        "y",
        "z",
        "intensity",
    )

    for name in required:

        if name not in fields:

            raise ValueError(
                f"PointCloud2 missing {name}. "
                f"Fields = {list(fields)}"
            )

    raw = bytes(
        msg.data
    )

    endian = (
        ">"
        if msg.is_bigendian
        else
        "<"
    )

    formats = {
        7: "f",
        8: "d",
    }

    height = int(
        msg.height
    )

    width = int(
        msg.width
    )

    points = []

    for row in range(
        height
    ):

        for col in range(
            width
        ):

            base = (
                row
                *
                int(
                    msg.row_step
                )
                +
                col
                *
                int(
                    msg.point_step
                )
            )

            values = []

            for name in (
                "x",
                "y",
                "z",
                "intensity",
            ):

                field = fields[
                    name
                ]

                datatype = int(
                    field.datatype
                )

                if datatype not in formats:

                    raise ValueError(
                        f"Unsupported "
                        f"PointCloud2 datatype "
                        f"{datatype} for {name}"
                    )

                value = (
                    struct.unpack_from(
                        endian
                        +
                        formats[
                            datatype
                        ],
                        raw,
                        base
                        +
                        int(
                            field.offset
                        ),
                    )[0]
                )

                values.append(
                    value
                )

            points.append(
                values
            )

    return np.asarray(
        points,
        dtype=np.float32
    )


# ============================================================================
# PLY
# ============================================================================

def save_ply_with_intensity(
    points,
    path
):

    points = np.asarray(
        points,
        dtype=np.float32
    )

    valid = np.isfinite(
        points[:, :3]
    ).all(
        axis=1
    )

    points = points[
        valid
    ]

    with open(
        path,
        "w"
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
            "property float intensity\n"
        )

        f.write(
            "end_header\n"
        )

        for point in points:

            f.write(
                f"{point[0]:.6f} "
                f"{point[1]:.6f} "
                f"{point[2]:.6f} "
                f"{point[3]:.6f}\n"
            )


def ply_has_intensity(
    path
):

    if not path.exists():

        return False

    try:

        with open(
            path,
            "r"
        ) as f:

            for line in f:

                line = line.strip()

                if line == "end_header":

                    break

                if (
                    line.startswith(
                        "property "
                    )
                    and
                    line.endswith(
                        " intensity"
                    )
                ):

                    return True

    except Exception:

        return False

    return False


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 80)
    print(
        "ROS2 BAG EXTRACTION"
    )
    print(
        "CAMERA FIRST -> LIDAR SECOND"
    )
    print("=" * 80)

    print()
    print(
        f"Bag:\n{BAG_PATH}"
    )

    print()
    print(
        f"Output:\n{OUT}"
    )

    print()
    print(
        "Camera processing:"
    )

    print(
        "  JPEG -> mono8: YES"
    )

    print(
        "  Bayer debayering: "
        f"{TRY_BAYER_DEBAYER}"
    )

    print(
        "  Brightness normalization: "
        f"{ENABLE_BAYER_NORMALISATION}"
    )

    if not BAG_PATH.exists():

        raise FileNotFoundError(
            f"Bag not found:\n"
            f"{BAG_PATH}"
        )

    register_sbg_messages()

    # =========================================================================
    # CREATE OUTPUT DIRECTORIES
    # =========================================================================

    for camera in CAMERAS:

        (
            OUT
            /
            camera
            /
            "images_rect"
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
        "calibration",
    ):

        (
            OUT
            /
            directory
        ).mkdir(
            parents=True,
            exist_ok=True
        )

    # =========================================================================
    # TOPIC LOOKUPS
    # =========================================================================

    camera_raw_topics = {
        topics[0]: camera
        for camera, topics in CAMERAS.items()
    }

    camera_info_topics = {
        topics[1]: camera
        for camera, topics in CAMERAS.items()
    }

    # =========================================================================
    # EXISTING OUTPUT COUNTS
    # =========================================================================

    print()
    print(
        "Existing output:"
    )

    for camera in CAMERAS:

        count = len(
            list(
                (
                    OUT
                    /
                    camera
                    /
                    "images_rect"
                ).glob(
                    "*.png"
                )
            )
        )

        print(
            f"  {camera}: "
            f"{count} existing images"
        )

    existing_lidar = list(
        (
            OUT
            /
            "lidar"
        ).glob(
            "*.ply"
        )
    )

    print(
        f"  LiDAR: "
        f"{len(existing_lidar)} existing PLYs"
    )

    # =========================================================================
    # ========================================================================
    #
    # PASS 1A
    #
    # CAMERA INFO ONLY
    #
    # This pass deliberately does NOT process LiDAR.
    #
    # =========================================================================
    # =========================================================================

    print("\n" + "=" * 80)
    print(
        "PASS 1A: EXTRACTING CAMERA INFO"
    )
    print("=" * 80)

    camera_info_found = {
        camera: False
        for camera in CAMERAS
    }

    with Reader(
        BAG_PATH
    ) as reader:

        for connection, bag_timestamp, raw in reader.messages():

            topic = connection.topic

            if topic not in camera_info_topics:

                continue

            camera = (
                camera_info_topics[
                    topic
                ]
            )

            msg = deserialize_cdr(
                raw,
                connection.msgtype
            )

            t = ros_timestamp_ns(
                msg,
                bag_timestamp
            )

            camera_info_file = (
                OUT
                /
                camera
                /
                "camera_info.csv"
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

            camera_info_found[
                camera
            ] = True

            print(
                f"{camera}: camera_info found"
            )

    # =========================================================================
    # BUILD CALIBRATIONS
    # =========================================================================

    print("\n" + "=" * 80)
    print(
        "PASS 1B: BUILDING CAMERA RECTIFICATION MAPS"
    )
    print("=" * 80)

    calibrations = {}

    for camera in CAMERAS:

        if not camera_info_found[
            camera
        ]:

            camera_info_file = (
                OUT
                /
                camera
                /
                "camera_info.csv"
            )

            if not camera_info_file.exists():

                raise FileNotFoundError(
                    f"No camera_info found for "
                    f"{camera}:\n"
                    f"{camera_info_file}"
                )

        calibrations[
            camera
        ] = build_camera_calibration(
            camera
        )

        calibration = calibrations[
            camera
        ]

        print(
            f"{camera}: "
            f"{calibration['width']} x "
            f"{calibration['height']}"
        )

        print(
            f"  fx = "
            f"{calibration['K'][0, 0]:.6f}"
        )

        print(
            f"  fy = "
            f"{calibration['K'][1, 1]:.6f}"
        )

        print(
            "  Fisheye map: ready"
        )

    # =========================================================================
    # =========================================================================
    #
    # PASS 1C
    #
    # CAMERA IMAGES
    #
    # THIS IS THE IMPORTANT PART:
    #
    # CAMERA PROCESSING COMPLETELY FINISHES HERE.
    #
    # NO LIDAR IS TOUCHED.
    #
    # =========================================================================
    # =========================================================================

    print("\n" + "=" * 80)
    print(
        "PASS 1C: CAMERA EXTRACTION"
    )
    print(
        "JPEG MONO8 -> BAYER DEBAYER -> RGB -> RECTIFY"
    )
    print("=" * 80)

    print()
    print(
        "CAMERAS WILL BE COMPLETED BEFORE LIDAR STARTS."
    )

    camera_timestamp_records = {
        camera: []
        for camera in CAMERAS
    }

    processed_counts = {
        camera: 0
        for camera in CAMERAS
    }

    skipped_counts = {
        camera: 0
        for camera in CAMERAS
    }

    brightness_rows = []

    # -------------------------------------------------------------------------
    # ONE CAMERA-ONLY BAG PASS
    #
    # We only inspect camera compressed image topics here.
    # -------------------------------------------------------------------------

    with Reader(
        BAG_PATH
    ) as reader:

        for connection, bag_timestamp, raw in reader.messages():

            topic = connection.topic

            if topic not in camera_raw_topics:

                continue

            camera = (
                camera_raw_topics[
                    topic
                ]
            )

            msg = deserialize_cdr(
                raw,
                connection.msgtype
            )

            t = ros_timestamp_ns(
                msg,
                bag_timestamp
            )

            camera_timestamp_records[
                camera
            ].append(
                t
            )

            output_file = (
                OUT
                /
                camera
                /
                "images_rect"
                /
                f"{t:019d}.png"
            )

            # ---------------------------------------------------------------
            # Existing image
            # ---------------------------------------------------------------

            if output_file.exists():

                skipped_counts[
                    camera
                ] += 1

                continue

            print(
                f"{camera}: "
                f"processing {t}"
            )

            # ---------------------------------------------------------------
            # JPEG -> Bayer -> RGB -> Rectify
            # ---------------------------------------------------------------

            (
                rectified_rgb,
                detected_encoding,
                format_string,
                brightness_stats,
            ) = debayer_and_undistort(
                msg,
                calibrations[
                    camera
                ],
            )

            # ---------------------------------------------------------------
            # Save RGB PNG
            # ---------------------------------------------------------------

            Image.fromarray(
                rectified_rgb
            ).save(
                output_file
            )

            processed_counts[
                camera
            ] += 1

            brightness_rows.append(
                (
                    camera,
                    t,
                    detected_encoding,
                    format_string,
                    brightness_stats[
                        "percentile"
                    ],
                    brightness_stats[
                        "gain"
                    ],
                    brightness_stats[
                        "normalised"
                    ],
                )
            )

    # =========================================================================
    # SAVE CAMERA TIMESTAMPS
    # =========================================================================

    print("\n" + "=" * 80)
    print(
        "CAMERA EXTRACTION COMPLETE"
    )
    print("=" * 80)

    for camera in CAMERAS:

        print(
            f"{camera}: "
            f"{processed_counts[camera]} new images, "
            f"{skipped_counts[camera]} skipped"
        )

        timestamp_file = (
            OUT
            /
            camera
            /
            "timestamps.csv"
        )

        write_csv(
            timestamp_file,
            [
                "ros_timestamp_ns",
                "stream",
            ],
            [
                (
                    t,
                    "raw"
                )
                for t in sorted(
                    set(
                        camera_timestamp_records[
                            camera
                        ]
                    )
                )
            ],
        )

    brightness_log_file = (
        OUT
        /
        "calibration"
        /
        "image_processing.csv"
    )

    write_csv(
        brightness_log_file,
        [
            "camera",
            "ros_timestamp_ns",
            "detected_encoding",
            "compressed_format",
            "raw_percentile",
            "gain_applied",
            "normalised",
        ],
        brightness_rows,
    )

    print()
    print(
        "Camera processing log:"
    )

    print(
        brightness_log_file
    )

    # =========================================================================
    # ========================================================================
    #
    # CAMERA IS NOW 100% FINISHED.
    #
    # ONLY NOW DO WE START LIDAR.
    #
    # =========================================================================
    # =========================================================================

    print("\n" + "=" * 80)
    print(
        "CAMERA PASS FINISHED"
    )
    print(
        "STARTING LIDAR PASS"
    )
    print("=" * 80)

    # =========================================================================
    # PASS 2A: LIDAR
    # =========================================================================

    lidar_records = []

    with Reader(
        BAG_PATH
    ) as reader:

        for connection, bag_timestamp, raw in reader.messages():

            topic = connection.topic

            if topic != LIDAR:

                continue

            msg = deserialize_cdr(
                raw,
                connection.msgtype
            )

            t = ros_timestamp_ns(
                msg,
                bag_timestamp
            )

            output_file = (
                OUT
                /
                "lidar"
                /
                f"{t:019d}.ply"
            )

            # ---------------------------------------------------------------
            # Existing PLY with intensity
            # ---------------------------------------------------------------

            if ply_has_intensity(
                output_file
            ):

                print(
                    f"LiDAR exists, skipping: "
                    f"{output_file.name}"
                )

                lidar_records.append(
                    (
                        t,
                        str(output_file),
                        -1,
                        "",
                    )
                )

                continue

            # ---------------------------------------------------------------
            # Extract LiDAR
            # ---------------------------------------------------------------

            print(
                f"Extracting LiDAR: "
                f"{t:019d}.ply"
            )

            points = (
                xyz_intensity_from_pc2(
                    msg
                )
            )

            save_ply_with_intensity(
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

    # =========================================================================
    # LIDAR TIMESTAMPS
    # =========================================================================

    lidar_timestamp_file = (
        OUT
        /
        "lidar"
        /
        "timestamps.csv"
    )

    write_csv(
        lidar_timestamp_file,
        [
            "ros_timestamp_ns",
            "ply_path",
            "point_count",
            "frame_id",
        ],
        lidar_records,
    )

    print("\n" + "=" * 80)
    print(
        "LIDAR EXTRACTION COMPLETE"
    )
    print("=" * 80)

    print(
        f"LiDAR frames: "
        f"{len(lidar_records)}"
    )

    # =========================================================================
    # =========================================================================
    #
    # PASS 3
    #
    # IMU + GPS + TF
    #
    # =========================================================================
    # =========================================================================

    print("\n" + "=" * 80)
    print(
        "PASS 3: IMU + GPS + TF"
    )
    print("=" * 80)

    imu_records = []
    gps_position_records = []
    gps_velocity_records = []
    tf_records = []

    with Reader(
        BAG_PATH
    ) as reader:

        for connection, bag_timestamp, raw in reader.messages():

            topic = connection.topic

            if topic not in (
                IMU,
                GPSPOS,
                GPSVEL,
                TF,
            ):

                continue

            msg = deserialize_cdr(
                raw,
                connection.msgtype
            )

            t = ros_timestamp_ns(
                msg,
                bag_timestamp
            )

            # ---------------------------------------------------------------
            # IMU
            # ---------------------------------------------------------------

            if topic == IMU:

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

                accuracy = (
                    msg.position_accuracy
                )

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
            # TF
            # ---------------------------------------------------------------

            elif topic == TF:

                tf_records.append(
                    (
                        t,
                        msg
                    )
                )

    # =========================================================================
    # SAVE IMU
    # =========================================================================

    imu_file = (
        OUT
        /
        "imu"
        /
        "imu.csv"
    )

    write_csv(
        imu_file,
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

    gps_position_file = (
        OUT
        /
        "gps"
        /
        "position.csv"
    )

    write_csv(
        gps_position_file,
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

    gps_velocity_file = (
        OUT
        /
        "gps"
        /
        "velocity.csv"
    )

    write_csv(
        gps_velocity_file,
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
        /
        "tf"
        /
        "tf_static.txt"
    )

    with open(
        tf_file,
        "w"
    ) as f:

        for t, msg in tf_records:

            f.write(
                f"ROS timestamp: {t}\n"
            )

            for transform in (
                msg.transforms
            ):

                translation = (
                    transform
                    .transform
                    .translation
                )

                rotation = (
                    transform
                    .transform
                    .rotation
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
    # PASS 4: POSES / EXTRINSICS
    # =========================================================================

    print("\n" + "=" * 80)
    print(
        "PASS 4: CAMERA POSES / EXTRINSICS"
    )
    print("=" * 80)

    gps_rows = load_csv(
        gps_position_file
    )

    velocity_rows = load_csv(
        gps_velocity_file
    )

    transforms = parse_tf_static(
        tf_file
    )

    extrinsics_file = (
        OUT
        /
        "calibration"
        /
        "extrinsics.txt"
    )

    if not extrinsics_file.exists():

        with open(
            extrinsics_file,
            "w"
        ) as f:

            for camera in CAMERAS:

                candidates = (
                    find_camera_extrinsic(
                        transforms,
                        camera
                    )
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
                        f"{camera}: "
                        f"NO CAMERA TF FOUND"
                    )

                    continue

                print(
                    f"\n{camera} TF candidates:"
                )

                for tr in candidates:

                    print(
                        f"  {tr['parent']} -> "
                        f"{tr['child']}"
                    )

                    q = tr[
                        "quaternion"
                    ]

                    R = (
                        quaternion_to_rotation(
                            q[0],
                            q[1],
                            q[2],
                            q[3]
                        )
                    )

                    T = np.eye(
                        4
                    )

                    T[
                        :3,
                        :3
                    ] = R

                    T[
                        :3,
                        3
                    ] = tr[
                        "translation"
                    ]

                    f.write(
                        f"\n"
                        f"{tr['parent']} -> "
                        f"{tr['child']}\n"
                    )

                    for row in T:

                        f.write(
                            " ".join(
                                f"{x:.12f}"
                                for x in row
                            )
                            +
                            "\n"
                        )

    # =========================================================================
    # BUILD POSES
    # =========================================================================

    for camera in CAMERAS:

        pose_dir = (
            OUT
            /
            "poses"
            /
            camera
        )

        if not pose_dir.exists():

            build_poses(
                camera,
                gps_rows,
                velocity_rows,
                transforms,
            )

        elif not list(
            pose_dir.glob(
                "*.txt"
            )
        ):

            build_poses(
                camera,
                gps_rows,
                velocity_rows,
                transforms,
            )

        else:

            print(
                f"{camera}: "
                f"poses already exist, skipping"
            )

    # =========================================================================
    # PASS 5: SYNCHRONISATION
    # =========================================================================

    print("\n" + "=" * 80)
    print(
        "PASS 5: SYNCHRONISATION"
    )
    print("=" * 80)

    sync_cam2_file = (
        OUT
        /
        "sync"
        /
        "CAM2_sync.csv"
    )

    if not sync_cam2_file.exists():

        camera2_times = sorted(
            camera_timestamp_records[
                "CAM2"
            ]
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

        rows = []

        for t in camera2_times:

            lidar_t, lidar_error = (
                nearest_timestamp(
                    t,
                    lidar_times
                )
            )

            imu_t, imu_error = (
                nearest_timestamp(
                    t,
                    imu_times
                )
            )

            gps_t, gps_error = (
                nearest_timestamp(
                    t,
                    gps_position_times
                )
            )

            gps_vel_t, gps_vel_error = (
                nearest_timestamp(
                    t,
                    gps_velocity_times
                )
            )

            lidar_ok = (
                lidar_error is not None
                and
                lidar_error <= TOL_NS
            )

            imu_ok = (
                imu_error is not None
                and
                imu_error <= TOL_NS
            )

            gps_ok = (
                gps_error is not None
                and
                gps_error <= TOL_NS
            )

            gps_vel_ok = (
                gps_vel_error is not None
                and
                gps_vel_error <= TOL_NS
            )

            rows.append(
                (
                    t,
                    lidar_t
                    if lidar_ok
                    else "",
                    lidar_error
                    if lidar_ok
                    else "",
                    imu_t
                    if imu_ok
                    else "",
                    imu_error
                    if imu_ok
                    else "",
                    gps_t
                    if gps_ok
                    else "",
                    gps_error
                    if gps_ok
                    else "",
                    gps_vel_t
                    if gps_vel_ok
                    else "",
                    gps_vel_error
                    if gps_vel_ok
                    else "",
                    int(lidar_ok),
                    int(imu_ok),
                    int(gps_ok),
                    int(gps_vel_ok),
                )
            )

        write_csv(
            sync_cam2_file,
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

        # ---------------------------------------------------------------------
        # CAM2 -> CAM1 / CAM6
        # ---------------------------------------------------------------------

        for other_camera in (
            "CAM1",
            "CAM6"
        ):

            sync_file = (
                OUT
                /
                "sync"
                /
                f"CAM2_{other_camera}_sync.csv"
            )

            if sync_file.exists():

                continue

            other_times = sorted(
                camera_timestamp_records[
                    other_camera
                ]
            )

            rows = []

            for t in camera2_times:

                matched, error = (
                    nearest_timestamp(
                        t,
                        other_times
                    )
                )

                good = (
                    error is not None
                    and
                    error <= TOL_NS
                )

                rows.append(
                    (
                        t,
                        matched
                        if good
                        else "",
                        error
                        if good
                        else "",
                        int(good),
                    )
                )

            write_csv(
                sync_file,
                [
                    "CAM2_timestamp_ns",
                    f"{other_camera}_timestamp_ns",
                    "delta_ns",
                    "synced",
                ],
                rows,
            )

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================

    print("\n" + "=" * 80)
    print(
        "EXTRACTION COMPLETE"
    )
    print("=" * 80)

    print()
    print(
        "PROCESSING ORDER:"
    )

    print(
        "  1. CAMERA"
    )

    print(
        "     - JPEG decode"
    )

    print(
        "     - mono8 Bayer interpretation"
    )

    print(
        "     - Bayer debayer"
    )

    print(
        "     - RGB conversion"
    )

    print(
        "     - fisheye rectification"
    )

    print(
        "     - PNG output"
    )

    print(
        "  2. LIDAR"
    )

    print(
        "     - XYZ + intensity"
    )

    print(
        "     - PLY output"
    )

    print(
        "  3. IMU + GPS + TF"
    )

    print(
        "  4. Poses"
    )

    print(
        "  5. Synchronisation"
    )

    print()
    print(
        "CAMERA RESULTS:"
    )

    for camera in CAMERAS:

        print(
            f"  {camera}: "
            f"{processed_counts[camera]} processed, "
            f"{skipped_counts[camera]} skipped"
        )

    print()
    print(
        "BAYER DEBAYERING:"
    )

    print(
        f"  Enabled: {TRY_BAYER_DEBAYER}"
    )

    print(
        "  Encoding is read from CompressedImage.format."
    )

    print(
        "  Original bag encoding is NOT modified."
    )

    print()
    print(
        "LIDAR RESULTS:"
    )

    print(
        f"  Frames: {len(lidar_records)}"
    )

    print()
    print(
        "Done."
    )

    print("=" * 80)


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":

    main()