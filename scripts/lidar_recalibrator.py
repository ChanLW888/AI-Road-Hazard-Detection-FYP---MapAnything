import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation


# ============================================================
# CONFIG
# ============================================================

BAG_ROOT = Path(
    "/home/lcha0115/bt60_scratch/lcha_data/"
    "map-anything/personal_data/"
    "camera_lidar_imu_test_04"
)

EXTRACTED = BAG_ROOT / "extracted_new"

OUTPUT_DIR = BAG_ROOT / "lidar_camera_gui_calibration"

CAMERAS = ["CAM1", "CAM2", "CAM6"]

REFERENCE_CAMERA = "CAM2"
REFERENCE_INDEX = 0

LIDAR_MIN_CAMERA_DEPTH = 0.05
BORDER_MARGIN_PIXELS = 2

# Maximum LiDAR points shown on screen.
MAX_DISPLAY_POINTS = 30000

RANDOM_SEED = 42


# ============================================================
# INITIAL GUI RANGES
# ============================================================
#
# Coarse mode
# ------------------------------------------------------------
# rotation:
#     ±15 degrees
#
# translation:
#     ±1 metre
#
# Fine mode
# ------------------------------------------------------------
# rotation:
#     ±2 degrees
#
# translation:
#     ±10 cm
#
# The sliders are relative to the ORIGINAL calibration for
# the current camera.
#
# ============================================================

COARSE_ROTATION_RANGE_DEG = 15.0
COARSE_TRANSLATION_RANGE_M = 1.0

FINE_ROTATION_RANGE_DEG = 2.0
FINE_TRANSLATION_RANGE_M = 0.10


# ============================================================
# POINT DISPLAY
# ============================================================

DEFAULT_POINT_SIZE = 2

MIN_POINT_SIZE = 1
MAX_POINT_SIZE = 6


# ============================================================
# EXTRINSICS
# ============================================================
#
# Stored calibration:
#
#     velodyne -> cam*_optical_frame
#
# Projection:
#
#     LiDAR -> camera = inverse(stored TF)
#
# ============================================================

EXTRINSICS = {
    "CAM1": np.array([
        [
            -0.724120173020,
             0.234431668711,
            -0.648607560649,
            -0.821321794487,
        ],
        [
             0.689673611850,
             0.245413606989,
            -0.681265345238,
            -0.621806408845,
        ],
        [
            -0.000533050740,
            -0.940645498692,
            -0.339390279246,
            -0.577415820549,
        ],
        [
             0.0,
             0.0,
             0.0,
             1.0,
        ],
    ], dtype=np.float64),

    "CAM2": np.array([
        [
             0.663115740417,
             0.096803134794,
            -0.742230872374,
            -0.358123144539,
        ],
        [
             0.747971972679,
            -0.123525740732,
             0.652134433583,
             0.741872257814,
        ],
        [
            -0.028555960826,
            -0.987608497569,
            -0.154317894720,
            -0.962231254134,
        ],
        [
             0.0,
             0.0,
             0.0,
             1.0,
        ],
    ], dtype=np.float64),

    "CAM6": np.array([
        [
            -0.702668737334,
            -0.130049294963,
             0.699531147593,
             0.950060206729,
        ],
        [
            -0.711456886515,
             0.141216082488,
            -0.688394593731,
            -0.464530853084,
        ],
        [
            -0.009259816670,
            -0.981399612251,
            -0.191752592861,
            -1.013403586989,
        ],
        [
             0.0,
             0.0,
             0.0,
             1.0,
        ],
    ], dtype=np.float64),
}


# ============================================================
# GLOBAL GUI STATE
# ============================================================

WINDOW_NAME = "LiDAR Camera Calibration"

current_camera_index = 0

fine_mode = False

show_depth_colour = True

point_size = DEFAULT_POINT_SIZE

current_adjustments = {
    "rx": 0.0,
    "ry": 0.0,
    "rz": 0.0,
    "tx": 0.0,
    "ty": 0.0,
    "tz": 0.0,
}


# ============================================================
# TRANSFORM HELPERS
# ============================================================

def invert_transform(T):

    R = T[:3, :3]
    t = T[:3, 3]

    T_inv = np.eye(
        4,
        dtype=np.float64
    )

    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t

    return T_inv


def transform_points(
    points,
    T
):
    points = np.asarray(
        points,
        dtype=np.float64
    )

    homogeneous = np.concatenate(
        [
            points,
            np.ones(
                (
                    len(points),
                    1
                ),
                dtype=np.float64
            )
        ],
        axis=1
    )

    return (
        homogeneous @ T.T
    )[:, :3]


# ============================================================
# 6-DOF HELPERS
# ============================================================

def transform_to_rotvec(T):

    return Rotation.from_matrix(
        T[:3, :3]
    ).as_rotvec()


def make_delta_transform(
    rx_deg,
    ry_deg,
    rz_deg,
    tx,
    ty,
    tz,
):
    rotvec = np.deg2rad(
        [
            rx_deg,
            ry_deg,
            rz_deg,
        ]
    )

    T = np.eye(
        4,
        dtype=np.float64
    )

    T[:3, :3] = (
        Rotation.from_rotvec(
            rotvec
        ).as_matrix()
    )

    T[:3, 3] = [
        tx,
        ty,
        tz,
    ]

    return T


# ============================================================
# GUI PARAMETER VALUES
# ============================================================

def get_range_values():

    if fine_mode:
        return (
            FINE_ROTATION_RANGE_DEG,
            FINE_TRANSLATION_RANGE_M,
        )

    return (
        COARSE_ROTATION_RANGE_DEG,
        COARSE_TRANSLATION_RANGE_M,
    )


def slider_to_value(
    slider_value,
    center_value,
    min_value,
    max_value,
):
    return (
        min_value
        +
        (
            max_value -
            min_value
        )
        *
        slider_value
        /
        1000.0
    )


def value_to_slider(
    value,
    min_value,
    max_value,
):
    if max_value == min_value:
        return 500

    normalized = (
        value - min_value
    ) / (
        max_value - min_value
    )

    return int(
        np.clip(
            normalized * 1000.0,
            0,
            1000,
        )
    )


# ============================================================
# CAMERA CALIBRATION
# ============================================================

def parse_array(value):

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
        raise RuntimeError(
            f"Could not parse array:\n{value}"
        )

    return values


def load_rectified_intrinsics(
    camera
):

    path = (
        EXTRACTED
        / camera
        / "camera_info.csv"
    )

    with open(
        path,
        "r"
    ) as f:

        rows = list(
            csv.DictReader(f)
        )

    if not rows:

        raise RuntimeError(
            f"No camera info found:\n{path}"
        )

    row = rows[0]

    P = parse_array(
        row["P"]
    )

    if P.size != 12:

        raise RuntimeError(
            f"{camera}: invalid P matrix"
        )

    P = P.reshape(
        3,
        4
    )

    K = P[:, :3]

    width = int(
        row["width"]
    )

    height = int(
        row["height"]
    )

    return (
        K,
        width,
        height
    )


# ============================================================
# FILE LOADING
# ============================================================

def load_image_records(
    camera
):

    directory = (
        EXTRACTED
        / camera
        / "images_rect"
    )

    records = []

    for path in directory.glob(
        "*.png"
    ):

        try:
            timestamp = int(
                path.stem
            )
        except ValueError:
            continue

        records.append(
            (
                timestamp,
                path
            )
        )

    records.sort(
        key=lambda x: x[0]
    )

    if not records:

        raise RuntimeError(
            f"No images found:\n{directory}"
        )

    return records


def load_lidar_records():

    directory = (
        EXTRACTED
        / "lidar"
    )

    records = []

    for path in directory.glob(
        "*.ply"
    ):

        try:
            timestamp = int(
                path.stem
            )
        except ValueError:
            continue

        records.append(
            (
                timestamp,
                path
            )
        )

    records.sort(
        key=lambda x: x[0]
    )

    if not records:

        raise RuntimeError(
            f"No LiDAR files found:\n{directory}"
        )

    return records


def nearest_timestamp(
    target,
    records
):

    timestamps = np.asarray(
        [
            x[0]
            for x in records
        ],
        dtype=np.int64
    )

    index = int(
        np.searchsorted(
            timestamps,
            target
        )
    )

    candidates = []

    if index > 0:
        candidates.append(
            index - 1
        )

    if index < len(records):
        candidates.append(
            index
        )

    if not candidates:

        return (
            None,
            None,
            None
        )

    best = min(
        candidates,
        key=lambda i:
        abs(
            int(records[i][0])
            -
            int(target)
        )
    )

    return (
        int(records[best][0]),
        records[best][1],
        abs(
            int(records[best][0])
            -
            int(target)
        )
    )


# ============================================================
# PLY
# ============================================================

def load_lidar_ply(
    path
):

    header = []

    with open(
        path,
        "r"
    ) as f:

        while True:

            line = f.readline()

            if not line:

                raise RuntimeError(
                    f"Unexpected EOF:\n{path}"
                )

            line = line.strip()

            header.append(
                line
            )

            if line == "end_header":
                break

    vertex_count = None
    properties = []

    for line in header:

        parts = line.split()

        if (
            len(parts) == 3
            and parts[0] == "element"
            and parts[1] == "vertex"
        ):

            vertex_count = int(
                parts[2]
            )

        if (
            len(parts) >= 3
            and parts[0] == "property"
        ):

            properties.append(
                parts[-1]
            )

    if vertex_count is None:

        raise RuntimeError(
            f"No vertex count found:\n{path}"
        )

    data = np.loadtxt(
        path,
        skiprows=len(header),
        max_rows=vertex_count
    )

    if data.ndim == 1:
        data = data.reshape(
            1,
            -1
        )

    x = properties.index(
        "x"
    )
    y = properties.index(
        "y"
    )
    z = properties.index(
        "z"
    )

    return data[
        :,
        [
            x,
            y,
            z
        ]
    ].astype(
        np.float64
    )


# ============================================================
# PROJECTION
# ============================================================

def project_lidar(
    lidar_points,
    T_lidar_to_camera,
    K,
    width,
    height,
):

    points_camera = transform_points(
        lidar_points,
        T_lidar_to_camera
    )

    X = points_camera[:, 0]
    Y = points_camera[:, 1]
    Z = points_camera[:, 2]

    valid = (
        np.isfinite(
            points_camera
        ).all(
            axis=1
        )
        &
        (
            Z >
            LIDAR_MIN_CAMERA_DEPTH
        )
    )

    X = X[valid]
    Y = Y[valid]
    Z = Z[valid]

    points_camera = (
        points_camera[
            valid
        ]
    )

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    u = (
        fx *
        X /
        Z
        +
        cx
    )

    v = (
        fy *
        Y /
        Z
        +
        cy
    )

    inside = (
        (u >= BORDER_MARGIN_PIXELS)
        &
        (
            u <
            width -
            BORDER_MARGIN_PIXELS
        )
        &
        (v >= BORDER_MARGIN_PIXELS)
        &
        (
            v <
            height -
            BORDER_MARGIN_PIXELS
        )
    )

    return (
        points_camera[inside],
        u[inside],
        v[inside],
        Z[inside],
    )


# ============================================================
# POINT COLOURING
# ============================================================

def make_depth_colours(
    depth
):

    if len(depth) == 0:

        return np.empty(
            (
                0,
                3
            ),
            dtype=np.uint8
        )

    d_min = np.percentile(
        depth,
        2
    )

    d_max = np.percentile(
        depth,
        98
    )

    norm = np.clip(
        (
            depth -
            d_min
        )
        /
        max(
            d_max -
            d_min,
            1e-8
        ),
        0,
        1
    )

    red = (
        255 *
        (1.0 - norm)
    ).astype(
        np.uint8
    )

    blue = (
        255 *
        norm
    ).astype(
        np.uint8
    )

    green = np.zeros(
        len(depth),
        dtype=np.uint8
    )

    return np.column_stack(
        [
            red,
            green,
            blue,
        ]
    )


# ============================================================
# DRAW PROJECTION
# ============================================================

def render_projection(
    image,
    u,
    v,
    depth,
    point_size,
    depth_colour
):

    output = image.copy()

    if len(u) == 0:
        return output

    if len(u) > MAX_DISPLAY_POINTS:

        rng = np.random.default_rng(
            RANDOM_SEED
        )

        indices = rng.choice(
            len(u),
            MAX_DISPLAY_POINTS,
            replace=False,
        )

        u = u[
            indices
        ]

        v = v[
            indices
        ]

        depth = depth[
            indices
        ]

    if depth_colour:

        colours = (
            make_depth_colours(
                depth
            )
        )

    else:

        colours = np.tile(
            np.array(
                [
                    0,
                    255,
                    0,
                ],
                dtype=np.uint8
            ),
            (
                len(u),
                1
            )
        )

    for x, y, colour in zip(
        u,
        v,
        colours
    ):

        x = int(
            round(x)
        )

        y = int(
            round(y)
        )

        r = point_size

        cv2.circle(
            output,
            (
                x,
                y
            ),
            r,
            (
                int(colour[2]),
                int(colour[1]),
                int(colour[0])
            ),
            -1
        )

    return output


# ============================================================
# GUI CONTROLS
# ============================================================

def nothing(_):
    pass


def create_trackbars():

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL
    )

    cv2.resizeWindow(
        WINDOW_NAME,
        1920,
        1250
    )

    cv2.createTrackbar(
        "RX",
        WINDOW_NAME,
        500,
        1000,
        nothing
    )

    cv2.createTrackbar(
        "RY",
        WINDOW_NAME,
        500,
        1000,
        nothing
    )

    cv2.createTrackbar(
        "RZ",
        WINDOW_NAME,
        500,
        1000,
        nothing
    )

    cv2.createTrackbar(
        "TX",
        WINDOW_NAME,
        500,
        1000,
        nothing
    )

    cv2.createTrackbar(
        "TY",
        WINDOW_NAME,
        500,
        1000,
        nothing
    )

    cv2.createTrackbar(
        "TZ",
        WINDOW_NAME,
        500,
        1000,
        nothing
    )

    cv2.createTrackbar(
        "POINT SIZE",
        WINDOW_NAME,
        DEFAULT_POINT_SIZE,
        MAX_POINT_SIZE,
        nothing
    )


def set_trackbars_from_adjustments():

    rot_range, trans_range = (
        get_range_values()
    )

    rx_min = -rot_range
    rx_max = rot_range

    ry_min = -rot_range
    ry_max = rot_range

    rz_min = -rot_range
    rz_max = rot_range

    tx_min = -trans_range
    tx_max = trans_range

    ty_min = -trans_range
    ty_max = trans_range

    tz_min = -trans_range
    tz_max = trans_range

    cv2.setTrackbarPos(
        "RX",
        WINDOW_NAME,
        value_to_slider(
            current_adjustments["rx"],
            rx_min,
            rx_max,
        )
    )

    cv2.setTrackbarPos(
        "RY",
        WINDOW_NAME,
        value_to_slider(
            current_adjustments["ry"],
            ry_min,
            ry_max,
        )
    )

    cv2.setTrackbarPos(
        "RZ",
        WINDOW_NAME,
        value_to_slider(
            current_adjustments["rz"],
            rz_min,
            rz_max,
        )
    )

    cv2.setTrackbarPos(
        "TX",
        WINDOW_NAME,
        value_to_slider(
            current_adjustments["tx"],
            tx_min,
            tx_max,
        )
    )

    cv2.setTrackbarPos(
        "TY",
        WINDOW_NAME,
        value_to_slider(
            current_adjustments["ty"],
            ty_min,
            ty_max,
        )
    )

    cv2.setTrackbarPos(
        "TZ",
        WINDOW_NAME,
        value_to_slider(
            current_adjustments["tz"],
            tz_min,
            tz_max,
        )
    )

    cv2.setTrackbarPos(
        "POINT SIZE",
        WINDOW_NAME,
        int(
            point_size
        )
    )


def read_trackbars():

    rot_range, trans_range = (
        get_range_values()
    )

    current_adjustments["rx"] = (
        slider_to_value(
            cv2.getTrackbarPos(
                "RX",
                WINDOW_NAME
            ),
            -rot_range,
            rot_range,
            rot_range,
        )
    )

    current_adjustments["ry"] = (
        slider_to_value(
            cv2.getTrackbarPos(
                "RY",
                WINDOW_NAME
            ),
            -rot_range,
            rot_range,
            rot_range,
        )
    )

    current_adjustments["rz"] = (
        slider_to_value(
            cv2.getTrackbarPos(
                "RZ",
                WINDOW_NAME
            ),
            -rot_range,
            rot_range,
            rot_range,
        )
    )

    current_adjustments["tx"] = (
        slider_to_value(
            cv2.getTrackbarPos(
                "TX",
                WINDOW_NAME
            ),
            -trans_range,
            trans_range,
            trans_range,
        )
    )

    current_adjustments["ty"] = (
        slider_to_value(
            cv2.getTrackbarPos(
                "TY",
                WINDOW_NAME
            ),
            -trans_range,
            trans_range,
            trans_range,
        )
    )

    current_adjustments["tz"] = (
        slider_to_value(
            cv2.getTrackbarPos(
                "TZ",
                WINDOW_NAME
            ),
            -trans_range,
            trans_range,
            trans_range,
        )
    )


def get_point_size():

    value = cv2.getTrackbarPos(
        "POINT SIZE",
        WINDOW_NAME
    )

    return max(
        MIN_POINT_SIZE,
        value
    )


# ============================================================
# RESET GUI
# ============================================================

def reset_adjustments():

    for key in current_adjustments:

        current_adjustments[
            key
        ] = 0.0

    cv2.setTrackbarPos(
        "RX",
        WINDOW_NAME,
        500
    )

    cv2.setTrackbarPos(
        "RY",
        WINDOW_NAME,
        500
    )

    cv2.setTrackbarPos(
        "RZ",
        WINDOW_NAME,
        500
    )

    cv2.setTrackbarPos(
        "TX",
        WINDOW_NAME,
        500
    )

    cv2.setTrackbarPos(
        "TY",
        WINDOW_NAME,
        500
    )

    cv2.setTrackbarPos(
        "TZ",
        WINDOW_NAME,
        500
    )


# ============================================================
# CREATE CURRENT TRANSFORM
# ============================================================

def get_current_transform(
    original_lidar_to_camera
):

    delta = make_delta_transform(
        current_adjustments["rx"],
        current_adjustments["ry"],
        current_adjustments["rz"],
        current_adjustments["tx"],
        current_adjustments["ty"],
        current_adjustments["tz"],
    )

    # Apply user adjustment relative to the original
    # calibrated transform.
    #
    # This means you can reset to the exact original
    # calibration at any point.

    return (
        delta @
        original_lidar_to_camera
    )


# ============================================================
# SAVE PLY
# ============================================================

def save_ply(
    path,
    points,
    colors
):

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
            "property uchar red\n"
        )

        f.write(
            "property uchar green\n"
        )

        f.write(
            "property uchar blue\n"
        )

        f.write(
            "end_header\n"
        )

        for point, colour in zip(
            points,
            colors
        ):

            f.write(
                f"{point[0]:.6f} "
                f"{point[1]:.6f} "
                f"{point[2]:.6f} "
                f"{int(colour[0])} "
                f"{int(colour[1])} "
                f"{int(colour[2])}\n"
            )


# ============================================================
# SAVE CURRENT CALIBRATION
# ============================================================

def save_current_calibration(
    camera,
    lidar_points,
    T_original,
    T_current,
    K,
    width,
    height,
):

    camera_dir = (
        OUTPUT_DIR
        / camera
    )

    camera_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Store LiDAR -> camera
    # --------------------------------------------------------

    save_transform(
        camera_dir /
        "T_lidar_to_camera.txt",
        T_current
    )

    # --------------------------------------------------------
    # Store inverse / original TF convention
    # --------------------------------------------------------

    T_stored = invert_transform(
        T_current
    )

    save_transform(
        camera_dir /
        "T_stored_tf.txt",
        T_stored
    )

    # --------------------------------------------------------
    # Project using current calibration
    # --------------------------------------------------------

    (
        points_camera,
        u,
        v,
        depth
    ) = project_lidar(
        lidar_points,
        T_current,
        K,
        width,
        height
    )

    # --------------------------------------------------------
    # Current camera-frame PLY
    # --------------------------------------------------------

    colours = (
        make_depth_colours(
            depth
        )
    )

    save_ply(
        camera_dir /
        "lidar_projected_camera_frame.ply",
        points_camera,
        colours
    )

    # --------------------------------------------------------
    # Original camera-frame PLY for comparison
    # --------------------------------------------------------

    (
        original_points_camera,
        original_u,
        original_v,
        original_depth
    ) = project_lidar(
        lidar_points,
        T_original,
        K,
        width,
        height
    )

    original_colours = (
        make_depth_colours(
            original_depth
        )
    )

    save_ply(
        camera_dir /
        "lidar_original_camera_frame.ply",
        original_points_camera,
        original_colours
    )

    print()
    print(
        f"SAVED {camera}"
    )

    print(
        "  LiDAR -> camera:"
    )

    print(
        T_current
    )

    print(
        "  Stored TF:"
    )

    print(
        T_stored
    )

    print(
        "  Output:",
        camera_dir
    )


# ============================================================
# SAVE ALL GUI STATE
# ============================================================

def save_combined_calibration(
    saved_transforms
):

    output = (
        OUTPUT_DIR /
        "gui_refined_extrinsics.txt"
    )

    with open(
        output,
        "w"
    ) as f:

        f.write(
            "=" * 80
            +
            "\n"
        )

        f.write(
            "GUI-REFINED LiDAR/CAMERA EXTRINSICS\n"
        )

        f.write(
            "=" * 80
            +
            "\n\n"
        )

        for camera in CAMERAS:

            if camera not in saved_transforms:
                continue

            T_lidar_to_camera = (
                saved_transforms[
                    camera
                ]
            )

            T_stored = (
                invert_transform(
                    T_lidar_to_camera
                )
            )

            f.write(
                f"{camera}\n"
            )

            f.write(
                "-" * 80
                +
                "\n"
            )

            f.write(
                "LiDAR -> camera\n"
            )

            f.write(
                str(
                    T_lidar_to_camera
                )
                +
                "\n\n"
            )

            f.write(
                "Stored TF convention\n"
            )

            f.write(
                str(
                    T_stored
                )
                +
                "\n\n"
            )

    return output


# ============================================================
# DISPLAY TEXT
# ============================================================

def draw_status(
    image,
    camera,
    adjustments,
    fine,
    point_size,
):

    output = image.copy()

    lines = [
        f"{camera}",
        "",
        f"Mode: {'FINE' if fine else 'COARSE'}",
        "",
        f"RX: {adjustments['rx']:+.4f} deg",
        f"RY: {adjustments['ry']:+.4f} deg",
        f"RZ: {adjustments['rz']:+.4f} deg",
        "",
        f"TX: {adjustments['tx']:+.4f} m",
        f"TY: {adjustments['ty']:+.4f} m",
        f"TZ: {adjustments['tz']:+.4f} m",
        "",
        f"Point size: {point_size}",
        "",
        "F = fine/coarse",
        "R = reset",
        "C = toggle depth colour",
        "S = save",
        "N = next camera",
        "P = previous camera",
        "Q/ESC = quit",
    ]

    x = 15
    y = 30

    for i, line in enumerate(lines):

        cv2.putText(
            output,
            line,
            (
                x,
                y + i * 25
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (
                255,
                255,
                255
            ),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            output,
            line,
            (
                x,
                y + i * 25
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (
                0,
                0,
                0
            ),
            1,
            cv2.LINE_AA
        )

    return output


# ============================================================
# MAIN GUI
# ============================================================

def main():

    global current_camera_index
    global fine_mode
    global show_depth_colour
    global point_size
    global current_adjustments

    # --------------------------------------------------------
    # Load all camera information
    # --------------------------------------------------------

    camera_records = {}

    camera_data = {}

    for camera in CAMERAS:

        camera_records[
            camera
        ] = load_image_records(
            camera
        )

        (
            K,
            width,
            height
        ) = load_rectified_intrinsics(
            camera
        )

        camera_data[
            camera
        ] = {
            "K": K,
            "width": width,
            "height": height,
        }

    # --------------------------------------------------------
    # Reference timestamp
    # --------------------------------------------------------

    reference_timestamp = int(
        camera_records[
            REFERENCE_CAMERA
        ][REFERENCE_INDEX][0]
    )

    # --------------------------------------------------------
    # LiDAR
    # --------------------------------------------------------

    lidar_records = (
        load_lidar_records()
    )

    (
        lidar_timestamp,
        lidar_path,
        lidar_delta
    ) = nearest_timestamp(
        reference_timestamp,
        lidar_records
    )

    print()
    print(
        "Reference timestamp:",
        reference_timestamp
    )

    print(
        "LiDAR timestamp:",
        lidar_timestamp
    )

    print(
        "LiDAR difference:",
        f"{lidar_delta / 1e6:.3f} ms"
    )

    print(
        "LiDAR file:",
        lidar_path
    )

    lidar_points = load_lidar_ply(
        lidar_path
    )

    print(
        "LiDAR points:",
        len(lidar_points)
    )

    # --------------------------------------------------------
    # Create GUI
    # --------------------------------------------------------

    create_trackbars()

    saved_transforms = {}

    # ========================================================
    # MAIN LOOP
    # ========================================================

    while True:

        camera = CAMERAS[
            current_camera_index
        ]

        data = camera_data[
            camera
        ]

        K = data["K"]
        width = data["width"]
        height = data["height"]

        # ----------------------------------------------------
        # Camera image
        # ----------------------------------------------------

        (
            image_timestamp,
            image_path,
            image_delta
        ) = nearest_timestamp(
            reference_timestamp,
            camera_records[
                camera
            ]
        )

        image = np.asarray(
            Image.open(
                image_path
            ).convert(
                "RGB"
            )
        )

        # ----------------------------------------------------
        # Existing calibration
        # ----------------------------------------------------

        T_original = (
            invert_transform(
                EXTRINSICS[
                    camera
                ]
            )
        )

        # ----------------------------------------------------
        # Reset transform when entering a new camera
        # ----------------------------------------------------

        # ----------------------------------------------------
        # Build current projection
        # ----------------------------------------------------

        T_current = (
            get_current_transform(
                T_original
            )
        )

        (
            points_camera,
            u,
            v,
            depth
        ) = project_lidar(
            lidar_points,
            T_current,
            K,
            width,
            height,
        )

        # ----------------------------------------------------
        # Render
        # ----------------------------------------------------

        display = render_projection(
            image,
            u,
            v,
            depth,
            get_point_size(),
            show_depth_colour,
        )

        # ----------------------------------------------------
        # Status panel
        # ----------------------------------------------------

        display = draw_status(
            display,
            camera,
            current_adjustments,
            fine_mode,
            get_point_size(),
        )

        # ----------------------------------------------------
        # Add timestamp info
        # ----------------------------------------------------

        cv2.putText(
            display,
            f"Image dt: {image_delta / 1e6:.2f} ms",
            (
                15,
                display.shape[0] - 60
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (
                255,
                255,
                255
            ),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            display,
            f"Visible LiDAR: {len(points_camera)}",
            (
                15,
                display.shape[0] - 30
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (
                255,
                255,
                255
            ),
            2,
            cv2.LINE_AA
        )

        cv2.imshow(
            WINDOW_NAME,
            display
        )

        # ----------------------------------------------------
        # Keyboard
        # ----------------------------------------------------

        key = cv2.waitKey(
            20
        ) & 0xFF

        if key in [
            ord("q"),
            ord("Q"),
            27,
        ]:
            break

        # ----------------------------------------------------
        # Fine/coarse
        # ----------------------------------------------------

        if key in [
            ord("f"),
            ord("F"),
        ]:

            fine_mode = not fine_mode

            # Keep current values within new range.

            rotation_range, translation_range = (
                get_range_values()
            )

            current_adjustments["rx"] = np.clip(
                current_adjustments["rx"],
                -rotation_range,
                rotation_range,
            )

            current_adjustments["ry"] = np.clip(
                current_adjustments["ry"],
                -rotation_range,
                rotation_range,
            )

            current_adjustments["rz"] = np.clip(
                current_adjustments["rz"],
                -rotation_range,
                rotation_range,
            )

            current_adjustments["tx"] = np.clip(
                current_adjustments["tx"],
                -translation_range,
                translation_range,
            )

            current_adjustments["ty"] = np.clip(
                current_adjustments["ty"],
                -translation_range,
                translation_range,
            )

            current_adjustments["tz"] = np.clip(
                current_adjustments["tz"],
                -translation_range,
                translation_range,
            )

            set_trackbars_from_adjustments()

        # ----------------------------------------------------
        # Reset
        # ----------------------------------------------------

        if key in [
            ord("r"),
            ord("R"),
        ]:

            reset_adjustments()

        # ----------------------------------------------------
        # Toggle depth colouring
        # ----------------------------------------------------

        if key in [
            ord("c"),
            ord("C"),
        ]:

            show_depth_colour = (
                not show_depth_colour
            )

        # ----------------------------------------------------
        # Point size
        # ----------------------------------------------------

        if key in [
            ord("+"),
            ord("="),
        ]:

            new_size = min(
                MAX_POINT_SIZE,
                get_point_size() + 1,
            )

            cv2.setTrackbarPos(
                "POINT SIZE",
                WINDOW_NAME,
                new_size,
            )

        if key in [
            ord("-"),
            ord("_"),
        ]:

            new_size = max(
                MIN_POINT_SIZE,
                get_point_size() - 1,
            )

            cv2.setTrackbarPos(
                "POINT SIZE",
                WINDOW_NAME,
                new_size,
            )

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        if key in [
            ord("s"),
            ord("S"),
        ]:

            save_current_calibration(
                camera,
                lidar_points,
                T_original,
                T_current,
                K,
                width,
                height,
            )

            saved_transforms[
                camera
            ] = T_current.copy()

            combined = (
                save_combined_calibration(
                    saved_transforms
                )
            )

            print()
            print(
                "Saved combined calibration:"
            )

            print(
                combined
            )

        # ----------------------------------------------------
        # NEXT CAMERA
        # ----------------------------------------------------

        if key in [
            ord("n"),
            ord("N"),
        ]:

            # Save current camera first.
            save_current_calibration(
                camera,
                lidar_points,
                T_original,
                T_current,
                K,
                width,
                height,
            )

            saved_transforms[
                camera
            ] = T_current.copy()

            combined = (
                save_combined_calibration(
                    saved_transforms
                )
            )

            current_camera_index = (
                current_camera_index + 1
            ) % len(CAMERAS)

            reset_adjustments()

            print()
            print(
                "Moved to:",
                CAMERAS[
                    current_camera_index
                ]
            )

            print(
                "Combined calibration:",
                combined
            )

        # ----------------------------------------------------
        # PREVIOUS CAMERA
        # ----------------------------------------------------

        if key in [
            ord("p"),
            ord("P"),
        ]:

            save_current_calibration(
                camera,
                lidar_points,
                T_original,
                T_current,
                K,
                width,
                height,
            )

            saved_transforms[
                camera
            ] = T_current.copy()

            current_camera_index = (
                current_camera_index - 1
            ) % len(CAMERAS)

            reset_adjustments()

            print()
            print(
                "Moved to:",
                CAMERAS[
                    current_camera_index
                ]
            )

    # ========================================================
    # CLEANUP
    # ========================================================

    cv2.destroyAllWindows()

    print()
    print("=" * 80)
    print("GUI CLOSED")
    print("=" * 80)

    if saved_transforms:

        combined = (
            save_combined_calibration(
                saved_transforms
            )
        )

        print()
        print(
            "Final combined calibration:"
        )

        print(
            combined
        )

    print()
    print(
        "Output directory:"
    )

    print(
        OUTPUT_DIR
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()