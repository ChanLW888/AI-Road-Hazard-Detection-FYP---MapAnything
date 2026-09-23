import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


# ============================================================
# CONFIG
# ============================================================

BAG_ROOT = Path(
    "/home/lcha0115/bt60_scratch/lcha_data/"
    "map-anything/personal_data/"
    "camera_lidar_imu_test_04"
)

EXTRACTED = BAG_ROOT / "extracted_new"

OUTPUT_DIR = BAG_ROOT / "lidar_image_overlay_new"

CAMERAS = ["CAM1", "CAM2", "CAM6"]

REFERENCE_CAMERA = "CAM2"
REFERENCE_INDEX = 0

LIDAR_MIN_CAMERA_DEPTH = 0.05
BORDER_MARGIN_PIXELS = 2

POINT_SIZE = 2


# ============================================================
# NEW GUI-REFINED LiDAR -> CAMERA MATRICES
# ============================================================
#
# These are the latest GUI-refined matrices.
#
# IMPORTANT:
#
# These are ALREADY:
#
#     LiDAR -> camera
#
# DO NOT INVERT THEM.
#
# ============================================================

LIDAR_TO_CAMERA = {

    "CAM1": np.array([
        [
            -0.74075182,
             0.67164194,
             0.01355925,
            -0.02493503,
        ],
        [
             0.24699625,
             0.29107072,
            -0.92426765,
            -0.10583372,
        ],
        [
            -0.62472362,
            -0.68130386,
            -0.38150420,
            -1.42902479,
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
             0.63311790,
             0.77355266,
            -0.02789272,
            -0.41198233,
        ],
        [
             0.12521772,
            -0.13791188,
            -0.98249724,
            -0.67023320,
        ],
        [
            -0.76386009,
             0.61854393,
            -0.18417698,
            -0.66565741,
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
            -0.70679283,
            -0.70736252,
             0.00906393,
             0.88808945,
        ],
        [
            -0.15894889,
             0.14630976,
            -0.97638553,
            -0.71249618,
        ],
        [
             0.68933239,
            -0.69154300,
            -0.21584518,
            -0.60088861,
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
# HELPERS
# ============================================================

def transform_points(
    points,
    T
):

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


def parse_array(
    value
):

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

    return np.fromstring(
        value,
        sep=" ",
        dtype=np.float64
    )


# ============================================================
# LOAD CAMERA INTRINSICS
# ============================================================

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
    ).reshape(
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

    return K, width, height


# ============================================================
# FIND IMAGE
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

    return records


# ============================================================
# FIND LiDAR
# ============================================================

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

    idx = int(
        np.searchsorted(
            timestamps,
            target
        )
    )

    candidates = []

    if idx > 0:
        candidates.append(
            idx - 1
        )

    if idx < len(records):
        candidates.append(
            idx
        )

    if not candidates:
        return (
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
        records[best][1]
    )


# ============================================================
# LOAD LiDAR PLY
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
            and
            parts[0] == "element"
            and
            parts[1] == "vertex"
        ):

            vertex_count = int(
                parts[2]
            )

        if (
            len(parts) >= 3
            and
            parts[0] == "property"
        ):

            properties.append(
                parts[-1]
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

    x = properties.index("x")
    y = properties.index("y")
    z = properties.index("z")

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
# PROJECT LiDAR ONTO IMAGE
# ============================================================

def project_lidar(
    lidar_points,
    T_lidar_to_camera,
    K,
    width,
    height
):

    camera_points = transform_points(
        lidar_points,
        T_lidar_to_camera
    )

    X = camera_points[:, 0]
    Y = camera_points[:, 1]
    Z = camera_points[:, 2]

    valid = (
        np.isfinite(
            camera_points
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
        u[inside],
        v[inside],
        Z[inside]
    )


# ============================================================
# DRAW OVERLAY
# ============================================================

def draw_overlay(
    image,
    u,
    v,
    Z
):

    output = image.copy()

    # Colour by camera Z only.
    z_min = np.percentile(
        Z,
        2
    )

    z_max = np.percentile(
        Z,
        98
    )

    norm = np.clip(
        (
            Z -
            z_min
        )
        /
        max(
            z_max -
            z_min,
            1e-8
        ),
        0,
        1
    )

    for x, y, value in zip(
        u,
        v,
        norm
    ):

        x = int(
            round(x)
        )

        y = int(
            round(y)
        )

        # Near = red
        # Far  = blue

        r = int(
            255 *
            (
                1 -
                value
            )
        )

        b = int(
            255 *
            value
        )

        cv2.circle(
            output,
            (
                x,
                y
            ),
            POINT_SIZE,
            (
                b,
                0,
                r
            ),
            -1
        )

    return output


# ============================================================
# MAIN
# ============================================================

print("=" * 80)
print("LiDAR → IMAGE OVERLAY USING GUI CALIBRATION")
print("=" * 80)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# REFERENCE FRAME
# ============================================================

camera_records = {}

for camera in CAMERAS:

    camera_records[
        camera
    ] = load_image_records(
        camera
    )

reference_timestamp = int(
    camera_records[
        REFERENCE_CAMERA
    ][REFERENCE_INDEX][0]
)

print()
print(
    "Reference timestamp:",
    reference_timestamp
)


# ============================================================
# LiDAR
# ============================================================

lidar_records = (
    load_lidar_records()
)

(
    lidar_timestamp,
    lidar_path
) = nearest_timestamp(
    reference_timestamp,
    lidar_records
)

print()
print(
    "LiDAR timestamp:",
    lidar_timestamp
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


# ============================================================
# PROCESS EACH CAMERA
# ============================================================

for camera in CAMERAS:

    print()
    print("=" * 80)

    print(
        camera
    )

    print(
        "=" * 80
    )

    # --------------------------------------------------------
    # Camera calibration
    # --------------------------------------------------------

    (
        K,
        width,
        height
    ) = load_rectified_intrinsics(
        camera
    )

    print(
        "\nK_rect:"
    )

    print(
        K
    )

    # --------------------------------------------------------
    # Image
    # --------------------------------------------------------

    image_timestamp, image_path = (
        nearest_timestamp(
            reference_timestamp,
            camera_records[
                camera
            ]
        )
    )

    print()
    print(
        "Image timestamp:",
        image_timestamp
    )

    print(
        "Image:",
        image_path
    )

    image = np.asarray(
        Image.open(
            image_path
        ).convert(
            "RGB"
        )
    )

    # Convert RGB -> BGR for OpenCV
    image = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR
    )

    # --------------------------------------------------------
    # NEW GUI REFINED EXTRINSIC
    # --------------------------------------------------------

    T_lidar_to_camera = (
        LIDAR_TO_CAMERA[
            camera
        ]
    )

    print()
    print(
        "Using GUI LiDAR -> camera:"
    )

    print(
        T_lidar_to_camera
    )

    # --------------------------------------------------------
    # Projection
    # --------------------------------------------------------

    (
        u,
        v,
        Z
    ) = project_lidar(
        lidar_points,
        T_lidar_to_camera,
        K,
        width,
        height
    )

    print()
    print(
        "Projected LiDAR points:",
        len(u)
    )

    if len(u) == 0:

        print(
            "WARNING: no LiDAR points projected "
            "inside the image."
        )

        continue

    # --------------------------------------------------------
    # Draw
    # --------------------------------------------------------

    overlay = draw_overlay(
        image,
        u,
        v,
        Z
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_file = (
        OUTPUT_DIR
        /
        f"{camera}_overlay.png"
    )

    cv2.imwrite(
        str(output_file),
        overlay
    )

    print()
    print(
        "Saved:",
        output_file
    )


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 80)

print(
    "OVERLAY TEST COMPLETE"
)

print(
    "=" * 80
)

print()
print(
    "Output directory:"
)

print(
    OUTPUT_DIR
)

print()
print(
    "Files:"
)

for camera in CAMERAS:

    print(
        OUTPUT_DIR
        /
        f"{camera}_overlay.png"
    )

print("=" * 80)