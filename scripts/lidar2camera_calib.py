import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw


# ============================================================
# CONFIG
# ============================================================

BAG_ROOT = Path(
    "/home/lcha0115/bt60_scratch/lcha_data/"
    "map-anything/personal_data/"
    "camera_lidar_imu_test_04"
)

EXTRACTED = BAG_ROOT / "extracted_new"

CAMERAS = [
    "CAM1",
    "CAM2",
    "CAM6",
]

# ------------------------------------------------------------
# Reference camera and frame
# ------------------------------------------------------------

REFERENCE_CAMERA = "CAM2"

# 0 = first CAM2 image
# 1 = second CAM2 image
# etc.
REFERENCE_INDEX = 0

# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

OUTPUT_DIR = (
    BAG_ROOT /
    "lidar_camera_projection"
)

# ------------------------------------------------------------
# Visualisation
# ------------------------------------------------------------

# 1 = every LiDAR point
# 5 = every fifth point
# 10 = every tenth point
POINT_STRIDE = 5

POINT_RADIUS = 2

# ------------------------------------------------------------
# Timestamp warning
# ------------------------------------------------------------

MAX_SYNC_ERROR_NS = 50_000_000  # 50 ms


# ============================================================
# ROS TF EXTRINSICS
# ============================================================
#
# These are the matrices from:
#
# extracted_new/calibration/extrinsics.txt
#
# They are labelled:
#
#     velodyne -> cam*_optical_frame
#
# For this dataset, testing showed that the INVERSE of these
# matrices gives the correct LiDAR-to-camera projection.
#
# Therefore this script uses:
#
#     inverse(TF)
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
# INVERT HOMOGENEOUS TRANSFORM
# ============================================================

def invert_transform(T):

    R = T[:3, :3]
    t = T[:3, 3]

    T_inv = np.eye(
        4,
        dtype=np.float64
    )

    T_inv[:3, :3] = R.T

    T_inv[:3, 3] = (
        -R.T @ t
    )

    return T_inv


# ============================================================
# TRANSFORM POINTS
# ============================================================

def transform_points(
    points,
    T
):

    points = np.asarray(
        points,
        dtype=np.float64
    )

    ones = np.ones(
        (
            len(points),
            1
        ),
        dtype=np.float64
    )

    homogeneous = np.concatenate(
        [
            points,
            ones
        ],
        axis=1
    )

    transformed = (
        homogeneous @ T.T
    )

    return transformed[:, :3]


# ============================================================
# LOAD TIMESTAMPED RECTIFIED IMAGES
# ============================================================

def load_images(camera):

    directory = (
        EXTRACTED /
        camera /
        "images_rect"
    )

    if not directory.exists():

        raise RuntimeError(
            f"Image directory does not exist:\n"
            f"{directory}"
        )

    files = []

    for path in directory.glob("*.png"):

        try:

            timestamp = int(
                path.stem
            )

        except ValueError:

            continue

        files.append(
            (
                timestamp,
                path
            )
        )

    files.sort(
        key=lambda x: x[0]
    )

    if not files:

        raise RuntimeError(
            f"No timestamped rectified images found:\n"
            f"{directory}"
        )

    return files


# ============================================================
# LOAD TIMESTAMPED LIDAR FILES
# ============================================================

def load_lidar_files():

    directory = (
        EXTRACTED /
        "lidar"
    )

    if not directory.exists():

        raise RuntimeError(
            f"LiDAR directory does not exist:\n"
            f"{directory}"
        )

    files = []

    for path in directory.glob("*.ply"):

        try:

            timestamp = int(
                path.stem
            )

        except ValueError:

            continue

        files.append(
            (
                timestamp,
                path
            )
        )

    files.sort(
        key=lambda x: x[0]
    )

    if not files:

        raise RuntimeError(
            f"No timestamped LiDAR PLY files found:\n"
            f"{directory}"
        )

    return files


# ============================================================
# FIND NEAREST TIMESTAMP
# ============================================================

def find_nearest_timestamp(
    target_timestamp,
    records
):

    timestamps = np.array(
        [
            x[0]
            for x in records
        ],
        dtype=np.int64
    )

    index = np.searchsorted(
        timestamps,
        target_timestamp
    )

    candidates = []

    if index > 0:

        candidates.append(
            index - 1
        )

    if index < len(timestamps):

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
            int(
                timestamps[i]
            )
            -
            int(
                target_timestamp
            )
        )
    )

    timestamp = int(
        timestamps[best]
    )

    path = records[
        best
    ][1]

    delta = abs(
        timestamp -
        int(
            target_timestamp
        )
    )

    return (
        timestamp,
        path,
        delta
    )


# ============================================================
# LOAD RECTIFIED CAMERA PROJECTION MATRIX
# ============================================================
#
# Because we are using images_rect:
#
#     K_rect = P[:3, :3]
#
# We do NOT use the raw distorted K.
#
# ============================================================

def load_rectified_projection(
    camera
):

    path = (
        EXTRACTED /
        "calibration" /
        f"{camera}_intrinsics.npz"
    )

    print(
        f"\nLoading calibration:"
    )

    print(
        path
    )

    data = np.load(
        path
    )

    print(
        "  Keys:",
        list(
            data.keys()
        )
    )

    if "P" not in data:

        raise RuntimeError(
            f"No P matrix found in:\n"
            f"{path}"
        )

    P = np.asarray(
        data["P"],
        dtype=np.float64
    )

    if P.shape == (
        3,
        4
    ):

        K_rect = P[:, :3]

    elif P.size == 12:

        P = P.reshape(
            3,
            4
        )

        K_rect = P[:, :3]

    else:

        raise RuntimeError(
            f"Unexpected P shape: "
            f"{P.shape}"
        )

    print(
        "\n  P:"
    )

    print(
        P
    )

    print(
        "\n  Rectified K:"
    )

    print(
        K_rect
    )

    return K_rect


# ============================================================
# LOAD PLY XYZ + INTENSITY
# ============================================================
#
# We load:
#
#     columns 0,1,2 -> XYZ
#     column 3      -> intensity
#
# if a fourth column exists.
#
# ============================================================

def load_lidar_ply(
    path
):

    print(
        "\nLoading LiDAR:"
    )

    print(
        path
    )

    header = []

    with open(
        path,
        "r"
    ) as f:

        while True:

            line = f.readline()

            if not line:

                raise RuntimeError(
                    "Unexpected end of PLY header."
                )

            line = line.strip()

            header.append(
                line
            )

            if line == "end_header":

                break

    # --------------------------------------------------------
    # Find number of vertices
    # --------------------------------------------------------

    vertex_count = None

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

            break

    if vertex_count is None:

        raise RuntimeError(
            "Could not determine "
            "PLY vertex count."
        )

    # --------------------------------------------------------
    # Print PLY properties
    # --------------------------------------------------------

    print(
        "\nPLY vertex properties:"
    )

    inside_vertex = False

    for line in header:

        parts = line.split()

        if (
            len(parts) >= 3
            and
            parts[0] == "element"
            and
            parts[1] == "vertex"
        ):

            inside_vertex = True

            continue

        if (
            inside_vertex
            and
            len(parts) >= 2
            and
            parts[0] == "property"
        ):

            print(
                " ",
                " ".join(parts)
            )

        if (
            inside_vertex
            and
            len(parts) >= 2
            and
            parts[0] == "element"
            and
            parts[1] != "vertex"
        ):

            break

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

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

    if data.shape[1] < 3:

        raise RuntimeError(
            "PLY does not contain "
            "at least XYZ."
        )

    # --------------------------------------------------------
    # XYZ
    # --------------------------------------------------------

    xyz = data[
        :,
        :3
    ].astype(
        np.float64
    )

    # --------------------------------------------------------
    # Intensity
    #
    # Assumption:
    #
    #     4th column = intensity
    #
    # If the PLY has a fourth field, we retain it.
    # --------------------------------------------------------

    if data.shape[1] >= 4:

        intensity = data[
            :,
            3
        ].astype(
            np.float64
        )

        has_intensity = True

    else:

        intensity = None

        has_intensity = False

    print(
        "\nNumber of vertices:",
        len(xyz)
    )

    print(
        "Number of columns:",
        data.shape[1]
    )

    print(
        "Intensity available:",
        has_intensity
    )

    if intensity is not None:

        print(
            "Intensity range:",
            intensity.min(),
            "to",
            intensity.max()
        )

    return (
        xyz,
        intensity
    )


# ============================================================
# PROJECT LIDAR INTO CAMERA
# ============================================================

def project_lidar(
    lidar_points,
    T_lidar_to_camera,
    K_rect,
    width,
    height
):

    # --------------------------------------------------------
    # LiDAR → camera
    # --------------------------------------------------------

    camera_points = transform_points(
        lidar_points,
        T_lidar_to_camera
    )

    X = camera_points[
        :,
        0
    ]

    Y = camera_points[
        :,
        1
    ]

    Z = camera_points[
        :,
        2
    ]

    # --------------------------------------------------------
    # ROS optical frame:
    #
    # +X = right
    # +Y = down
    # +Z = forward
    #
    # Keep points in front of camera.
    # --------------------------------------------------------

    valid_front = (
        Z > 0.05
    )

    original_indices = np.arange(
        len(lidar_points)
    )

    original_indices = (
        original_indices[
            valid_front
        ]
    )

    X = X[
        valid_front
    ]

    Y = Y[
        valid_front
    ]

    Z = Z[
        valid_front
    ]

    # --------------------------------------------------------
    # Rectified pinhole projection
    # --------------------------------------------------------

    fx = K_rect[
        0,
        0
    ]

    fy = K_rect[
        1,
        1
    ]

    cx = K_rect[
        0,
        2
    ]

    cy = K_rect[
        1,
        2
    ]

    u = (
        fx * X / Z
        +
        cx
    )

    v = (
        fy * Y / Z
        +
        cy
    )

    # --------------------------------------------------------
    # Keep only points inside image
    # --------------------------------------------------------

    inside = (
        (u >= 0)
        &
        (u < width)
        &
        (v >= 0)
        &
        (v < height)
    )

    u = u[
        inside
    ]

    v = v[
        inside
    ]

    Z = Z[
        inside
    ]

    original_indices = (
        original_indices[
            inside
        ]
    )

    return (
        u,
        v,
        Z,
        original_indices
    )


# ============================================================
# DEPTH OVERLAY
# ============================================================
#
# RED  = near
# BLUE = far
#
# Depth = Z_camera.
#
# ============================================================

def draw_depth_overlay(
    image,
    u,
    v,
    depths
):

    output = Image.fromarray(
        image.copy()
    )

    draw = ImageDraw.Draw(
        output
    )

    if len(depths) == 0:

        return output

    # --------------------------------------------------------
    # Robust colour range
    # --------------------------------------------------------

    d_min = np.percentile(
        depths,
        2
    )

    d_max = np.percentile(
        depths,
        98
    )

    d_range = max(
        d_max - d_min,
        1e-6
    )

    indices = np.arange(
        0,
        len(depths),
        POINT_STRIDE
    )

    for i in indices:

        x = int(
            round(
                u[i]
            )
        )

        y = int(
            round(
                v[i]
            )
        )

        depth_norm = (
            depths[i] -
            d_min
        ) / d_range

        depth_norm = np.clip(
            depth_norm,
            0.0,
            1.0
        )

        # Near = red
        # Far = blue

        r = int(
            255 *
            (1.0 - depth_norm)
        )

        b = int(
            255 *
            depth_norm
        )

        draw.ellipse(
            (
                x - POINT_RADIUS,
                y - POINT_RADIUS,
                x + POINT_RADIUS,
                y + POINT_RADIUS
            ),
            fill=(
                r,
                50,
                b
            )
        )

    # --------------------------------------------------------
    # Legend
    # --------------------------------------------------------

    legend_x = 20
    legend_y = 20

    legend_width = 250
    legend_height = 85

    draw.rectangle(
        (
            legend_x,
            legend_y,
            legend_x + legend_width,
            legend_y + legend_height
        ),
        fill=(
            0,
            0,
            0
        )
    )

    draw.text(
        (
            legend_x + 10,
            legend_y + 8
        ),
        f"Near: {d_min:.1f} m",
        fill=(
            255,
            0,
            0
        )
    )

    draw.text(
        (
            legend_x + 10,
            legend_y + 30
        ),
        f"Far:  {d_max:.1f} m",
        fill=(
            0,
            50,
            255
        )
    )

    draw.text(
        (
            legend_x + 10,
            legend_y + 52
        ),
        "Colour = camera depth",
        fill=(
            255,
            255,
            255
        )
    )

    return output


# ============================================================
# INTENSITY OVERLAY
# ============================================================
#
# DARK = low intensity
# BRIGHT = high intensity
#
# ============================================================

def draw_intensity_overlay(
    image,
    u,
    v,
    intensities
):

    output = Image.fromarray(
        image.copy()
    )

    draw = ImageDraw.Draw(
        output
    )

    if (
        intensities is None
        or
        len(intensities) == 0
    ):

        return output

    # --------------------------------------------------------
    # Robust intensity range
    # --------------------------------------------------------

    i_min = np.percentile(
        intensities,
        2
    )

    i_max = np.percentile(
        intensities,
        98
    )

    i_range = max(
        i_max - i_min,
        1e-6
    )

    indices = np.arange(
        0,
        len(intensities),
        POINT_STRIDE
    )

    for i in indices:

        x = int(
            round(
                u[i]
            )
        )

        y = int(
            round(
                v[i]
            )
        )

        intensity_norm = (
            intensities[i] -
            i_min
        ) / i_range

        intensity_norm = np.clip(
            intensity_norm,
            0.0,
            1.0
        )

        value = int(
            255 *
            intensity_norm
        )

        draw.ellipse(
            (
                x - POINT_RADIUS,
                y - POINT_RADIUS,
                x + POINT_RADIUS,
                y + POINT_RADIUS
            ),
            fill=(
                value,
                value,
                value
            )
        )

    # --------------------------------------------------------
    # Legend
    # --------------------------------------------------------

    legend_x = 20
    legend_y = 20

    legend_width = 300
    legend_height = 85

    draw.rectangle(
        (
            legend_x,
            legend_y,
            legend_x + legend_width,
            legend_y + legend_height
        ),
        fill=(
            0,
            0,
            0
        )
    )

    draw.text(
        (
            legend_x + 10,
            legend_y + 8
        ),
        f"Low:  {i_min:.1f}",
        fill=(
            80,
            80,
            80
        )
    )

    draw.text(
        (
            legend_x + 10,
            legend_y + 30
        ),
        f"High: {i_max:.1f}",
        fill=(
            255,
            255,
            255
        )
    )

    draw.text(
        (
            legend_x + 10,
            legend_y + 52
        ),
        "Brightness = LiDAR intensity",
        fill=(
            255,
            255,
            255
        )
    )

    return output


# ============================================================
# MAIN
# ============================================================

print(
    "=" * 80
)

print(
    "LiDAR → CAMERA DEPTH + INTENSITY PROJECTION"
)

print(
    "=" * 80
)

print(
    "\nDataset:"
)

print(
    EXTRACTED
)

print(
    "\nCamera images:"
)

print(
    "images_rect"
)

print(
    "\nCamera calibration:"
)

print(
    "P[:3,:3]"
)

print(
    "\nTransform:"
)

print(
    "INVERSE of stored velodyne → camera TF"
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD CAMERA IMAGE LISTS
# ============================================================

camera_records = {}

for camera in CAMERAS:

    records = load_images(
        camera
    )

    camera_records[
        camera
    ] = records

    print(
        f"\n{camera}: "
        f"{len(records)} rectified images"
    )


# ============================================================
# REFERENCE CAMERA FRAME
# ============================================================

reference_records = (
    camera_records[
        REFERENCE_CAMERA
    ]
)

if (
    REFERENCE_INDEX < 0
    or
    REFERENCE_INDEX >= len(
        reference_records
    )
):

    raise RuntimeError(
        "REFERENCE_INDEX is outside "
        "the available image range."
    )

reference_timestamp = (
    reference_records[
        REFERENCE_INDEX
    ][0]
)

reference_path = (
    reference_records[
        REFERENCE_INDEX
    ][1]
)

print(
    "\n" + "=" * 80
)

print(
    "REFERENCE FRAME"
)

print(
    "=" * 80
)

print(
    "Reference camera:",
    REFERENCE_CAMERA
)

print(
    "Reference image:",
    reference_path.name
)

print(
    "Reference timestamp:",
    reference_timestamp
)


# ============================================================
# FIND CLOSEST LIDAR
# ============================================================

lidar_records = (
    load_lidar_files()
)

(
    lidar_timestamp,
    lidar_path,
    lidar_delta
) = find_nearest_timestamp(
    reference_timestamp,
    lidar_records
)

if lidar_path is None:

    raise RuntimeError(
        "Could not find matching LiDAR scan."
    )

print(
    "\n" + "=" * 80
)

print(
    "TIME SYNCHRONISATION"
)

print(
    "=" * 80
)

print(
    "Camera timestamp:",
    reference_timestamp
)

print(
    "LiDAR timestamp:",
    lidar_timestamp
)

print(
    "LiDAR file:",
    lidar_path.name
)

print(
    "Difference:",
    f"{lidar_delta / 1e6:.3f} ms"
)

if (
    lidar_delta >
    MAX_SYNC_ERROR_NS
):

    print(
        "\nWARNING:"
    )

    print(
        "Camera/LiDAR timestamp difference "
        "exceeds "
        f"{MAX_SYNC_ERROR_NS / 1e6:.1f} ms."
    )

else:

    print(
        "Synchronization looks good."
    )


# ============================================================
# LOAD LIDAR
# ============================================================

(
    lidar_points,
    lidar_intensity
) = load_lidar_ply(
    lidar_path
)


# ============================================================
# RAW LIDAR STATISTICS
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "RAW LIDAR COORDINATES"
)

print(
    "=" * 80
)

print(
    "Points:",
    len(lidar_points)
)

print(
    "X:",
    lidar_points[:, 0].min(),
    "to",
    lidar_points[:, 0].max()
)

print(
    "Y:",
    lidar_points[:, 1].min(),
    "to",
    lidar_points[:, 1].max()
)

print(
    "Z:",
    lidar_points[:, 2].min(),
    "to",
    lidar_points[:, 2].max()
)

print(
    "Mean XYZ:",
    lidar_points.mean(
        axis=0
    )
)

if lidar_intensity is not None:

    print(
        "Intensity:",
        lidar_intensity.min(),
        "to",
        lidar_intensity.max()
    )

else:

    print(
        "\nWARNING:"
    )

    print(
        "No fourth PLY column was found."
    )

    print(
        "Intensity overlay will NOT be generated."
    )


# ============================================================
# PROCESS EACH CAMERA
# ============================================================

for camera in CAMERAS:

    print(
        "\n" + "=" * 80
    )

    print(
        camera
    )

    print(
        "=" * 80
    )

    # --------------------------------------------------------
    # Find camera image nearest to reference timestamp
    # --------------------------------------------------------

    (
        image_timestamp,
        image_path,
        image_delta
    ) = find_nearest_timestamp(
        reference_timestamp,
        camera_records[
            camera
        ]
    )

    print(
        "\nImage:"
    )

    print(
        image_path.name
    )

    print(
        "Image timestamp:",
        image_timestamp
    )

    print(
        "Difference from reference:",
        f"{image_delta / 1e6:.3f} ms"
    )

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = np.asarray(
        Image.open(
            image_path
        ).convert(
            "RGB"
        )
    )

    height, width = (
        image.shape[:2]
    )

    print(
        "Image resolution:",
        width,
        "x",
        height
    )

    # --------------------------------------------------------
    # Load rectified projection matrix
    # --------------------------------------------------------

    K_rect = (
        load_rectified_projection(
            camera
        )
    )

    # --------------------------------------------------------
    # Original TF
    # --------------------------------------------------------

    T_velodyne_to_camera = (
        EXTRINSICS[
            camera
        ]
    )

    # --------------------------------------------------------
    # Inverse transform
    #
    # This is the transform that produced the correct
    # projection in your previous test.
    # --------------------------------------------------------

    T_lidar_to_camera = (
        invert_transform(
            T_velodyne_to_camera
        )
    )

    print(
        "\nUsing inverse transform:"
    )

    print(
        T_lidar_to_camera
    )

    # --------------------------------------------------------
    # Project LiDAR
    # --------------------------------------------------------

    (
        u,
        v,
        depths,
        lidar_indices
    ) = project_lidar(
        lidar_points=lidar_points,
        T_lidar_to_camera=T_lidar_to_camera,
        K_rect=K_rect,
        width=width,
        height=height
    )

    print(
        "\nProjection statistics:"
    )

    print(
        "Original LiDAR points:",
        len(lidar_points)
    )

    print(
        "Visible in camera:",
        len(u)
    )

    if len(depths) > 0:

        print(
            "Camera depth range:",
            f"{depths.min():.3f}",
            "to",
            f"{depths.max():.3f}",
            "m"
        )

        print(
            "Median camera depth:",
            f"{np.median(depths):.3f}",
            "m"
        )

    # --------------------------------------------------------
    # Get projected intensity values
    # --------------------------------------------------------

    if lidar_intensity is not None:

        projected_intensity = (
            lidar_intensity[
                lidar_indices
            ]
        )

    else:

        projected_intensity = None

    # ========================================================
    # DEPTH OVERLAY
    # ========================================================

    depth_overlay = (
        draw_depth_overlay(
            image=image,
            u=u,
            v=v,
            depths=depths
        )
    )

    depth_path = (
        OUTPUT_DIR /
        f"{camera}_DEPTH.png"
    )

    depth_overlay.save(
        depth_path
    )

    print(
        "\nSaved depth overlay:"
    )

    print(
        depth_path
    )

    # ========================================================
    # INTENSITY OVERLAY
    # ========================================================

    if projected_intensity is not None:

        intensity_overlay = (
            draw_intensity_overlay(
                image=image,
                u=u,
                v=v,
                intensities=projected_intensity
            )
        )

        intensity_path = (
            OUTPUT_DIR /
            f"{camera}_INTENSITY.png"
        )

        intensity_overlay.save(
            intensity_path
        )

        print(
            "Saved intensity overlay:"
        )

        print(
            intensity_path
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "COMPLETE"
)

print(
    "=" * 80
)

print(
    "\nOutput directory:"
)

print(
    OUTPUT_DIR
)

print(
    "\nExpected files:"
)

for camera in CAMERAS:

    print(
        f"  {camera}_DEPTH.png"
    )

    if lidar_intensity is not None:

        print(
            f"  {camera}_INTENSITY.png"
        )

print(
    "\nDEPTH:"
)

print(
    "  Red  = close"
)

print(
    "  Blue = far"
)

print(
    "  Distance shown is Z_camera."
)

print(
    "\nINTENSITY:"
)

if lidar_intensity is not None:

    print(
        "  Dark  = low intensity"
    )

    print(
        "  Bright = high intensity"
    )

else:

    print(
        "  Not available in this PLY."
    )

print(
    "\nTransform:"
)

print(
    "  inverse(velodyne → camera TF)"
)

print(
    "\nCamera images:"
)

print(
    "  images_rect"
)

print(
    "\nCamera projection:"
)

print(
    "  P[:3,:3]"
)

print(
    "=" * 80
)