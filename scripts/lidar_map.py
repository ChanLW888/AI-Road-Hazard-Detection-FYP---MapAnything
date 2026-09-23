import csv
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from mapanything.models import MapAnything
from mapanything.utils.image import preprocess_inputs


# ============================================================
# CONFIG
# ============================================================

BAG_ROOT = Path(
    "/home/lcha0115/bt60_scratch/lcha_data/"
    "map-anything/personal_data/hazard_test_101"
)

EXTRACTED = BAG_ROOT / "extracted"

CAMERAS = ["CAM1", "CAM2", "CAM6"]

REFERENCE_CAMERA = "CAM2"
REFERENCE_INDEX = 6500

OUTPUT_DIR = (
    BAG_ROOT /
    f"mapanything_lidar_{REFERENCE_INDEX}_50m"
)

SYNC_TOLERANCE_NS = 50_000_000

LIDAR_MIN_CAMERA_DEPTH = 0.05
BORDER_MARGIN_PIXELS = 2

# ONLY LiDAR parameter being changed.
LIDAR_MAX_RANGE = 50.0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# HARD-CODED REFINED EXTRINSICS
# LiDAR -> Camera
# ============================================================

LIDAR_TO_CAMERA = {

    "CAM1": np.array([
        [-0.77287266,  0.63426662,  0.01933150, -0.45922380],
        [ 0.22607452,  0.30368823, -0.92556133, -0.36791854],
        [-0.59292340, -0.71097069, -0.37810385, -1.18339019],
        [ 0.00000000,  0.00000000,  0.00000000,  1.00000000],
    ], dtype=np.float64),

    "CAM2": np.array([
        [ 0.64901045,  0.76012775, -0.03148390, -0.18778682],
        [ 0.09876220, -0.12521404, -0.98720184, -0.85765461],
        [-0.75434174,  0.63759489, -0.15633711, -0.93159365],
        [ 0.00000000,  0.00000000,  0.00000000,  1.00000000],
    ], dtype=np.float64),

    "CAM6": np.array([
        [-0.67094235, -0.74109244, -0.02486673,  0.26797529],
        [-0.11488898,  0.13702719, -0.98388214, -0.69426488],
        [ 0.73255504, -0.65727128, -0.17708070, -0.91874840],
        [ 0.00000000,  0.00000000,  0.00000000,  1.00000000],
    ], dtype=np.float64),
}


# ============================================================
# TRANSFORMS
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

    flat = points.reshape(
        -1,
        3
    )

    homogeneous = np.concatenate(
        [
            flat,
            np.ones(
                (len(flat), 1),
                dtype=np.float64
            )
        ],
        axis=1
    )

    transformed = (
        homogeneous @ T.T
    )[:, :3]

    return transformed.reshape(
        points.shape
    )


# ============================================================
# CAMERA DATA
# ============================================================

def parse_array(value):

    value = str(value).strip()

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
            f"{camera}: invalid P size "
            f"{P.size}"
        )

    K = P.reshape(
        3,
        4
    )[:, :3]

    return (
        K,
        int(row["width"]),
        int(row["height"])
    )


def load_images(
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


def load_lidar_files():

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
    target_timestamp,
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
            target_timestamp
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
            - int(target_timestamp)
        )
    )

    timestamp = int(
        records[best][0]
    )

    path = records[best][1]

    delta = abs(
        timestamp
        - int(target_timestamp)
    )

    return (
        timestamp,
        path,
        delta
    )


# ============================================================
# PLY
# ============================================================

def load_ascii_ply(
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

            break

    if vertex_count is None:
        raise RuntimeError(
            f"No vertex count found:\n{path}"
        )

    properties = []

    in_vertex = False

    for line in header:

        parts = line.split()

        if (
            len(parts) >= 3
            and parts[0] == "element"
            and parts[1] == "vertex"
        ):

            in_vertex = True
            continue

        if (
            in_vertex
            and len(parts) >= 3
            and parts[0] == "property"
        ):

            properties.append(
                parts[-1]
            )

        elif (
            in_vertex
            and len(parts) >= 2
            and parts[0] == "element"
        ):

            break

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

    if data.shape[1] != len(
        properties
    ):

        raise RuntimeError(
            f"{path}: "
            f"{data.shape[1]} columns vs "
            f"{len(properties)} properties."
        )

    return (
        data,
        properties
    )


def load_lidar_ply(
    path
):

    data, properties = (
        load_ascii_ply(path)
    )

    x_idx = properties.index(
        "x"
    )

    y_idx = properties.index(
        "y"
    )

    z_idx = properties.index(
        "z"
    )

    return data[
        :,
        [x_idx, y_idx, z_idx]
    ].astype(
        np.float64
    )


# ============================================================
# LiDAR PROJECTION
# ============================================================

def project_lidar_to_image(
    lidar_points,
    T_lidar_to_camera,
    K,
    width,
    height
):

    points = np.asarray(
        lidar_points,
        dtype=np.float64
    )

    # --------------------------------------------------------
    # 50 m LiDAR range filter
    # --------------------------------------------------------

    ranges = np.linalg.norm(
        points[:, :3],
        axis=1
    )

    range_mask = (
        ranges <= LIDAR_MAX_RANGE
    )

    points = points[
        range_mask
    ]

    # --------------------------------------------------------
    # Transform LiDAR -> Camera
    # --------------------------------------------------------

    camera_points = transform_points(
        points,
        T_lidar_to_camera
    )

    X = camera_points[:, 0]
    Y = camera_points[:, 1]
    Z = camera_points[:, 2]

    # --------------------------------------------------------
    # Valid depth
    # --------------------------------------------------------

    valid = (
        np.isfinite(
            camera_points
        ).all(axis=1)
        &
        (
            Z >
            LIDAR_MIN_CAMERA_DEPTH
        )
    )

    X = X[valid]
    Y = Y[valid]
    Z = Z[valid]

    if len(Z) == 0:

        return (
            np.empty(
                0,
                dtype=np.intp
            ),
            np.empty(
                0,
                dtype=np.intp
            ),
            np.empty(
                0,
                dtype=np.float64
            )
        )

    # --------------------------------------------------------
    # Project into image
    # --------------------------------------------------------

    fx = K[0, 0]
    fy = K[1, 1]

    cx = K[0, 2]
    cy = K[1, 2]

    u = (
        fx * X / Z
        + cx
    )

    v = (
        fy * Y / Z
        + cy
    )

    # Round BEFORE bounds check.
    u = np.rint(
        u
    ).astype(
        np.intp
    )

    v = np.rint(
        v
    ).astype(
        np.intp
    )

    # --------------------------------------------------------
    # Image bounds
    # --------------------------------------------------------

    inside = (
        (u >= BORDER_MARGIN_PIXELS)
        &
        (
            u <
            width -
            BORDER_MARGIN_PIXELS
        )
        &
        (
            v >=
            BORDER_MARGIN_PIXELS
        )
        &
        (
            v <
            height -
            BORDER_MARGIN_PIXELS
        )
        &
        (u >= 0)
        &
        (u < width)
        &
        (v >= 0)
        &
        (v < height)
    )

    return (
        u[inside],
        v[inside],
        Z[inside]
    )


# ============================================================
# LiDAR DEPTH MAP
# ============================================================

def lidar_to_camera_depth(
    lidar_points,
    T_lidar_to_camera,
    K,
    width,
    height
):

    u, v, depth_values = (
        project_lidar_to_image(
            lidar_points,
            T_lidar_to_camera,
            K,
            width,
            height
        )
    )

    depth = np.zeros(
        (height, width),
        dtype=np.float32
    )

    if len(u) == 0:
        return depth

    # --------------------------------------------------------
    # Multiple LiDAR points can project
    # to the same pixel.
    #
    # Keep the nearest one.
    # --------------------------------------------------------

    pixel_ids = (
        v.astype(np.int64)
        * width
        + u.astype(np.int64)
    )

    order = np.lexsort(
        (
            depth_values,
            pixel_ids
        )
    )

    sorted_pixels = (
        pixel_ids[order]
    )

    keep = np.ones(
        len(order),
        dtype=bool
    )

    if len(order) > 1:

        keep[1:] = (
            sorted_pixels[1:]
            !=
            sorted_pixels[:-1]
        )

    selected = order[
        keep
    ]

    depth[
        v[selected],
        u[selected]
    ] = (
        depth_values[selected]
        .astype(np.float32)
    )

    return depth


# ============================================================
# SAVE LiDAR PROJECTION IMAGE
# ============================================================

def save_lidar_projected_image(
    image,
    lidar_points,
    camera,
    K,
    width,
    height,
    output_path
):

    # Project LiDAR.
    u, v, depth_values = (
        project_lidar_to_image(
            lidar_points,
            LIDAR_TO_CAMERA[camera],
            K,
            width,
            height
        )
    )

    # Start from the ORIGINAL image.
    overlay = Image.fromarray(
        image.copy()
    ).convert("RGB")

    draw = ImageDraw.Draw(
        overlay
    )

    if len(u) == 0:

        overlay.save(
            output_path
        )

        return 0

    # --------------------------------------------------------
    # Depth colouring
    #
    # Near = one end of the colour scale
    # Far  = the other end
    # --------------------------------------------------------

    depth_min = float(
        np.min(depth_values)
    )

    depth_max = float(
        np.max(depth_values)
    )

    if depth_max <= depth_min:
        normalized = np.zeros_like(
            depth_values
        )

    else:
        normalized = (
            depth_values
            - depth_min
        ) / (
            depth_max
            - depth_min
        )

    # --------------------------------------------------------
    # Draw 3x3 LiDAR points.
    #
    # Using a small square makes sparse
    # LiDAR easier to see.
    # --------------------------------------------------------

    for px, py, d in zip(
        u,
        v,
        normalized
    ):

        # Simple blue -> red depth mapping.
        r = int(
            255 * d
        )

        b = int(
            255 * (1.0 - d)
        )

        g = 0

        x0 = max(
            0,
            int(px) - 1
        )

        y0 = max(
            0,
            int(py) - 1
        )

        x1 = min(
            width - 1,
            int(px) + 1
        )

        y1 = min(
            height - 1,
            int(py) + 1
        )

        draw.rectangle(
            [
                x0,
                y0,
                x1,
                y1
            ],
            fill=(
                r,
                g,
                b
            )
        )

    overlay.save(
        output_path
    )

    return len(u)


# ============================================================
# SAVE LiDAR DEPTH MAP
# ============================================================

def save_depth_visualization(
    depth,
    output_path
):

    valid = depth > 0

    visualization = np.zeros(
        (*depth.shape, 3),
        dtype=np.uint8
    )

    if not np.any(valid):

        Image.fromarray(
            visualization
        ).save(
            output_path
        )

        return

    values = depth[
        valid
    ]

    d_min = values.min()
    d_max = values.max()

    if d_max > d_min:

        normalized = (
            depth - d_min
        ) / (
            d_max - d_min
        )

    else:

        normalized = np.zeros_like(
            depth
        )

    normalized = np.clip(
        normalized,
        0,
        1
    )

    visualization[..., 0] = (
        normalized * 255
    ).astype(
        np.uint8
    )

    visualization[..., 2] = (
        (1.0 - normalized)
        * 255
    ).astype(
        np.uint8
    )

    visualization[
        ~valid
    ] = 0

    Image.fromarray(
        visualization
    ).save(
        output_path
    )


# ============================================================
# MAPANYTHING
# ============================================================

def run_mapanything(
    model,
    images,
    calibration,
    lidar_points
):

    views = []

    depth_stats = {}

    for camera in CAMERAS:

        image = images[
            camera
        ]

        K = calibration[
            camera
        ]["K"]

        width = calibration[
            camera
        ]["width"]

        height = calibration[
            camera
        ]["height"]

        # ----------------------------------------------------
        # Create sparse LiDAR depth map.
        # ----------------------------------------------------

        depth = lidar_to_camera_depth(
            lidar_points,
            LIDAR_TO_CAMERA[camera],
            K,
            width,
            height
        )

        depth_stats[camera] = int(
            (depth > 0).sum()
        )

        # Save depth map for inspection.
        save_depth_visualization(
            depth,
            OUTPUT_DIR /
            f"{camera}_lidar_depth.png"
        )

        views.append({

            "img": image,

            "intrinsics":
                K.astype(
                    np.float32
                ),

            "depth_z":
                torch.from_numpy(
                    depth
                ).float(),

            "is_metric_scale":
                torch.tensor(
                    [True],
                    device=DEVICE
                ),
        })

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    processed_views = (
        preprocess_inputs(
            views
        )
    )

    # --------------------------------------------------------
    # MapAnything inference
    # --------------------------------------------------------

    with torch.no_grad():

        predictions = model.infer(

            processed_views,

            memory_efficient_inference=True,

            minibatch_size=1,

            use_amp=True,

            amp_dtype="bf16",

            apply_mask=True,

            mask_edges=True,

            apply_confidence_mask=False,

            confidence_percentile=10,

            use_multiview_confidence=True,

            ignore_calibration_inputs=False,

            ignore_depth_inputs=False,

            ignore_pose_inputs=True,

            ignore_depth_scale_inputs=False,

            ignore_pose_scale_inputs=True,
        )

    return (
        predictions,
        depth_stats
    )


# ============================================================
# SAVE MERGED MAPANYTHING CLOUD
# ============================================================

def save_merged_cloud(
    path,
    predictions,
    images
):

    all_points = []
    all_colors = []

    for camera_index, camera in enumerate(
        CAMERAS
    ):

        prediction = predictions[
            camera_index
        ]

        points_camera = (
            prediction[
                "pts3d_cam"
            ]
            .detach()
            .cpu()
            .numpy()
        )

        if points_camera.ndim == 4:
            points_camera = (
                points_camera[0]
            )

        mask = (
            prediction[
                "mask"
            ]
            .detach()
            .cpu()
            .numpy()
        )

        if mask.ndim == 4:
            mask = mask[0]

        if mask.ndim == 3:
            mask = mask[..., 0]

        image = images[
            camera
        ]

        H, W = (
            points_camera.shape[:2]
        )

        resized_image = np.asarray(
            Image.fromarray(
                image
            ).resize(
                (W, H)
            )
        )

        valid = (
            mask.astype(bool)
            &
            np.isfinite(
                points_camera
            ).all(axis=-1)
        )

        points_camera_valid = (
            points_camera[
                valid
            ]
        )

        colors_valid = (
            resized_image[
                valid
            ]
        )

        # ----------------------------------------------------
        # Camera -> LiDAR
        # ----------------------------------------------------

        T_camera_to_lidar = (
            invert_transform(
                LIDAR_TO_CAMERA[
                    camera
                ]
            )
        )

        points_lidar = transform_points(
            points_camera_valid,
            T_camera_to_lidar
        )

        all_points.append(
            points_lidar
        )

        all_colors.append(
            colors_valid
        )

    if not all_points:

        raise RuntimeError(
            "No MapAnything points were produced."
        )

    all_points = np.concatenate(
        all_points,
        axis=0
    )

    all_colors = np.concatenate(
        all_colors,
        axis=0
    )

    # --------------------------------------------------------
    # Write PLY
    # --------------------------------------------------------

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
            f"element vertex "
            f"{len(all_points)}\n"
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

        for point, color in zip(
            all_points,
            all_colors
        ):

            f.write(
                f"{point[0]:.6f} "
                f"{point[1]:.6f} "
                f"{point[2]:.6f} "
                f"{int(color[0])} "
                f"{int(color[1])} "
                f"{int(color[2])}\n"
            )

    return len(
        all_points
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)

    print(
        "MAPANYTHING + LiDAR "
        "SINGLE FRAME"
    )

    print("=" * 80)

    print(
        f"\nDevice: {DEVICE}"
    )

    if DEVICE.type == "cuda":

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # 1. LOAD CAMERA DATA
    # ========================================================

    calibration = {}
    camera_images = {}

    print(
        "\n1. Loading camera data..."
    )

    for camera in CAMERAS:

        (
            K,
            width,
            height
        ) = load_rectified_intrinsics(
            camera
        )

        camera_images[
            camera
        ] = load_images(
            camera
        )

        calibration[
            camera
        ] = {

            "K": K,

            "width":
                width,

            "height":
                height,
        }

        print(
            f"  {camera}: "
            f"{width} x {height}, "
            f"{len(camera_images[camera])} images"
        )

    # ========================================================
    # 2. SELECT REFERENCE FRAME
    # ========================================================

    reference_records = (
        camera_images[
            REFERENCE_CAMERA
        ]
    )

    if not (
        0 <= REFERENCE_INDEX
        < len(reference_records)
    ):

        raise RuntimeError(
            "REFERENCE_INDEX must be between "
            f"0 and "
            f"{len(reference_records) - 1}"
        )

    reference_timestamp = int(
        reference_records[
            REFERENCE_INDEX
        ][0]
    )

    print(
        f"\n2. Reference: "
        f"{REFERENCE_CAMERA} "
        f"index {REFERENCE_INDEX}"
    )

    print(
        f"   Timestamp: "
        f"{reference_timestamp}"
    )

    # ========================================================
    # 3. SYNCHRONIZE CAMERAS
    # ========================================================

    selected_images = {}

    print(
        "\n3. Synchronizing cameras..."
    )

    for camera in CAMERAS:

        (
            timestamp,
            path,
            delta
        ) = nearest_timestamp(
            reference_timestamp,
            camera_images[
                camera
            ]
        )

        if path is None:

            raise RuntimeError(
                f"No image found "
                f"for {camera}"
            )

        selected_images[
            camera
        ] = path

        print(
            f"  {camera}: "
            f"{delta / 1e6:.3f} ms, "
            f"{path.name}"
        )

    # ========================================================
    # 4. SYNCHRONIZE LiDAR
    # ========================================================

    print(
        "\n4. Synchronizing LiDAR..."
    )

    lidar_records = (
        load_lidar_files()
    )

    (
        lidar_timestamp,
        lidar_path,
        lidar_delta
    ) = nearest_timestamp(
        reference_timestamp,
        lidar_records
    )

    if lidar_path is None:

        raise RuntimeError(
            "No LiDAR scan found."
        )

    print(
        f"  LiDAR: "
        f"{lidar_path.name}"
    )

    print(
        f"  Timestamp difference: "
        f"{lidar_delta / 1e6:.3f} ms"
    )

    if lidar_delta > (
        SYNC_TOLERANCE_NS
    ):

        print(
            "  WARNING: LiDAR is outside "
            "the 50 ms synchronization "
            "tolerance."
        )

    # ========================================================
    # 5. LOAD LiDAR
    # ========================================================

    print(
        "\n5. Loading LiDAR..."
    )

    lidar_points = load_lidar_ply(
        lidar_path
    )

    print(
        f"  Raw LiDAR points: "
        f"{len(lidar_points)}"
    )

    lidar_ranges = np.linalg.norm(
        lidar_points[:, :3],
        axis=1
    )

    range_mask = (
        lidar_ranges
        <= LIDAR_MAX_RANGE
    )

    print(
        f"  LiDAR points <= "
        f"{LIDAR_MAX_RANGE:.1f} m: "
        f"{int(range_mask.sum())}"
    )

    # ========================================================
    # 6. LOAD AND SAVE ORIGINAL IMAGES
    # ========================================================

    print(
        "\n6. Loading and saving "
        "camera images..."
    )

    images = {}

    for camera in CAMERAS:

        image = np.asarray(
            Image.open(
                selected_images[
                    camera
                ]
            ).convert(
                "RGB"
            )
        )

        expected = (
            calibration[
                camera
            ]["height"],

            calibration[
                camera
            ]["width"],
        )

        if image.shape[:2] != expected:

            raise RuntimeError(
                f"{camera}: image shape "
                f"{image.shape[:2]} "
                f"does not match "
                f"camera_info.csv "
                f"{expected}"
            )

        images[
            camera
        ] = image

        # ----------------------------------------------------
        # SAVE ORIGINAL IMAGE
        # ----------------------------------------------------

        raw_output = (
            OUTPUT_DIR
            / f"{camera}.png"
        )

        Image.fromarray(
            image
        ).save(
            raw_output
        )

        print(
            f"  {camera} original: "
            f"{raw_output}"
        )

    # ========================================================
    # 7. SAVE LiDAR PROJECTIONS
    # ========================================================

    print(
        "\n7. Saving LiDAR projections..."
    )

    for camera in CAMERAS:

        projected_output = (
            OUTPUT_DIR
            / f"{camera}_lidar_projected.png"
        )

        projected_count = (
            save_lidar_projected_image(
                image=images[camera],

                lidar_points=lidar_points,

                camera=camera,

                K=calibration[
                    camera
                ]["K"],

                width=calibration[
                    camera
                ]["width"],

                height=calibration[
                    camera
                ]["height"],

                output_path=projected_output
            )
        )

        print(
            f"  {camera}: "
            f"{projected_count} "
            f"projected LiDAR points"
        )

        print(
            f"    Saved: "
            f"{projected_output}"
        )

    # ========================================================
    # 8. LOAD MAPANYTHING
    # ========================================================

    print(
        "\n8. Loading MapAnything..."
    )

    model = (
        MapAnything
        .from_pretrained(
            "facebook/map-anything"
        )
        .to(DEVICE)
    )

    model.eval()

    print(
        "  MapAnything loaded."
    )

    # ========================================================
    # 9. RUN MAPANYTHING
    # ========================================================

    print(
        "\n9. Running MapAnything..."
    )

    print(
        f"  LiDAR maximum range: "
        f"{LIDAR_MAX_RANGE:.1f} m"
    )

    print(
        f"  Minimum camera depth: "
        f"{LIDAR_MIN_CAMERA_DEPTH:.2f} m"
    )

    print(
        f"  Border margin: "
        f"{BORDER_MARGIN_PIXELS} px"
    )

    start_time = time.perf_counter()

    predictions, depth_stats = (
        run_mapanything(
            model=model,

            images=images,

            calibration=calibration,

            lidar_points=lidar_points
        )
    )

    if DEVICE.type == "cuda":

        torch.cuda.synchronize()

    elapsed = (
        time.perf_counter()
        - start_time
    )

    print(
        f"\n  Depth pixels:"
    )

    for camera in CAMERAS:

        print(
            f"    {camera}: "
            f"{depth_stats[camera]}"
        )

    print(
        f"  Runtime: "
        f"{elapsed:.3f} s"
    )

    # ========================================================
    # 10. SAVE MERGED MAPANYTHING CLOUD
    # ========================================================

    print(
        "\n10. Saving merged MapAnything cloud..."
    )

    output_ply = (
        OUTPUT_DIR
        / "merged_mapanything.ply"
    )

    point_count = (
        save_merged_cloud(
            output_ply,
            predictions,
            images
        )
    )

    print(
        f"  MapAnything points: "
        f"{point_count}"
    )

    print(
        f"  Saved: "
        f"{output_ply}"
    )

    # ========================================================
    # 11. CLEANUP
    # ========================================================

    del predictions

    if DEVICE.type == "cuda":

        torch.cuda.empty_cache()

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
        f"\nOutput directory:"
    )

    print(
        OUTPUT_DIR
    )

    print(
        "\nFiles:"
    )

    for path in sorted(
        OUTPUT_DIR.iterdir()
    ):

        print(
            f"  {path.name}"
        )


if __name__ == "__main__":
    main()