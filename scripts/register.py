import csv
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from mapanything.models import MapAnything
from mapanything.utils.image import preprocess_inputs


# ============================================================
# CONFIG
# ============================================================

BAG_ROOT = Path(
    "/home/lcha0115/bt60_scratch/lcha_data/"
    "map-anything/personal_data/"
    "camera_lidar_imu_test_04"
)

EXTRACTED = BAG_ROOT / "extracted_new"

OUTPUT_DIR = (
    BAG_ROOT /
    "mapanything_lidar_registration"
)

CAMERAS = [
    "CAM1",
    "CAM2",
    "CAM6",
]

# ------------------------------------------------------------
# Reference camera / frame
# ------------------------------------------------------------

REFERENCE_CAMERA = "CAM2"

REFERENCE_INDEX = 0

# ------------------------------------------------------------
# Synchronisation
# ------------------------------------------------------------

SYNC_TOLERANCE_NS = 50_000_000  # 50 ms

# ------------------------------------------------------------
# Correspondence sampling
# ------------------------------------------------------------

MAX_CORRESPONDENCES_PER_CAMERA = 500000

# ------------------------------------------------------------
# RANSAC
# ------------------------------------------------------------

RANSAC_ITERATIONS = 20000

RANSAC_INLIER_THRESHOLD = 1.0

MIN_INLIERS = 5000

# ------------------------------------------------------------
# Refinement
# ------------------------------------------------------------

REFINEMENT_THRESHOLDS = [
    1.0,
    0.75,
    0.50,
    0.35,
    0.25,
    0.15,
]

# ------------------------------------------------------------
# LiDAR projection
# ------------------------------------------------------------

LIDAR_MIN_CAMERA_DEPTH = 0.05

# ------------------------------------------------------------
# Random seed
# ------------------------------------------------------------

RNG_SEED = 42


# ============================================================
# DEVICE
# ============================================================

device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)


# ============================================================
# EXTRINSICS
# ============================================================
#
# Your calibration file says:
#
#     velodyne -> cam*_optical_frame
#
# Empirical test showed:
#
#     inverse(TF)
#
# gives the correct LiDAR -> image projection.
#
# Therefore:
#
#     T_lidar_to_camera = inverse(TF)
#
# and:
#
#     T_camera_to_lidar = TF
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
    )

    return transformed[
        :, :3
    ].reshape(
        points.shape
    )


# ============================================================
# ARRAY PARSER
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

    value = value.replace(
        ")",
        ""
    )

    value = value.replace(
        "[",
        ""
    )

    value = value.replace(
        "]",
        ""
    )

    value = value.replace(
        ",",
        " "
    )

    values = np.fromstring(
        value,
        sep=" ",
        dtype=np.float64
    )

    if values.size == 0:

        raise ValueError(
            f"Could not parse array:\n{value}"
        )

    return values


# ============================================================
# CAMERA INFO
# ============================================================

def load_rectified_intrinsics(camera):

    path = (
        EXTRACTED /
        camera /
        "camera_info.csv"
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
            f"No CameraInfo rows found:\n{path}"
        )

    row = rows[0]

    P = parse_array(
        row["P"]
    )

    if P.size != 12:

        raise ValueError(
            f"{camera}: P has "
            f"{P.size} values instead of 12."
        )

    P = P.reshape(
        3,
        4
    )

    K_rect = P[:, :3]

    width = int(
        row["width"]
    )

    height = int(
        row["height"]
    )

    return (
        K_rect,
        width,
        height
    )


# ============================================================
# TIMESTAMPED CAMERA IMAGES
# ============================================================

def load_images(camera):

    directory = (
        EXTRACTED /
        camera /
        "images_rect"
    )

    files = []

    for path in directory.glob(
        "*.png"
    ):

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
            f"No rectified images found:\n"
            f"{directory}"
        )

    return files


# ============================================================
# TIMESTAMPED LIDAR
# ============================================================

def load_lidar_files():

    directory = (
        EXTRACTED /
        "lidar"
    )

    files = []

    for path in directory.glob(
        "*.ply"
    ):

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
            f"No LiDAR PLY files found:\n"
            f"{directory}"
        )

    return files


# ============================================================
# ROBUST NEAREST TIMESTAMP
# ============================================================

def nearest_timestamp(
    target_timestamp,
    records
):

    if not records:

        return (
            None,
            None,
            None
        )

    timestamps = np.asarray(
        [
            record[0]
            for record in records
        ],
        dtype=np.int64
    )

    insert_index = int(
        np.searchsorted(
            timestamps,
            target_timestamp
        )
    )

    candidate_indices = []

    if insert_index > 0:

        candidate_indices.append(
            insert_index - 1
        )

    if insert_index < len(records):

        candidate_indices.append(
            insert_index
        )

    if not candidate_indices:

        return (
            None,
            None,
            None
        )

    best_index = min(
        candidate_indices,
        key=lambda idx:
        abs(
            int(
                records[idx][0]
            )
            -
            int(
                target_timestamp
            )
        )
    )

    timestamp = int(
        records[best_index][0]
    )

    path = records[
        best_index
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
# LOAD LIDAR PLY
# ============================================================

def load_lidar_ply(path):

    header = []

    with open(
        path,
        "r"
    ) as f:

        while True:

            line = f.readline()

            if not line:

                raise RuntimeError(
                    f"Unexpected end of PLY:\n{path}"
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
            f"Could not determine vertex count:\n{path}"
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

    if data.shape[1] < 3:

        raise RuntimeError(
            f"LiDAR PLY has fewer than XYZ columns:\n{path}"
        )

    points = data[
        :,
        :3
    ].astype(
        np.float64
    )

    return points


# ============================================================
# PROJECT LIDAR TO IMAGE
# ============================================================

def project_lidar_to_image(
    lidar_points,
    T_lidar_to_camera,
    K_rect,
    width,
    height
):

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

    valid_front = (
        Z >
        LIDAR_MIN_CAMERA_DEPTH
    )

    original_indices = np.arange(
        len(lidar_points)
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

    original_indices = (
        original_indices[
            valid_front
        ]
    )

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

    valid_image = (
        (u >= 0)
        &
        (u < width)
        &
        (v >= 0)
        &
        (v < height)
    )

    u = u[
        valid_image
    ]

    v = v[
        valid_image
    ]

    Z = Z[
        valid_image
    ]

    original_indices = (
        original_indices[
            valid_image
        ]
    )

    return (
        u,
        v,
        Z,
        original_indices
    )


# ============================================================
# LIDAR Z-BUFFER
# ============================================================

def lidar_zbuffer(
    u,
    v,
    depth,
    lidar_indices,
    width
):

    if len(u) == 0:

        return (
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.int64)
        )

    u_int = np.rint(
        u
    ).astype(
        np.int32
    )

    v_int = np.rint(
        v
    ).astype(
        np.int32
    )

    pixel_id = (
        v_int.astype(
            np.int64
        )
        *
        int(width)
        +
        u_int.astype(
            np.int64
        )
    )

    order = np.lexsort(
        (
            depth,
            pixel_id
        )
    )

    sorted_pixels = (
        pixel_id[
            order
        ]
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

    return (
        u_int[selected],
        v_int[selected],
        depth[selected],
        lidar_indices[selected]
    )


# ============================================================
# MAPANYTHING OUTPUT EXTRACTION
# ============================================================

def get_mapanything_grid(
    prediction
):

    points = (
        prediction["pts3d_cam"]
        .detach()
        .cpu()
        .numpy()
    )

    if points.ndim == 4:

        points = points[0]

    if (
        points.ndim != 3
        or
        points.shape[-1] != 3
    ):

        raise RuntimeError(
            f"Unexpected pts3d_cam shape: "
            f"{points.shape}"
        )

    mask = (
        prediction["mask"]
        .detach()
        .cpu()
        .numpy()
    )

    if mask.ndim == 4:

        mask = mask[0]

    if mask.ndim == 3:

        mask = mask[..., 0]

    mask = mask.astype(
        bool
    )

    return (
        points,
        mask
    )


# ============================================================
# IMAGE PIXEL -> MAPANYTHING GRID
# ============================================================

def image_to_mapanything_grid(
    u,
    v,
    image_width,
    image_height,
    map_width,
    map_height
):

    u_ma = (
        u
        *
        float(map_width)
        /
        float(image_width)
    )

    v_ma = (
        v
        *
        float(map_height)
        /
        float(image_height)
    )

    u_ma = np.rint(
        u_ma
    ).astype(
        np.int32
    )

    v_ma = np.rint(
        v_ma
    ).astype(
        np.int32
    )

    valid = (
        (u_ma >= 0)
        &
        (u_ma < map_width)
        &
        (v_ma >= 0)
        &
        (v_ma < map_height)
    )

    return (
        u_ma,
        v_ma,
        valid
    )


# ============================================================
# BUILD CORRESPONDENCES
# ============================================================

def build_correspondences(
    camera,
    prediction,
    lidar_points,
    K_rect,
    image_width,
    image_height,
    T_lidar_to_camera,
    T_camera_to_lidar
):

    (
        points_grid,
        map_mask
    ) = get_mapanything_grid(
        prediction
    )

    H_ma, W_ma = (
        points_grid.shape[:2]
    )

    print(
        f"\n{camera}"
    )

    print(
        "  MapAnything grid:",
        W_ma,
        "x",
        H_ma
    )

    (
        u_img,
        v_img,
        depth,
        lidar_indices
    ) = project_lidar_to_image(
        lidar_points=lidar_points,
        T_lidar_to_camera=T_lidar_to_camera,
        K_rect=K_rect,
        width=image_width,
        height=image_height
    )

    print(
        "  Projected LiDAR points:",
        len(u_img)
    )

    (
        u_img,
        v_img,
        depth,
        lidar_indices
    ) = lidar_zbuffer(
        u=u_img,
        v=v_img,
        depth=depth,
        lidar_indices=lidar_indices,
        width=image_width
    )

    print(
        "  After LiDAR z-buffer:",
        len(u_img)
    )

    if len(u_img) == 0:

        return (
            np.empty(
                (0, 3),
                dtype=np.float64
            ),
            np.empty(
                (0, 3),
                dtype=np.float64
            )
        )

    (
        u_ma,
        v_ma,
        valid_grid
    ) = image_to_mapanything_grid(
        u=u_img,
        v=v_img,
        image_width=image_width,
        image_height=image_height,
        map_width=W_ma,
        map_height=H_ma
    )

    u_ma = u_ma[
        valid_grid
    ]

    v_ma = v_ma[
        valid_grid
    ]

    lidar_indices = lidar_indices[
        valid_grid
    ]

    ma_cam = (
        points_grid[
            v_ma,
            u_ma
        ]
    )

    valid = (
        map_mask[
            v_ma,
            u_ma
        ]
        &
        np.isfinite(
            ma_cam
        ).all(
            axis=1
        )
    )

    ma_cam = (
        ma_cam[
            valid
        ]
    )

    lidar_indices = (
        lidar_indices[
            valid
        ]
    )

    lidar_corr = (
        lidar_points[
            lidar_indices
        ]
    )

    print(
        "  Valid correspondences:",
        len(
            lidar_corr
        )
    )

    if len(lidar_corr) == 0:

        return (
            np.empty(
                (0, 3),
                dtype=np.float64
            ),
            np.empty(
                (0, 3),
                dtype=np.float64
            )
        )

    ma_lidar_frame = (
        transform_points(
            ma_cam,
            T_camera_to_lidar
        )
    )

    if len(
        lidar_corr
    ) > MAX_CORRESPONDENCES_PER_CAMERA:

        rng = np.random.default_rng(
            RNG_SEED
        )

        keep = rng.choice(
            len(lidar_corr),
            size=MAX_CORRESPONDENCES_PER_CAMERA,
            replace=False
        )

        ma_lidar_frame = (
            ma_lidar_frame[
                keep
            ]
        )

        lidar_corr = (
            lidar_corr[
                keep
            ]
        )

    print(
        "  Final correspondences:",
        len(
            lidar_corr
        )
    )

    return (
        ma_lidar_frame,
        lidar_corr
    )


# ============================================================
# UMeyAMA SIMILARITY TRANSFORM
# ============================================================

def estimate_similarity(
    src,
    dst
):

    src = np.asarray(
        src,
        dtype=np.float64
    )

    dst = np.asarray(
        dst,
        dtype=np.float64
    )

    if (
        len(src) != len(dst)
        or
        len(src) < 3
    ):

        raise ValueError(
            "Need at least 3 paired points."
        )

    src_mean = (
        src.mean(
            axis=0
        )
    )

    dst_mean = (
        dst.mean(
            axis=0
        )
    )

    src_centered = (
        src -
        src_mean
    )

    dst_centered = (
        dst -
        dst_mean
    )

    covariance = (
        dst_centered.T
        @
        src_centered
        /
        len(src)
    )

    U, singular_values, Vt = (
        np.linalg.svd(
            covariance
        )
    )

    D = np.eye(
        3,
        dtype=np.float64
    )

    if np.linalg.det(
        U @ Vt
    ) < 0:

        D[
            2,
            2
        ] = -1

    R = (
        U
        @
        D
        @
        Vt
    )

    src_variance = (
        np.sum(
            src_centered ** 2
        )
        /
        len(src)
    )

    if src_variance <= 1e-12:

        raise ValueError(
            "Degenerate MapAnything point geometry."
        )

    scale = (
        np.sum(
            singular_values *
            np.diag(D)
        )
        /
        src_variance
    )

    if (
        not np.isfinite(scale)
        or
        scale <= 0
    ):

        raise ValueError(
            f"Invalid scale: {scale}"
        )

    t = (
        dst_mean
        -
        scale *
        (
            R @ src_mean
        )
    )

    T = np.eye(
        4,
        dtype=np.float64
    )

    T[:3, :3] = (
        scale *
        R
    )

    T[:3, 3] = t

    return (
        T,
        scale,
        R,
        t
    )


# ============================================================
# APPLY TRANSFORM
# ============================================================

def apply_transform(
    points,
    T
):

    return transform_points(
        points,
        T
    )


# ============================================================
# RANSAC SIMILARITY
# ============================================================

def ransac_similarity(
    src,
    dst,
    iterations,
    threshold
):

    n = len(src)

    if n < 4:

        raise RuntimeError(
            "Need at least 4 correspondences "
            "for RANSAC."
        )

    rng = np.random.default_rng(
        RNG_SEED
    )

    best_count = 0
    best_rmse = np.inf
    best_T = None
    best_inliers = None

    for _ in range(
        iterations
    ):

        sample = rng.choice(
            n,
            size=4,
            replace=False
        )

        src_sample = src[
            sample
        ]

        dst_sample = dst[
            sample
        ]

        src_rank = np.linalg.matrix_rank(
            src_sample -
            src_sample.mean(
                axis=0
            )
        )

        dst_rank = np.linalg.matrix_rank(
            dst_sample -
            dst_sample.mean(
                axis=0
            )
        )

        if (
            src_rank < 2
            or
            dst_rank < 2
        ):

            continue

        try:

            (
                T,
                scale,
                R,
                t
            ) = estimate_similarity(
                src_sample,
                dst_sample
            )

        except ValueError:

            continue

        if (
            scale < 0.01
            or
            scale > 100.0
        ):

            continue

        predicted = (
            apply_transform(
                src,
                T
            )
        )

        residuals = np.linalg.norm(
            predicted -
            dst,
            axis=1
        )

        inliers = (
            residuals <
            threshold
        )

        count = int(
            inliers.sum()
        )

        if count == 0:

            continue

        rmse = np.sqrt(
            np.mean(
                residuals[
                    inliers
                ] ** 2
            )
        )

        if (
            count > best_count
            or
            (
                count == best_count
                and
                rmse < best_rmse
            )
        ):

            best_count = count
            best_rmse = rmse
            best_T = T
            best_inliers = inliers

    if best_T is None:

        raise RuntimeError(
            "RANSAC could not find "
            "a valid transformation."
        )

    (
        best_T,
        best_scale,
        best_R,
        best_t
    ) = estimate_similarity(
        src[
            best_inliers
        ],
        dst[
            best_inliers
        ]
    )

    return (
        best_T,
        best_scale,
        best_R,
        best_t,
        best_inliers
    )


# ============================================================
# METRIC HELPERS
# ============================================================

def residuals_for_transform(
    src,
    dst,
    T
):

    transformed = (
        apply_transform(
            src,
            T
        )
    )

    residuals = np.linalg.norm(
        transformed -
        dst,
        axis=1
    )

    return residuals


def summarize_residuals(
    residuals
):

    if len(residuals) == 0:

        return {
            "count": 0,
            "rmse": np.nan,
            "mean": np.nan,
            "median": np.nan,
            "p90": np.nan,
            "p95": np.nan,
            "max": np.nan,
        }

    residuals = np.asarray(
        residuals,
        dtype=np.float64
    )

    return {
        "count":
            int(len(residuals)),

        "rmse":
            float(
                np.sqrt(
                    np.mean(
                        residuals ** 2
                    )
                )
            ),

        "mean":
            float(
                np.mean(
                    residuals
                )
            ),

        "median":
            float(
                np.median(
                    residuals
                )
            ),

        "p90":
            float(
                np.percentile(
                    residuals,
                    90
                )
            ),

        "p95":
            float(
                np.percentile(
                    residuals,
                    95
                )
            ),

        "max":
            float(
                np.max(
                    residuals
                )
            ),
    }


def print_metrics(
    title,
    residuals
):

    metrics = summarize_residuals(
        residuals
    )

    print(
        f"\n{title}"
    )

    print(
        f"  Count:   {metrics['count']}"
    )

    print(
        f"  RMSE:    {metrics['rmse']:.4f} m"
    )

    print(
        f"  Mean:    {metrics['mean']:.4f} m"
    )

    print(
        f"  Median:  {metrics['median']:.4f} m"
    )

    print(
        f"  P90:     {metrics['p90']:.4f} m"
    )

    print(
        f"  P95:     {metrics['p95']:.4f} m"
    )

    print(
        f"  Max:     {metrics['max']:.4f} m"
    )

    return metrics


# ============================================================
# ITERATIVE REFINEMENT
# ============================================================
#
# IMPORTANT:
#
# This now returns:
#
#   final transform
#   actual final inlier mask
#   actual final threshold
#
# Therefore the final reporting uses the SAME population
# that the final refinement stage used.
#
# ============================================================

def refine_similarity(
    src,
    dst,
    T_initial
):

    T = T_initial.copy()

    last_successful_threshold = None

    last_successful_inliers = None

    for threshold in REFINEMENT_THRESHOLDS:

        residuals = residuals_for_transform(
            src,
            dst,
            T
        )

        inliers = (
            residuals <
            threshold
        )

        count = int(
            inliers.sum()
        )

        print(
            f"\nRefinement threshold: "
            f"{threshold:.3f} m"
        )

        print(
            "  Inliers:",
            count,
            "/",
            len(
                residuals
            )
        )

        if count < MIN_INLIERS:

            print(
                "  Too few inliers; "
                "stopping refinement."
            )

            break

        (
            T_new,
            scale,
            R,
            t
        ) = estimate_similarity(
            src[
                inliers
            ],
            dst[
                inliers
            ]
        )

        T = T_new

        new_residuals = residuals_for_transform(
            src,
            dst,
            T
        )

        new_inliers = (
            new_residuals <
            threshold
        )

        inlier_residuals = (
            new_residuals[
                new_inliers
            ]
        )

        inlier_rmse = np.sqrt(
            np.mean(
                inlier_residuals ** 2
            )
        )

        print(
            "  New inliers:",
            int(
                new_inliers.sum()
            )
        )

        print(
            "  Inlier RMSE:",
            f"{inlier_rmse:.4f} m"
        )

        last_successful_threshold = (
            threshold
        )

        last_successful_inliers = (
            new_inliers.copy()
        )

    # --------------------------------------------------------
    # Fallback if no refinement threshold succeeded.
    # --------------------------------------------------------

    if (
        last_successful_inliers is None
    ):

        final_residuals = residuals_for_transform(
            src,
            dst,
            T
        )

        last_successful_inliers = (
            final_residuals <
            RANSAC_INLIER_THRESHOLD
        )

        last_successful_threshold = (
            RANSAC_INLIER_THRESHOLD
        )

    return (
        T,
        last_successful_inliers,
        last_successful_threshold
    )


# ============================================================
# PLY WRITER
# ============================================================

def save_xyz_ply(
    path,
    points,
    colors=None
):

    points = np.asarray(
        points,
        dtype=np.float32
    )

    if colors is not None:

        colors = np.asarray(
            colors,
            dtype=np.uint8
        )

        if len(colors) != len(points):

            raise ValueError(
                "Colour count does not "
                "match point count."
            )

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

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

        if colors is not None:

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

        if colors is None:

            for p in points:

                f.write(
                    f"{p[0]:.6f} "
                    f"{p[1]:.6f} "
                    f"{p[2]:.6f}\n"
                )

        else:

            for p, c in zip(
                points,
                colors
            ):

                f.write(
                    f"{p[0]:.6f} "
                    f"{p[1]:.6f} "
                    f"{p[2]:.6f} "
                    f"{int(c[0])} "
                    f"{int(c[1])} "
                    f"{int(c[2])}\n"
                )


# ============================================================
# SAVE TRANSFORM
# ============================================================

def save_transform(
    path,
    T,
    scale,
    R,
    t,
    final_threshold,
    all_metrics,
    inlier_metrics
):

    with open(
        path,
        "w"
    ) as f:

        f.write(
            "MapAnything -> LiDAR similarity transform\n"
        )

        f.write(
            "p_lidar = scale * R * p_mapanything + t\n\n"
        )

        f.write(
            f"scale = {scale:.12f}\n\n"
        )

        f.write(
            "R =\n"
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
            "\nt =\n"
        )

        for x in t:

            f.write(
                f"{x:.12f}\n"
            )

        f.write(
            "\nT =\n"
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

        f.write(
            "\n"
        )

        f.write(
            f"final_inlier_threshold = "
            f"{final_threshold:.6f} m\n"
        )

        f.write(
            "\nALL CORRESPONDENCE METRICS\n"
        )

        for key, value in all_metrics.items():

            f.write(
                f"{key} = {value}\n"
            )

        f.write(
            "\nFINAL INLIER METRICS\n"
        )

        for key, value in inlier_metrics.items():

            f.write(
                f"{key} = {value}\n"
            )


# ============================================================
# MAIN
# ============================================================

print(
    "=" * 80
)

print(
    "MAPANYTHING -> LiDAR REGISTRATION"
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
    "\nDevice:",
    device
)

if device.type == "cuda":

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 1. LOAD CAMERA DATA
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "1. LOAD CAMERA DATA"
)

print(
    "=" * 80
)

camera_images = {}

camera_calibration = {}

for camera in CAMERAS:

    records = load_images(
        camera
    )

    camera_images[
        camera
    ] = records

    (
        K_rect,
        width,
        height
    ) = load_rectified_intrinsics(
        camera
    )

    camera_calibration[
        camera
    ] = {
        "K": K_rect,
        "width": width,
        "height": height,
    }

    print(
        f"\n{camera}"
    )

    print(
        "  images:",
        len(records)
    )

    print(
        "  resolution:",
        width,
        "x",
        height
    )

    print(
        "  K_rect:"
    )

    print(
        K_rect
    )


# ============================================================
# 2. REFERENCE TIMESTAMP
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "2. REFERENCE TIMESTAMP"
)

print(
    "=" * 80
)

reference_records = (
    camera_images[
        REFERENCE_CAMERA
    ]
)

if not reference_records:

    raise RuntimeError(
        f"No images for reference camera "
        f"{REFERENCE_CAMERA}."
    )

if (
    REFERENCE_INDEX < 0
    or
    REFERENCE_INDEX >= len(
        reference_records
    )
):

    raise RuntimeError(
        f"REFERENCE_INDEX={REFERENCE_INDEX} "
        f"but {REFERENCE_CAMERA} has "
        f"{len(reference_records)} images."
    )

reference_timestamp = int(
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
    "Reference camera:",
    REFERENCE_CAMERA
)

print(
    "Reference index:",
    REFERENCE_INDEX
)

print(
    "Reference timestamp:",
    reference_timestamp
)

print(
    "Reference image:",
    reference_path
)


# ============================================================
# 3. SELECT SYNCHRONIZED CAMERA IMAGES
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "3. SYNCHRONIZED CAMERA FRAMES"
)

print(
    "=" * 80
)

selected_images = {}

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
            f"Could not find image for {camera}."
        )

    selected_images[
        camera
    ] = {
        "timestamp":
            timestamp,

        "path":
            path,

        "delta":
            delta,
    }

    print(
        f"\n{camera}"
    )

    print(
        "  timestamp:",
        timestamp
    )

    print(
        "  delta:",
        f"{delta / 1e6:.3f} ms"
    )

    print(
        "  path:",
        path
    )

    if (
        delta >
        SYNC_TOLERANCE_NS
    ):

        print(
            "  WARNING: > 50 ms"
        )


# ============================================================
# 4. SELECT SYNCHRONIZED LIDAR
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "4. SYNCHRONIZED LiDAR"
)

print(
    "=" * 80
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
        "Could not find synchronized LiDAR scan."
    )

print(
    "LiDAR timestamp:",
    lidar_timestamp
)

print(
    "LiDAR delta:",
    f"{lidar_delta / 1e6:.3f} ms"
)

print(
    "LiDAR file:",
    lidar_path
)

if (
    lidar_delta >
    SYNC_TOLERANCE_NS
):

    print(
        "WARNING: LiDAR difference > 50 ms"
    )


# ============================================================
# 5. LOAD LIDAR
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "5. LOAD LiDAR"
)

print(
    "=" * 80
)

lidar_points = load_lidar_ply(
    lidar_path
)

print(
    "LiDAR points:",
    len(lidar_points)
)

print(
    "MIN:",
    lidar_points.min(
        axis=0
    )
)

print(
    "MAX:",
    lidar_points.max(
        axis=0
    )
)


# ============================================================
# 6. BUILD TRANSFORMS
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "6. CAMERA / LiDAR TRANSFORMS"
)

print(
    "=" * 80
)

T_lidar_to_camera = {}

T_camera_to_lidar = {}

for camera in CAMERAS:

    T_camera_to_lidar[
        camera
    ] = EXTRINSICS[
        camera
    ]

    T_lidar_to_camera[
        camera
    ] = invert_transform(
        EXTRINSICS[
            camera
        ]
    )

    print(
        f"\n{camera}"
    )

    print(
        "  LiDAR -> camera = inverse(TF)"
    )

    print(
        "  camera -> LiDAR = TF"
    )


# ============================================================
# 7. LOAD MAPANYTHING
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "7. MAPANYTHING"
)

print(
    "=" * 80
)

model = MapAnything.from_pretrained(
    "facebook/map-anything"
).to(
    device
)

model.eval()

print(
    "MapAnything loaded."
)


# ============================================================
# 8. BUILD VIEWS
# ============================================================

views = []

images = {}

for camera in CAMERAS:

    path = selected_images[
        camera
    ]["path"]

    image = np.asarray(
        Image.open(
            path
        ).convert(
            "RGB"
        )
    )

    images[
        camera
    ] = image

    K_rect = (
        camera_calibration[
            camera
        ]["K"]
    )

    views.append(
        {
            "img":
                image,

            "intrinsics":
                K_rect.astype(
                    np.float32
                ),
        }
    )

    print(
        f"\n{camera}:",
        image.shape
    )


# ============================================================
# 9. PREPROCESS + INFERENCE
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "8. MAPANYTHING INFERENCE"
)

print(
    "=" * 80
)

processed_views = (
    preprocess_inputs(
        views
    )
)

start = time.perf_counter()

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

    ignore_depth_inputs=True,

    ignore_pose_inputs=True,

    ignore_depth_scale_inputs=True,

    ignore_pose_scale_inputs=True,
)

if device.type == "cuda":

    torch.cuda.synchronize()

print(
    "Inference time:",
    f"{time.perf_counter() - start:.3f} s"
)


# ============================================================
# 10. BUILD CORRESPONDENCES
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "9. BUILD LiDAR <-> MAPANYTHING CORRESPONDENCES"
)

print(
    "=" * 80
)

all_ma = []

all_lidar = []

for i, camera in enumerate(
    CAMERAS
):

    cal = camera_calibration[
        camera
    ]

    ma_points, lidar_corr = (
        build_correspondences(
            camera=camera,

            prediction=predictions[i],

            lidar_points=lidar_points,

            K_rect=cal["K"],

            image_width=cal["width"],

            image_height=cal["height"],

            T_lidar_to_camera=
                T_lidar_to_camera[camera],

            T_camera_to_lidar=
                T_camera_to_lidar[camera],
        )
    )

    all_ma.append(
        ma_points
    )

    all_lidar.append(
        lidar_corr
    )


# ============================================================
# MERGE CORRESPONDENCES
# ============================================================

ma_corr = np.concatenate(
    all_ma,
    axis=0
)

lidar_corr = np.concatenate(
    all_lidar,
    axis=0
)

print(
    "\nTotal correspondences:",
    len(ma_corr)
)

if len(ma_corr) < MIN_INLIERS:

    raise RuntimeError(
        "Too few valid correspondences "
        "for registration."
    )


# ============================================================
# SAVE CORRESPONDENCE CLOUDS
# ============================================================

save_xyz_ply(
    OUTPUT_DIR /
    "correspondence_mapanything.ply",
    ma_corr
)

save_xyz_ply(
    OUTPUT_DIR /
    "correspondence_lidar.ply",
    lidar_corr
)


# ============================================================
# 11. RANSAC SIMILARITY
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "10. SIMILARITY REGISTRATION"
)

print(
    "=" * 80
)

print(
    "Solving:"
)

print(
    "    LiDAR = s * R * MapAnything + t"
)

(
    T_initial,
    scale_initial,
    R_initial,
    t_initial,
    inliers_initial
) = ransac_similarity(
    src=ma_corr,
    dst=lidar_corr,
    iterations=RANSAC_ITERATIONS,
    threshold=RANSAC_INLIER_THRESHOLD
)

initial_residuals = residuals_for_transform(
    ma_corr,
    lidar_corr,
    T_initial
)

initial_inlier_residuals = (
    initial_residuals[
        inliers_initial
    ]
)

print(
    "\nInitial scale:",
    scale_initial
)

print(
    "Initial inliers:",
    int(
        inliers_initial.sum()
    ),
    "/",
    len(
        inliers_initial
    )
)

print_metrics(
    "Initial ALL-correspondence metrics",
    initial_residuals
)

print_metrics(
    "Initial RANSAC-inlier metrics",
    initial_inlier_residuals
)


# ============================================================
# 12. REFINE
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "11. REFINE"
)

print(
    "=" * 80
)

(
    T_final,
    final_inliers,
    final_inlier_threshold
) = refine_similarity(
    ma_corr,
    lidar_corr,
    T_initial
)


# ============================================================
# FINAL RESIDUALS
# ============================================================

final_residuals = residuals_for_transform(
    ma_corr,
    lidar_corr,
    T_final
)

final_inlier_residuals = (
    final_residuals[
        final_inliers
    ]
)

final_outlier_residuals = (
    final_residuals[
        ~final_inliers
    ]
)


# ============================================================
# FINAL TRANSFORM PARAMETERS
# ============================================================

M = T_final[
    :3,
    :3
]

scale_final = np.cbrt(
    np.linalg.det(
        M
    )
)

R_final = (
    M /
    scale_final
)

t_final = (
    T_final[
        :3,
        3
    ]
)


# ============================================================
# FINAL METRICS
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "12. FINAL CALIBRATION"
)

print(
    "=" * 80
)

print(
    "\nScale:",
    scale_final
)

print(
    "\nRotation:"
)

print(
    R_final
)

print(
    "\nTranslation:"
)

print(
    t_final
)

print(
    "\nTransform:"
)

print(
    T_final
)

print(
    "\nFinal inlier threshold:",
    f"{final_inlier_threshold:.3f} m"
)

print(
    "\nTotal correspondences:",
    len(ma_corr)
)

print(
    "Final inliers:",
    int(
        final_inliers.sum()
    )
)

print(
    "Final outliers:",
    int(
        (~final_inliers).sum()
    )
)

print(
    "Final inlier ratio:",
    f"{100 * final_inliers.mean():.2f}%"
)

# ------------------------------------------------------------
# ALL points
# ------------------------------------------------------------

all_metrics = print_metrics(
    "FINAL ALL-CORRESPONDENCE METRICS",
    final_residuals
)

# ------------------------------------------------------------
# Final inliers
# ------------------------------------------------------------

inlier_metrics = print_metrics(
    "FINAL INLIER-ONLY METRICS",
    final_inlier_residuals
)

# ------------------------------------------------------------
# Outliers
# ------------------------------------------------------------

if len(
    final_outlier_residuals
) > 0:

    print_metrics(
        "FINAL OUTLIER-ONLY METRICS",
        final_outlier_residuals
    )


# ============================================================
# SAVE TRANSFORM
# ============================================================

transform_path = (
    OUTPUT_DIR /
    "mapanything_to_lidar_transform.txt"
)

save_transform(
    transform_path,
    T_final,
    scale_final,
    R_final,
    t_final,
    final_inlier_threshold,
    all_metrics,
    inlier_metrics
)

print(
    "\nSaved:"
)

print(
    transform_path
)


# ============================================================
# 13. BUILD FULL MAPANYTHING CLOUD
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "13. TRANSFORM COMPLETE MAPANYTHING CLOUD"
)

print(
    "=" * 80
)

all_points_calibrated = []

all_colors = []

for i, camera in enumerate(
    CAMERAS
):

    (
        points_grid,
        map_mask
    ) = get_mapanything_grid(
        predictions[i]
    )

    H_ma, W_ma = (
        points_grid.shape[:2]
    )

    image = images[
        camera
    ]

    image_grid = np.asarray(
        Image.fromarray(
            image
        ).resize(
            (
                W_ma,
                H_ma
            )
        )
    )

    valid = (
        map_mask
        &
        np.isfinite(
            points_grid
        ).all(
            axis=-1
        )
    )

    points_cam = (
        points_grid[
            valid
        ]
    )

    colors = (
        image_grid[
            valid
        ]
    )

    points_lidar_frame = (
        transform_points(
            points_cam,
            T_camera_to_lidar[
                camera
            ]
        )
    )

    points_calibrated = (
        apply_transform(
            points_lidar_frame,
            T_final
        )
    )

    all_points_calibrated.append(
        points_calibrated
    )

    all_colors.append(
        colors
    )

    print(
        f"{camera}:",
        len(points_calibrated),
        "points"
    )


all_points_calibrated = np.concatenate(
    all_points_calibrated,
    axis=0
)

all_colors = np.concatenate(
    all_colors,
    axis=0
)

print(
    "\nTotal calibrated MapAnything points:",
    len(
        all_points_calibrated
    )
)


# ============================================================
# 14. SAVE CALIBRATED MAPANYTHING
# ============================================================

aligned_path = (
    OUTPUT_DIR /
    "mapanything_aligned_to_lidar.ply"
)

save_xyz_ply(
    aligned_path,
    all_points_calibrated,
    all_colors
)

print(
    "\nSaved:"
)

print(
    aligned_path
)


# ============================================================
# 15. SAVE LIDAR REFERENCE
# ============================================================

lidar_reference_path = (
    OUTPUT_DIR /
    "lidar_reference.ply"
)

save_xyz_ply(
    lidar_reference_path,
    lidar_points
)


# ============================================================
# 16. SAVE COMBINED CLOUD
# ============================================================

lidar_colors = np.full(
    (
        len(lidar_points),
        3
    ),
    255,
    dtype=np.uint8
)

combined_points = np.concatenate(
    [
        all_points_calibrated,
        lidar_points
    ],
    axis=0
)

combined_colors = np.concatenate(
    [
        all_colors,
        lidar_colors
    ],
    axis=0
)

combined_path = (
    OUTPUT_DIR /
    "mapanything_aligned_plus_lidar.ply"
)

save_xyz_ply(
    combined_path,
    combined_points,
    combined_colors
)


# ============================================================
# 17. SAVE FINAL INLIERS
# ============================================================

save_xyz_ply(
    OUTPUT_DIR /
    "final_inlier_mapanything.ply",
    ma_corr[
        final_inliers
    ]
)

save_xyz_ply(
    OUTPUT_DIR /
    "final_inlier_lidar.ply",
    lidar_corr[
        final_inliers
    ]
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "REGISTRATION COMPLETE"
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
    "\nImportant outputs:"
)

print(
    "  mapanything_to_lidar_transform.txt"
)

print(
    "  mapanything_aligned_to_lidar.ply"
)

print(
    "  lidar_reference.ply"
)

print(
    "  mapanything_aligned_plus_lidar.ply"
)

print(
    "  correspondence_mapanything.ply"
)

print(
    "  correspondence_lidar.ply"
)

print(
    "  final_inlier_mapanything.ply"
)

print(
    "  final_inlier_lidar.ply"
)

print(
    "\nFinal scale:",
    scale_final
)

print(
    "Final inlier threshold:",
    f"{final_inlier_threshold:.3f} m"
)

print(
    "Final inlier ratio:",
    f"{100 * final_inliers.mean():.2f}%"
)

print(
    "\nFINAL INLIER RMSE:",
    f"{inlier_metrics['rmse']:.4f} m"
)

print(
    "FINAL INLIER MEDIAN:",
    f"{inlier_metrics['median']:.4f} m"
)

print(
    "FINAL INLIER P90:",
    f"{inlier_metrics['p90']:.4f} m"
)

print(
    "FINAL INLIER P95:",
    f"{inlier_metrics['p95']:.4f} m"
)

print(
    "\nFor reference, ALL-correspondence RMSE:",
    f"{all_metrics['rmse']:.4f} m"
)

print(
    "=" * 80
)