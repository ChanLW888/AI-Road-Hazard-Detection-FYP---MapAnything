import csv
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.spatial import cKDTree

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

DIAGNOSTICS_DIR = (
    BAG_ROOT /
    "mapanything_3camera_diagnostics"
)

OUTPUT_DIR = (
    BAG_ROOT /
    "mapanything_lidar_direct_correction"
)


# ============================================================
# INPUT FILES
# ============================================================

SEMANTIC_VOXEL_FILE = (
    DIAGNOSTICS_DIR /
    "semantic_voxels_direct.ply"
)

RGB_POINT_FILE = (
    DIAGNOSTICS_DIR /
    "semantic_3d_merged_direct.ply"
)


# ============================================================
# FRAME SELECTION
# ============================================================

REFERENCE_CAMERA = "CAM2"

REFERENCE_INDEX = 0

CAMERAS = [
    "CAM1",
    "CAM2",
    "CAM6",
]

SYNC_TOLERANCE_NS = 50_000_000  # 50 ms


# ============================================================
# VOXEL SETTINGS
# ============================================================

VOXEL_SIZE = 0.05  # 5 cm


# ============================================================
# LOCAL CORRECTION SETTINGS
# ============================================================

# Number of nearby direct LiDAR↔MapAnything correspondences
# considered for each voxel.
K_NEIGHBORS = 24

# Maximum distance from a voxel to a correspondence point.
MAX_NEIGHBOR_DISTANCE = 30  # m

# Minimum number of locally agreeing correspondences.
MIN_NEIGHBORS = 3

# Distance-weighting exponent.
DISTANCE_POWER = 4.0

# Maximum amount that an individual voxel may move.
MAX_CORRECTION = 2  # m

# Maximum disagreement allowed between local displacement
# vectors.
LOCAL_DISPLACEMENT_GATE = 0.50  # m

# Ignore movements smaller than this.
MIN_CORRECTION = 0.005  # m


# ============================================================
# DIRECT CORRESPONDENCE FILTER
# ============================================================

MAX_DIRECT_DISPLACEMENT = 10.0  # metres


# ============================================================
# CAMERA PROJECTION
# ============================================================

LIDAR_MIN_CAMERA_DEPTH = 0.05

BORDER_MARGIN_PIXELS = 2


# ============================================================
# RANDOM
# ============================================================

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
# CAMERA/LIDAR EXTRINSICS
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
# TRANSFORM FUNCTIONS
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
                (
                    len(flat),
                    1
                ),
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
# PARSE CAMERA MATRIX
# ============================================================

def parse_array(value):

    value = str(value).strip()

    wrappers = [
        "np.float64(",
        "np.float32(",
        "np.int64(",
        "np.int32(",
    ]

    for wrapper in wrappers:

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
            f"Could not parse:\n{value}"
        )

    return values


# ============================================================
# CAMERA INTRINSICS
# ============================================================

def load_rectified_intrinsics(
    camera
):

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

        raise RuntimeError(
            f"{camera}: invalid P matrix size "
            f"{P.size}"
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
# TIMESTAMPED IMAGES
# ============================================================

def load_images(
    camera
):

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
# NEAREST TIMESTAMP
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
            int(
                records[i][0]
            )
            -
            int(
                target_timestamp
            )
        )
    )

    timestamp = int(
        records[best][0]
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
# GENERIC ASCII PLY LOADER
# ============================================================

def load_ascii_ply(
    path
):

    print(
        f"\nLoading:\n{path}"
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
            f"No vertex count found:\n{path}"
        )

    properties = []

    in_vertex = False

    for line in header:

        parts = line.split()

        if (
            len(parts) >= 3
            and
            parts[0] == "element"
            and
            parts[1] == "vertex"
        ):

            in_vertex = True

            continue

        if (
            in_vertex
            and
            len(parts) >= 3
            and
            parts[0] == "property"
        ):

            properties.append(
                parts[-1]
            )

        elif (
            in_vertex
            and
            len(parts) >= 2
            and
            parts[0] == "element"
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
            f"{data.shape[1]} columns but "
            f"{len(properties)} properties."
        )

    print(
        "  Points:",
        vertex_count
    )

    print(
        "  Properties:",
        properties
    )

    return (
        data,
        properties
    )


# ============================================================
# XYZ / RGB
# ============================================================

def get_xyz(
    data,
    properties
):

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
        [
            x_idx,
            y_idx,
            z_idx
        ]
    ].astype(
        np.float64
    )


def get_rgb(
    data,
    properties
):

    r_idx = properties.index(
        "red"
    )

    g_idx = properties.index(
        "green"
    )

    b_idx = properties.index(
        "blue"
    )

    return np.clip(
        np.rint(
            data[
                :,
                [
                    r_idx,
                    g_idx,
                    b_idx
                ]
            ]
        ),
        0,
        255
    ).astype(
        np.uint8
    )


# ============================================================
# LOAD LIDAR PLY
# ============================================================

def load_lidar_ply(
    path
):

    data, properties = (
        load_ascii_ply(
            path
        )
    )

    required = [
        "x",
        "y",
        "z",
    ]

    for name in required:

        if name not in properties:

            raise RuntimeError(
                f"LiDAR PLY is missing "
                f"'{name}':\n{path}"
            )

    points = get_xyz(
        data,
        properties
    )

    return points


# ============================================================
# PROJECT LiDAR INTO CAMERA
# ============================================================

def project_lidar_to_camera(
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

    X = camera_points[:, 0]
    Y = camera_points[:, 1]
    Z = camera_points[:, 2]

    valid_front = (
        Z >
        LIDAR_MIN_CAMERA_DEPTH
    )

    source_indices = np.arange(
        len(lidar_points),
        dtype=np.intp
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

    source_indices = (
        source_indices[
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

    inside = (
        (u >= 2)
        &
        (u < width - 2)
        &
        (v >= 2)
        &
        (v < height - 2)
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

    source_indices = (
        source_indices[
            inside
        ]
    )

    return (
        u,
        v,
        Z,
        source_indices
    )


# ============================================================
# MAPANYTHING OUTPUT
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

    return (
        points,
        mask.astype(
            bool
        )
    )


# ============================================================
# IMAGE PIXEL -> MAPANYTHING GRID
# ============================================================

def image_to_map_grid(
    u,
    v,
    image_width,
    image_height,
    map_width,
    map_height
):

    u = np.asarray(
        u,
        dtype=np.float64
    )

    v = np.asarray(
        v,
        dtype=np.float64
    )

    u_map = (
        u
        *
        float(map_width)
        /
        float(image_width)
    )

    v_map = (
        v
        *
        float(map_height)
        /
        float(image_height)
    )

    u_map = np.rint(
        u_map
    ).astype(
        np.intp
    )

    v_map = np.rint(
        v_map
    ).astype(
        np.intp
    )

    valid = (
        (u_map >= 0)
        &
        (u_map < map_width)
        &
        (v_map >= 0)
        &
        (v_map < map_height)
    )

    return (
        u_map,
        v_map,
        valid
    )


# ============================================================
# DIRECT CORRESPONDENCES
# ============================================================

def build_direct_correspondences(
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
        map_points_cam,
        map_mask
    ) = get_mapanything_grid(
        prediction
    )

    H_map, W_map = (
        map_points_cam.shape[:2]
    )

    print(
        f"\n{camera}"
    )

    print(
        "  MapAnything grid:",
        W_map,
        "x",
        H_map
    )

    (
        u,
        v,
        lidar_depth,
        lidar_indices
    ) = project_lidar_to_camera(
        lidar_points,
        T_lidar_to_camera,
        K_rect,
        image_width,
        image_height
    )

    print(
        "  Projected base LiDAR points:",
        len(u)
    )

    if len(u) == 0:

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
        u_map,
        v_map,
        valid_grid
    ) = image_to_map_grid(
        u,
        v,
        image_width,
        image_height,
        W_map,
        H_map
    )

    u_map = np.asarray(
        u_map[
            valid_grid
        ],
        dtype=np.intp
    )

    v_map = np.asarray(
        v_map[
            valid_grid
        ],
        dtype=np.intp
    )

    lidar_depth = (
        lidar_depth[
            valid_grid
        ]
    )

    lidar_indices = np.asarray(
        lidar_indices[
            valid_grid
        ],
        dtype=np.intp
    )

    pixel_ids = (
        v_map.astype(
            np.int64
        )
        *
        int(W_map)
        +
        u_map.astype(
            np.int64
        )
    )

    order = np.lexsort(
        (
            lidar_depth,
            pixel_ids
        )
    )

    order = np.asarray(
        order,
        dtype=np.intp
    )

    sorted_pixel_ids = (
        pixel_ids[
            order
        ]
    )

    keep = np.ones(
        len(order),
        dtype=bool
    )

    if len(order) > 1:

        keep[1:] = (
            sorted_pixel_ids[1:]
            !=
            sorted_pixel_ids[:-1]
        )

    selected = np.asarray(
        order[
            keep
        ],
        dtype=np.intp
    )

    u_map = np.asarray(
        u_map[
            selected
        ],
        dtype=np.intp
    )

    v_map = np.asarray(
        v_map[
            selected
        ],
        dtype=np.intp
    )

    lidar_indices = np.asarray(
        lidar_indices[
            selected
        ],
        dtype=np.intp
    )

    print(
        "  Unique MapAnything pixels:",
        len(u_map)
    )

    map_points_cam_sampled = (
        map_points_cam[
            v_map,
            u_map
        ]
    )

    sampled_mask = (
        map_mask[
            v_map,
            u_map
        ]
    )

    valid = (
        sampled_mask
        &
        np.isfinite(
            map_points_cam_sampled
        ).all(
            axis=1
        )
    )

    map_points_cam_sampled = (
        map_points_cam_sampled[
            valid
        ]
    )

    lidar_indices = np.asarray(
        lidar_indices[
            valid
        ],
        dtype=np.intp
    )

    print(
        "  Valid MapAnything pixels:",
        len(
            map_points_cam_sampled
        )
    )

    if len(
        map_points_cam_sampled
    ) == 0:

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

    map_points_lidar = (
        transform_points(
            map_points_cam_sampled,
            T_camera_to_lidar
        )
    )

    lidar_points_sampled = (
        lidar_points[
            lidar_indices
        ]
    )

    displacement = (
        lidar_points_sampled -
        map_points_lidar
    )

    displacement_magnitude = (
        np.linalg.norm(
            displacement,
            axis=1
        )
    )

    print(
        "  Direct displacement median:",
        f"{np.median(displacement_magnitude):.4f} m"
    )

    print(
        "  Direct displacement P90:",
        f"{np.percentile(displacement_magnitude, 90):.4f} m"
    )

    print(
        "  Direct displacement P95:",
        f"{np.percentile(displacement_magnitude, 95):.4f} m"
    )

    valid_direct = (
        np.isfinite(
            displacement_magnitude
        )
        &
        (
            displacement_magnitude
            <=
            MAX_DIRECT_DISPLACEMENT
        )
    )

    map_points_lidar = (
        map_points_lidar[
            valid_direct
        ]
    )

    lidar_points_sampled = (
        lidar_points_sampled[
            valid_direct
        ]
    )

    print(
        "  After direct correspondence filter:",
        len(
            map_points_lidar
        )
    )

    return (
        map_points_lidar,
        lidar_points_sampled
    )


# ============================================================
# LOCAL CORRECTION FIELD
# ============================================================

def calculate_local_corrections(
    voxels,
    map_corr,
    lidar_corr
):

    if len(map_corr) == 0:

        return (
            np.zeros(
                (
                    len(voxels),
                    3
                ),
                dtype=np.float64
            ),
            np.zeros(
                len(voxels),
                dtype=np.int32
            )
        )

    displacement = (
        lidar_corr -
        map_corr
    )

    magnitude = np.linalg.norm(
        displacement,
        axis=1
    )

    print(
        "\nLocal displacement stats:"
    )

    print(
        "  Correspondences:",
        len(displacement)
    )

    print(
        "  Median:",
        f"{np.median(magnitude):.4f} m"
    )

    print(
        "  P90:",
        f"{np.percentile(magnitude, 90):.4f} m"
    )

    print(
        "  P95:",
        f"{np.percentile(magnitude, 95):.4f} m"
    )

    tree = cKDTree(
        map_corr
    )

    distances, indices = (
        tree.query(
            voxels,
            k=K_NEIGHBORS,
            distance_upper_bound=
                MAX_NEIGHBOR_DISTANCE
        )
    )

    if K_NEIGHBORS == 1:

        distances = (
            distances[:, None]
        )

        indices = (
            indices[:, None]
        )

    corrections = np.zeros(
        (
            len(voxels),
            3
        ),
        dtype=np.float64
    )

    support = np.zeros(
        len(voxels),
        dtype=np.int32
    )

    for i in range(
        len(voxels)
    ):

        d = distances[i]
        idx = indices[i]

        valid = (
            np.isfinite(d)
            &
            (idx < len(map_corr))
        )

        d = d[
            valid
        ]

        idx = np.asarray(
            idx[
                valid
            ],
            dtype=np.intp
        )

        if len(d) < MIN_NEIGHBORS:

            continue

        local_vectors = (
            displacement[
                idx
            ]
        )

        median_vector = np.median(
            local_vectors,
            axis=0
        )

        deviation = np.linalg.norm(
            local_vectors -
            median_vector,
            axis=1
        )

        consistent = (
            deviation
            <=
            LOCAL_DISPLACEMENT_GATE
        )

        d = d[
            consistent
        ]

        local_vectors = (
            local_vectors[
                consistent
            ]
        )

        if len(
            local_vectors
        ) < MIN_NEIGHBORS:

            continue

        weights = (
            1.0
            /
            (
                np.maximum(
                    d,
                    1e-4
                )
                **
                DISTANCE_POWER
            )
        )

        correction = (
            np.sum(
                local_vectors *
                weights[:, None],
                axis=0
            )
            /
            np.sum(
                weights
            )
        )

        magnitude = np.linalg.norm(
            correction
        )

        if magnitude > MAX_CORRECTION:

            correction = (
                correction
                /
                magnitude
                *
                MAX_CORRECTION
            )

            magnitude = (
                MAX_CORRECTION
            )

        if magnitude < MIN_CORRECTION:

            correction[:] = 0.0

            magnitude = 0.0

        corrections[i] = (
            correction
        )

        support[i] = (
            len(local_vectors)
        )

    return (
        corrections,
        support
    )


# ============================================================
# RGB VOXELISATION
# ============================================================

def voxelize_rgb(
    points,
    rgb,
    voxel_size
):

    valid = (
        np.isfinite(
            points
        ).all(
            axis=1
        )
        &
        np.isfinite(
            rgb
        ).all(
            axis=1
        )
    )

    points = points[
        valid
    ]

    rgb = rgb[
        valid
    ]

    voxel_index = np.floor(
        points /
        voxel_size
    ).astype(
        np.int64
    )

    unique_voxels, inverse = (
        np.unique(
            voxel_index,
            axis=0,
            return_inverse=True
        )
    )

    num_voxels = len(
        unique_voxels
    )

    xyz_sum = np.zeros(
        (
            num_voxels,
            3
        ),
        dtype=np.float64
    )

    np.add.at(
        xyz_sum,
        inverse,
        points
    )

    counts = np.bincount(
        inverse,
        minlength=num_voxels
    )

    voxel_xyz = (
        xyz_sum /
        counts[:, None]
    )

    rgb_sum = np.zeros(
        (
            num_voxels,
            3
        ),
        dtype=np.float64
    )

    np.add.at(
        rgb_sum,
        inverse,
        rgb
    )

    voxel_rgb = np.rint(
        rgb_sum /
        counts[:, None]
    )

    voxel_rgb = np.clip(
        voxel_rgb,
        0,
        255
    ).astype(
        np.uint8
    )

    return (
        voxel_xyz,
        voxel_rgb
    )


# ============================================================
# SAVE RGB VOXELS
# ============================================================

def save_rgb_voxels(
    path,
    xyz,
    rgb,
    correction,
    magnitude,
    support
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
            f"element vertex {len(xyz)}\n"
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
            "property float correction_dx\n"
        )

        f.write(
            "property float correction_dy\n"
        )

        f.write(
            "property float correction_dz\n"
        )

        f.write(
            "property float correction_magnitude\n"
        )

        f.write(
            "property int lidar_support\n"
        )

        f.write(
            "end_header\n"
        )

        for i in range(
            len(xyz)
        ):

            p = xyz[i]

            c = rgb[i]

            d = correction[i]

            f.write(
                f"{p[0]:.6f} "
                f"{p[1]:.6f} "
                f"{p[2]:.6f} "
                f"{int(c[0])} "
                f"{int(c[1])} "
                f"{int(c[2])} "
                f"{d[0]:.6f} "
                f"{d[1]:.6f} "
                f"{d[2]:.6f} "
                f"{magnitude[i]:.6f} "
                f"{int(support[i])}\n"
            )


# ============================================================
# SAVE SEMANTIC VOXELS
# ============================================================

def save_semantic_voxels(
    path,
    xyz,
    original_data,
    properties,
    correction,
    magnitude,
    support
):

    x_idx = properties.index(
        "x"
    )

    y_idx = properties.index(
        "y"
    )

    z_idx = properties.index(
        "z"
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
            f"element vertex {len(xyz)}\n"
        )

        for name in properties:

            if name == "label":

                f.write(
                    "property int label\n"
                )

            elif name in [
                "red",
                "green",
                "blue",
                "semantic_red",
                "semantic_green",
                "semantic_blue",
            ]:

                f.write(
                    f"property uchar {name}\n"
                )

            else:

                f.write(
                    f"property float {name}\n"
                )

        f.write(
            "property float correction_dx\n"
        )

        f.write(
            "property float correction_dy\n"
        )

        f.write(
            "property float correction_dz\n"
        )

        f.write(
            "property float correction_magnitude\n"
        )

        f.write(
            "property int lidar_support\n"
        )

        f.write(
            "end_header\n"
        )

        for i in range(
            len(xyz)
        ):

            row = (
                original_data[
                    i
                ].copy()
            )

            row[x_idx] = (
                xyz[i, 0]
            )

            row[y_idx] = (
                xyz[i, 1]
            )

            row[z_idx] = (
                xyz[i, 2]
            )

            values = []

            for j, name in enumerate(
                properties
            ):

                if name in [
                    "label",
                    "red",
                    "green",
                    "blue",
                    "semantic_red",
                    "semantic_green",
                    "semantic_blue",
                ]:

                    values.append(
                        str(
                            int(
                                round(
                                    row[j]
                                )
                            )
                        )
                    )

                else:

                    values.append(
                        f"{row[j]:.6f}"
                    )

            d = correction[i]

            values.extend(
                [
                    f"{d[0]:.6f}",
                    f"{d[1]:.6f}",
                    f"{d[2]:.6f}",
                    f"{magnitude[i]:.6f}",
                    str(
                        int(
                            support[i]
                        )
                    ),
                ]
            )

            f.write(
                " ".join(
                    values
                )
                +
                "\n"
            )


# ============================================================
# SAVE CORRECTION VISUALISATION
# ============================================================

def save_correction_visualization(
    path,
    xyz,
    correction
):

    magnitude = np.linalg.norm(
        correction,
        axis=1
    )

    vmin = np.percentile(
        magnitude,
        2
    )

    vmax = np.percentile(
        magnitude,
        98
    )

    scale = max(
        vmax - vmin,
        1e-8
    )

    norm = np.clip(
        (
            magnitude -
            vmin
        )
        /
        scale,
        0,
        1
    )

    red = (
        255 *
        norm
    ).astype(
        np.uint8
    )

    blue = (
        255 *
        (1.0 - norm)
    ).astype(
        np.uint8
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
            f"element vertex {len(xyz)}\n"
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

        for i in range(
            len(xyz)
        ):

            p = xyz[i]

            f.write(
                f"{p[0]:.6f} "
                f"{p[1]:.6f} "
                f"{p[2]:.6f} "
                f"{int(red[i])} "
                f"50 "
                f"{int(blue[i])}\n"
            )


# ============================================================
# MAIN
# ============================================================

print(
    "=" * 80
)

print(
    "DIRECT LiDAR-BASED LOCAL VOXEL CORRECTION"
)

print(
    "=" * 80
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
# 1. CAMERA CALIBRATION
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "1. CAMERA CALIBRATION"
)

print(
    "=" * 80
)

camera_calibration = {}
camera_images = {}

for camera in CAMERAS:

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

    camera_images[
        camera
    ] = load_images(
        camera
    )

    print(
        f"\n{camera}"
    )

    print(
        "  Resolution:",
        width,
        "x",
        height
    )

    print(
        "  Images:",
        len(
            camera_images[
                camera
            ]
        )
    )

    print(
        "  K_rect:"
    )

    print(
        K_rect
    )


# ============================================================
# 2. SYNCHRONIZED FRAME
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "2. SYNCHRONIZED FRAME"
)

print(
    "=" * 80
)

reference_records = (
    camera_images[
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
        "REFERENCE_INDEX out of range."
    )

reference_timestamp = int(
    reference_records[
        REFERENCE_INDEX
    ][0]
)

print(
    "Reference timestamp:",
    reference_timestamp
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
            f"No synchronized "
            f"{camera} image."
        )

    selected_images[
        camera
    ] = path

    print(
        f"{camera}: "
        f"{timestamp} "
        f"({delta / 1e6:.3f} ms)"
    )


# ============================================================
# 3. BASE LIDAR
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "3. BASE LiDAR"
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
        "No LiDAR scan found."
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
        "WARNING: synchronization > 50 ms"
    )

lidar_points = load_lidar_ply(
    lidar_path
)

print(
    "LiDAR points:",
    len(lidar_points)
)


# ============================================================
# 4. LOAD MAPANYTHING
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "4. MAPANYTHING"
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


# ============================================================
# 5. BUILD VIEWS
# ============================================================

views = []
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

    images[
        camera
    ] = image

    views.append(
        {
            "img":
                image,

            "intrinsics":
                camera_calibration[
                    camera
                ]["K"].astype(
                    np.float32
                ),
        }
    )


# ============================================================
# 6. MAPANYTHING INFERENCE
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "5. MAPANYTHING INFERENCE"
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
# 7. DIRECT LiDAR ↔ MAPANYTHING CORRESPONDENCES
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "6. DIRECT LiDAR <-> MAPANYTHING CORRESPONDENCES"
)

print(
    "=" * 80
)

all_map_corr = []
all_lidar_corr = []

for i, camera in enumerate(
    CAMERAS
):

    cal = camera_calibration[
        camera
    ]

    T_lidar_to_camera = (
        invert_transform(
            EXTRINSICS[
                camera
            ]
        )
    )

    T_camera_to_lidar = (
        EXTRINSICS[
            camera
        ]
    )

    map_corr, lidar_corr = (
        build_direct_correspondences(
            camera=camera,

            prediction=predictions[i],

            lidar_points=lidar_points,

            K_rect=cal["K"],

            image_width=cal["width"],

            image_height=cal["height"],

            T_lidar_to_camera=
                T_lidar_to_camera,

            T_camera_to_lidar=
                T_camera_to_lidar,
        )
    )

    all_map_corr.append(
        map_corr
    )

    all_lidar_corr.append(
        lidar_corr
    )


map_corr = np.concatenate(
    all_map_corr,
    axis=0
)

lidar_corr = np.concatenate(
    all_lidar_corr,
    axis=0
)

print(
    "\nTOTAL DIRECT CORRESPONDENCES:",
    len(map_corr)
)

if len(map_corr) < MIN_NEIGHBORS:

    raise RuntimeError(
        "Too few direct correspondences."
    )


# ============================================================
# 8. DIRECT CORRESPONDENCE STATISTICS
# ============================================================

displacement = (
    lidar_corr -
    map_corr
)

displacement_magnitude = np.linalg.norm(
    displacement,
    axis=1
)

print(
    "\n" + "=" * 80
)

print(
    "7. DIRECT CORRESPONDENCE STATISTICS"
)

print(
    "=" * 80
)

print(
    "Count:",
    len(map_corr)
)

print(
    "Median displacement:",
    f"{np.median(displacement_magnitude):.4f} m"
)

print(
    "P90 displacement:",
    f"{np.percentile(displacement_magnitude, 90):.4f} m"
)

print(
    "P95 displacement:",
    f"{np.percentile(displacement_magnitude, 95):.4f} m"
)


# ============================================================
# 9. LOAD SEMANTIC VOXELS
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "8. LOAD SEMANTIC VOXELS"
)

print(
    "=" * 80
)

semantic_data, semantic_properties = (
    load_ascii_ply(
        SEMANTIC_VOXEL_FILE
    )
)

semantic_voxels = get_xyz(
    semantic_data,
    semantic_properties
)

print(
    "Semantic voxels:",
    len(semantic_voxels)
)


# ============================================================
# 10. LOAD RGB CLOUD
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "9. LOAD RGB MAPANYTHING"
)

print(
    "=" * 80
)

rgb_data, rgb_properties = (
    load_ascii_ply(
        RGB_POINT_FILE
    )
)

rgb_points = get_xyz(
    rgb_data,
    rgb_properties
)

rgb_colors = get_rgb(
    rgb_data,
    rgb_properties
)

print(
    "RGB points:",
    len(rgb_points)
)


# ============================================================
# 11. RGB VOXELISATION
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "10. RGB VOXELISATION"
)

print(
    "=" * 80
)

rgb_voxels, rgb_voxel_colors = (
    voxelize_rgb(
        rgb_points,
        rgb_colors,
        VOXEL_SIZE
    )
)

print(
    "RGB voxels:",
    len(rgb_voxels)
)


# ============================================================
# 12. SEMANTIC LOCAL CORRECTION
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "11. LOCAL SEMANTIC CORRECTION"
)

print(
    "=" * 80
)

semantic_correction, semantic_support = (
    calculate_local_corrections(
        semantic_voxels,
        map_corr,
        lidar_corr
    )
)

semantic_corrected = (
    semantic_voxels +
    semantic_correction
)

semantic_magnitude = np.linalg.norm(
    semantic_correction,
    axis=1
)


# ============================================================
# 13. RGB LOCAL CORRECTION
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "12. LOCAL RGB CORRECTION"
)

print(
    "=" * 80
)

rgb_correction, rgb_support = (
    calculate_local_corrections(
        rgb_voxels,
        map_corr,
        lidar_corr
    )
)

rgb_corrected = (
    rgb_voxels +
    rgb_correction
)

rgb_magnitude = np.linalg.norm(
    rgb_correction,
    axis=1
)


# ============================================================
# 14. ACCEPTANCE MASKS
# ============================================================
#
# IMPORTANT:
#
# The final RGB and semantic outputs contain ONLY voxels that:
#
#   1. Have at least MIN_NEIGHBORS locally consistent LiDAR
#      correspondences.
#
#   2. Actually received a correction larger than
#      MIN_CORRECTION.
#
# Non-accepted MapAnything voxels are completely removed.
#
# ============================================================

semantic_accepted = (
    (semantic_support >= MIN_NEIGHBORS)
    &
    (semantic_magnitude > MIN_CORRECTION)
)

rgb_accepted = (
    (rgb_support >= MIN_NEIGHBORS)
    &
    (rgb_magnitude > MIN_CORRECTION)
)


# ============================================================
# 15. STATISTICS
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "13. ACCEPTANCE STATISTICS"
)

print(
    "=" * 80
)

print(
    "\nSEMANTIC"
)

print(
    "  Total voxels:",
    len(semantic_voxels)
)

print(
    "  Accepted/corrected:",
    int(
        semantic_accepted.sum()
    )
)

print(
    "  Rejected:",
    int(
        (~semantic_accepted).sum()
    )
)

if semantic_accepted.any():

    m = (
        semantic_magnitude[
            semantic_accepted
        ]
    )

    print(
        "  Accepted correction mean:",
        f"{m.mean():.4f} m"
    )

    print(
        "  Accepted correction median:",
        f"{np.median(m):.4f} m"
    )

    print(
        "  Accepted correction P90:",
        f"{np.percentile(m, 90):.4f} m"
    )

    print(
        "  Accepted correction P95:",
        f"{np.percentile(m, 95):.4f} m"
    )


print(
    "\nRGB"
)

print(
    "  Total voxels:",
    len(rgb_voxels)
)

print(
    "  Accepted/corrected:",
    int(
        rgb_accepted.sum()
    )
)

print(
    "  Rejected:",
    int(
        (~rgb_accepted).sum()
    )
)

if rgb_accepted.any():

    m = (
        rgb_magnitude[
            rgb_accepted
        ]
    )

    print(
        "  Accepted correction mean:",
        f"{m.mean():.4f} m"
    )

    print(
        "  Accepted correction median:",
        f"{np.median(m):.4f} m"
    )

    print(
        "  Accepted correction P90:",
        f"{np.percentile(m, 90):.4f} m"
    )

    print(
        "  Accepted correction P95:",
        f"{np.percentile(m, 95):.4f} m"
    )


# ============================================================
# 16. SAVE ONLY ACCEPTED SEMANTIC VOXELS
# ============================================================

semantic_output = (
    OUTPUT_DIR /
    "semantic_voxels_lidar_corrected_only.ply"
)

save_semantic_voxels(
    path=semantic_output,

    xyz=semantic_corrected[
        semantic_accepted
    ],

    original_data=semantic_data[
        semantic_accepted
    ],

    properties=semantic_properties,

    correction=semantic_correction[
        semantic_accepted
    ],

    magnitude=semantic_magnitude[
        semantic_accepted
    ],

    support=semantic_support[
        semantic_accepted
    ]
)

print(
    "\nSemantic accepted-only output:"
)

print(
    semantic_output
)


# ============================================================
# 17. SAVE ONLY ACCEPTED RGB VOXELS
# ============================================================

rgb_output = (
    OUTPUT_DIR /
    "rgb_voxels_lidar_corrected_only.ply"
)

save_rgb_voxels(
    path=rgb_output,

    xyz=rgb_corrected[
        rgb_accepted
    ],

    rgb=rgb_voxel_colors[
        rgb_accepted
    ],

    correction=rgb_correction[
        rgb_accepted
    ],

    magnitude=rgb_magnitude[
        rgb_accepted
    ],

    support=rgb_support[
        rgb_accepted
    ]
)

print(
    "\nRGB accepted-only output:"
)

print(
    rgb_output
)


# ============================================================
# 18. SAVE CORRECTION VISUALISATION
# ============================================================

correction_visualization = (
    OUTPUT_DIR /
    "semantic_voxel_correction_magnitude.ply"
)

save_correction_visualization(
    path=correction_visualization,

    xyz=semantic_corrected[
        semantic_accepted
    ],

    correction=semantic_correction[
        semantic_accepted
    ]
)


# ============================================================
# 19. SAVE DIRECT CORRESPONDENCES
# ============================================================

correspondence_file = (
    OUTPUT_DIR /
    "direct_mapanything_lidar_correspondences.ply"
)

with open(
    correspondence_file,
    "w"
) as f:

    f.write(
        "ply\n"
    )

    f.write(
        "format ascii 1.0\n"
    )

    f.write(
        f"element vertex {len(map_corr)}\n"
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
        "property float lidar_x\n"
    )

    f.write(
        "property float lidar_y\n"
    )

    f.write(
        "property float lidar_z\n"
    )

    f.write(
        "property float displacement\n"
    )

    f.write(
        "end_header\n"
    )

    for i in range(
        len(map_corr)
    ):

        mp = map_corr[i]

        lp = lidar_corr[i]

        f.write(
            f"{mp[0]:.6f} "
            f"{mp[1]:.6f} "
            f"{mp[2]:.6f} "
            f"{lp[0]:.6f} "
            f"{lp[1]:.6f} "
            f"{lp[2]:.6f} "
            f"{displacement_magnitude[i]:.6f}\n"
        )


# ============================================================
# FINAL
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "DIRECT LiDAR VOXEL CORRECTION COMPLETE"
)

print(
    "=" * 80
)

print(
    "\nNo register.py outputs were used."
)

print(
    "\nBase LiDAR:"
)

print(
    lidar_path
)

print(
    "\nSEMANTIC ACCEPTED-ONLY:"
)

print(
    semantic_output
)

print(
    "  Voxels:",
    int(
        semantic_accepted.sum()
    )
)

print(
    "\nRGB ACCEPTED-ONLY:"
)

print(
    rgb_output
)

print(
    "  Voxels:",
    int(
        rgb_accepted.sum()
    )
)

print(
    "\nCorrection visualization:"
)

print(
    correction_visualization
)

print(
    "\nDirect correspondence diagnostic:"
)

print(
    correspondence_file
)

print(
    "\nCurrent settings:"
)

print(
    "  VOXEL_SIZE =",
    VOXEL_SIZE,
    "m"
)

print(
    "  K_NEIGHBORS =",
    K_NEIGHBORS
)

print(
    "  MAX_NEIGHBOR_DISTANCE =",
    MAX_NEIGHBOR_DISTANCE,
    "m"
)

print(
    "  MIN_NEIGHBORS =",
    MIN_NEIGHBORS
)

print(
    "  DISTANCE_POWER =",
    DISTANCE_POWER
)

print(
    "  MAX_CORRECTION =",
    MAX_CORRECTION,
    "m"
)

print(
    "  MAX_DIRECT_DISPLACEMENT =",
    MAX_DIRECT_DISPLACEMENT,
    "m"
)

print(
    "  LOCAL_DISPLACEMENT_GATE =",
    LOCAL_DISPLACEMENT_GATE,
    "m"
)

print(
    "  MIN_CORRECTION =",
    MIN_CORRECTION,
    "m"
)

print(
    "=" * 80
)