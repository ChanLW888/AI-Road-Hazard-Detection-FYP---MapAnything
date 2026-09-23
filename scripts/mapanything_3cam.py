import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from mapanything.models import MapAnything
from mapanything.utils.image import preprocess_inputs


# ============================================================
# CONFIGURATION
# ============================================================

BAG_ROOT = Path(
    "/home/lcha0115/bt60_scratch/lcha_data/"
    "map-anything/personal_data/"
    "camera_lidar_imu_test_04"
)

EXTRACTED = BAG_ROOT / "extracted"

OUTPUT_DIR = (
    BAG_ROOT /
    "mapanything_3camera_diagnostics"
)

CAMERAS = [
    "CAM1",
    "CAM2",
    "CAM6",
]

IMAGE_STREAM = "rect"

REFERENCE_CAMERA = "CAM2"

REFERENCE_INDEX = 0

# We will print the real synchronization error.
# This is deliberately not used as a hard failure.
SYNC_TOLERANCE_NS = 500_000_000


# ============================================================
# TF STATIC
#
# ROS BAG:
#
# velodyne -> cam*_optical_frame
#
# translation: x y z
# quaternion:  x y z w
# ============================================================

TF_STATIC = {

    "CAM1": {
        "translation": np.array([
            -0.8213217944865953,
            -0.6218064088450744,
            -0.5774158205492054,
        ]),

        "quaternion": np.array([
            -0.30407914234483474,
            -0.7597572078064244,
            0.5336937995671186,
            0.2132505303173423,
        ]),
    },

    "CAM6": {
        "translation": np.array([
            0.9500602067289753,
            -0.46453085308427877,
            -1.0134035869893538,
        ]),

        "quaternion": np.array([
            -0.2949015823285541,
            0.7133788286531555,
            -0.5851709284442818,
            0.2483922061442834,
        ]),
    },

    "CAM2": {
        "translation": np.array([
            -0.35812314453904737,
            0.7418722578139009,
            -0.9622312541336254,
        ]),

        "quaternion": np.array([
            -0.696591590508534,
            -0.30318163432606277,
            0.27662795664738776,
            0.5884879151191293,
        ]),
    },
}


# ============================================================
# DEVICE
# ============================================================

device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)


# ============================================================
# BASIC PATHS
# ============================================================

def image_dir(camera):

    if IMAGE_STREAM == "rect":

        return (
            EXTRACTED /
            camera /
            "images_rect"
        )

    return (
        EXTRACTED /
        camera /
        "images_raw"
    )


def camera_info_path(camera):

    return (
        EXTRACTED /
        camera /
        "camera_info.csv"
    )


# ============================================================
# PARSE ARRAYS FROM CSV
# ============================================================

def parse_array(value):

    value = str(value).strip()

    # Handle things like:
    #
    # np.float64(517.2), np.float64(0.0), ...
    #
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
# LOAD RECTIFIED CAMERA INTRINSICS
# ============================================================
#
# IMPORTANT:
#
# We are using images_rect.
#
# Therefore use:
#
# P[:3,:3]
#
# rather than the original K.
# ============================================================

def load_rectified_intrinsics(camera):

    path = camera_info_path(camera)

    with open(path, "r") as f:

        rows = list(
            csv.DictReader(f)
        )

    if not rows:

        raise RuntimeError(
            f"No CameraInfo found:\n{path}"
        )

    row = rows[0]

    P = parse_array(
        row["P"]
    )

    if P.size != 12:

        raise ValueError(
            f"{camera} P has "
            f"{P.size} values instead of 12"
        )

    P = P.reshape(
        3,
        4
    )

    K_rect = P[:, :3]

    return (
        K_rect,
        P,
        int(row["width"]),
        int(row["height"]),
        row["distortion_model"],
    )


# ============================================================
# LOAD IMAGES
# ============================================================

def load_images(camera):

    directory = image_dir(camera)

    files = sorted(
        directory.glob("*.png")
    )

    records = []

    for path in files:

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
            f"No timestamped images for {camera}:\n"
            f"{directory}"
        )

    return records


# ============================================================
# NEAREST TIMESTAMP
# ============================================================

def nearest_timestamp(
    target,
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
        target
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

        return None, None, None

    best = min(
        candidates,
        key=lambda i:
        abs(
            int(
                timestamps[i]
            )
            -
            target
        )
    )

    timestamp = int(
        timestamps[best]
    )

    delta = abs(
        timestamp -
        target
    )

    return (
        best,
        timestamp,
        delta
    )


# ============================================================
# QUATERNION -> ROTATION
# ============================================================

def quaternion_to_rotation(q):

    x, y, z, w = q

    R = np.array([

        [
            1 - 2 * (y*y + z*z),
            2 * (x*y - z*w),
            2 * (x*z + y*w),
        ],

        [
            2 * (x*y + z*w),
            1 - 2 * (x*x + z*z),
            2 * (y*z - x*w),
        ],

        [
            2 * (x*z - y*w),
            2 * (y*z + x*w),
            1 - 2 * (x*x + y*y),
        ],

    ], dtype=np.float64)

    return R


# ============================================================
# BUILD HOMOGENEOUS TRANSFORM
# ============================================================

def make_transform(
    translation,
    quaternion
):

    R = quaternion_to_rotation(
        quaternion
    )

    T = np.eye(
        4,
        dtype=np.float64
    )

    T[:3, :3] = R

    T[:3, 3] = translation

    return T


# ============================================================
# INVERSE TRANSFORM
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

    flat = points.reshape(
        -1,
        3
    )

    ones = np.ones(
        (
            flat.shape[0],
            1
        ),
        dtype=np.float64
    )

    homogeneous = np.concatenate(
        [
            flat,
            ones
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
# PLY WRITER
# ============================================================

def save_ply(
    path,
    points,
    colors
):

    points = np.asarray(
        points,
        dtype=np.float32
    )

    colors = np.asarray(
        colors,
        dtype=np.uint8
    )

    valid = (
        np.isfinite(
            points
        ).all(
            axis=1
        )
    )

    points = points[
        valid
    ]

    colors = colors[
        valid
    ]

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
# PRINT MATRIX
# ============================================================

def print_matrix(
    name,
    T
):

    print(
        f"\n{name}:"
    )

    print(
        np.array2string(
            T,
            precision=6,
            suppress_small=True
        )
    )


# ============================================================
# LOAD CAMERA DATA
# ============================================================

print("=" * 80)
print("MAPANYTHING / THREE CAMERA TRANSFORMATION DIAGNOSTIC")
print("=" * 80)

print(
    "\nReference camera:",
    REFERENCE_CAMERA
)

print(
    "Reference index:",
    REFERENCE_INDEX
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


print("\n" + "=" * 80)
print("1. LOAD CAMERA DATA")
print("=" * 80)


camera_images = {}
camera_K = {}
camera_P = {}


for camera in CAMERAS:

    camera_images[camera] = (
        load_images(camera)
    )

    (
        K_rect,
        P,
        width,
        height,
        distortion
    ) = load_rectified_intrinsics(
        camera
    )

    camera_K[camera] = K_rect

    camera_P[camera] = P

    print(
        f"\n{camera}"
    )

    print(
        "  Images:",
        len(camera_images[camera])
    )

    print(
        "  Resolution:",
        width,
        "x",
        height
    )

    print(
        "  Distortion:",
        distortion
    )

    print(
        "  RECTIFIED K = P[:3,:3]:"
    )

    print(
        K_rect
    )


# ============================================================
# SELECT REFERENCE FRAME
# ============================================================

print("\n" + "=" * 80)
print("2. TIME SYNCHRONISATION")
print("=" * 80)


reference_records = (
    camera_images[
        REFERENCE_CAMERA
    ]
)


reference_timestamp = (
    reference_records[
        REFERENCE_INDEX
    ][0]
)


print(
    "\nReference timestamp:",
    reference_timestamp
)


selected = {}


for camera in CAMERAS:

    records = camera_images[camera]

    if camera == REFERENCE_CAMERA:

        index = REFERENCE_INDEX

        timestamp = (
            records[index][0]
        )

        delta = 0

    else:

        (
            index,
            timestamp,
            delta
        ) = nearest_timestamp(
            reference_timestamp,
            records
        )

    if index is None:

        raise RuntimeError(
            f"No frame found for {camera}"
        )

    selected[camera] = {

        "index": index,

        "timestamp": timestamp,

        "delta": delta,

        "path": records[index][1],
    }

    print(
        f"\n{camera}"
    )

    print(
        "  index:",
        index
    )

    print(
        "  timestamp:",
        timestamp
    )

    print(
        "  delta:",
        f"{delta / 1e6:.3f} ms"
    )

    if delta > SYNC_TOLERANCE_NS:

        print(
            "  WARNING: outside nominal "
            "20 ms synchronization window"
        )


# ============================================================
# BUILD TF MATRICES
# ============================================================

print("\n" + "=" * 80)
print("3. TF STATIC MATRICES")
print("=" * 80)


T_tf = {}


for camera in CAMERAS:

    data = TF_STATIC[camera]

    T_tf[camera] = make_transform(
        data["translation"],
        data["quaternion"]
    )

    print_matrix(
        f"{camera}: TF",
        T_tf[camera]
    )

    R = T_tf[camera][:3, :3]

    print(
        "det(R) =",
        np.linalg.det(R)
    )

    print(
        "translation =",
        T_tf[camera][:3, 3]
    )


# ============================================================
# BUILD INVERSES
# ============================================================

T_tf_inv = {}


for camera in CAMERAS:

    T_tf_inv[camera] = (
        invert_transform(
            T_tf[camera]
        )
    )


print(
    "\nInverse matrices:"
)

for camera in CAMERAS:

    print_matrix(
        f"{camera}: inverse TF",
        T_tf_inv[camera]
    )


# ============================================================
# CAMERA -> CAMERA TRANSFORMS
# ============================================================
#
# These are useful because they remove the question of
# whether VELODYNE is actually the correct global reference.
#
# We construct:
#
#       CAM1 -> CAM2
#       CAM6 -> CAM2
#
# ============================================================

print("\n" + "=" * 80)
print("4. CAMERA-TO-CAMERA TRANSFORMS")
print("=" * 80)


T_cam_to_cam2 = {}


for camera in CAMERAS:

    if camera == "CAM2":

        T_cam_to_cam2[camera] = np.eye(
            4,
            dtype=np.float64
        )

    else:

        T_cam_to_cam2[camera] = (
            T_tf_inv["CAM2"]
            @
            T_tf[camera]
        )

    print_matrix(
        f"{camera} -> CAM2",
        T_cam_to_cam2[camera]
    )


# ============================================================
# LOAD MAPANYTHING
# ============================================================

print("\n" + "=" * 80)
print("5. LOAD MAPANYTHING")
print("=" * 80)


model = MapAnything.from_pretrained(
    "facebook/map-anything"
).to(device)


model.eval()


print(
    "Model loaded."
)


# ============================================================
# BUILD VIEWS
# ============================================================

print("\n" + "=" * 80)
print("6. BUILD THREE VIEWS")
print("=" * 80)


views = []

images = {}


for camera in CAMERAS:

    path = selected[camera]["path"]

    image = np.asarray(
        Image.open(
            path
        ).convert(
            "RGB"
        )
    )

    images[camera] = image

    print(
        f"\n{camera}"
    )

    print(
        "  image:",
        path
    )

    print(
        "  shape:",
        image.shape
    )

    print(
        "  intrinsics:"
    )

    print(
        camera_K[camera]
    )

    views.append({

        "img": image,

        "intrinsics":
            camera_K[camera].astype(
                np.float32
            ),
    })


# ============================================================
# PREPROCESS
# ============================================================

print("\n" + "=" * 80)
print("7. PREPROCESS")
print("=" * 80)


processed_views = (
    preprocess_inputs(
        views
    )
)


print(
    "Preprocessing successful."
)


# ============================================================
# MAPANYTHING INFERENCE
# ============================================================

print("\n" + "=" * 80)
print("8. MAPANYTHING MULTIVIEW INFERENCE")
print("=" * 80)


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


print(
    "Inference complete."
)


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

RAW_DIR = (
    OUTPUT_DIR /
    "00_raw_camera"
)

DIRECT_DIR = (
    OUTPUT_DIR /
    "01_tf_direct"
)

INVERSE_DIR = (
    OUTPUT_DIR /
    "02_tf_inverse"
)

ROT_DIR = (
    OUTPUT_DIR /
    "03_rotation_only"
)

INV_ROT_DIR = (
    OUTPUT_DIR /
    "04_inverse_rotation_only"
)

CAM2_DIR = (
    OUTPUT_DIR /
    "05_cam2_reference"
)


for directory in [
    RAW_DIR,
    DIRECT_DIR,
    INVERSE_DIR,
    ROT_DIR,
    INV_ROT_DIR,
    CAM2_DIR,
]:

    directory.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# EXTRACT RAW POINT CLOUDS
# ============================================================

print("\n" + "=" * 80)
print("9. EXTRACT RAW MAPANYTHING POINT CLOUDS")
print("=" * 80)


raw_points = {}
raw_colors = {}


for i, camera in enumerate(
    CAMERAS
):

    prediction = (
        predictions[i]
    )

    points = (
        prediction["pts3d_cam"]
        .detach()
        .cpu()
        .numpy()
    )

    if points.ndim == 4:

        points = points[0]

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

    image = images[camera]

    if image.shape[:2] != points.shape[:2]:

        image = np.asarray(
            Image.fromarray(
                image
            ).resize(
                (
                    points.shape[1],
                    points.shape[0]
                )
            )
        )

    valid = (
        mask
        &
        np.isfinite(
            points
        ).all(
            axis=-1
        )
    )

    points = points[
        valid
    ]

    colors = image[
        valid
    ]

    raw_points[camera] = points

    raw_colors[camera] = colors

    print(
        f"{camera}:",
        len(points),
        "valid points"
    )

    save_ply(
        RAW_DIR /
        f"{camera}_raw_camera.ply",
        points,
        colors
    )


# ============================================================
# TRANSFORMATION TESTS
# ============================================================

print("\n" + "=" * 80)
print("10. GENERATE TRANSFORMATION VARIANTS")
print("=" * 80)


merged = {

    "direct": [],

    "inverse": [],

    "rotation": [],

    "inverse_rotation": [],

    "cam2": [],
}


merged_colors = []


for camera in CAMERAS:

    points = raw_points[camera]

    colors = raw_colors[camera]

    R = T_tf[camera][:3, :3]

    t = T_tf[camera][:3, 3]


    # --------------------------------------------------------
    # A: DIRECT TF
    # --------------------------------------------------------
    #
    # p_v = T_tf p_c
    #

    points_direct = transform_points(
        points,
        T_tf[camera]
    )


    save_ply(
        DIRECT_DIR /
        f"{camera}_direct.ply",
        points_direct,
        colors
    )


    merged["direct"].append(
        points_direct
    )


    # --------------------------------------------------------
    # B: INVERSE TF
    # --------------------------------------------------------
    #
    # p_v = inv(T_tf) p_c
    #

    points_inverse = transform_points(
        points,
        T_tf_inv[camera]
    )


    save_ply(
        INVERSE_DIR /
        f"{camera}_inverse.ply",
        points_inverse,
        colors
    )


    merged["inverse"].append(
        points_inverse
    )


    # --------------------------------------------------------
    # C: ROTATION ONLY
    # --------------------------------------------------------

    points_rotation = (
        points @ R.T
    )


    save_ply(
        ROT_DIR /
        f"{camera}_rotation_only.ply",
        points_rotation,
        colors
    )


    merged["rotation"].append(
        points_rotation
    )


    # --------------------------------------------------------
    # D: INVERSE ROTATION ONLY
    # --------------------------------------------------------

    points_inverse_rotation = (
        points @ R
    )


    save_ply(
        INV_ROT_DIR /
        f"{camera}_inverse_rotation_only.ply",
        points_inverse_rotation,
        colors
    )


    merged["inverse_rotation"].append(
        points_inverse_rotation
    )


    # --------------------------------------------------------
    # E: CAM2 REFERENCE
    # --------------------------------------------------------
    #
    # CAM -> VELODYNE -> CAM2
    #

    points_cam2 = transform_points(
        points,
        T_cam_to_cam2[camera]
    )


    save_ply(
        CAM2_DIR /
        f"{camera}_in_cam2.ply",
        points_cam2,
        colors
    )


    merged["cam2"].append(
        points_cam2
    )


    merged_colors.append(
        colors
    )


    print(
        f"\n{camera}"
    )

    print(
        "  raw:",
        len(points)
    )

    print(
        "  direct:",
        len(points_direct)
    )

    print(
        "  inverse:",
        len(points_inverse)
    )

    print(
        "  rotation only:",
        len(points_rotation)
    )

    print(
        "  inverse rotation:",
        len(points_inverse_rotation)
    )

    print(
        "  CAM2:",
        len(points_cam2)
    )


# ============================================================
# MERGED VARIANTS
# ============================================================

print("\n" + "=" * 80)
print("11. MERGED POINT CLOUDS")
print("=" * 80)


for name, clouds in merged.items():

    points = np.concatenate(
        clouds,
        axis=0
    )

    colors = np.concatenate(
        merged_colors,
        axis=0
    )

    save_ply(
        OUTPUT_DIR /
        f"merged_{name}.ply",
        points,
        colors
    )

    print(
        f"{name:25s}:",
        len(points),
        "points"
    )


# ============================================================
# SAVE DEBUG MATRICES
# ============================================================

print("\n" + "=" * 80)
print("12. SAVE DEBUG DATA")
print("=" * 80)


np.savez(

    OUTPUT_DIR /
    "transformation_debug.npz",

    reference_timestamp=
        reference_timestamp,

    camera_timestamps=np.array(
        [
            selected[c]["timestamp"]
            for c in CAMERAS
        ],
        dtype=np.int64
    ),

    camera_indices=np.array(
        [
            selected[c]["index"]
            for c in CAMERAS
        ],
        dtype=np.int64
    ),

    K_rect=np.array(
        [
            camera_K[c]
            for c in CAMERAS
        ]
    ),

    P=np.array(
        [
            camera_P[c]
            for c in CAMERAS
        ]
    ),

    T_tf=np.array(
        [
            T_tf[c]
            for c in CAMERAS
        ]
    ),

    T_tf_inverse=np.array(
        [
            T_tf_inv[c]
            for c in CAMERAS
        ]
    ),

    T_cam_to_cam2=np.array(
        [
            T_cam_to_cam2[c]
            for c in CAMERAS
        ]
    ),
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)


print(
    "\nOutput directory:"
)

print(
    OUTPUT_DIR
)


print(
    "\nGenerated variants:"
)

print(
    "00_raw_camera/"
)

print(
    "01_tf_direct/"
)

print(
    "02_tf_inverse/"
)

print(
    "03_rotation_only/"
)

print(
    "04_inverse_rotation_only/"
)

print(
    "05_cam2_reference/"
)


print(
    "\nMerged:"
)

print(
    "merged_direct.ply"
)

print(
    "merged_inverse.ply"
)

print(
    "merged_rotation.ply"
)

print(
    "merged_inverse_rotation.ply"
)

print(
    "merged_cam2.ply"
)


print(
    "\nIMPORTANT:"
)

print(
    "No GPS poses were used."
)

print(
    "No LiDAR points were used."
)

print(
    "Rectified images use P[:3,:3]."
)

print(
    "All transformations are applied "
    "to raw pts3d_cam."
)

print(
    "The purpose is to determine which "
    "coordinate-frame convention is correct."
)

print("=" * 80)