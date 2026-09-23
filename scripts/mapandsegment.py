import csv
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from mapanything.models import MapAnything
from mapanything.utils.image import preprocess_inputs

from transformers import (
    AutoImageProcessor,
    SegformerForSemanticSegmentation,
)

CLASS_COLORS = {
    0:  (128, 128, 128),   # wall       -> grey
    1:  (70, 130, 180),    # building   -> steel blue
    2:  (135, 206, 235),   # sky        -> sky blue
    3:  (180, 180, 180),   # floor      -> light grey
    4:  (34, 139, 34),     # tree       -> forest green
    6:  (80, 80, 80),      # road       -> dark grey
    9:  (144, 238, 144),   # grass      -> light green
    11: (211, 211, 211),   # sidewalk   -> light grey
    13: (139, 90, 43),     # earth      -> brown
    17: (0, 200, 0),       # plant      -> bright green
    20: (220, 50, 50),     # car        -> red
    21: (0, 200, 220),     # water      -> cyan
    32: (255, 165, 0),     # fence      -> orange
    43: (255, 215, 0),     # signboard  -> yellow
    72: (0, 100, 0),       # palm       -> dark green
}
# ============================================================
# CONFIGURATION
# ============================================================

BAG_ROOT = Path(
    "/home/lcha0115/bt60_scratch/lcha_data/"
    "map-anything/personal_data/"
    "camera_lidar_test_09"
)

EXTRACTED = BAG_ROOT / "extracted_dark"

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

REFERENCE_INDEX = 250

# We will print the real synchronization error.
# This is deliberately not used as a hard failure.
SYNC_TOLERANCE_NS = 500_000_000

# ============================================================
# SEMANTIC SEGMENTATION
# ============================================================
# SegFormer is used only for 2D semantic segmentation.
# Its output is resized to the exact MapAnything point-grid size
# before labels are attached to 3D points.
SEGMENTATION_MODEL_NAME = (
    "nvidia/segformer-b0-finetuned-ade-512-512"
)

SEGMENTATION_CONFIDENCE_THRESHOLD = 0.0

# ============================================================
# VOXELISATION
# ============================================================
# Points are first transformed into the common DIRECT frame.
# Then points are grouped into 3D voxels. Each voxel receives
# the majority semantic class of the points inside it.
#
# Smaller value = finer voxel grid.
# Larger value = smoother / more consolidated segmentation.
VOXEL_SIZE = 0.05  # metres


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
# TIMING
# ============================================================

timings = {
    "mapanything_total": 0.0,
    "segmentation": {},
    "point_cloud_generation": {},
    "mask_projection": {},
    "voxelisation": 0.0,
}


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
# SEMANTIC PLY WRITER
# ============================================================

def save_semantic_ply(
    path,
    points,
    rgb_colors,
    labels,
    semantic_colors
):

    points = np.asarray(
        points,
        dtype=np.float32
    )

    rgb_colors = np.asarray(
        rgb_colors,
        dtype=np.uint8
    )

    labels = np.asarray(
        labels,
        dtype=np.int32
    )

    semantic_colors = np.asarray(
        semantic_colors,
        dtype=np.uint8
    )

    valid = (
        np.isfinite(points).all(axis=1)
        &
        np.isfinite(labels)
    )

    points = points[valid]
    rgb_colors = rgb_colors[valid]
    labels = labels[valid]
    semantic_colors = semantic_colors[valid]

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(path, "w") as f:

        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")

        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")

        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")

        f.write("property int label\n")

        f.write("property uchar semantic_red\n")
        f.write("property uchar semantic_green\n")
        f.write("property uchar semantic_blue\n")

        f.write("end_header\n")

        for p, c, label, sc in zip(
            points,
            rgb_colors,
            labels,
            semantic_colors
        ):

            f.write(
                f"{p[0]:.6f} "
                f"{p[1]:.6f} "
                f"{p[2]:.6f} "
                f"{int(c[0])} "
                f"{int(c[1])} "
                f"{int(c[2])} "
                f"{int(label)} "
                f"{int(sc[0])} "
                f"{int(sc[1])} "
                f"{int(sc[2])}\n"
            )


# ============================================================
# SEGFORMER
# ============================================================

def load_segmentation_model(device):

    print("\n" + "=" * 80)
    print("LOAD SEGFORMER")
    print("=" * 80)

    print(
        "\nModel:",
        SEGMENTATION_MODEL_NAME
    )

    processor = AutoImageProcessor.from_pretrained(
        SEGMENTATION_MODEL_NAME
    )

    model = (
        SegformerForSemanticSegmentation
        .from_pretrained(
            SEGMENTATION_MODEL_NAME
        )
    )

    model = model.to(device)
    model.eval()

    print("SegFormer loaded.")

    return processor, model


def segment_image(
    image,
    processor,
    model,
    device,
    output_size
):

    """
    Run SegFormer on the ORIGINAL camera image.

    output_size is (width, height) and should be exactly the
    spatial size of MapAnything's pts3d_cam grid.

    Returns:
        labels      : H x W integer class IDs
        confidence  : H x W max softmax probability
        id2label    : class-name dictionary
    """

    image_pil = Image.fromarray(
        image
    ).convert("RGB")

    inputs = processor(
        images=image_pil,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model(
            **inputs
        )

    logits = outputs.logits

    # First resize to the MapAnything point-grid resolution.
    logits = torch.nn.functional.interpolate(
        logits,
        size=(
            output_size[1],
            output_size[0]
        ),
        mode="bilinear",
        align_corners=False
    )

    probabilities = torch.softmax(
        logits,
        dim=1
    )

    confidence, labels = probabilities.max(
        dim=1
    )

    labels = (
        labels[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.int32)
    )

    confidence = (
        confidence[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    return (
        labels,
        confidence,
        model.config.id2label
    )


def make_class_colors(num_classes):

    """
    Deterministic colours so the same semantic class always
    has the same colour.
    """

    rng = np.random.default_rng(42)

    colors = rng.integers(
        0,
        256,
        size=(num_classes, 3),
        dtype=np.uint8
    )

    return colors


def save_segmentation_visualization(
    path,
    labels,
    class_colors
):

    colored = class_colors[
        labels
    ]

    Image.fromarray(
        colored
    ).save(path)


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
    "MapAnything model loaded."
)


# ============================================================
# LOAD SEGFORMER
# ============================================================

seg_processor, seg_model = load_segmentation_model(
    device
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


mapanything_start = time.perf_counter()

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

# Synchronise CUDA before stopping the timer so GPU work is included.
if device.type == "cuda":
    torch.cuda.synchronize()

timings["mapanything_total"] = (
    time.perf_counter() -
    mapanything_start
)

print(
    "Inference complete."
)

print(
    f"MapAnything inference: "
    f"{timings['mapanything_total']:.3f} s"
)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================
#
# Only these outputs are saved:
#
#   semantic_3d_merged_direct.ply
#   semantic_voxels_direct.ply
#   CAM1_raw.png / CAM1_segmentation.png
#   CAM2_raw.png / CAM2_segmentation.png
#   CAM6_raw.png / CAM6_segmentation.png
#
# No per-camera PLYs, NumPy arrays, TF diagnostic clouds,
# confidence files, or intermediate files are saved.
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# EXTRACT RAW POINT CLOUDS + PROJECT 2D SEMANTIC MASKS
# ============================================================

print("\n" + "=" * 80)
print("9. EXTRACT MAPANYTHING POINTS + PROJECT 2D SEGMENTATION")
print("=" * 80)

raw_points = {}
raw_colors = {}

semantic_labels = {}
semantic_confidence = {}
semantic_colors = {}

num_seg_classes = seg_model.config.num_labels

class_colors = np.zeros(
    (num_seg_classes, 3),
    dtype=np.uint8
)

for label, color in CLASS_COLORS.items():
    class_colors[label] = color

for i, camera in enumerate(CAMERAS):

    prediction = predictions[i]

    # --------------------------------------------------------
    # POINT CLOUD GENERATION / EXTRACTION TIMING
    # --------------------------------------------------------

    pointcloud_start = time.perf_counter()

    # --------------------------------------------------------
    # MAPANYTHING 3D POINT GRID
    # --------------------------------------------------------

    points_grid = (
        prediction["pts3d_cam"]
        .detach()
        .cpu()
        .numpy()
    )

    if points_grid.ndim == 4:
        points_grid = points_grid[0]

    # points_grid is H_map x W_map x 3

    H_map, W_map = points_grid.shape[:2]

    # This measures extracting the MapAnything 3D grid from the
    # prediction, not the neural-network inference itself.
    timings["point_cloud_generation"][camera] = (
        time.perf_counter() -
        pointcloud_start
    )

    print(
        f"\n{camera}"
    )

    print(
        "  MapAnything point grid:",
        points_grid.shape
    )

    print(
        f"  Point-cloud extraction: "
        f"{timings['point_cloud_generation'][camera]:.4f} s"
    )

    # --------------------------------------------------------
    # MAPANYTHING VALIDITY MASK
    # --------------------------------------------------------

    map_mask = (
        prediction["mask"]
        .detach()
        .cpu()
        .numpy()
    )

    if map_mask.ndim == 4:
        map_mask = map_mask[0]

    if map_mask.ndim == 3:
        map_mask = map_mask[..., 0]

    map_mask = map_mask.astype(bool)

    # --------------------------------------------------------
    # ORIGINAL IMAGE
    # --------------------------------------------------------

    image = images[camera]

    # Save the ORIGINAL camera image into the semantic-3D
    # output directory so each segmentation/point-cloud result
    # has its source image alongside it.
    Image.fromarray(
        image
    ).save(
        OUTPUT_DIR /
        f"{camera}_raw.png"
    )

    # --------------------------------------------------------
    # RUN SEGFORMER
    #
    # IMPORTANT:
    #
    # SegFormer is run on the original image.
    # Its logits are then resized directly to the exact
    # MapAnything point-grid resolution.
    #
    # Therefore:
    #
    # labels[v,u] <-> pts3d_cam[v,u]
    #
    # --------------------------------------------------------

    segmentation_start = time.perf_counter()

    labels, confidence, id2label = segment_image(
        image=image,
        processor=seg_processor,
        model=seg_model,
        device=device,
        output_size=(W_map, H_map)
    )

    if device.type == "cuda":
        torch.cuda.synchronize()

    timings["segmentation"][camera] = (
        time.perf_counter() -
        segmentation_start
    )

    print(
        f"  Segmentation: "
        f"{timings['segmentation'][camera]:.4f} s"
    )

    print(
        "  Segmentation grid:",
        labels.shape
    )

    # Sanity check
    if labels.shape != points_grid.shape[:2]:

        raise RuntimeError(
            f"{camera}: segmentation shape "
            f"{labels.shape} does not match "
            f"MapAnything point grid "
            f"{points_grid.shape[:2]}"
        )

    # --------------------------------------------------------
    # MASK PROJECTION TIMING
    # --------------------------------------------------------
    #
    # This is the actual 2D -> 3D label attachment:
    #
    #   labels[v,u] + pts3d_cam[v,u]
    #
    # It excludes the SegFormer neural-network inference time.
    # --------------------------------------------------------

    projection_start = time.perf_counter()

    # --------------------------------------------------------
    # RESIZE RGB IMAGE TO POINT GRID
    # --------------------------------------------------------

    if image.shape[:2] != (H_map, W_map):

        image_grid = np.asarray(
            Image.fromarray(
                image
            ).resize(
                (
                    W_map,
                    H_map
                )
            )
        )

    else:

        image_grid = image

    # --------------------------------------------------------
    # VALID POINTS
    # --------------------------------------------------------

    valid = (
        map_mask
        &
        np.isfinite(
            points_grid
        ).all(axis=-1)
        &
        np.isfinite(
            confidence
        )
    )

    # Optional semantic confidence filtering
    if SEGMENTATION_CONFIDENCE_THRESHOLD > 0:

        valid &= (
            confidence >=
            SEGMENTATION_CONFIDENCE_THRESHOLD
        )

    # --------------------------------------------------------
    # PROJECT 2D LABELS ONTO 3D POINTS
    # --------------------------------------------------------

    points = points_grid[valid]

    colors = image_grid[valid]

    labels_flat = labels[valid]

    confidence_flat = confidence[valid]

    sem_colors = class_colors[
        labels_flat
    ]

    timings["mask_projection"][camera] = (
        time.perf_counter() -
        projection_start
    )

    print(
        f"  Mask projection: "
        f"{timings['mask_projection'][camera]:.4f} s"
    )

    raw_points[camera] = points
    raw_colors[camera] = colors

    semantic_labels[camera] = labels_flat
    semantic_confidence[camera] = confidence_flat
    semantic_colors[camera] = sem_colors

    # --------------------------------------------------------
    # PRINT SEMANTIC DISTRIBUTION
    # --------------------------------------------------------

    unique_labels, label_counts = np.unique(
        labels_flat,
        return_counts=True
    )

    print(
        "  Valid 3D points:",
        len(points)
    )

    print(
        "  Semantic classes:"
    )

    for cls, count in zip(
        unique_labels,
        label_counts
    ):

        percentage = (
            count /
            len(labels_flat) *
            100
        )

        name = id2label.get(
            int(cls),
            "unknown"
        )

        print(
            f"    {int(cls):3d} "
            f"{name:30s} "
            f"{percentage:6.2f}%"
        )

    # --------------------------------------------------------
    # SAVE SEGMENTATION FOR DEBUGGING
    # --------------------------------------------------------

    save_segmentation_visualization(
        OUTPUT_DIR /
        f"{camera}_segmentation.png",
        labels,
        class_colors
    )


# ============================================================
# DIRECT TRANSFORM + SEMANTIC 3D VOXELS
# ============================================================
#
# You confirmed that the DIRECT TF transform is the correct one.
#
# For every camera:
#
#   image pixel (u,v)
#          ↓
#   SegFormer class
#          +
#   MapAnything pts3d_cam[u,v]
#          ↓
#   DIRECT camera -> common frame transform
#          ↓
#   3D point + semantic label
#
# Then all three cameras are merged and voxelised.
#
# Each voxel receives the MAJORITY semantic class of the
# 3D points falling inside that voxel.
# ============================================================

print("\n" + "=" * 80)
print("10. DIRECT TRANSFORM + SEMANTIC 3D VOXELS")
print("=" * 80)

merged_direct_points = []
merged_direct_colors = []
merged_direct_labels = []
merged_direct_semantic_colors = []
merged_direct_confidence = []


for camera in CAMERAS:

    points = raw_points[camera]
    colors = raw_colors[camera]

    labels = semantic_labels[camera]
    confidence = semantic_confidence[camera]
    sem_colors = semantic_colors[camera]

    # --------------------------------------------------------
    # DIRECT CAMERA -> COMMON FRAME TRANSFORM
    # --------------------------------------------------------

    points_direct = transform_points(
        points,
        T_tf[camera]
    )

    # --------------------------------------------------------
    # COLLECT FOR THREE-CAMERA MERGE
    # --------------------------------------------------------

    merged_direct_points.append(
        points_direct
    )

    merged_direct_colors.append(
        colors
    )

    merged_direct_labels.append(
        labels
    )

    merged_direct_semantic_colors.append(
        sem_colors
    )

    merged_direct_confidence.append(
        confidence
    )

    print(
        f"\n{camera}: "
        f"{len(points_direct)} semantic 3D points"
    )


# ============================================================
# MERGE THREE CAMERAS
# ============================================================

merged_points = np.concatenate(
    merged_direct_points,
    axis=0
)

merged_colors = np.concatenate(
    merged_direct_colors,
    axis=0
)

merged_labels = np.concatenate(
    merged_direct_labels,
    axis=0
)

merged_semantic_colors = np.concatenate(
    merged_direct_semantic_colors,
    axis=0
)

merged_confidence = np.concatenate(
    merged_direct_confidence,
    axis=0
)


print(
    "\nTotal merged 3D points:",
    len(merged_points)
)


# ============================================================
# SAVE MERGED SEMANTIC POINT CLOUD
# ============================================================

save_semantic_ply(
    OUTPUT_DIR /
    "semantic_3d_merged_direct.ply",
    merged_points,
    merged_colors,
    merged_labels,
    merged_semantic_colors
)


# ============================================================
# VOXELISE
# ============================================================

print("\n" + "=" * 80)
print("11. VOXELISE SEMANTIC POINT CLOUD")
print("=" * 80)

print(
    "Voxel size:",
    VOXEL_SIZE,
    "m"
)

voxel_start = time.perf_counter()


# ------------------------------------------------------------
# Remove invalid points
# ------------------------------------------------------------

valid = (
    np.isfinite(
        merged_points
    ).all(axis=1)
)

points_v = merged_points[valid]
labels_v = merged_labels[valid]
confidence_v = merged_confidence[valid]


# ------------------------------------------------------------
# Convert each point to an integer voxel coordinate
#
# Example:
#
# point = [1.23, 2.41, 0.17]
# voxel size = 0.05
#
# voxel index =
# [24, 48, 3]
# ------------------------------------------------------------

voxel_indices = np.floor(
    points_v / VOXEL_SIZE
).astype(
    np.int64
)


# ------------------------------------------------------------
# Find unique voxels
# ------------------------------------------------------------

unique_voxels, inverse = np.unique(
    voxel_indices,
    axis=0,
    return_inverse=True
)


num_voxels = len(
    unique_voxels
)

print(
    "Unique voxels:",
    num_voxels
)


# ============================================================
# MAJORITY-VOTE SEMANTIC LABEL PER VOXEL
# ============================================================
#
# If a voxel contains:
#
#   road road road grass
#
# it becomes:
#
#   road
#
# This prevents a single incorrectly labelled point from
# determining the entire voxel.
# ============================================================

voxel_labels = np.zeros(
    num_voxels,
    dtype=np.int32
)

voxel_confidence = np.zeros(
    num_voxels,
    dtype=np.float32
)


for voxel_id in range(
    num_voxels
):

    point_ids = np.where(
        inverse == voxel_id
    )[0]

    labels_here = labels_v[
        point_ids
    ]

    confidence_here = confidence_v[
        point_ids
    ]

    # Count semantic classes
    classes_here, counts_here = np.unique(
        labels_here,
        return_counts=True
    )

    # Majority class
    majority_index = np.argmax(
        counts_here
    )

    majority_class = int(
        classes_here[
            majority_index
        ]
    )

    voxel_labels[
        voxel_id
    ] = majority_class

    # Mean confidence of points belonging to
    # the selected class
    selected = (
        labels_here ==
        majority_class
    )

    voxel_confidence[
        voxel_id
    ] = confidence_here[
        selected
    ].mean()


# ============================================================
# VOXEL CENTRES
# ============================================================
#
# unique_voxels contains the integer voxel coordinates.
#
# Convert them back to metric coordinates using the centre
# of each voxel.
# ============================================================

voxel_centres = (
    (
        unique_voxels.astype(
            np.float64
        )
        + 0.5
    )
    *
    VOXEL_SIZE
)


# ============================================================
# VOXEL SEMANTIC COLOURS
# ============================================================

voxel_semantic_colors = class_colors[
    voxel_labels
]


# ============================================================
# SAVE VOXEL CLOUD
# ============================================================
#
# Each row now represents ONE VOXEL rather than one original
# MapAnything point.
#
# PLY fields:
#
#   x y z
#   RGB
#   semantic label
#   semantic RGB
# ============================================================

voxel_rgb = voxel_semantic_colors.copy()

save_semantic_ply(
    OUTPUT_DIR /
    "semantic_voxels_direct.ply",
    voxel_centres,
    voxel_rgb,
    voxel_labels,
    voxel_semantic_colors
)


# ============================================================
# VOXEL TIMING
# ============================================================

timings["voxelisation"] = (
    time.perf_counter() -
    voxel_start
)

print(
    f"\nVoxelisation: "
    f"{timings['voxelisation']:.4f} s"
)


# ============================================================
# VOXEL CLASS SUMMARY
# ============================================================

print(
    "\nVoxel semantic classes:"
)

unique_labels, label_counts = np.unique(
    voxel_labels,
    return_counts=True
)

for cls, count in zip(
    unique_labels,
    label_counts
):

    name = seg_model.config.id2label.get(
        int(cls),
        "unknown"
    )

    percentage = (
        count /
        len(voxel_labels) *
        100
    )

    print(
        f"  {int(cls):3d} "
        f"{name:30s} "
        f"{percentage:6.2f}%"
    )


print(
    "\nSaved voxel cloud:"
)

print(
    OUTPUT_DIR /
    "semantic_voxels_direct.ply"
)


# ============================================================
# NO OTHER TRANSFORMATION DIAGNOSTICS
# ============================================================
#
# Direct TF is the selected/correct convention.
# No inverse, rotation-only, inverse-rotation, or CAM2
# diagnostic clouds are generated.
# ============================================================


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 80)
print("SEMANTIC 3D / VOXELISATION COMPLETE")
print("=" * 80)

print(
    "\nOutput directory:"
)

print(
    OUTPUT_DIR
)


# ============================================================
# TIMING SUMMARY
# ============================================================

print("\n" + "-" * 80)
print("TIMING SUMMARY")
print("-" * 80)

print(
    f"MapAnything 3-camera inference: "
    f"{timings['mapanything_total']:.3f} s"
)

print(
    "\nPer-camera:"
)

for camera in CAMERAS:

    print(
        f"  {camera}:"
    )

    print(
        f"    Point-cloud extraction: "
        f"{timings['point_cloud_generation'][camera]:.4f} s"
    )

    print(
        f"    Segmentation:          "
        f"{timings['segmentation'][camera]:.4f} s"
    )

    print(
        f"    Mask projection:       "
        f"{timings['mask_projection'][camera]:.4f} s"
    )

print(
    f"\nVoxelisation: "
    f"{timings['voxelisation']:.4f} s"
)

seg_total = sum(
    timings["segmentation"].values()
)

pc_total = sum(
    timings["point_cloud_generation"].values()
)

projection_total = sum(
    timings["mask_projection"].values()
)

print(
    "\nStage totals:"
)

print(
    f"  Segmentation total:       {seg_total:.3f} s"
)

print(
    f"  Point-cloud extraction:   {pc_total:.3f} s"
)

print(
    f"  Mask projection total:    {projection_total:.3f} s"
)

print(
    f"  Voxelisation:             "
    f"{timings['voxelisation']:.3f} s"
)


# ============================================================
# OUTPUTS
# ============================================================

print(
    "\nMain voxel output:"
)

print(
    OUTPUT_DIR /
    "semantic_voxels_direct.ply"
)

print(
    "\nMerged semantic point cloud:"
)

print(
    OUTPUT_DIR /
    "semantic_3d_merged_direct.ply"
)

print(
    "\nVoxel size:",
    VOXEL_SIZE,
    "m"
)

print(
    "\nSemantic 3D points:",
    len(merged_points)
)

print(
    "Semantic voxels:",
    len(voxel_centres)
)

print(
    "\nEach voxel receives the majority semantic "
    "class of the MapAnything 3D points inside it."
)

print(
    "Only the DIRECT camera -> common-frame "
    "TF transform is used."
)

print("=" * 80)