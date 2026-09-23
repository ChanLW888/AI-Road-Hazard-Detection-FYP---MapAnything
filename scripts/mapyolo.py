import csv
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from mapanything.models import MapAnything
from mapanything.utils.image import preprocess_inputs
from ultralytics import YOLO

CLASS_COLORS = {
    0: (255, 0, 255),      # blocked_footpath_trip_hazard -> magenta
    1: (255, 165, 0),      # fallen_fence                 -> orange
    2: (144, 238, 144),    # grass                        -> light green
    3: (255, 215, 0),      # loose_trash                  -> yellow
    4: (139, 90, 43),      # mud                          -> brown
    5: (180, 0, 255),      # paint                        -> purple
    6: (80, 80, 80),       # road                         -> dark grey
    7: (211, 211, 211),    # sidewalk                     -> light grey
    8: (0, 180, 200),      # skip_bin                     -> teal
}
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

REFERENCE_INDEX = 1250

# We will print the real synchronization error.
# This is deliberately not used as a hard failure.
SYNC_TOLERANCE_NS = 500_000_000

# ============================================================
# YOLO SEMANTIC / INSTANCE SEGMENTATION
# ============================================================
#
# YOLO is a YOLO segmentation model ("segment" task).
#
# YOLO returns INSTANCE masks:
#
#   instance mask + class + confidence
#
# We convert those instance masks into one semantic pixel mask
# for each camera. If multiple instances overlap, the instance
# with the highest confidence wins at that pixel.
#
# The resulting semantic mask is then resized to the exact
# MapAnything point-grid resolution.
# ============================================================

YOLO_MODEL_PATH = Path(
    "/home/lcha0115/bt60_scratch/lcha_data/"
    "map-anything/weights/best.pt"
)

YOLO_CONFIDENCE_THRESHOLD = 0.25
YOLO_MASK_THRESHOLD = 0.50

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
# PLY WRITERS
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
        np.isfinite(points).all(axis=1)
    )

    points = points[valid]
    colors = colors[valid]

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
        f.write("end_header\n")

        for p, c in zip(points, colors):

            f.write(
                f"{p[0]:.6f} "
                f"{p[1]:.6f} "
                f"{p[2]:.6f} "
                f"{int(c[0])} "
                f"{int(c[1])} "
                f"{int(c[2])}\n"
            )


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

        # Original camera RGB
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")

        # Semantic class ID
        f.write("property int label\n")

        # Semantic visualization colour
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
# YOLO SEGMENTATION
# ============================================================

def load_yolo_model(device):

    print("\n" + "=" * 80)
    print("LOAD YOLO SEGMENTATION MODEL")
    print("=" * 80)

    print(
        "\nModel:",
        YOLO_MODEL_PATH
    )

    if not YOLO_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"YOLO model not found:\n{YOLO_MODEL_PATH}\n\n"
            "Set YOLO_MODEL_PATH to the location of your .pt file."
        )

    model = YOLO(
        str(YOLO_MODEL_PATH)
    )

    # Force a quick model/task sanity check.
    print(
        "Task:",
        getattr(model, "task", "unknown")
    )

    if getattr(model, "task", None) != "segment":
        raise RuntimeError(
            "The supplied YOLO model is not a YOLO segmentation model. "
            f"Detected task: {getattr(model, 'task', 'unknown')}"
        )

    names = model.names

    print(
        "Classes:",
        names
    )

    return model


def segment_image(
    image,
    model,
    device,
    output_size
):

    """
    Run YOLO segmentation on the original camera image.

    YOLO produces instance masks. These are converted into one
    semantic mask:

        labels[v,u] = class ID of the highest-confidence instance
                       covering pixel (v,u)

    output_size is (width, height) and should be exactly the
    spatial size of MapAnything's pts3d_cam grid.

    Returns:
        labels      : H x W integer class IDs; -1 = no YOLO mask
        confidence  : H x W confidence of selected instance
        id2label    : YOLO class-name dictionary
    """

    # YOLO handles the image preprocessing internally.
    results = model.predict(
        source=image,
        conf=YOLO_CONFIDENCE_THRESHOLD,
        verbose=False,
        device=0 if device.type == "cuda" else "cpu",
    )

    result = results[0]

    H_out = output_size[1]
    W_out = output_size[0]

    # Default: no semantic class at this pixel.
    labels = np.full(
        (H_out, W_out),
        -1,
        dtype=np.int32
    )

    confidence = np.zeros(
        (H_out, W_out),
        dtype=np.float32
    )

    if result.masks is None or len(result.masks.data) == 0:

        return (
            labels,
            confidence,
            result.names
        )

    # --------------------------------------------------------
    # YOLO INSTANCE MASKS
    # --------------------------------------------------------

    masks = result.masks.data

    if not torch.is_tensor(masks):
        masks = torch.as_tensor(masks)

    masks = masks.float()

    # Resize every instance mask directly to the MapAnything
    # point-grid resolution.
    masks = torch.nn.functional.interpolate(
        masks.unsqueeze(1),
        size=(H_out, W_out),
        mode="bilinear",
        align_corners=False
    ).squeeze(1)

    # --------------------------------------------------------
    # INSTANCE CLASSES + CONFIDENCES
    # --------------------------------------------------------

    class_ids = (
        result.boxes.cls
        .detach()
        .cpu()
        .numpy()
        .astype(np.int32)
    )

    instance_conf = (
        result.boxes.conf
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    # Keep only detections that have a corresponding mask.
    n = min(
        masks.shape[0],
        len(class_ids),
        len(instance_conf)
    )

    masks = masks[:n]
    class_ids = class_ids[:n]
    instance_conf = instance_conf[:n]

    # --------------------------------------------------------
    # OVERLAPPING INSTANCES
    #
    # For each pixel, use the highest-confidence instance
    # whose mask covers that pixel.
    # --------------------------------------------------------

    mask_binary = (
        masks >=
        YOLO_MASK_THRESHOLD
    )

    # Score each instance at each pixel.
    # Outside its mask the score is -1.
    scores = (
        masks *
        torch.from_numpy(
            instance_conf
        ).to(masks.device)[:, None, None]
    )

    scores = torch.where(
        mask_binary,
        scores,
        torch.full_like(
            scores,
            -1.0
        )
    )

    best_scores, best_indices = torch.max(
        scores,
        dim=0
    )

    covered = (
        best_scores >=
        YOLO_CONFIDENCE_THRESHOLD
    )

    best_indices_np = (
        best_indices
        .detach()
        .cpu()
        .numpy()
    )

    best_scores_np = (
        best_scores
        .detach()
        .cpu()
        .numpy()
    )

    # IMPORTANT:
    # Everything used for NumPy indexing must be on CPU.
    # `covered` was still a CUDA tensor in the previous version.
    covered_np = (
        best_scores_np >=
        YOLO_CONFIDENCE_THRESHOLD
    )

    # Convert YOLO class IDs to a NumPy array as well.
    if torch.is_tensor(class_ids):
        class_ids_np = (
            class_ids
            .detach()
            .cpu()
            .numpy()
            .astype(np.int32)
        )
    else:
        class_ids_np = np.asarray(
            class_ids,
            dtype=np.int32
        )

    labels[covered_np] = class_ids_np[
        best_indices_np[covered_np]
    ]

    confidence[covered_np] = (
        best_scores_np[covered_np]
    )

    return (
        labels,
        confidence,
        result.names
    )


def make_class_colors(
    class_names
):

    """
    Fixed semantic colours.

    The YOLO class IDs are used as the keys so the colour
    convention is stable across every run.
    """

    num_classes = len(
        class_names
    )

    colors = np.zeros(
        (num_classes, 3),
        dtype=np.uint8
    )

    for label in range(
        num_classes
    ):

        if label in CLASS_COLORS:

            colors[label] = (
                CLASS_COLORS[label]
            )

        else:

            # Fallback for an unexpected class.
            colors[label] = (
                128,
                128,
                128
            )

    return colors


def save_segmentation_visualization(
    path,
    labels,
    class_colors
):

    H, W = labels.shape

    colored = np.zeros(
        (H, W, 3),
        dtype=np.uint8
    )

    valid = (
        labels >= 0
    )

    colored[valid] = class_colors[
        labels[valid]
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
print("MAPANYTHING + YOLO 3D SEMANTIC VOXEL PIPELINE")
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
# LOAD YOLO SEGMENTATION
# ============================================================

yolo_model = load_yolo_model(
    device
)

YOLO_CLASS_NAMES = yolo_model.names

class_colors = make_class_colors(
    YOLO_CLASS_NAMES
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
# OUTPUT DIRECTORIES
# ============================================================

RAW_DIR = (
    OUTPUT_DIR /
    "00_raw_camera"
)

SEMANTIC_DIR = (
    OUTPUT_DIR /
    "06_semantic_3d"
)


for directory in [
    RAW_DIR,
    SEMANTIC_DIR,
]:

    directory.mkdir(
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
        SEMANTIC_DIR /
        f"{camera}_raw.png"
    )

    # --------------------------------------------------------
    # RUN YOLO SEGMENTATION
    #
    # YOLO runs on the original image and returns instance
    # masks. We convert them into one semantic mask and resize
    # it directly to the MapAnything point-grid resolution.
    #
    # Therefore:
    #
    # labels[v,u] <-> pts3d_cam[v,u]
    #
    # --------------------------------------------------------

    segmentation_start = time.perf_counter()

    labels, confidence, id2label = segment_image(
        image=image,
        model=yolo_model,
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
    # It excludes the YOLO neural-network inference time.
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
        (labels >= 0)
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
    if YOLO_CONFIDENCE_THRESHOLD > 0:

        valid &= (
            confidence >=
            YOLO_CONFIDENCE_THRESHOLD
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
    # SAVE RAW CAMERA CLOUD
    # --------------------------------------------------------

    save_ply(
        RAW_DIR /
        f"{camera}_raw_camera.ply",
        points,
        colors
    )

    # --------------------------------------------------------
    # SAVE SEMANTIC CAMERA CLOUD
    #
    # Coordinates remain in MapAnything camera coordinates
    # here.
    #
    # Each point now has:
    #
    # XYZ
    # original RGB
    # semantic label
    # semantic RGB
    #
    # --------------------------------------------------------

    save_semantic_ply(
        SEMANTIC_DIR /
        f"{camera}_semantic_camera.ply",
        points,
        colors,
        labels_flat,
        sem_colors
    )

    # --------------------------------------------------------
    # SAVE SEGMENTATION FOR DEBUGGING
    # --------------------------------------------------------

    save_segmentation_visualization(
        SEMANTIC_DIR /
        f"{camera}_segmentation.png",
        labels,
        class_colors
    )

    np.save(
        SEMANTIC_DIR /
        f"{camera}_labels.npy",
        labels
    )

    np.save(
        SEMANTIC_DIR /
        f"{camera}_confidence.npy",
        confidence
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
#   YOLO class
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
    # SAVE DIRECT PER-CAMERA SEMANTIC POINT CLOUD
    # --------------------------------------------------------

    save_semantic_ply(
        SEMANTIC_DIR /
        f"{camera}_semantic_direct.ply",
        points_direct,
        colors,
        labels,
        sem_colors
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
# SAVE NUMPY VOXEL DATA
# ============================================================

np.save(
    OUTPUT_DIR /
    "semantic_voxel_centres.npy",
    voxel_centres
)

np.save(
    OUTPUT_DIR /
    "semantic_voxel_labels.npy",
    voxel_labels
)

np.save(
    OUTPUT_DIR /
    "semantic_voxel_confidence.npy",
    voxel_confidence
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

    name = YOLO_CLASS_NAMES.get(
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
# SAVE ORIGINAL MERGED POINT DATA
# ============================================================

np.save(
    OUTPUT_DIR /
    "semantic_3d_points_direct.npy",
    merged_points
)

np.save(
    OUTPUT_DIR /
    "semantic_3d_labels_direct.npy",
    merged_labels
)

np.save(
    OUTPUT_DIR /
    "semantic_3d_confidence_direct.npy",
    merged_confidence
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