import csv
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from mapanything.models import MapAnything
from mapanything.utils.image import preprocess_inputs
from rfdetr import RFDETRSegSmall

# RF-DETR-Seg-Small class colours.
# These are deliberately HIGH-CONTRAST colours so the semantic regions
# remain obvious against the original camera image.
CLASS_COLORS = {
    0: (255, 0, 255),      # fallen_fence    -> bright magenta
    1: (0, 255, 255),      # loose_trash     -> cyan
    2: (255, 80, 0),       # mud_spill       -> bright orange-red
    3: (0, 255, 0),        # public_greenway -> bright green
    4: (0, 120, 255),      # sidewalk        -> bright blue
}
# ============================================================
# CONFIGURATION
# ============================================================

BAG_ROOT = Path(
    "/home/lcha0115/bt60_scratch/lcha_data/"
    "map-anything/personal_data/"
    "camera_lidar_test_101"
)

EXTRACTED = BAG_ROOT / "extracted"

OUTPUT_DIR = (
    BAG_ROOT /
    "map_cam2"
)

CAMERAS = [
    "CAM2"
]

IMAGE_STREAM = "rect"

REFERENCE_CAMERA = "CAM2"

REFERENCE_INDEX = 7000

# We will print the real synchronization error.
# This is deliberately not used as a hard failure.
SYNC_TOLERANCE_NS = 500_000_000

# ============================================================
# RF-DETR SEMANTIC / INSTANCE SEGMENTATION
# ============================================================
#
# RF-DETR-Seg-Small is an INSTANCE segmentation model.
#
# RF-DETR returns, for each detection:
#
#   instance mask + class + confidence
#
# We convert those instance masks into one semantic pixel mask
# for each camera. If multiple instances overlap, the instance
# with the highest confidence wins at that pixel.
#
# The resulting semantic mask is resized to the exact
# MapAnything point-grid resolution.
#
# RF-DETR-Seg-Small uses a 384x384 native inference resolution,
# but its returned masks are postprocessed back to the source
# image size. We therefore perform the final resize ourselves
# so that:
#
#     labels[v,u] <-> pts3d_cam[v,u]
# ============================================================

RFDETR_MODEL_PATH = Path(
"/home/lcha0115/bt60_scratch/lcha_data/map-anything/personal_data/2D-Seg-Road-Nissan.v6i.coco-segmentation/outputs/checkpoint_best_total.pth"
)

RFDETR_CONFIDENCE_THRESHOLD = 0.25
RFDETR_MASK_THRESHOLD = 0.50

# Transparency of semantic highlights over the original RGB image.
# 0.0 = original image only
# 1.0 = semantic colours only
SEMANTIC_OVERLAY_ALPHA = 0.65
SEMANTIC_LABEL_MIN_AREA = 300  # ignore tiny fragments when placing labels

# ============================================================
# VOXELISATION
# ============================================================
# Points are first transformed into the common DIRECT frame using the supplied TF.
# Then points are grouped into 3D voxels. Each voxel receives
# the majority semantic class of the points inside it.
#
# Smaller value = finer voxel grid.
# Larger value = smoother / more consolidated segmentation.
VOXEL_SIZE = 0.05  # metres


# ============================================================
# TF STATIC
#
# SUPPLIED TRANSFORM CONVENTION
#
# The matrices below are applied DIRECTLY to MapAnything pts3d_cam.
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
# RF-DETR SEGMENTATION
# ============================================================

def load_rfdetr_model(device):
    """
    Load the RF-DETR Segmentation Small model.

    If RFDETR_MODEL_PATH points to a fine-tuned RF-DETR checkpoint,
    from_checkpoint() automatically reconstructs the appropriate
    architecture and class count from the checkpoint metadata/weights.

    If the checkpoint is a standard official RF-DETR-Seg-Small
    checkpoint, the same loader can also be used when its filename
    identifies the architecture.
    """
    print("\n" + "=" * 80)
    print("LOAD RF-DETR SEGMENTATION SMALL MODEL")
    print("=" * 80)

    print("\nModel checkpoint:")
    print(RFDETR_MODEL_PATH)

    if not RFDETR_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"RF-DETR model not found:\n{RFDETR_MODEL_PATH}\n\n"
            "Set RFDETR_MODEL_PATH to your RF-DETR-Seg-Small "
            "checkpoint (.pth)."
        )

    # from_checkpoint() is preferable for a fine-tuned checkpoint:
    # it reconstructs the model architecture and class count from
    # the checkpoint rather than assuming COCO's 90 classes.
    model = RFDETRSegSmall.from_checkpoint(
        str(RFDETR_MODEL_PATH)
    )

    # RF-DETR internally manages its inference device. Move the
    # underlying PyTorch model when possible for consistency with
    # the rest of this pipeline.
    try:
        model.model.to(device)
    except AttributeError:
        pass

    try:
        model.optimize_for_inference()
        print("Inference optimisation enabled.")
    except Exception as exc:
        print(
            "Inference optimisation unavailable; "
            "continuing without it:",
            exc
        )

    print("RF-DETR model loaded.")

    try:
        print("Classes:", model.class_names)
    except Exception:
        print("Classes: unavailable from model.class_names")

    return model


def _get_rfdetr_class_names(model, detections):
    """
    Return a {class_id: class_name} mapping.

    Prefer the model's class_names property. If that is not
    available, use RF-DETR's per-detection class_name data.
    """
    try:
        names = model.class_names

        if isinstance(names, dict):
            return {
                int(k): str(v)
                for k, v in names.items()
            }

        if isinstance(names, (list, tuple)):
            return {
                i: str(name)
                for i, name in enumerate(names)
            }
    except Exception:
        pass

    # Newer RF-DETR versions expose class_name directly in
    # detections.data.
    try:
        names = detections.data["class_name"]
        class_ids = detections.class_id

        mapping = {}
        for cid, name in zip(class_ids, names):
            mapping[int(cid)] = str(name)

        return mapping
    except Exception:
        return {}


def _as_numpy(value):
    """Convert torch/numpy-like values to NumPy."""
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()

    return np.asarray(value)


def _resize_instance_masks(masks, output_size):
    """
    Resize RF-DETR instance masks to (H_out, W_out).

    RF-DETR normally returns masks already upsampled to the
    original image size. This function also handles native/
    lower-resolution masks robustly.
    """
    H_out = int(output_size[1])
    W_out = int(output_size[0])

    masks = _as_numpy(masks)

    # Expected forms:
    #   N x H x W
    #   N x 1 x H x W
    if masks.ndim == 4:
        if masks.shape[1] != 1:
            raise ValueError(
                f"Unexpected RF-DETR mask shape: {masks.shape}"
            )

        masks = masks[:, 0]

    if masks.ndim != 3:
        raise ValueError(
            f"Unexpected RF-DETR mask shape: {masks.shape}"
        )

    if masks.shape[1:] == (H_out, W_out):
        return masks.astype(np.float32)

    # Resize all masks together with PyTorch interpolation.
    masks_t = torch.from_numpy(
        masks.astype(np.float32)
    )

    masks_t = torch.nn.functional.interpolate(
        masks_t.unsqueeze(1),
        size=(H_out, W_out),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)

    return masks_t.numpy()


def segment_image(
    image,
    model,
    device,
    output_size
):
    """
    Run RF-DETR-Seg-Small on the original camera image.

    RF-DETR produces instance masks. These are converted into one
    semantic mask:

        labels[v,u] = class ID of the highest-confidence instance
                       covering pixel (v,u)

    output_size is (width, height) and should be exactly the
    spatial size of MapAnything's pts3d_cam grid.

    Returns:
        labels      : H x W integer class IDs; -1 = no RF-DETR mask
        confidence  : H x W confidence of selected instance
        id2label    : RF-DETR class-name dictionary
    """
    # RF-DETR's predict() accepts a PIL image or NumPy image.
    detections = model.predict(
        image,
        threshold=RFDETR_CONFIDENCE_THRESHOLD,
        include_source_image=False,
    )

    # Some versions return a one-element list for a single image.
    if isinstance(detections, (list, tuple)):
        if len(detections) == 0:
            raise RuntimeError(
                "RF-DETR returned no prediction object."
            )
        detections = detections[0]

    H_out = int(output_size[1])
    W_out = int(output_size[0])

    labels = np.full(
        (H_out, W_out),
        -1,
        dtype=np.int32
    )

    confidence = np.zeros(
        (H_out, W_out),
        dtype=np.float32
    )

    # RF-DETR detection count.
    try:
        n_detections = len(detections)
    except TypeError:
        n_detections = 0

    id2label = _get_rfdetr_class_names(
        model,
        detections
    )

    if n_detections == 0:
        return (
            labels,
            confidence,
            id2label
        )

    if not hasattr(detections, "mask"):
        raise RuntimeError(
            "The loaded RF-DETR model did not return masks. "
            "Make sure this is RF-DETR-Seg-Small, not the "
            "detection-only RF-DETR-Small model."
        )

    masks = detections.mask

    if masks is None:
        return (
            labels,
            confidence,
            id2label
        )

    masks = _resize_instance_masks(
        masks,
        output_size
    )

    class_ids = _as_numpy(
        detections.class_id
    ).astype(np.int32).reshape(-1)

    instance_conf = _as_numpy(
        detections.confidence
    ).astype(np.float32).reshape(-1)

    n = min(
        masks.shape[0],
        len(class_ids),
        len(instance_conf)
    )

    masks = masks[:n]
    class_ids = class_ids[:n]
    instance_conf = instance_conf[:n]

    if n == 0:
        return (
            labels,
            confidence,
            id2label
        )

    # --------------------------------------------------------
    # OVERLAPPING INSTANCES
    # --------------------------------------------------------
    #
    # For each pixel, select the highest-confidence instance
    # whose mask covers that pixel.
    #
    # We deliberately use detection confidence as the primary
    # winner criterion, matching the semantic conversion that
    # the previous RF-DETR pipeline used.
    # --------------------------------------------------------

    mask_binary = (
        masks >= RFDETR_MASK_THRESHOLD
    )

    # Score = detection confidence inside the mask.
    # Outside the mask the score is -1.
    scores = (
        mask_binary.astype(np.float32)
        * instance_conf[:, None, None]
    )

    scores = np.where(
        mask_binary,
        scores,
        -1.0
    )

    best_indices = np.argmax(
        scores,
        axis=0
    )

    best_scores = np.max(
        scores,
        axis=0
    )

    covered = (
        best_scores >=
        RFDETR_CONFIDENCE_THRESHOLD
    )

    labels[covered] = class_ids[
        best_indices[covered]
    ]

    confidence[covered] = (
        best_scores[covered]
    )

    return (
        labels,
        confidence,
        id2label
    )


def make_class_colors(
    class_names
):
    """
    Fixed semantic colours.

    RF-DETR class IDs are used as the keys so the colour
    convention is stable across every run.
    """
    if not class_names:
        raise RuntimeError(
            "RF-DETR returned no class-name mapping. "
            "Check the checkpoint and RF-DETR version."
        )

    num_classes = (
        max(class_names.keys()) + 1
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


def save_semantic_overlay(
    path,
    image,
    labels,
    class_colors,
    alpha=0.65,
    class_names=None
):
    """
    Save the ORIGINAL RGB camera image with strong semantic highlights,
    plus a readable label/legend.

    - Original image remains visible underneath.
    - Semantic colours are intentionally high-contrast.
    - Each detected class gets a label placed near the largest region.
    - A legend is drawn in the top-left corner.
    - Unlabelled pixels remain unchanged.
    """
    from PIL import ImageDraw, ImageFont

    image = np.asarray(image, dtype=np.uint8)

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"Expected RGB image HxWx3, got {image.shape}"
        )

    if labels.shape != image.shape[:2]:
        labels_for_image = np.asarray(
            Image.fromarray(
                labels.astype(np.int32),
                mode="I"
            ).resize(
                (image.shape[1], image.shape[0]),
                Image.Resampling.NEAREST
            ),
            dtype=np.int32
        )
    else:
        labels_for_image = np.asarray(labels, dtype=np.int32)

    overlay = image.astype(np.float32).copy()

    valid = (
        (labels_for_image >= 0)
        & (labels_for_image < len(class_colors))
    )

    if np.any(valid):
        semantic_rgb = np.zeros_like(overlay, dtype=np.float32)
        semantic_rgb[valid] = class_colors[labels_for_image[valid]]

        overlay[valid] = (
            (1.0 - alpha) * overlay[valid]
            + alpha * semantic_rgb[valid]
        )

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    # --------------------------------------------------------
    # Find a readable label position for each class.
    # We use the centroid of the largest connected component.
    # This avoids dumping text in the middle of scattered pixels.
    # --------------------------------------------------------
    try:
        import cv2
    except ImportError:
        cv2 = None

    label_positions = {}
    present_classes = [
        int(c) for c in np.unique(labels_for_image)
        if int(c) >= 0 and int(c) < len(class_colors)
    ]

    for class_id in present_classes:
        mask = (labels_for_image == class_id).astype(np.uint8)

        if cv2 is not None:
            n_components, component_labels, stats, centroids = cv2.connectedComponentsWithStats(
                mask, connectivity=8
            )
            if n_components > 1:
                areas = stats[1:, cv2.CC_STAT_AREA]
                best_idx = 1 + int(np.argmax(areas))
                best_area = int(stats[best_idx, cv2.CC_STAT_AREA])
                if best_area >= SEMANTIC_LABEL_MIN_AREA:
                    cx, cy = centroids[best_idx]
                    label_positions[class_id] = (int(cx), int(cy))
        else:
            ys, xs = np.where(mask > 0)
            if len(xs) >= SEMANTIC_LABEL_MIN_AREA:
                label_positions[class_id] = (int(xs.mean()), int(ys.mean()))

    result = Image.fromarray(overlay, mode="RGB")
    draw = ImageDraw.Draw(result)

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 24)
        small_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
    except Exception:
        font = ImageFont.load_default()
        small_font = font

    def display_name(class_id):
        if isinstance(class_names, dict):
            return str(class_names.get(class_id, f"class_{class_id}"))
        if class_names is not None and class_id < len(class_names):
            return str(class_names[class_id])
        return f"class_{class_id}"

    # --------------------------------------------------------
    # Per-region labels: black box + white text for readability.
    # --------------------------------------------------------
    for class_id, (cx, cy) in label_positions.items():
        text = display_name(class_id)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        x0 = max(0, min(cx - tw // 2 - 8, result.width - tw - 16))
        y0 = max(0, min(cy - th // 2 - 8, result.height - th - 16))
        x1 = x0 + tw + 16
        y1 = y0 + th + 16

        draw.rounded_rectangle(
            (x0, y0, x1, y1),
            radius=7,
            fill=(0, 0, 0),
            outline=tuple(int(v) for v in class_colors[class_id]),
            width=3
        )
        draw.text(
            (x0 + 8, y0 + 8),
            text,
            fill=(255, 255, 255),
            font=font
        )

    # --------------------------------------------------------
    # Legend: always shows every class that is actually present.
    # --------------------------------------------------------
    if present_classes:
        legend_texts = [display_name(c) for c in present_classes]
        widths = []
        heights = []
        for text in legend_texts:
            bb = draw.textbbox((0, 0), text, font=small_font)
            widths.append(bb[2] - bb[0])
            heights.append(bb[3] - bb[1])

        row_h = max(30, max(heights, default=20) + 14)
        legend_w = max(widths, default=80) + 60
        legend_h = 18 + row_h * len(present_classes)

        # Semi-opaque black panel.
        panel = Image.new("RGBA", (legend_w, legend_h), (0, 0, 0, 205))
        result_rgba = result.convert("RGBA")
        result_rgba.alpha_composite(panel, (10, 10))
        result = result_rgba.convert("RGB")
        draw = ImageDraw.Draw(result)

        for row, class_id in enumerate(present_classes):
            y = 18 + row * row_h
            rgb = tuple(int(v) for v in class_colors[class_id])
            draw.rectangle((22, y + 4, 22 + 22, y + 26), fill=rgb, outline=(255, 255, 255), width=1)
            draw.text((54, y), display_name(class_id), fill=(255, 255, 255), font=small_font)

    result.save(path)


def save_segmentation_visualization(
    path,
    image,
    labels,
    class_colors,
    alpha=0.65
):
    """Backwards-compatible wrapper for the semantic highlight overlay."""
    save_semantic_overlay(
        path=path,
        image=image,
        labels=labels,
        class_colors=class_colors,
        alpha=alpha,
        class_names=RFDETR_CLASS_NAMES
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
print("MAPANYTHING + RF-DETR-SEG-SMALL 3D PIPELINE")
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

    # IMPORTANT: this matrix is the matrix used directly during merging.
    # Do not invert it.
    assert T_tf[camera].shape == (4, 4)

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
# LOAD RF-DETR SEGMENTATION
# ============================================================

rfdetr_model = load_rfdetr_model(
    device
)

RFDETR_CLASS_NAMES = _get_rfdetr_class_names(
    rfdetr_model,
    None
)

# class_names is retrieved from the model directly when possible.
try:
    RFDETR_CLASS_NAMES = {
        int(k): str(v)
        for k, v in rfdetr_model.class_names.items()
    } if isinstance(rfdetr_model.class_names, dict) else {
        i: str(v)
        for i, v in enumerate(rfdetr_model.class_names)
    }
except Exception:
    RFDETR_CLASS_NAMES = {}

if not RFDETR_CLASS_NAMES:
    raise RuntimeError(
        "Could not obtain RF-DETR class names from the loaded model."
    )

class_colors = make_class_colors(
    RFDETR_CLASS_NAMES
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

SEMANTIC_DIR = (
    OUTPUT_DIR /
    "06_semantic_3d"
)


for directory in [
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
semantic_points = {}
semantic_colors_rgb = {}

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
    # RUN RF-DETR SEGMENTATION
    #
    # RF-DETR runs on the original image and returns instance
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
        model=rfdetr_model,
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
    # It excludes the RF-DETR neural-network inference time.
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
    # FULL / RAW VALID POINTS
    # --------------------------------------------------------
    # Keep ALL valid MapAnything 3D points.  RF-DETR segmentation
    # must NOT determine whether a point belongs in the full cloud.
    full_valid = (
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

    # --------------------------------------------------------
    # SEGMENTED VALID POINTS
    # --------------------------------------------------------
    # The segmented cloud is a separate subset: only points for which
    # RF-DETR produced a valid class and which pass the confidence
    # threshold are included here.
    semantic_valid = (
        full_valid
        &
        (labels >= 0)
    )

    if RFDETR_CONFIDENCE_THRESHOLD > 0:
        semantic_valid &= (
            confidence >=
            RFDETR_CONFIDENCE_THRESHOLD
        )

    # --------------------------------------------------------
    # EXTRACT FULL CLOUD
    # --------------------------------------------------------

    full_points = points_grid[full_valid]
    full_colors = image_grid[full_valid]

    # --------------------------------------------------------
    # EXTRACT SEGMENTED CLOUD
    # --------------------------------------------------------

    points = points_grid[semantic_valid]
    colors = image_grid[semantic_valid]

    labels_flat = labels[semantic_valid]
    confidence_flat = confidence[semantic_valid]

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

    # FULL cloud -- independent of segmentation
    raw_points[camera] = full_points
    raw_colors[camera] = full_colors

    # SEGMENTED cloud -- RF-DETR-labelled subset
    semantic_points[camera] = points
    semantic_colors_rgb[camera] = colors
    semantic_labels[camera] = labels_flat
    semantic_confidence[camera] = confidence_flat
    semantic_colors[camera] = sem_colors

    # --------------------------------------------------------
    # SAVE INDIVIDUAL CAMERA PLYs
    # --------------------------------------------------------
    # Raw RGB point cloud in MapAnything camera coordinates.
    save_ply(
        SEMANTIC_DIR / f"{camera}_raw.ply",
        full_points,
        full_colors
    )

    # Segmented point cloud in MapAnything camera coordinates.
    save_semantic_ply(
        SEMANTIC_DIR / f"{camera}_semantic.ply",
        points,
        colors,
        labels_flat,
        sem_colors
    )

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
    # SAVE ORIGINAL IMAGE + SEMANTIC HIGHLIGHT OVERLAY
    # --------------------------------------------------------

    save_segmentation_visualization(
        SEMANTIC_DIR /
        f"{camera}_segmentation_overlay.png",
        image,
        labels,
        class_colors,
        alpha=SEMANTIC_OVERLAY_ALPHA
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
#   RF-DETR class
#          +
#   MapAnything pts3d_cam[u,v]
#          ↓
#   DIRECT supplied camera -> common-frame transform
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

merged_full_points = []
merged_full_colors = []

merged_semantic_points = []
merged_semantic_rgb = []
merged_semantic_labels = []
merged_semantic_colors = []
merged_semantic_confidence = []


for camera in CAMERAS:

    # --------------------------------------------------------
    # FULL CLOUD -- NO SEGMENTATION FILTER
    # --------------------------------------------------------

    full_points = raw_points[camera]
    full_colors = raw_colors[camera]

    # DIRECT TRANSFORM CONVENTION:
    # Apply the supplied TF matrix exactly as constructed.
    # NO inversion.
    T_direct = T_tf[camera]

    full_points_direct = transform_points(
        full_points,
        T_direct
    )

    merged_full_points.append(full_points_direct)
    merged_full_colors.append(full_colors)

    # --------------------------------------------------------
    # SEGMENTED CLOUD -- RF-DETR SUBSET ONLY
    # --------------------------------------------------------

    points = semantic_points[camera]
    colors = semantic_colors_rgb[camera]
    labels = semantic_labels[camera]
    confidence = semantic_confidence[camera]
    sem_colors = semantic_colors[camera]

    # Same DIRECT transform for the RF-DETR-labelled points.
    points_direct = transform_points(
        points,
        T_direct
    )

    merged_semantic_points.append(points_direct)
    merged_semantic_rgb.append(colors)
    merged_semantic_labels.append(labels)
    merged_semantic_colors.append(sem_colors)
    merged_semantic_confidence.append(confidence)

    print(
        f"\n{camera}: "
        f"{len(full_points_direct)} full points, "
        f"{len(points_direct)} segmented points"
    )


# ============================================================
# MERGE THREE CAMERAS
# ============================================================

merged_full_points = np.concatenate(
    merged_full_points,
    axis=0
)

merged_full_colors = np.concatenate(
    merged_full_colors,
    axis=0
)

merged_points = np.concatenate(
    merged_semantic_points,
    axis=0
)

merged_colors = np.concatenate(
    merged_semantic_rgb,
    axis=0
)

merged_labels = np.concatenate(
    merged_semantic_labels,
    axis=0
)

merged_semantic_colors = np.concatenate(
    merged_semantic_colors,
    axis=0
)

merged_confidence = np.concatenate(
    merged_semantic_confidence,
    axis=0
)


print(
    "\nTotal FULL merged 3D points:",
    len(merged_full_points)
)

print(
    "Total SEGMENTED merged 3D points:",
    len(merged_points)
)


# ============================================================
# SAVE FULL MERGED CLOUD
# ============================================================
# This is the complete valid MapAnything reconstruction from
# CAM1 + CAM2 + CAM6. RF-DETR segmentation does not remove points.
# ============================================================

save_ply(
    OUTPUT_DIR / "fully_merged_3d.ply",
    merged_full_points,
    merged_full_colors
)

print(
    "Saved FULL merged 3D cloud:"
)
print(
    OUTPUT_DIR / "fully_merged_3d.ply"
)


# ============================================================
# SAVE SEGMENTED MERGED CLOUD
# ============================================================
# This contains only the RF-DETR-labelled subset.
# ============================================================

save_semantic_ply(
    OUTPUT_DIR /
    "semantic_3d_merged_direct.ply",
    merged_points,
    merged_colors,
    merged_labels,
    merged_semantic_colors
)

print(
    "Saved SEGMENTED merged 3D cloud:"
)
print(
    OUTPUT_DIR / "semantic_3d_merged_direct.ply"
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

if voxel_labels.max(initial=-1) >= len(class_colors):
    raise RuntimeError(
        "A voxel label exceeds the configured CLASS_COLORS table."
    )

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

    name = RFDETR_CLASS_NAMES.get(
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