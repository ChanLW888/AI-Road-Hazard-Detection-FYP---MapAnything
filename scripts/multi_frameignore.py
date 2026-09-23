#!/usr/bin/env python3

from pathlib import Path
import gc
import re

import cv2
import folium
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from folium.plugins import MeasureControl, MarkerCluster

from mapanything.models import MapAnything
from mapanything.utils.image import load_images
from rfdetr import RFDETRSegSmall


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(
    "/home/lcha0115/bt60_scratch/lcha_data/map-anything/"
    "personal_data/camera_lidar_gps_test_101/extracted"
)

IMAGE_DIR = BASE_DIR / "CAM2" / "images_rect"
GPS_FILE = BASE_DIR / "gps" / "position.csv"
CALIB_FILE = BASE_DIR / "calibration" / "CAM2_calib.txt"

START_IMAGE = 9600
END_IMAGE = 10600

# Inspect every image when doing GPS-distance selection.
STRIDE = 1

# Only reconstruct another frame after the vehicle has moved this far.
MIN_TRAVEL_DISTANCE_M = 3.50

# Maximum permitted image/GPS timestamp difference.
MAX_GPS_TIME_DIFF_NS = 1_000_000_000

# MapAnything load_images() determines the actual model resolution.
# These are informational only; images are NOT manually resized.
MAPANYTHING_TARGET_WIDTH = 960
MAPANYTHING_TARGET_HEIGHT = 600

# Semantic point fusion.
VOXEL_SIZE_M = 0.05
MIN_POINTS_PER_VOXEL = 1

# RF-DETR checkpoint.
RFDETR_MODEL_PATH = "/home/lcha0115/bt60_scratch/lcha_data/map-anything/personal_data/2D-Seg-Road-Nissan.v6i.coco-segmentation/outputs/checkpoint_best_total.pth"

RFDETR_CONFIDENCE_THRESHOLD = 0.5
RFDETR_MASK_THRESHOLD = 0.50
SEMANTIC_ALPHA = 0.85

CLASS_NAMES = {
    0: "fallen_fence",
    1: "loose_trash",
    2: "mud_spill",
    3: "public_greenway",
    4: "sidewalk",
}

CLASS_COLORS = {
    0: (255, 0, 255),
    1: (0, 255, 255),
    2: (255, 80, 0),
    3: (0, 255, 0),
    4: (0, 120, 255),
}

HAZARD_MARKER_COLORS = {
    "fallen_fence": "purple",
    "loose_trash": "orange",
    "mud_spill": "red",
    "public_greenway": "green",
    "sidewalk": "blue",
}

OUTPUT_DIR = (
    BASE_DIR.parent / f"mapanything_{START_IMAGE}_{END_IMAGE}_fused"
)

LABELLED_DIR = OUTPUT_DIR / "labelled_images"
PREDICTION_FILE = OUTPUT_DIR / "predictions.pt"
FULL_RECONSTRUCTION_PLY = OUTPUT_DIR / "full_reconstruction.ply"
SEMANTIC_OVERLAY_PLY = OUTPUT_DIR / "semantic_overlay_full_reconstruction.ply"
HAZARD_CSV = OUTPUT_DIR / "hazards_gps.csv"
MAP_HTML = OUTPUT_DIR / "hazards_map.html"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LABELLED_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CALIBRATION
# ============================================================

def load_calibration(calib_file):
    """
    Read the calibration file directly.

    Expected format:

        width: 1920
        height: 1200

        K:
        row
        row
        row

        D:
        row

        R:
        row
        row
        row

        P:
        row
        row
        row
    """

    calib_file = Path(calib_file)

    if not calib_file.exists():
        raise FileNotFoundError(
            f"Calibration file not found:\n{calib_file}"
        )

    text = calib_file.read_text(encoding="utf-8")

    width_match = re.search(
        r"(?m)^\s*width\s*:\s*(\d+)\s*$",
        text,
    )
    height_match = re.search(
        r"(?m)^\s*height\s*:\s*(\d+)\s*$",
        text,
    )

    if width_match is None or height_match is None:
        raise RuntimeError(
            "Could not find width/height in calibration file."
        )

    width = int(width_match.group(1))
    height = int(height_match.group(1))

    def parse_matrix(name, rows, cols):
        """
        Find NAME: and read exactly the next non-empty numeric rows.
        This intentionally does NOT expect square brackets.
        """

        marker = re.search(
            rf"(?m)^\s*{re.escape(name)}\s*:\s*$",
            text,
        )

        if marker is None:
            raise RuntimeError(
                f"Could not find matrix '{name}' in:\n{calib_file}"
            )

        lines = text[marker.end():].splitlines()

        values = []

        for line in lines:
            stripped = line.strip()

            if not stripped:
                continue

            # Stop if the next calibration field begins.
            if re.match(
                r"^[A-Za-z_][A-Za-z0-9_]*\s*:",
                stripped,
            ):
                break

            parts = stripped.split()

            try:
                row_values = [float(x) for x in parts]
            except ValueError:
                continue

            if len(row_values) != cols:
                raise RuntimeError(
                    f"Matrix '{name}' expected {cols} values per row, "
                    f"but got {len(row_values)} in line:\n{line}"
                )

            values.extend(row_values)

            if len(values) == rows * cols:
                break

            if len(values) > rows * cols:
                raise RuntimeError(
                    f"Matrix '{name}' contains too many values."
                )

        if len(values) != rows * cols:
            raise RuntimeError(
                f"Matrix '{name}' contains {len(values)} values; "
                f"expected {rows * cols}."
            )

        return np.asarray(
            values,
            dtype=np.float32,
        ).reshape(rows, cols)

    K = parse_matrix("K", 3, 3)
    D = parse_matrix("D", 1, 4)
    R = parse_matrix("R", 3, 3)
    P = parse_matrix("P", 3, 4)

    return {
        "width": width,
        "height": height,
        "K": K,
        "D": D,
        "R": R,
        "P": P,
    }


def scale_intrinsics(
    K,
    original_width,
    original_height,
    new_width,
    new_height,
):
    """Scale pinhole intrinsics to the actual image tensor resolution."""

    sx = new_width / float(original_width)
    sy = new_height / float(original_height)

    K_scaled = K.copy()

    K_scaled[0, 0] *= sx
    K_scaled[1, 1] *= sy
    K_scaled[0, 2] *= sx
    K_scaled[1, 2] *= sy

    return K_scaled


print("\n" + "=" * 70)
print("LOADING CAM2 CALIBRATION")
print("=" * 70)

calib = load_calibration(CALIB_FILE)

CALIB_WIDTH = calib["width"]
CALIB_HEIGHT = calib["height"]

K = calib["K"]
D = calib["D"]
R = calib["R"]
P = calib["P"]

# images_rect are already rectified, so use the rectified P matrix.
K_RECTIFIED = P[:3, :3].copy()

print(f"Calibration file: {CALIB_FILE}")
print(f"Calibration resolution: {CALIB_WIDTH} x {CALIB_HEIGHT}")

print("\nK:")
print(K)

print("\nD:")
print(D)

print("\nR:")
print(R)

print("\nP:")
print(P)

print("\nRectified P[:3,:3]:")
print(K_RECTIFIED)


# ============================================================
# GPS
# ============================================================

print("\n" + "=" * 70)
print("LOADING GPS")
print("=" * 70)

gps_df = pd.read_csv(GPS_FILE)

required_gps_columns = [
    "ros_timestamp_ns",
    "latitude_deg",
    "longitude_deg",
    "altitude_m",
]

for column in required_gps_columns:
    if column not in gps_df.columns:
        raise RuntimeError(
            f"GPS CSV missing required column: {column}"
        )

gps_df = (
    gps_df
    .dropna(subset=required_gps_columns)
    .copy()
)

gps_df["ros_timestamp_ns"] = (
    gps_df["ros_timestamp_ns"]
    .astype(np.int64)
)

gps_df = (
    gps_df
    .sort_values("ros_timestamp_ns")
    .reset_index(drop=True)
)

gps_timestamps = gps_df[
    "ros_timestamp_ns"
].to_numpy(dtype=np.int64)

gps_lat = gps_df[
    "latitude_deg"
].to_numpy(dtype=float)

gps_lon = gps_df[
    "longitude_deg"
].to_numpy(dtype=float)

gps_alt = gps_df[
    "altitude_m"
].to_numpy(dtype=float)

print(f"GPS records: {len(gps_df):,}")


def nearest_gps(image_timestamp_ns):
    """Return the GPS record closest to an image ROS timestamp."""

    idx = np.searchsorted(
        gps_timestamps,
        image_timestamp_ns,
    )

    candidates = []

    if idx > 0:
        candidates.append(idx - 1)

    if idx < len(gps_timestamps):
        candidates.append(idx)

    if not candidates:
        return None

    best_idx = min(
        candidates,
        key=lambda i: abs(
            int(gps_timestamps[i])
            - int(image_timestamp_ns)
        ),
    )

    gps_timestamp = int(
        gps_timestamps[best_idx]
    )

    delta_ns = abs(
        gps_timestamp
        - int(image_timestamp_ns)
    )

    if (
        MAX_GPS_TIME_DIFF_NS is not None
        and delta_ns > MAX_GPS_TIME_DIFF_NS
    ):
        return None

    return {
        "gps_timestamp_ns": gps_timestamp,
        "gps_delta_ns": delta_ns,
        "gps_delta_ms": delta_ns / 1_000_000.0,
        "latitude_deg": float(gps_lat[best_idx]),
        "longitude_deg": float(gps_lon[best_idx]),
        "altitude_m": float(gps_alt[best_idx]),
    }


# ============================================================
# LOCAL GPS DISTANCE
# ============================================================

def gps_to_local_xy(
    latitude_deg,
    longitude_deg,
    origin_lat_deg,
    origin_lon_deg,
):
    """
    Convert nearby GPS coordinates to local metres.
    Good for the short road segment being processed.
    """

    earth_radius = 6378137.0

    lat = np.deg2rad(latitude_deg)
    lon = np.deg2rad(longitude_deg)

    origin_lat = np.deg2rad(origin_lat_deg)
    origin_lon = np.deg2rad(origin_lon_deg)

    x = (
        (lon - origin_lon)
        * earth_radius
        * np.cos(origin_lat)
    )

    y = (
        (lat - origin_lat)
        * earth_radius
    )

    return float(x), float(y)


# ============================================================
# FIND IMAGES
# ============================================================

print("\n" + "=" * 70)
print("FINDING IMAGES")
print("=" * 70)

all_images = sorted(
    p
    for p in IMAGE_DIR.iterdir()
    if p.suffix.lower()
    in {".png", ".jpg", ".jpeg", ".bmp"}
)

if not all_images:
    raise RuntimeError(
        f"No images found in:\n{IMAGE_DIR}"
    )

print(f"Total images: {len(all_images):,}")

if START_IMAGE < 1 or END_IMAGE > len(all_images):
    raise RuntimeError(
        f"Frame range {START_IMAGE}->{END_IMAGE} is outside "
        f"the available image range 1->{len(all_images)}."
    )

candidate_frames = list(
    range(
        START_IMAGE,
        END_IMAGE + 1,
        STRIDE,
    )
)

candidate_images = [
    all_images[frame - 1]
    for frame in candidate_frames
]

print(f"Candidate frames: {len(candidate_frames):,}")


# ============================================================
# MATCH CANDIDATES TO GPS
# ============================================================

print("\n" + "=" * 70)
print("MATCHING CANDIDATE FRAMES TO GPS")
print("=" * 70)

candidate_metadata = []

for frame, image_path in zip(
    candidate_frames,
    candidate_images,
):
    try:
        image_timestamp_ns = int(image_path.stem)
    except ValueError:
        print(
            f"WARNING: Could not parse timestamp from "
            f"{image_path.name}; skipping."
        )
        continue

    gps = nearest_gps(image_timestamp_ns)

    if gps is None:
        print(
            f"Frame {frame}: no valid GPS match; skipping."
        )
        continue

    candidate_metadata.append(
        {
            "frame": frame,
            "image_path": str(image_path),
            "image_filename": image_path.name,
            "image_timestamp_ns": image_timestamp_ns,
            **gps,
        }
    )

print(
    f"GPS-matched candidates: "
    f"{len(candidate_metadata):,}"
)

if not candidate_metadata:
    raise RuntimeError(
        "No frames have valid GPS matches."
    )


# ============================================================
# GPS-BASED SPATIAL FRAME SELECTION
# ============================================================

print("\n" + "=" * 70)
print("SPATIAL FRAME SELECTION")
print("=" * 70)

origin_lat = candidate_metadata[0]["latitude_deg"]
origin_lon = candidate_metadata[0]["longitude_deg"]

selected_metadata = []

last_x = None
last_y = None

for metadata in candidate_metadata:

    x, y = gps_to_local_xy(
        metadata["latitude_deg"],
        metadata["longitude_deg"],
        origin_lat,
        origin_lon,
    )

    metadata["local_x_m"] = x
    metadata["local_y_m"] = y

    if last_x is None:
        metadata["distance_from_previous_selected_m"] = 0.0
        selected_metadata.append(metadata)
        last_x = x
        last_y = y
        continue

    distance = np.hypot(
        x - last_x,
        y - last_y,
    )

    metadata["distance_from_previous_selected_m"] = float(
        distance
    )

    if distance >= MIN_TRAVEL_DISTANCE_M:
        selected_metadata.append(metadata)
        last_x = x
        last_y = y

selected_frames = [
    metadata["frame"]
    for metadata in selected_metadata
]

selected_images = [
    Path(metadata["image_path"])
    for metadata in selected_metadata
]

print(
    f"\nMinimum travel distance: "
    f"{MIN_TRAVEL_DISTANCE_M:.2f} m"
)

print(
    f"Candidate frames: "
    f"{len(candidate_metadata):,}"
)

print(
    f"Selected frames: "
    f"{len(selected_metadata):,}"
)

print(
    f"Frames removed: "
    f"{len(candidate_metadata) - len(selected_metadata):,}"
)

print("\nSelected frames:")
print(selected_frames[:20])

if len(selected_frames) > 20:
    print("...")


# ============================================================
# LOAD MAPANYTHING
# ============================================================

print("\n" + "=" * 70)
print("LOADING MAPANYTHING")
print("=" * 70)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"Device: {device}")

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

model = (
    MapAnything
    .from_pretrained("facebook/map-anything")
    .to(device)
)

model.eval()

print("MapAnything loaded.")


# ============================================================
# LOAD ORIGINAL IMAGES
# ============================================================

print("\nLoading original images...")

views = load_images(
    [
        str(path)
        for path in selected_images
    ]
)

print(
    f"Loaded views: {len(views)}"
)

print(
    f"Input tensor shape: "
    f"{views[0]['img'].shape}"
)

input_tensor = views[0]["img"]

mapanything_height = int(
    input_tensor.shape[-2]
)

mapanything_width = int(
    input_tensor.shape[-1]
)

print("\nActual MapAnything resolution:")
print(
    f"{mapanything_width} x "
    f"{mapanything_height}"
)


# ============================================================
# SCALE CALIBRATION TO MAPANYTHING RESOLUTION
# ============================================================

K_RESIZED = scale_intrinsics(
    K_RECTIFIED,
    CALIB_WIDTH,
    CALIB_HEIGHT,
    mapanything_width,
    mapanything_height,
)

print("\nScaled intrinsics:")
print(K_RESIZED)


# ============================================================
# ADD INTRINSICS TO EVERY VIEW
# ============================================================

K_torch = torch.from_numpy(
    K_RESIZED
).float().unsqueeze(0)

for view in views:
    view["intrinsics"] = K_torch.clone()


print("\n" + "=" * 70)
print("MAPANYTHING INPUT SANITY CHECK")
print("=" * 70)

print(
    f"Image shape       : {views[0]['img'].shape}"
)

print(
    f"Image dtype       : {views[0]['img'].dtype}"
)

print(
    f"Intrinsics shape  : {views[0]['intrinsics'].shape}"
)

print(
    f"Intrinsics dtype  : {views[0]['intrinsics'].dtype}"
)

print("\nIntrinsics:")
print(views[0]["intrinsics"])

print(
    "\nRay directions: NOT PROVIDED"
)

print(
    "MapAnything will derive them from the supplied intrinsics."
)


# ============================================================
# MAPANYTHING INFERENCE
# ============================================================

print("\n" + "=" * 70)
print("RUNNING MAPANYTHING")
print("=" * 70)

print(
    f"Reconstructing {len(views)} spatially selected frames."
)

print(
    f"Minimum vehicle movement: "
    f"{MIN_TRAVEL_DISTANCE_M:.2f} m"
)

with torch.inference_mode():

    predictions = model.infer(
        views,
        memory_efficient_inference=True,
        minibatch_size=1,
        use_amp=True,
        amp_dtype="fp16",
        apply_mask=True,
        mask_edges=True,
        apply_confidence_mask=False,
    )

print("\nMapAnything inference complete.")

print(
    f"Prediction entries: {len(predictions)}"
)


# ============================================================
# MOVE PREDICTIONS TO CPU
# ============================================================

def move_to_cpu(obj):

    if torch.is_tensor(obj):
        return obj.detach().cpu()

    if isinstance(obj, dict):
        return {
            key: move_to_cpu(value)
            for key, value in obj.items()
        }

    if isinstance(obj, list):
        return [
            move_to_cpu(value)
            for value in obj
        ]

    if isinstance(obj, tuple):
        return tuple(
            move_to_cpu(value)
            for value in obj
        )

    return obj


print("\n" + "=" * 70)
print("SAVING RAW MAPANYTHING PREDICTIONS")
print("=" * 70)

predictions_cpu = move_to_cpu(predictions)

torch.save(
    {
        "predictions": predictions_cpu,
        "metadata": selected_metadata,
        "calibration": {
            "calibration_file": str(CALIB_FILE),
            "original_width": CALIB_WIDTH,
            "original_height": CALIB_HEIGHT,
            "K": K,
            "D": D,
            "R": R,
            "P": P,
            "mapanything_intrinsics": K_RESIZED,
            "mapanything_width": mapanything_width,
            "mapanything_height": mapanything_height,
        },
        "config": {
            "start_image": START_IMAGE,
            "end_image": END_IMAGE,
            "stride": STRIDE,
            "min_travel_distance_m": MIN_TRAVEL_DISTANCE_M,
            "voxel_size_m": VOXEL_SIZE_M,
            "min_points_per_voxel": MIN_POINTS_PER_VOXEL,
            "rfdetr_confidence_threshold": RFDETR_CONFIDENCE_THRESHOLD,
            "rfdetr_mask_threshold": RFDETR_MASK_THRESHOLD,
        },
    },
    PREDICTION_FILE,
)

print(f"Saved:\n{PREDICTION_FILE}")


# ============================================================
# PREPARE MAPANYTHING 3D FRAME DATA
# ============================================================

print("\n" + "=" * 70)
print("PREPARING 3D FRAME DATA")
print("=" * 70)

frame_points = []
frame_rgb = []
frame_shapes = []
frame_valid_masks = []

for i, prediction in enumerate(predictions_cpu):

    frame = selected_frames[i]

    pts3d = prediction["pts3d"]

    if torch.is_tensor(pts3d):
        pts3d = pts3d.numpy()

    pts3d = np.asarray(pts3d)

    while (
        pts3d.ndim > 3
        and pts3d.shape[0] == 1
    ):
        pts3d = pts3d[0]

    if (
        pts3d.ndim != 3
        or pts3d.shape[-1] != 3
    ):
        raise RuntimeError(
            f"Unexpected pts3d shape for frame "
            f"{frame}: {pts3d.shape}"
        )

    ph, pw = pts3d.shape[:2]

    image = Image.open(
        selected_images[i]
    ).convert("RGB")

    image = image.resize(
        (pw, ph),
        Image.Resampling.BILINEAR,
    )

    image = np.asarray(image)

    valid = np.isfinite(
        pts3d
    ).all(axis=2)

    points = pts3d[valid].astype(
        np.float32
    )

    rgb = image[valid].astype(
        np.uint8
    )

    frame_points.append(points)
    frame_rgb.append(rgb)
    frame_shapes.append((ph, pw))
    frame_valid_masks.append(valid)

    print(
        f"Frame {frame}: "
        f"{ph}x{pw}, "
        f"{len(points):,} valid points"
    )


# ============================================================
# FREE MAPANYTHING
# ============================================================

del model
del views
del predictions

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

print("\nMapAnything memory released.")


# ============================================================
# LOAD RF-DETR
# ============================================================

print("\n" + "=" * 70)
print("LOADING RF-DETR")
print("=" * 70)

print(
    f"Checkpoint:\n{RFDETR_MODEL_PATH}"
)

# IMPORTANT:
# This checkpoint is the 1300-query checkpoint.
# Do NOT use RFDETRSegPreview.
# Do NOT manually specify num_queries.
rfdetr_model = RFDETRSegSmall.from_checkpoint(
    str(RFDETR_MODEL_PATH)
)

print("RF-DETR loaded.")

# RF-DETR warns when the checkpoint has not been optimized for inference.
# Use the model's own optimization API when available. Do not change the
# checkpoint architecture or num_queries.
try:
    if hasattr(rfdetr_model, "optimize_for_inference"):
        rfdetr_model.optimize_for_inference()
        print("RF-DETR optimized for inference.")
except Exception as e:
    print(
        f"RF-DETR inference optimization skipped: {e}"
    )


# ============================================================
# FONT
# ============================================================

try:
    font = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        30,
    )
except Exception:
    font = ImageFont.load_default()


# ============================================================
# SEMANTIC STORAGE
# ============================================================

overlay_points = []
overlay_rgb = []

# Complete MapAnything reconstruction from every selected frame.
# These retain ALL valid 3D points, not just semantic points.
all_reconstruction_points = []
all_reconstruction_rgb = []

all_semantic_points = []
all_semantic_colors = []
all_semantic_confidences = []
all_semantic_classes = []

hazard_rows = []


# ============================================================
# RF-DETR + 3D SEMANTIC PROJECTION
# ============================================================

print("\n" + "=" * 70)
print("RUNNING RF-DETR")
print("=" * 70)

for pred_idx, metadata in enumerate(selected_metadata):

    frame = metadata["frame"]
    image_path = Path(metadata["image_path"])

    print(
        f"\nRF-DETR frame {frame} "
        f"({pred_idx + 1}/{len(selected_metadata)})"
    )

    image_bgr = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR,
    )

    if image_bgr is None:
        print("Failed to load image.")
        continue

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    ).copy()

    original_h, original_w = image_rgb.shape[:2]

    # --------------------------------------------------------
    # RF-DETR
    # --------------------------------------------------------

    detections = rfdetr_model.predict(
        image_rgb,
        threshold=RFDETR_CONFIDENCE_THRESHOLD,
        include_source_image=False,
    )

    # --------------------------------------------------------
    # Label image
    # --------------------------------------------------------

    labels = np.full(
        (original_h, original_w),
        -1,
        dtype=np.int16,
    )

    confidence_map = np.zeros(
        (original_h, original_w),
        dtype=np.float32,
    )

    if detections is not None:

        masks = getattr(
            detections,
            "mask",
            None,
        )

        class_ids = getattr(
            detections,
            "class_id",
            None,
        )

        confidences = getattr(
            detections,
            "confidence",
            None,
        )

        if (
            masks is not None
            and class_ids is not None
        ):

            masks = np.asarray(masks)
            class_ids = np.asarray(class_ids)

            if confidences is None:
                confidences = np.ones(
                    len(class_ids),
                    dtype=np.float32,
                )
            else:
                confidences = np.asarray(
                    confidences
                )

            for det_idx in range(
                len(class_ids)
            ):

                class_id = int(
                    class_ids[det_idx]
                )

                confidence = float(
                    confidences[det_idx]
                )

                if confidence < RFDETR_CONFIDENCE_THRESHOLD:
                    continue

                if class_id not in CLASS_NAMES:
                    continue

                mask = np.squeeze(
                    masks[det_idx]
                ).astype(np.float32)

                if mask.shape != (
                    original_h,
                    original_w,
                ):

                    mask = np.asarray(
                        Image.fromarray(
                            (
                                mask * 255
                            ).astype(np.uint8)
                        ).resize(
                            (
                                original_w,
                                original_h,
                            ),
                            Image.Resampling.NEAREST,
                        )
                    ).astype(np.float32) / 255.0

                mask = (
                    mask >= RFDETR_MASK_THRESHOLD
                )

                update = (
                    mask
                    & (
                        confidence
                        > confidence_map
                    )
                )

                labels[update] = class_id
                confidence_map[update] = confidence

    # --------------------------------------------------------
    # Save labelled 2D image
    # --------------------------------------------------------

    labelled = image_rgb.copy()

    detected_classes = []

    for class_id, colour in CLASS_COLORS.items():

        mask = labels == class_id

        if not np.any(mask):
            continue

        detected_classes.append(class_id)

        colour_array = np.asarray(
            colour,
            dtype=np.float32,
        )

        labelled[mask] = (
            SEMANTIC_ALPHA * colour_array
            + (
                1.0 - SEMANTIC_ALPHA
            )
            * labelled[mask]
        ).astype(np.uint8)

    labelled_image = Image.fromarray(
        labelled
    )

    draw = ImageDraw.Draw(
        labelled_image
    )

    if detected_classes:

        x = 25
        y = 25
        line_height = 45
        box_width = 430

        box_height = (
            20
            + len(detected_classes)
            * line_height
        )

        draw.rectangle(
            [
                x,
                y,
                x + box_width,
                y + box_height,
            ],
            fill=(0, 0, 0),
        )

        for j, class_id in enumerate(
            detected_classes
        ):

            yy = (
                y
                + 12
                + j * line_height
            )

            draw.rectangle(
                [
                    x + 10,
                    yy,
                    x + 35,
                    yy + 25,
                ],
                fill=CLASS_COLORS[class_id],
            )

            draw.text(
                (
                    x + 50,
                    yy - 3,
                ),
                (
                    f"{CLASS_NAMES[class_id]} "
                    f"({confidence_map[labels == class_id].max():.2f})"
                ),
                fill=(255, 255, 255),
                font=font,
            )

    labelled_path = (
        LABELLED_DIR
        / image_path.name
    )

    labelled_image.save(
        labelled_path
    )

    # --------------------------------------------------------
    # MapAnything 3D points for this frame
    # --------------------------------------------------------

    pts3d = frame_points[pred_idx]

    original_rgb = frame_rgb[pred_idx]

    # Keep the complete reconstruction: every valid MapAnything 3D point
    # and its original RGB colour.
    all_reconstruction_points.append(
        pts3d.astype(np.float32)
    )
    all_reconstruction_rgb.append(
        original_rgb.astype(np.uint8)
    )

    ph, pw = frame_shapes[pred_idx]

    valid = frame_valid_masks[pred_idx]

    # RF-DETR labels are at original image resolution.
    # Resize the semantic label map to the MapAnything point grid.
    labels_small = np.asarray(
        Image.fromarray(
            labels
        ).resize(
            (pw, ph),
            Image.Resampling.NEAREST,
        )
    )

    labels_flat = labels_small.reshape(-1)
    labels_flat = labels_flat[
        valid.reshape(-1)
    ]

    overlay = original_rgb.copy()

    # Colour the original MapAnything RGB only where a
    # semantic class is present.
    for class_id, colour in CLASS_COLORS.items():

        semantic_mask = (
            labels_flat == class_id
        )

        if not np.any(semantic_mask):
            continue

        colour_array = np.asarray(
            colour,
            dtype=np.float32,
        )

        overlay[semantic_mask] = (
            SEMANTIC_ALPHA * colour_array
            + (
                1.0 - SEMANTIC_ALPHA
            )
            * overlay[semantic_mask]
        ).astype(np.uint8)

    overlay_points.append(
        pts3d
    )

    overlay_rgb.append(
        overlay
    )

    # --------------------------------------------------------
    # Extract ONLY semantic points for fused semantic PLY
    # --------------------------------------------------------

    semantic_pixel_mask = (
        labels_flat >= 0
    )

    if np.any(semantic_pixel_mask):

        semantic_points = pts3d[
            semantic_pixel_mask
        ]

        semantic_labels = labels_flat[
            semantic_pixel_mask
        ]

        semantic_colors = np.asarray(
            [
                CLASS_COLORS[int(class_id)]
                for class_id in semantic_labels
            ],
            dtype=np.uint8,
        )

        # Resize the confidence map to the same MapAnything
        # pixel grid, then apply the exact same valid-pixel mask
        # and semantic-pixel mask as the 3D points.
        confidence_small = np.asarray(
            Image.fromarray(
                confidence_map.astype(np.float32)
            ).resize(
                (pw, ph),
                Image.Resampling.NEAREST,
            )
        )

        confidence_flat = confidence_small.reshape(-1)
        confidence_flat = confidence_flat[
            valid.reshape(-1)
        ]

        semantic_conf = confidence_flat[
            semantic_pixel_mask
        ]

        all_semantic_points.append(
            semantic_points.astype(np.float32)
        )

        all_semantic_colors.append(
            semantic_colors
        )

        all_semantic_confidences.append(
            semantic_conf.astype(np.float32)
        )

        all_semantic_classes.append(
            semantic_labels.astype(np.int16)
        )

    # --------------------------------------------------------
    # Record EVERY detected instance with image GPS
    # --------------------------------------------------------

    if detections is not None:

        masks = getattr(
            detections,
            "mask",
            None,
        )

        class_ids = getattr(
            detections,
            "class_id",
            None,
        )

        confidences = getattr(
            detections,
            "confidence",
            None,
        )

        if (
            masks is not None
            and class_ids is not None
        ):

            masks = np.asarray(masks)
            class_ids = np.asarray(class_ids)

            if confidences is None:
                confidences = np.ones(
                    len(class_ids),
                    dtype=np.float32,
                )
            else:
                confidences = np.asarray(
                    confidences
                )

            for det_idx in range(
                len(class_ids)
            ):

                class_id = int(
                    class_ids[det_idx]
                )

                confidence = float(
                    confidences[det_idx]
                )

                if confidence < RFDETR_CONFIDENCE_THRESHOLD:
                    continue

                if class_id not in CLASS_NAMES:
                    continue

                mask = np.squeeze(
                    masks[det_idx]
                ).astype(np.float32)

                if mask.shape != (
                    original_h,
                    original_w,
                ):

                    mask = np.asarray(
                        Image.fromarray(
                            (
                                mask * 255
                            ).astype(np.uint8)
                        ).resize(
                            (
                                original_w,
                                original_h,
                            ),
                            Image.Resampling.NEAREST,
                        )
                    ).astype(np.float32) / 255.0

                mask = (
                    mask >= RFDETR_MASK_THRESHOLD
                )

                pixel_count = int(
                    np.sum(mask)
                )

                hazard_name = CLASS_NAMES[
                    class_id
                ]

                hazard_rows.append(
                    {
                        "frame": frame,
                        "image_filename": image_path.name,
                        "image_timestamp_ns": metadata[
                            "image_timestamp_ns"
                        ],
                        "gps_timestamp_ns": metadata[
                            "gps_timestamp_ns"
                        ],
                        "gps_delta_ms": metadata[
                            "gps_delta_ms"
                        ],
                        "latitude_deg": metadata[
                            "latitude_deg"
                        ],
                        "longitude_deg": metadata[
                            "longitude_deg"
                        ],
                        "altitude_m": metadata[
                            "altitude_m"
                        ],
                        "class_id": class_id,
                        "class_name": hazard_name,
                        "confidence": confidence,
                        "mask_pixels": pixel_count,
                    }
                )

                print(
                    f"  HAZARD: {hazard_name} | "
                    f"confidence={confidence:.3f} | "
                    f"GPS={metadata['latitude_deg']:.8f}, "
                    f"{metadata['longitude_deg']:.8f} | "
                    f"Δ={metadata['gps_delta_ms']:.2f} ms"
                )


def save_ply(
    path,
    points,
    rgb,
):
    """Write a binary little-endian RGB PLY."""

    data = np.empty(
        len(points),
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )

    data["x"] = points[:, 0]
    data["y"] = points[:, 1]
    data["z"] = points[:, 2]

    data["red"] = rgb[:, 0]
    data["green"] = rgb[:, 1]
    data["blue"] = rgb[:, 2]

    with open(path, "wb") as f:

        f.write(
            b"ply\n"
        )

        f.write(
            b"format binary_little_endian 1.0\n"
        )

        f.write(
            f"element vertex {len(data)}\n".encode()
        )

        f.write(
            b"property float x\n"
        )

        f.write(
            b"property float y\n"
        )

        f.write(
            b"property float z\n"
        )

        f.write(
            b"property uchar red\n"
        )

        f.write(
            b"property uchar green\n"
        )

        f.write(
            b"property uchar blue\n"
        )

        f.write(
            b"end_header\n"
        )

        data.tofile(f)

    print(
        f"Saved:\n{path}"
    )


# ============================================================
# BUILD FULL RECONSTRUCTION + SEMANTIC OVERLAY
# ============================================================

print("\n" + "=" * 70)
print("BUILDING FULL RECONSTRUCTION")
print("=" * 70)

if all_reconstruction_points:

    full_reconstruction_points = np.concatenate(
        all_reconstruction_points,
        axis=0,
    )

    full_reconstruction_rgb = np.concatenate(
        all_reconstruction_rgb,
        axis=0,
    )

else:

    full_reconstruction_points = np.empty(
        (0, 3),
        dtype=np.float32,
    )

    full_reconstruction_rgb = np.empty(
        (0, 3),
        dtype=np.uint8,
    )

print(
    f"Full reconstruction points: "
    f"{len(full_reconstruction_points):,}"
)

# Save the complete RGB reconstruction exactly as reconstructed.
if len(full_reconstruction_points) > 0:
    save_ply(
        FULL_RECONSTRUCTION_PLY,
        full_reconstruction_points,
        full_reconstruction_rgb,
    )
else:
    print("No full reconstruction points to save.")


print("\n" + "=" * 70)
print("BUILDING SEMANTIC OVERLAY OVER FULL RECONSTRUCTION")
print("=" * 70)

# `overlay_points` contains the complete per-frame point grids.
# `overlay_rgb` contains the same points with RF-DETR semantic colours
# applied only where a semantic class was detected.
#
# Concatenating these therefore gives:
#   - ALL reconstruction points remain present
#   - normal MapAnything RGB remains on non-semantic regions
#   - RF-DETR semantic colours appear on detected regions
if overlay_points:

    semantic_overlay_points = np.concatenate(
        overlay_points,
        axis=0,
    )

    semantic_overlay_rgb = np.concatenate(
        overlay_rgb,
        axis=0,
    )

else:

    semantic_overlay_points = np.empty(
        (0, 3),
        dtype=np.float32,
    )

    semantic_overlay_rgb = np.empty(
        (0, 3),
        dtype=np.uint8,
    )

print(
    f"Semantic overlay points: "
    f"{len(semantic_overlay_points):,}"
)

if len(semantic_overlay_points) > 0:
    save_ply(
        SEMANTIC_OVERLAY_PLY,
        semantic_overlay_points,
        semantic_overlay_rgb,
    )
else:
    print("No semantic overlay points to save.")


# ============================================================
# VOXEL FUSION
# ============================================================

def voxel_fuse(
    points,
    colors,
    confidences,
    classes,
    voxel_size,
    min_points=1,
):
    """
    Fuse semantic 3D points that fall into the same voxel.

    Position:
        confidence-weighted centroid

    Colour:
        confidence-weighted RGB average

    Class:
        class with the largest summed confidence
    """

    if len(points) == 0:
        return (
            points,
            colors,
            confidences,
            classes,
        )

    finite = (
        np.isfinite(points).all(axis=1)
        & np.isfinite(confidences)
    )

    points = points[finite]
    colors = colors[finite]
    confidences = confidences[finite]
    classes = classes[finite]

    if len(points) == 0:
        return (
            points,
            colors,
            confidences,
            classes,
        )

    voxel_coords = np.floor(
        points / float(voxel_size)
    ).astype(np.int64)

    order = np.lexsort(
        (
            voxel_coords[:, 2],
            voxel_coords[:, 1],
            voxel_coords[:, 0],
        )
    )

    voxel_coords = voxel_coords[order]
    points = points[order]
    colors = colors[order]
    confidences = confidences[order]
    classes = classes[order]

    if len(voxel_coords) == 1:
        starts = np.array([0])
        ends = np.array([1])
    else:
        changes = np.any(
            voxel_coords[1:]
            != voxel_coords[:-1],
            axis=1,
        )

        boundaries = (
            np.flatnonzero(changes)
            + 1
        )

        starts = np.concatenate(
            [
                np.array([0]),
                boundaries,
            ]
        )

        ends = np.concatenate(
            [
                boundaries,
                np.array([len(points)]),
            ]
        )

    fused_points = []
    fused_colors = []
    fused_confidences = []
    fused_classes = []

    for start, end in zip(
        starts,
        ends,
    ):

        count = end - start

        if count < min_points:
            continue

        p = points[start:end]
        c = colors[start:end]
        conf = confidences[start:end]
        cls = classes[start:end]

        weights = np.maximum(
            conf.astype(np.float64),
            1e-6,
        )

        weight_sum = weights.sum()

        weights /= weight_sum

        fused_position = (
            p.astype(np.float64)
            * weights[:, None]
        ).sum(axis=0)

        fused_colour = (
            c.astype(np.float64)
            * weights[:, None]
        ).sum(axis=0)

        class_scores = {}

        for class_id, weight in zip(
            cls,
            conf,
        ):
            class_id = int(class_id)

            class_scores[class_id] = (
                class_scores.get(
                    class_id,
                    0.0,
                )
                + float(weight)
            )

        fused_class = max(
            class_scores,
            key=class_scores.get,
        )

        fused_confidence = float(
            np.max(conf)
        )

        fused_points.append(
            fused_position.astype(np.float32)
        )

        fused_colors.append(
            np.clip(
                fused_colour,
                0,
                255,
            ).astype(np.uint8)
        )

        fused_confidences.append(
            fused_confidence
        )

        fused_classes.append(
            fused_class
        )

    if not fused_points:

        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.uint8),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int16),
        )

    return (
        np.asarray(
            fused_points,
            dtype=np.float32,
        ),
        np.asarray(
            fused_colors,
            dtype=np.uint8,
        ),
        np.asarray(
            fused_confidences,
            dtype=np.float32,
        ),
        np.asarray(
            fused_classes,
            dtype=np.int16,
        ),
    )


# ============================================================
# SAVE HAZARD GPS CSV
# ============================================================

print("\n" + "=" * 70)
print("SAVING HAZARD GPS CSV")
print("=" * 70)

hazards_df = pd.DataFrame(
    hazard_rows
)

hazards_df.to_csv(
    HAZARD_CSV,
    index=False,
)

print(
    f"Detections: {len(hazards_df):,}"
)

print(
    f"Saved:\n{HAZARD_CSV}"
)


# ============================================================
# INTERACTIVE GPS MAP
# ============================================================

print("\n" + "=" * 70)
print("CREATING GPS HAZARD MAP")
print("=" * 70)

try:

    map_center = [
        float(
            gps_df["latitude_deg"].mean()
        ),
        float(
            gps_df["longitude_deg"].mean()
        ),
    ]

    fmap = folium.Map(
        location=map_center,
        zoom_start=18,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    # --------------------------------------------------------
    # Full GPS trajectory
    # --------------------------------------------------------

    trajectory = list(
        zip(
            gps_df["latitude_deg"].astype(float),
            gps_df["longitude_deg"].astype(float),
        )
    )

    if trajectory:

        folium.PolyLine(
            trajectory,
            weight=4,
            opacity=0.7,
            tooltip="Full vehicle GPS trajectory",
        ).add_to(fmap)

    # --------------------------------------------------------
    # Selected reconstruction trajectory
    # --------------------------------------------------------

    selected_trajectory = [
        [
            metadata["latitude_deg"],
            metadata["longitude_deg"],
        ]
        for metadata in selected_metadata
    ]

    if selected_trajectory:

        folium.PolyLine(
            selected_trajectory,
            weight=6,
            opacity=0.9,
            tooltip="Selected reconstruction frames",
        ).add_to(fmap)

    # --------------------------------------------------------
    # Start
    # --------------------------------------------------------

    if trajectory:

        folium.Marker(
            trajectory[0],
            popup="START",
            tooltip="START",
            icon=folium.Icon(
                color="green",
                icon="play",
            ),
        ).add_to(fmap)

        # ----------------------------------------------------
        # End
        # ----------------------------------------------------

        folium.Marker(
            trajectory[-1],
            popup="END",
            tooltip="END",
            icon=folium.Icon(
                color="red",
                icon="stop",
            ),
        ).add_to(fmap)

    # --------------------------------------------------------
    # Selected frame layer
    # --------------------------------------------------------

    selected_layer = folium.FeatureGroup(
        name="Selected reconstruction frames"
    )

    selected_layer.add_to(fmap)

    for metadata in selected_metadata:

        folium.CircleMarker(
            [
                metadata["latitude_deg"],
                metadata["longitude_deg"],
            ],
            radius=3,
            tooltip=(
                f"Frame {metadata['frame']}"
            ),
            popup=(
                f"Frame: {metadata['frame']}<br>"
                f"GPS Δ: {metadata['gps_delta_ms']:.2f} ms"
            ),
        ).add_to(selected_layer)

    # --------------------------------------------------------
    # Hazard clusters
    # --------------------------------------------------------

    feature_groups = {}

    for class_name in CLASS_NAMES.values():

        feature_groups[class_name] = (
            MarkerCluster(
                name=class_name
            ).add_to(fmap)
        )

    # --------------------------------------------------------
    # Hazard markers
    # --------------------------------------------------------

    for _, row in hazards_df.iterrows():

        hazard = row["class_name"]

        marker_color = HAZARD_MARKER_COLORS.get(
            hazard,
            "black",
        )

        popup_html = f"""
        <div style="width:350px">
            <h4>{hazard}</h4>

            <b>Frame:</b> {int(row["frame"])}<br>
            <b>Confidence:</b> {row["confidence"]:.3f}<br>
            <b>Mask pixels:</b> {int(row["mask_pixels"])}<br><br>

            <b>Latitude:</b> {row["latitude_deg"]:.9f}<br>
            <b>Longitude:</b> {row["longitude_deg"]:.9f}<br>
            <b>Altitude:</b> {row["altitude_m"]:.3f} m<br><br>

            <b>Image timestamp:</b>
            {int(row["image_timestamp_ns"])}<br>

            <b>GPS timestamp:</b>
            {int(row["gps_timestamp_ns"])}<br>

            <b>GPS Δ:</b>
            {row["gps_delta_ms"]:.3f} ms<br><br>

            <b>Image:</b>
            {row["image_filename"]}
        </div>
        """

        folium.Marker(
            [
                row["latitude_deg"],
                row["longitude_deg"],
            ],
            popup=folium.Popup(
                popup_html,
                max_width=450,
            ),
            tooltip=(
                f"{hazard} | "
                f"Frame {int(row['frame'])} | "
                f"Conf {row['confidence']:.2f}"
            ),
            icon=folium.Icon(
                color=marker_color,
                icon="warning-sign",
            ),
        ).add_to(
            feature_groups[hazard]
        )

    # --------------------------------------------------------
    # Controls
    # --------------------------------------------------------

    fmap.add_child(
        MeasureControl()
    )

    folium.LayerControl(
        collapsed=False
    ).add_to(fmap)

    if trajectory:
        fmap.fit_bounds(trajectory)

    fmap.save(MAP_HTML)

    print(
        f"Saved:\n{MAP_HTML}"
    )

except Exception as e:

    print(
        f"\nCould not create map: {repr(e)}"
    )


# ============================================================
# CLEANUP
# ============================================================

del rfdetr_model
del predictions_cpu
del frame_points
del frame_rgb
del frame_shapes
del frame_valid_masks

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

print(
    f"Original candidate frames : "
    f"{len(candidate_metadata):,}"
)

print(
    f"Selected frames           : "
    f"{len(selected_metadata):,}"
)

print(
    f"Minimum travel distance   : "
    f"{MIN_TRAVEL_DISTANCE_M:.2f} m"
)

print(
    f"Voxel size                : "
    f"{VOXEL_SIZE_M:.3f}"
)

print(
    f"Raw semantic points       : "
    f"{len(semantic_points):,}"
)

print(
    f"Full reconstruction pts   : "
    f"{len(full_reconstruction_points):,}"
)

print(
    f"Semantic overlay pts      : "
    f"{len(semantic_overlay_points):,}"
)

print("\nOutputs:")

print(
    f"  Predictions : {PREDICTION_FILE}"
)

print(
    f"  Full PLY    : {FULL_RECONSTRUCTION_PLY}"
)

print(
    f"  Overlay PLY  : {SEMANTIC_OVERLAY_PLY}"
)

print(
    f"  Hazard CSV  : {HAZARD_CSV}"
)

print(
    f"  GPS map     : {MAP_HTML}"
)

print(
    f"  Images      : {LABELLED_DIR}"
)

print("=" * 70)
