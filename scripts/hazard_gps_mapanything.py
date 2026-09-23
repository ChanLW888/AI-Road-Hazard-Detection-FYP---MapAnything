#!/usr/bin/env python3

# ============================================================
# RF-DETR SEG SMALL + GPS HAZARD CLUSTERING
# + CENTRAL-FRAME MAPANYTHING 3D HAZARD RECONSTRUCTION
# + FOLIUM HTML MAP
#
# WHAT THIS VERSION DOES
# ============================================================
#
# 1. Runs RF-DETR Seg Small on all CAM2 images.
# 2. Saves the ACTUAL RF-DETR segmentation masks.
# 3. Matches detections to GPS.
# 4. Clusters same-class detections into physical hazards.
# 5. For EACH physical hazard cluster:
#
#       -> sorts detections chronologically
#       -> chooses the CENTRAL detection/frame
#       -> runs MapAnything ONCE on that image
#       -> uses the RF-DETR mask to keep ONLY hazard pixels
#       -> converts those pixels into 3D points
#
# 6. Saves one hazard-only coloured PLY per cluster.
# 7. Saves an interactive Plotly 3D HTML per cluster.
# 8. Saves a lightweight PNG 3D preview per cluster.
# 9. Displays the 3D preview + link to interactive 3D in the
#    Folium hazard popup.
#
# IMPORTANT
# ============================================================
#
# MapAnything is NOT given a cropped image.
#
# It receives the original CAM2 image with the same calibrated
# intrinsics/preprocessing as the working MapAnything script.
#
# AFTER MapAnything inference, the RF-DETR segmentation mask is
# projected onto the image-aligned pts3d output.
#
# Therefore:
#
#     MapAnything = geometry
#     RF-DETR     = hazard mask
#
# This preserves the camera geometry while giving a
# hazard-only 3D point cloud.
#
# The MapAnything run is SINGLE-VIEW for the central frame.
#
# ============================================================
#
# MODES
# ============================================================
#
# --mode inference
#     Rerun RF-DETR, then cluster, MapAnything, and map.
#
# --mode map
#     Reuse existing RF-DETR CSV + masks.
#     Recluster, rerun only missing MapAnything results,
#     then rebuild the HTML map.
#
# --mode auto
#     Run RF-DETR if saved results are missing.
#     Otherwise reuse them.
#
# Examples:
#
#   python scripts/hazard_tracker_mapanything.py \
#       --mode inference --cluster-radius 10
#
#   python scripts/hazard_tracker_mapanything.py \
#       --mode map --cluster-radius 20
#
#   python scripts/hazard_tracker_mapanything.py \
#       --mode auto --cluster-radius 10
#
# ============================================================


import argparse
import base64
import csv
import gc
import gzip
import hashlib
import html
import math
import random
import time

import cv2
import numpy as np
import pandas as pd
import torch
import folium

from pathlib import Path

from rfdetr import RFDETRSegSmall

from mapanything.models import MapAnything
from mapanything.utils.image import load_images

from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description=(
        "RF-DETR GPS hazard clustering with central-frame "
        "MapAnything hazard-only 3D reconstruction"
    )
)

parser.add_argument(
    "--mode",
    choices=["auto", "inference", "map"],
    default="auto",
)

parser.add_argument(
    "--cluster-radius",
    type=float,
    default=10.0,
    help="Hazard clustering radius in metres",
)

parser.add_argument(
    "--confidence",
    type=float,
    default=0.30,
    help="RF-DETR confidence threshold",
)

parser.add_argument(
    "--skip-mapanything",
    action="store_true",
    help="Skip MapAnything and only rebuild the RF-DETR map",
)

parser.add_argument(
    "--max-3d-points",
    type=int,
    default=100000,
    help="Maximum points saved to each hazard PLY",
)

parser.add_argument(
    "--max-plot-points",
    type=int,
    default=0,
    help=(
        "Maximum embedded scene points. 0 = embed ALL valid MapAnything points."
    ),
)

args = parser.parse_args()


# ============================================================
# SETTINGS
# ============================================================

CONFIDENCE_THRESHOLD = args.confidence
CLUSTER_RADIUS_METERS = args.cluster_radius

BATCH_SIZE = 128
TRAJECTORY_STRIDE = 20

MASK_ALPHA = 0.40
MASK_CONTOUR_THICKNESS = 3

EMBED_IMAGE_MAX_WIDTH = 900
EMBED_JPEG_QUALITY = 70

# Original calibrated CAM2 rectified image size.
CALIBRATION_WIDTH = 1920
CALIBRATION_HEIGHT = 1200

# Working CAM2 rectified P matrix from the verified
# MapAnything script.
P_RECTIFIED = np.array(
    [
        [510.651676055731, 0.0, 949.416735545071],
        [0.0, 510.456423883940, 607.626749742175],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(
    "/home/lcha0115/bt60_scratch/lcha_data/"
    "map-anything/personal_data/"
    "camera_lidar_gps_test_101/"
)

IMAGE_DIR = (
    BASE_DIR
    / "extracted"
    / "CAM2"
    / "images_rect"
)

GPS_FILE = (
    BASE_DIR
    / "extracted"
    / "gps"
    / "position.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "rfdetr_CAM2_results_clustered"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DETAILED_DETECTIONS_CSV = (
    OUTPUT_DIR
    / "rfdetr_detections_detailed.csv"
)

FRAME_SUMMARY_CSV = (
    OUTPUT_DIR
    / "detections_gps.csv"
)

CLUSTER_CSV = (
    OUTPUT_DIR
    / "hazard_clusters.csv"
)

MAP_OUTPUT = (
    OUTPUT_DIR
    / "hazard_map_clustered_mapanything.html"
)

MASK_DIR = (
    OUTPUT_DIR
    / "mask_cache"
)

MASK_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPRESENTATIVE_DIR = (
    OUTPUT_DIR
    / "hazard_clusters"
)

REPRESENTATIVE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MAPANYTHING_DIR = (
    OUTPUT_DIR
    / "mapanything_hazards"
)

MAPANYTHING_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MAPANYTHING_SUMMARY_CSV = (
    OUTPUT_DIR
    / "mapanything_hazard_summary.csv"
)

RFDETR_MODEL_PATH = Path(
    "/home/lcha0115/bt60_scratch/lcha_data/"
    "map-anything/personal_data/"
    "2D-Seg-Road-Nissan.v6i.coco-segmentation/"
    "outputs/checkpoint_best_total.pth"
)


# ============================================================
# CLASSES
# ============================================================

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

HAZARD_CLASSES = {
    "loose_trash",
    "mud_spill",
}


# ============================================================
# GPU
# ============================================================

NEEDS_GPU = (
    args.mode != "map"
    or not args.skip_mapanything
)

if NEEDS_GPU and not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU not available. "
        "RF-DETR and/or MapAnything require CUDA."
    )

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


print()
print("=" * 80)
print("RF-DETR + MAPANYTHING HAZARD TRACKER")
print("=" * 80)

print(f"Mode                 : {args.mode}")
print(f"Cluster radius       : {CLUSTER_RADIUS_METERS:.2f} m")
print(f"Confidence threshold : {CONFIDENCE_THRESHOLD:.2f}")
print(f"RF-DETR batch size   : {BATCH_SIZE}")
print(
    f"MapAnything          : "
    f"{'SKIPPED' if args.skip_mapanything else 'ENABLED'}"
)

if torch.cuda.is_available():
    print(
        f"GPU                  : "
        f"{torch.cuda.get_device_name(0)}"
    )
    print(
        f"CUDA                 : "
        f"{torch.version.cuda}"
    )
    print(
        f"PyTorch              : "
        f"{torch.__version__}"
    )

print("=" * 80)


# ============================================================
# IMAGE HELPERS
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def image_sort_key(path):
    if path.stem.isdigit():
        return (0, int(path.stem))
    return (1, path.stem)


def list_images():
    images = [
        p
        for p in IMAGE_DIR.iterdir()
        if (
            p.is_file()
            and p.suffix.lower() in IMAGE_EXTENSIONS
        )
    ]

    images.sort(key=image_sort_key)
    return images


# ============================================================
# GPS
# ============================================================

print()
print("=" * 80)
print("LOADING GPS")
print("=" * 80)

gps_df = pd.read_csv(GPS_FILE)

required_gps_columns = [
    "ros_timestamp_ns",
    "latitude_deg",
    "longitude_deg",
]

for col in required_gps_columns:
    if col not in gps_df.columns:
        raise RuntimeError(
            f"Missing GPS column: {col}"
        )

gps_df["ros_timestamp_ns"] = pd.to_numeric(
    gps_df["ros_timestamp_ns"],
    errors="coerce",
)

gps_df["latitude_deg"] = pd.to_numeric(
    gps_df["latitude_deg"],
    errors="coerce",
)

gps_df["longitude_deg"] = pd.to_numeric(
    gps_df["longitude_deg"],
    errors="coerce",
)

if "altitude_m" in gps_df.columns:
    gps_df["altitude_m"] = pd.to_numeric(
        gps_df["altitude_m"],
        errors="coerce",
    )
else:
    gps_df["altitude_m"] = 0.0

gps_df = gps_df.dropna(
    subset=[
        "ros_timestamp_ns",
        "latitude_deg",
        "longitude_deg",
    ]
).copy()

gps_df = gps_df[
    gps_df["latitude_deg"].between(-90, 90)
]

gps_df = gps_df[
    gps_df["longitude_deg"].between(-180, 180)
]

gps_df = gps_df.sort_values(
    "ros_timestamp_ns"
).reset_index(drop=True)

gps_timestamps = gps_df[
    "ros_timestamp_ns"
].to_numpy(
    dtype=np.int64
)


def get_closest_gps(timestamp_ns):
    idx = np.searchsorted(
        gps_timestamps,
        timestamp_ns,
    )

    if idx <= 0:
        idx = 0

    elif idx >= len(gps_timestamps):
        idx = len(gps_timestamps) - 1

    else:
        before = idx - 1
        after = idx

        before_diff = abs(
            gps_timestamps[before]
            - timestamp_ns
        )

        after_diff = abs(
            gps_timestamps[after]
            - timestamp_ns
        )

        if after_diff < before_diff:
            idx = after
        else:
            idx = before

    row = gps_df.iloc[idx]

    return {
        "gps_timestamp_ns": int(
            row["ros_timestamp_ns"]
        ),
        "latitude": float(
            row["latitude_deg"]
        ),
        "longitude": float(
            row["longitude_deg"]
        ),
        "altitude": float(
            row["altitude_m"]
        ),
        "time_difference_ms": (
            abs(
                int(row["ros_timestamp_ns"])
                - int(timestamp_ns)
            )
            / 1e6
        ),
    }


print(
    f"GPS points: {len(gps_df):,}"
)


# ============================================================
# HAVERSINE
# ============================================================

def haversine_meters(
    lat1,
    lon1,
    lat2,
    lon2,
):
    R = 6371000.0

    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)

    dlat = math.radians(
        lat2 - lat1
    )

    dlon = math.radians(
        lon2 - lon1
    )

    a = (
        math.sin(dlat / 2.0) ** 2
        +
        math.cos(lat1)
        *
        math.cos(lat2)
        *
        math.sin(dlon / 2.0) ** 2
    )

    c = (
        2.0
        *
        math.atan2(
            math.sqrt(a),
            math.sqrt(1.0 - a),
        )
    )

    return R * c


# ============================================================
# MASK FUNCTIONS
# ============================================================

def normalise_mask(
    mask,
    image_height,
    image_width,
):
    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()

    mask = np.asarray(mask)
    mask = np.squeeze(mask)

    if mask.dtype != np.bool_:
        mask = mask > 0.5

    if mask.ndim != 2:
        raise ValueError(
            f"Unexpected mask shape: {mask.shape}"
        )

    if (
        mask.shape[0] != image_height
        or mask.shape[1] != image_width
    ):
        mask = cv2.resize(
            mask.astype(np.uint8),
            (image_width, image_height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

    return mask


def save_mask(mask, path):
    mask_uint8 = (
        mask.astype(np.uint8) * 255
    )

    ok = cv2.imwrite(
        str(path),
        mask_uint8,
        [
            cv2.IMWRITE_PNG_COMPRESSION,
            9,
        ],
    )

    if not ok:
        raise RuntimeError(
            f"Could not save mask: {path}"
        )


def load_mask(path):
    mask = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE,
    )

    if mask is None:
        raise RuntimeError(
            f"Could not load mask: {path}"
        )

    return mask > 127


# ============================================================
# SEGMENTATION DRAWING
# ============================================================

def get_class_id(class_name):
    for class_id, name in CLASS_NAMES.items():
        if name == class_name:
            return class_id
    return -1


def draw_segmentation(
    image,
    mask,
    class_id,
    confidence,
):
    class_name = CLASS_NAMES.get(
        int(class_id),
        str(class_id),
    )

    color = CLASS_COLORS.get(
        int(class_id),
        (255, 255, 255),
    )

    overlay = image.copy()
    overlay[mask] = color

    image[:] = cv2.addWeighted(
        overlay,
        MASK_ALPHA,
        image,
        1.0 - MASK_ALPHA,
        0,
    )

    mask_uint8 = (
        mask.astype(np.uint8) * 255
    )

    contours, _ = cv2.findContours(
        mask_uint8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    cv2.drawContours(
        image,
        contours,
        -1,
        color,
        MASK_CONTOUR_THICKNESS,
    )

    ys, xs = np.where(mask)

    if len(xs) == 0:
        return

    x = int(xs.min())
    y = int(ys.min())

    label = (
        f"{class_name} "
        f"{confidence:.2f}"
    )

    (tw, th), baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        2,
    )

    label_y1 = max(
        0,
        y - th - baseline - 8,
    )

    cv2.rectangle(
        image,
        (x, label_y1),
        (x + tw + 10, y),
        color,
        -1,
    )

    cv2.putText(
        image,
        label,
        (x + 5, y - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )


# ============================================================
# DETERMINE RF-DETR RUN
# ============================================================

if args.mode == "inference":
    RUN_INFERENCE = True

elif args.mode == "map":
    RUN_INFERENCE = False

else:
    RUN_INFERENCE = not (
        DETAILED_DETECTIONS_CSV.exists()
        and MASK_DIR.exists()
        and any(MASK_DIR.glob("*.png"))
    )


if RUN_INFERENCE:
    print()
    print("=" * 80)
    print("RF-DETR WILL RUN")
    print("=" * 80)
else:
    print()
    print("=" * 80)
    print("RF-DETR WILL BE SKIPPED")
    print("=" * 80)

    if not DETAILED_DETECTIONS_CSV.exists():
        raise FileNotFoundError(
            f"Missing:\n{DETAILED_DETECTIONS_CSV}\n\n"
            "Run --mode inference first."
        )

    test_df = pd.read_csv(
        DETAILED_DETECTIONS_CSV,
        nrows=1,
    )

    if "mask_path" not in test_df.columns:
        raise RuntimeError(
            "Existing RF-DETR CSV does not contain "
            "'mask_path'. Run a fresh --mode inference."
        )

    if not any(MASK_DIR.glob("*.png")):
        raise RuntimeError(
            f"No masks found in {MASK_DIR}. "
            "Run a fresh --mode inference."
        )


# ============================================================
# RF-DETR INFERENCE
# ============================================================

if RUN_INFERENCE:

    print()
    print("=" * 80)
    print("LOADING RF-DETR SEG SMALL")
    print("=" * 80)

    t_model = time.perf_counter()

    rfdetr_model = RFDETRSegSmall.from_checkpoint(
        str(RFDETR_MODEL_PATH)
    )

    print(
        f"Checkpoint loaded in "
        f"{time.perf_counter() - t_model:.2f} sec"
    )

    print()
    print("=" * 80)
    print("COMPILING OPTIMIZED FP16 RF-DETR")
    print("=" * 80)

    t_compile = time.perf_counter()

    rfdetr_model.inference(
        dtype=torch.float16,
        batch_size=BATCH_SIZE,
    )

    print(
        f"Optimized model ready in "
        f"{time.perf_counter() - t_compile:.2f} sec"
    )

    images = list_images()

    print(
        f"Images found: {len(images):,}"
    )

    # Clear old masks.
    for old_mask in MASK_DIR.glob("*.png"):
        old_mask.unlink()

    detailed_file = open(
        DETAILED_DETECTIONS_CSV,
        "w",
        newline="",
    )

    detailed_writer = csv.writer(
        detailed_file
    )

    detailed_writer.writerow([
        "image",
        "image_ros_timestamp_ns",
        "gps_timestamp_ns",
        "gps_time_difference_ms",
        "latitude",
        "longitude",
        "altitude",
        "class",
        "class_id",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "mask_path",
    ])

    frame_file = open(
        FRAME_SUMMARY_CSV,
        "w",
        newline="",
    )

    frame_writer = csv.writer(
        frame_file
    )

    frame_writer.writerow([
        "image",
        "image_ros_timestamp_ns",
        "gps_timestamp_ns",
        "gps_time_difference_ms",
        "latitude",
        "longitude",
        "altitude",
        "hazard_detected",
        "hazard_classes",
        "loose_trash_count",
        "mud_spill_count",
        "all_detections",
    ])

    processed = 0
    hazard_frames = 0
    hazard_detections = 0

    inference_start = time.perf_counter()
    last_print = time.perf_counter()

    for batch_start in range(
        0,
        len(images),
        BATCH_SIZE,
    ):

        batch_paths = images[
            batch_start:
            batch_start + BATCH_SIZE
        ]

        batch_images = []
        valid_paths = []

        for image_path in batch_paths:

            image = cv2.imread(
                str(image_path),
                cv2.IMREAD_COLOR,
            )

            if image is None:
                print(
                    f"\nWARNING: failed to read "
                    f"{image_path}"
                )
                continue

            batch_images.append(image)
            valid_paths.append(image_path)

        if not batch_images:
            continue

        real_batch_size = len(
            batch_images
        )

        if real_batch_size < BATCH_SIZE:
            last_image = batch_images[-1]

            for _ in range(
                BATCH_SIZE - real_batch_size
            ):
                batch_images.append(
                    last_image
                )

        with torch.inference_mode():
            batch_detections = (
                rfdetr_model.predict(
                    batch_images
                )
            )

        batch_detections = batch_detections[
            :real_batch_size
        ]

        for (
            image,
            image_path,
            detections,
        ) in zip(
            batch_images[:real_batch_size],
            valid_paths,
            batch_detections,
        ):

            processed += 1

            try:
                image_timestamp_ns = int(
                    image_path.stem
                )
            except ValueError:
                print(
                    f"\nWARNING: invalid timestamp: "
                    f"{image_path.name}"
                )
                continue

            gps = get_closest_gps(
                image_timestamp_ns
            )

            image_height, image_width = (
                image.shape[:2]
            )

            boxes = np.asarray(
                getattr(
                    detections,
                    "xyxy",
                    [],
                )
            )

            class_ids = np.asarray(
                getattr(
                    detections,
                    "class_id",
                    [],
                )
            )

            confidences = np.asarray(
                getattr(
                    detections,
                    "confidence",
                    [],
                )
            )

            masks = getattr(
                detections,
                "mask",
                None,
            )

            if masks is None:
                masks = getattr(
                    detections,
                    "masks",
                    None,
                )

            hazard_classes_this_frame = set()
            all_detections = []

            loose_trash_count = 0
            mud_spill_count = 0
            hazard_count = 0

            if masks is None:
                raise RuntimeError(
                    "RF-DETR did not return segmentation masks."
                )

            for detection_index in range(
                len(confidences)
            ):

                confidence = float(
                    confidences[detection_index]
                )

                if confidence < CONFIDENCE_THRESHOLD:
                    continue

                class_id = int(
                    class_ids[detection_index]
                )

                class_name = CLASS_NAMES.get(
                    class_id,
                    str(class_id),
                )

                all_detections.append(
                    f"{class_name}:{confidence:.3f}"
                )

                if class_name not in HAZARD_CLASSES:
                    continue

                hazard_count += 1
                hazard_classes_this_frame.add(
                    class_name
                )

                box = np.asarray(
                    boxes[detection_index],
                    dtype=float,
                )

                mask = normalise_mask(
                    masks[detection_index],
                    image_height,
                    image_width,
                )

                mask_filename = (
                    f"{image_timestamp_ns}_"
                    f"{detection_index}_"
                    f"{class_name}.png"
                )

                mask_path = (
                    MASK_DIR
                    / mask_filename
                )

                save_mask(
                    mask,
                    mask_path,
                )

                detailed_writer.writerow([
                    image_path.name,
                    image_timestamp_ns,
                    gps["gps_timestamp_ns"],
                    gps["time_difference_ms"],
                    gps["latitude"],
                    gps["longitude"],
                    gps["altitude"],
                    class_name,
                    class_id,
                    confidence,
                    float(box[0]),
                    float(box[1]),
                    float(box[2]),
                    float(box[3]),
                    str(mask_path),
                ])

                if class_name == "loose_trash":
                    loose_trash_count += 1

                elif class_name == "mud_spill":
                    mud_spill_count += 1

            hazard_detected = (
                hazard_count > 0
            )

            if hazard_detected:
                hazard_frames += 1
                hazard_detections += hazard_count

            frame_writer.writerow([
                image_path.name,
                image_timestamp_ns,
                gps["gps_timestamp_ns"],
                gps["time_difference_ms"],
                gps["latitude"],
                gps["longitude"],
                gps["altitude"],
                hazard_detected,
                ";".join(
                    sorted(
                        hazard_classes_this_frame
                    )
                ),
                loose_trash_count,
                mud_spill_count,
                ";".join(all_detections),
            ])

        now = time.perf_counter()

        if now - last_print >= 2.0:

            elapsed = (
                now - inference_start
            )

            fps = (
                processed / elapsed
                if elapsed > 0
                else 0
            )

            remaining = (
                len(images) - processed
            )

            eta = (
                remaining / fps
                if fps > 0
                else 0
            )

            print(
                f"\r"
                f"{processed:,}/"
                f"{len(images):,} "
                f"({100 * processed / len(images):.1f}%) | "
                f"{fps:.2f} img/s | "
                f"ETA {eta / 60:.1f} min | "
                f"hazard frames {hazard_frames:,} | "
                f"hazard detections {hazard_detections:,}",
                end="",
                flush=True,
            )

            last_print = now

    detailed_file.close()
    frame_file.close()

    print()
    print()
    print("=" * 80)
    print("RF-DETR COMPLETE")
    print("=" * 80)
    print(
        f"Images processed  : {processed:,}"
    )
    print(
        f"Hazard frames     : {hazard_frames:,}"
    )
    print(
        f"Hazard detections : {hazard_detections:,}"
    )


# ============================================================
# LOAD RF-DETR DETECTIONS
# ============================================================

print()
print("=" * 80)
print("LOADING SAVED HAZARD DETECTIONS")
print("=" * 80)

detection_df = pd.read_csv(
    DETAILED_DETECTIONS_CSV
)

if len(detection_df) == 0:
    raise RuntimeError(
        "No hazard detections found."
    )

required_detection_columns = [
    "image",
    "image_ros_timestamp_ns",
    "latitude",
    "longitude",
    "altitude",
    "class",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "mask_path",
]

for col in required_detection_columns:
    if col not in detection_df.columns:
        raise RuntimeError(
            f"Missing detection CSV column: {col}"
        )

detection_df = detection_df.sort_values(
    "image_ros_timestamp_ns"
).reset_index(drop=True)

print(
    f"Hazard detections loaded: "
    f"{len(detection_df):,}"
)


# ============================================================
# CLUSTER
# ============================================================

print()
print("=" * 80)
print("CLUSTERING HAZARDS")
print("=" * 80)

print(
    "Rule: first detection becomes permanent anchor."
)

clusters = []

cluster_start = time.perf_counter()

for _, row in detection_df.iterrows():

    hazard_class = str(
        row["class"]
    )

    latitude = float(
        row["latitude"]
    )

    longitude = float(
        row["longitude"]
    )

    altitude = float(
        row["altitude"]
    )

    confidence = float(
        row["confidence"]
    )

    image_name = str(
        row["image"]
    )

    timestamp_ns = int(
        row["image_ros_timestamp_ns"]
    )

    mask_path = str(
        row["mask_path"]
    )

    box = np.array(
        [
            float(row["x1"]),
            float(row["y1"]),
            float(row["x2"]),
            float(row["y2"]),
        ],
        dtype=float,
    )

    detection_record = {
        "image": image_name,
        "timestamp_ns": timestamp_ns,
        "latitude": latitude,
        "longitude": longitude,
        "altitude": altitude,
        "confidence": confidence,
        "box": box,
        "mask_path": mask_path,
    }

    closest_cluster = None
    closest_distance = float("inf")

    for cluster in clusters:

        if cluster["class"] != hazard_class:
            continue

        distance = haversine_meters(
            latitude,
            longitude,
            cluster["anchor_latitude"],
            cluster["anchor_longitude"],
        )

        if distance < closest_distance:
            closest_distance = distance
            closest_cluster = cluster

    if (
        closest_cluster is not None
        and closest_distance <= CLUSTER_RADIUS_METERS
    ):

        cluster = closest_cluster

        cluster["detections"].append(
            detection_record
        )

        cluster["detection_count"] += 1

        cluster["max_distance_m"] = max(
            cluster["max_distance_m"],
            closest_distance,
        )

        if confidence > cluster["best_confidence"]:

            cluster["best_confidence"] = confidence
            cluster["best_detection"] = (
                detection_record
            )

    else:

        cluster_id = len(clusters) + 1

        cluster = {
            "cluster_id": cluster_id,
            "class": hazard_class,

            "anchor_latitude": latitude,
            "anchor_longitude": longitude,
            "anchor_altitude": altitude,
            "anchor_image": image_name,
            "anchor_timestamp_ns": timestamp_ns,

            "detection_count": 1,
            "max_distance_m": 0.0,

            "best_confidence": confidence,
            "best_detection": detection_record,

            "detections": [
                detection_record
            ],
        }

        clusters.append(cluster)


cluster_time = (
    time.perf_counter()
    - cluster_start
)

print(
    f"Original detections   : "
    f"{len(detection_df):,}"
)

print(
    f"Physical hazard sites : "
    f"{len(clusters):,}"
)

print(
    f"Clustering time       : "
    f"{cluster_time:.2f} sec"
)


# ============================================================
# CENTRAL DETECTION
# ============================================================

print()
print("=" * 80)
print("SELECTING CENTRAL FRAME FOR EACH HAZARD")
print("=" * 80)

for cluster in clusters:

    detections = sorted(
        cluster["detections"],
        key=lambda d: d["timestamp_ns"],
    )

    cluster["detections"] = detections

    timestamps = np.array(
        [
            d["timestamp_ns"]
            for d in detections
        ],
        dtype=np.int64,
    )

    median_timestamp = np.median(
        timestamps
    )

    central_detection = min(
        detections,
        key=lambda d: abs(
            d["timestamp_ns"]
            - median_timestamp
        ),
    )

    cluster["central_detection"] = (
        central_detection
    )

    print(
        f"Hazard #{cluster['cluster_id']:04d} | "
        f"{cluster['class']:<12} | "
        f"{len(detections):>4} detections | "
        f"central = {central_detection['image']}"
    )


# ============================================================
# REPRESENTATIVE / CENTRAL SEGMENTATION IMAGES
# ============================================================

print()
print("=" * 80)
print("SAVING CENTRAL HAZARD IMAGES")
print("=" * 80)

for old_file in REPRESENTATIVE_DIR.iterdir():
    if old_file.is_file():
        old_file.unlink()

for cluster in clusters:

    central = cluster[
        "central_detection"
    ]

    source_path = (
        IMAGE_DIR
        / central["image"]
    )

    mask_path = Path(
        central["mask_path"]
    )

    if not source_path.exists():
        print(
            f"WARNING: missing central image: "
            f"{source_path}"
        )
        continue

    if not mask_path.exists():
        print(
            f"WARNING: missing central mask: "
            f"{mask_path}"
        )
        continue

    image = cv2.imread(
        str(source_path),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        continue

    mask = load_mask(
        mask_path
    )

    image_height, image_width = (
        image.shape[:2]
    )

    mask = normalise_mask(
        mask,
        image_height,
        image_width,
    )

    draw_segmentation(
        image,
        mask,
        get_class_id(cluster["class"]),
        central["confidence"],
    )

    cv2.putText(
        image,
        (
            f"HAZARD #{cluster['cluster_id']:04d} | "
            f"{cluster['class']} | CENTRAL FRAME"
        ),
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.90,
        (0, 0, 255),
        3,
        cv2.LINE_AA,
    )

    cv2.putText(
        image,
        (
            f"GPS: "
            f"{central['latitude']:.8f}, "
            f"{central['longitude']:.8f}"
        ),
        (30, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    output_name = (
        f"hazard_"
        f"{cluster['cluster_id']:04d}_"
        f"{cluster['class']}_central.jpg"
    )

    output_path = (
        REPRESENTATIVE_DIR
        / output_name
    )

    cv2.imwrite(
        str(output_path),
        image,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            EMBED_JPEG_QUALITY,
        ],
    )

    cluster["representative_image"] = (
        output_name
    )


# ============================================================
# PLY WRITER
# ============================================================

def save_ply(
    path,
    points,
    rgb,
):
    if len(points) == 0:
        raise RuntimeError(
            "Cannot save empty PLY."
        )

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
            b"format binary_little_endian 1.0\n"
        )

        f.write(
            f"element vertex {len(data)}\n".encode()
        )

        f.write(
            b"property float x\n"
            b"property float y\n"
            b"property float z\n"
            b"property uchar red\n"
            b"property uchar green\n"
            b"property uchar blue\n"
            b"end_header\n"
        )

        data.tofile(f)


# ============================================================
# MAPANYTHING HELPERS
# ============================================================

def tensor_to_numpy(value):
    if torch.is_tensor(value):
        return (
            value.detach()
            .cpu()
            .numpy()
        )
    return np.asarray(value)


def extract_pts3d(prediction):
    if "pts3d" not in prediction:
        raise RuntimeError(
            "MapAnything prediction does not contain 'pts3d'."
        )

    pts3d = tensor_to_numpy(
        prediction["pts3d"]
    )

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
            f"Unexpected pts3d shape: "
            f"{pts3d.shape}"
        )

    return pts3d.astype(
        np.float32,
        copy=False,
    )


def extract_prediction_mask(
    prediction,
    height,
    width,
):
    if "mask" not in prediction:
        return np.ones(
            (height, width),
            dtype=bool,
        )

    map_mask = tensor_to_numpy(
        prediction["mask"]
    )

    map_mask = np.squeeze(
        map_mask
    )

    if map_mask.ndim != 2:
        return np.ones(
            (height, width),
            dtype=bool,
        )

    if (
        map_mask.shape[0] != height
        or map_mask.shape[1] != width
    ):
        map_mask = cv2.resize(
            map_mask.astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )

    return map_mask.astype(bool)


def resize_rgb_to_shape(
    image_path,
    width,
    height,
):
    image = Image.open(
        image_path
    ).convert("RGB")

    image = image.resize(
        (width, height),
        Image.Resampling.BILINEAR,
    )

    return np.asarray(
        image,
        dtype=np.uint8,
    )


# ============================================================
# STATIC 3D PREVIEW
# ============================================================

def save_3d_preview(
    path,
    points,
    rgb,
    title,
):
    if len(points) == 0:
        return

    display_points = points
    display_rgb = rgb

    if len(display_points) > 15000:

        rng = np.random.default_rng(
            12345
        )

        indices = rng.choice(
            len(display_points),
            size=15000,
            replace=False,
        )

        display_points = (
            display_points[indices]
        )

        display_rgb = (
            display_rgb[indices]
        )

    center = np.median(
        display_points,
        axis=0,
    )

    local_points = (
        display_points - center
    )

    fig = plt.figure(
        figsize=(8.5, 6.0),
        dpi=120,
    )

    ax = fig.add_subplot(
        111,
        projection="3d",
    )

    rgb_float = (
        display_rgb.astype(np.float32)
        / 255.0
    )

    ax.scatter(
        local_points[:, 0],
        local_points[:, 1],
        local_points[:, 2],
        c=rgb_float,
        s=1.5,
        alpha=0.85,
        linewidths=0,
    )

    ax.set_title(title)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.view_init(
        elev=18,
        azim=-65,
    )

    fig.tight_layout()

    fig.savefig(
        path,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# INTERACTIVE 3D HTML
# ============================================================

def save_interactive_3d(
    path,
    points,
    rgb,
    title,
):
    # Plotly is imported globally because the embedded popup
    # viewer is built inside reconstruct_hazard().
    plot_points = points
    plot_rgb = rgb

    if len(plot_points) > args.max_plot_points:

        rng = np.random.default_rng(
            12345
        )

        indices = rng.choice(
            len(plot_points),
            size=args.max_plot_points,
            replace=False,
        )

        plot_points = (
            plot_points[indices]
        )

        plot_rgb = (
            plot_rgb[indices]
        )

    center = np.median(
        plot_points,
        axis=0,
    )

    local_points = (
        plot_points - center
    )

    marker_colors = [
        (
            f"rgb({int(c[0])},"
            f"{int(c[1])},"
            f"{int(c[2])})"
        )
        for c in plot_rgb
    ]

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=local_points[:, 0],
                y=local_points[:, 1],
                z=local_points[:, 2],
                mode="markers",
                marker=dict(
                    size=2,
                    color=marker_colors,
                    opacity=0.85,
                ),
            )
        ]
    )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=45,
        ),
    )

    fig.write_html(
        str(path),
        include_plotlyjs="cdn",
        full_html=True,
    )


# ============================================================
# MAPANYTHING SINGLE-HAZARD INFERENCE
# ============================================================

def _sample_points(points, rgb, max_points, seed=12345):
    """Return a display sample while keeping RGB aligned."""
    if max_points <= 0 or len(points) <= max_points:
        return points, rgb

    rng = np.random.default_rng(seed)
    idx = rng.choice(
        len(points),
        size=max_points,
        replace=False,
    )
    return points[idx], rgb[idx]


def _encode_pointcloud(points, rgb, max_points=0):
    """
    Encode XYZ float32 + RGB uint8 into gzip-compressed binary Base64.

    The payload is embedded directly in the final Folium HTML.  No
    external point-cloud file is required by the browser.
    """
    points, rgb = _sample_points(
        np.asarray(points, dtype=np.float32),
        np.asarray(rgb, dtype=np.uint8),
        max_points,
    )

    # 15 bytes / point: x,y,z float32 + r,g,b uint8.
    raw = np.empty(
        (len(points), 15),
        dtype=np.uint8,
    )
    raw[:, 0:12] = points.view(np.uint8).reshape(-1, 12)
    raw[:, 12:15] = rgb

    compressed = gzip.compress(
        raw.tobytes(),
        compresslevel=6,
    )

    return base64.b64encode(compressed).decode("ascii"), len(points)


def build_embedded_3d_payload(
    scene_points,
    scene_rgb,
    hazard_points,
    hazard_rgb,
):
    """
    Build the complete self-contained browser payload.

    The COMPLETE local scene is embedded. Hazard points are embedded
    separately so the browser can draw them as a highlighted overlay.
    """
    scene_payload, scene_count = _encode_pointcloud(
        scene_points,
        scene_rgb,
        args.max_plot_points,
    )

    hazard_payload, hazard_count = _encode_pointcloud(
        hazard_points,
        hazard_rgb,
        0,
    )

    # Centre the displayed coordinates around the complete scene median.
    centre = np.median(
        np.asarray(scene_points, dtype=np.float32),
        axis=0,
    ).astype(np.float32)

    return {
        "scene": scene_payload,
        "hazard": hazard_payload,
        "scene_count": int(scene_count),
        "hazard_count": int(hazard_count),
        "centre": [float(x) for x in centre],
    }


def save_full_scene_viewer(
    path,
    points,
    rgb,
    hazard_points,
    hazard_rgb,
    title,
):
    """
    Retain a standalone viewer as an optional debugging artifact.

    The FINAL Folium map does NOT load this file. The actual 3D data
    is embedded directly in the main Folium HTML by
    inject_embedded_3d_runtime().
    """
    scene_points, scene_rgb = _sample_points(
        points,
        rgb,
        args.max_plot_points,
        seed=12345,
    )
    hz_points, hz_rgb = _sample_points(
        hazard_points,
        hazard_rgb,
        0,
        seed=54321,
    )

    centre = np.median(scene_points, axis=0)
    scene_local = (scene_points - centre).astype(np.float32)
    hz_local = (hz_points - centre).astype(np.float32)

    scene_colors = [
        f"rgb({int(c[0])},{int(c[1])},{int(c[2])})"
        for c in scene_rgb
    ]
    hazard_colors = [
        f"rgb({int(c[0])},{int(c[1])},{int(c[2])})"
        for c in hz_rgb
    ]

    import json

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=scene_local[:, 0].tolist(),
                y=scene_local[:, 1].tolist(),
                z=scene_local[:, 2].tolist(),
                mode="markers",
                name="Full local point cloud",
                marker=dict(size=1.8, color=scene_colors, opacity=0.72),
            ),
            go.Scatter3d(
                x=hz_local[:, 0].tolist(),
                y=hz_local[:, 1].tolist(),
                z=hz_local[:, 2].tolist(),
                mode="markers",
                name="RF-DETR hazard",
                marker=dict(size=4.0, color=hazard_colors, opacity=1.0),
            ),
        ]
    )
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode="data",
        ),
        height=760,
        margin=dict(l=0, r=0, b=0, t=45),
    )
    fig.write_html(
        str(path),
        include_plotlyjs="cdn",
        full_html=True,
        config={
            "displaylogo": False,
            "responsive": True,
            "scrollZoom": True,
        },
    )


def inject_embedded_3d_runtime(
    map_path,
    embedded_payloads,
):
    """
    Inject Plotly + ALL point-cloud data directly into the final map.

    The point clouds are embedded in the main Folium HTML as compressed
    Base64 binary.  No external viewer/data file is required.

    Rendering is deliberately triggered AFTER a Leaflet popup is opened.
    This avoids Plotly/WebGL being initialised while the popup is hidden,
    which can result in a completely blank 3D canvas.
    """
    import json
    import plotly.offline

    map_path = Path(map_path)
    html_text = map_path.read_text(encoding="utf-8")

    payload_json = json.dumps(
        embedded_payloads,
        separators=(",", ":"),
    )

    plotly_js = plotly.offline.get_plotlyjs()

    runtime = f"""
<script>
{plotly_js}
</script>

<script>
const HAZARD_3D_PAYLOADS = {payload_json};

function _base64ToBytes(b64) {{
    const bin = atob(b64);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; ++i) {{
        out[i] = bin.charCodeAt(i);
    }}
    return out;
}}

async function _gunzipBase64(b64) {{
    const compressed = _base64ToBytes(b64);

    if (!('DecompressionStream' in window)) {{
        throw new Error(
            'This browser does not support gzip decompression. ' +
            'Please use a current Chrome, Edge, or Firefox.'
        );
    }}

    const stream = new Blob([compressed])
        .stream()
        .pipeThrough(new DecompressionStream('gzip'));

    return new Uint8Array(
        await new Response(stream).arrayBuffer()
    );
}}

async function _decodeCloud(b64, centre) {{
    const bytes = await _gunzipBase64(b64);

    const stride = 15;
    const n = Math.floor(bytes.length / stride);

    const x = new Float32Array(n);
    const y = new Float32Array(n);
    const z = new Float32Array(n);

    const rgb = new Uint8Array(n * 3);

    const view = new DataView(
        bytes.buffer,
        bytes.byteOffset,
        bytes.byteLength
    );

    for (let i = 0; i < n; ++i) {{
        const o = i * stride;

        x[i] = view.getFloat32(o, true) - centre[0];
        y[i] = view.getFloat32(o + 4, true) - centre[1];
        z[i] = view.getFloat32(o + 8, true) - centre[2];

        rgb[3 * i]     = bytes[o + 12];
        rgb[3 * i + 1] = bytes[o + 13];
        rgb[3 * i + 2] = bytes[o + 14];
    }}

    return {{
        x: x,
        y: y,
        z: z,
        rgb: rgb,
        count: n
    }};
}}

/*
 * Convert RGB bytes to Plotly colour strings.
 *
 * This is intentionally done only when the viewer is opened, rather
 * than storing 125k+ colour strings in the HTML payload.
 */
function _rgbStrings(rgb) {{
    const n = Math.floor(rgb.length / 3);
    const result = new Array(n);

    for (let i = 0, j = 0; i < n; ++i, j += 3) {{
        result[i] =
            'rgb(' +
            rgb[j] + ',' +
            rgb[j + 1] + ',' +
            rgb[j + 2] +
            ')';
    }}

    return result;
}}

function _show3DError(div, err) {{
    div.dataset.rendered = '0';

    div.innerHTML =
        '<div style="' +
        'padding:20px;' +
        'font-family:Arial,sans-serif;' +
        'font-size:15px;' +
        'color:#b00020;' +
        '">' +
        '<b>3D viewer error</b><br>' +
        String(err) +
        '<br><br>' +
        '<small>Open the browser developer console (F12) for details.</small>' +
        '</div>';

    console.error('Embedded MapAnything 3D error:', err);
}}

async function renderEmbeddedHazard3D(key) {{
    const payload = HAZARD_3D_PAYLOADS[key];
    const div = document.getElementById('hazard3d_' + key);

    if (!payload || !div) {{
        return;
    }}

    /*
     * If Plotly is already there, just resize it.  This is important
     * when the user reopens the same Leaflet popup.
     */
    if (div.dataset.rendered === '1') {{
        setTimeout(function() {{
            try {{
                Plotly.Plots.resize(div);
            }} catch (e) {{}}
        }}, 100);
        return;
    }}

    if (div.dataset.loading === '1') {{
        return;
    }}

    div.dataset.loading = '1';

    div.innerHTML =
        '<div style="' +
        'padding:24px;' +
        'font-family:Arial,sans-serif;' +
        'font-size:16px;' +
        'text-align:center;' +
        '">' +
        'Loading ' +
        Number(payload.scene_count).toLocaleString() +
        ' embedded MapAnything points...' +
        '</div>';

    try {{
        /*
         * Wait until the Leaflet popup has actually been laid out.
         * Plotly is very sensitive to zero/incorrect dimensions.
         */
        await new Promise(function(resolve) {{
            requestAnimationFrame(function() {{
                requestAnimationFrame(resolve);
            }});
        }});

        const scene = await _decodeCloud(
            payload.scene,
            payload.centre
        );

        const hazard = await _decodeCloud(
            payload.hazard,
            payload.centre
        );

        if (scene.count === 0) {{
            throw new Error(
                'Decoded complete scene contains 0 points.'
            );
        }}

        const sceneColors = _rgbStrings(scene.rgb);
        const hazardColors = _rgbStrings(hazard.rgb);

        console.log(
            'MapAnything 3D:',
            key,
            'scene points =',
            scene.count,
            'hazard points =',
            hazard.count
        );

        const data = [
            {{
                x: scene.x,
                y: scene.y,
                z: scene.z,
                mode: 'markers',
                type: 'scatter3d',
                name: 'Complete local point cloud',
                hovertemplate:
                    'X: %{{x:.3f}} m<br>' +
                    'Y: %{{y:.3f}} m<br>' +
                    'Z: %{{z:.3f}} m' +
                    '<extra>Scene</extra>',
                marker: {{
                    size: 2.2,
                    color: sceneColors,
                    opacity: 0.78
                }}
            }},
            {{
                x: hazard.x,
                y: hazard.y,
                z: hazard.z,
                mode: 'markers',
                type: 'scatter3d',
                name: 'RF-DETR hazard',
                hovertemplate:
                    'X: %{{x:.3f}} m<br>' +
                    'Y: %{{y:.3f}} m<br>' +
                    'Z: %{{z:.3f}} m' +
                    '<extra>Hazard</extra>',
                marker: {{
                    size: 5.5,
                    color: hazardColors,
                    opacity: 1.0
                }}
            }}
        ];

        const layout = {{
            title: {{
                text: 'Complete localised MapAnything point cloud',
                font: {{
                    size: 20
                }}
            }},

            scene: {{
                xaxis: {{
                    title: {{
                        text: 'X (m)',
                        font: {{size: 16}}
                    }},
                    tickfont: {{size: 13}},
                    showgrid: true,
                    zeroline: true
                }},

                yaxis: {{
                    title: {{
                        text: 'Y (m)',
                        font: {{size: 16}}
                    }},
                    tickfont: {{size: 13}},
                    showgrid: true,
                    zeroline: true
                }},

                zaxis: {{
                    title: {{
                        text: 'Z (m)',
                        font: {{size: 16}}
                    }},
                    tickfont: {{size: 13}},
                    showgrid: true,
                    zeroline: true
                }},

                aspectmode: 'data',

                /*
                 * A slightly closer default camera makes the cloud
                 * easier to see immediately.
                 */
                camera: {{
                    eye: {{
                        x: 1.35,
                        y: 1.35,
                        z: 1.05
                    }},
                    center: {{
                        x: 0,
                        y: 0,
                        z: 0
                    }}
                }},

                bgcolor: '#f8f8f8'
            }},

            height: 760,

            margin: {{
                l: 0,
                r: 0,
                b: 0,
                t: 55
            }},

            paper_bgcolor: '#ffffff',

            legend: {{
                orientation: 'h',
                y: 1.01,
                x: 0,
                font: {{
                    size: 14
                }}
            }}
        }};

        div.innerHTML = '';

        /*
         * Use Plotly.react/newPlot only after the popup is visible.
         */
        await Plotly.newPlot(
            div,
            data,
            layout,
            {{
                responsive: true,
                displaylogo: false,
                scrollZoom: true,
                displayModeBar: true,
                modeBarButtonsToRemove: [
                    'toImage'
                ]
            }}
        );

        div.dataset.loading = '0';
        div.dataset.rendered = '1';

        /*
         * Resize multiple times because Leaflet can change the popup
         * dimensions after the initial WebGL canvas is created.
         */
        setTimeout(function() {{
            try {{
                Plotly.Plots.resize(div);
            }} catch (e) {{}}
        }}, 50);

        setTimeout(function() {{
            try {{
                Plotly.Plots.resize(div);
            }} catch (e) {{}}
        }}, 300);

        setTimeout(function() {{
            try {{
                Plotly.Plots.resize(div);
            }} catch (e) {{}}
        }}, 800);

    }} catch (err) {{
        div.dataset.loading = '0';
        _show3DError(div, err);
    }}
}}

/*
 * Leaflet inserts popup HTML dynamically.  A MutationObserver is much
 * more reliable here than IntersectionObserver because the viewer
 * starts life inside a hidden popup and becomes visible later.
 */
function hookEmbedded3DPopups() {{
    document
        .querySelectorAll('[id^="hazard3d_"]')
        .forEach(function(div) {{

            const key =
                div.id.substring('hazard3d_'.length);

            if (div.dataset.hooked === '1') {{
                return;
            }}

            div.dataset.hooked = '1';

            /*
             * Give Leaflet one layout cycle after popup insertion.
             */
            setTimeout(function() {{
                renderEmbeddedHazard3D(key);
            }}, 250);
        }});
}}

/*
 * Watch the entire page for Leaflet popup insertion.
 */
function startEmbedded3DObserver() {{
    hookEmbedded3DPopups();

    const observer = new MutationObserver(
        function(mutations) {{
            let found = false;

            for (const mutation of mutations) {{
                if (mutation.addedNodes.length > 0) {{
                    found = true;
                    break;
                }}
            }}

            if (found) {{
                hookEmbedded3DPopups();
            }}
        }}
    );

    observer.observe(
        document.body,
        {{
            childList: true,
            subtree: true
        }}
    );
}}

window.addEventListener(
    'load',
    function() {{
        setTimeout(
            startEmbedded3DObserver,
            500
        );
    }}
);
</script>
"""

    html_text = html_text.replace(
        "</body>",
        runtime + "\n</body>",
        1,
    )

    map_path.write_text(
        html_text,
        encoding="utf-8",
    )


def reconstruct_hazard(
    cluster,
    model,
):
    central = cluster[
        "central_detection"
    ]

    source_path = (
        IMAGE_DIR
        / central["image"]
    )

    mask_path = Path(
        central["mask_path"]
    )

    if not source_path.exists():
        raise FileNotFoundError(
            f"Missing image: {source_path}"
        )

    if not mask_path.exists():
        raise FileNotFoundError(
            f"Missing mask: {mask_path}"
        )

    cache_key = mask_path.stem

    cache_dir = (
        MAPANYTHING_DIR
        / cache_key
    )

    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # NEW OUTPUTS
    #
    # localized_points.ply = COMPLETE local scene
    # hazard_points.ply   = RF-DETR hazard subset
    # viewer.html          = interactive complete scene
    # --------------------------------------------------------

    scene_ply_path = (
        cache_dir
        / "localized_points.ply"
    )

    scene_npy_path = (
        cache_dir
        / "localized_points.npz"
    )

    hazard_ply_path = (
        cache_dir
        / "hazard_points.ply"
    )

    hazard_npy_path = (
        cache_dir
        / "hazard_points.npz"
    )

    preview_path = (
        cache_dir
        / "hazard_3d_preview.png"
    )

    interactive_path = (
        cache_dir
        / "localized_3d_viewer.html"
    )

    metadata_path = (
        cache_dir
        / "metadata.csv"
    )

    # --------------------------------------------------------
    # CACHE
    #
    # Old hazard-only caches are deliberately NOT accepted.
    # We need a complete local scene for the new viewer.
    # --------------------------------------------------------

    if (
        scene_ply_path.exists()
        and scene_npy_path.exists()
        and hazard_ply_path.exists()
        and hazard_npy_path.exists()
        and interactive_path.exists()
        and preview_path.exists()
    ):
        scene_data = np.load(scene_npy_path)
        hazard_data = np.load(hazard_npy_path)

        scene_points = scene_data["points"]
        scene_rgb = scene_data["rgb"]

        hazard_points = hazard_data["points"]
        hazard_rgb = hazard_data["rgb"]

        embedded_payload = build_embedded_3d_payload(
            scene_points,
            scene_rgb,
            hazard_points,
            hazard_rgb,
        )

        cluster["mapanything"] = {
            "cache_key": cache_key,
            "ply_path": str(hazard_ply_path),
            "scene_ply_path": str(scene_ply_path),
            "preview_path": str(preview_path),
            "interactive_path": str(interactive_path),
            "embedded_3d": embedded_payload,
            "point_count": int(len(hazard_points)),
            "scene_point_count": int(len(scene_points)),
            "x_extent_m": float(np.ptp(scene_points[:, 0])) if len(scene_points) else 0.0,
            "y_extent_m": float(np.ptp(scene_points[:, 1])) if len(scene_points) else 0.0,
            "z_extent_m": float(np.ptp(scene_points[:, 2])) if len(scene_points) else 0.0,
        }

        return

    print()
    print("-" * 80)
    print(
        f"MAPANYTHING | "
        f"HAZARD #{cluster['cluster_id']:04d}"
    )
    print("-" * 80)

    print(
        f"Central image : {central['image']}"
    )

    print(
        f"Mask          : {mask_path.name}"
    )

    # --------------------------------------------------------
    # MAPANYTHING PREPROCESSING
    # --------------------------------------------------------

    views = load_images(
        [str(source_path)]
    )

    if len(views) != 1:
        raise RuntimeError(
            "Expected exactly one MapAnything view."
        )

    view = views[0]
    img_tensor = view["img"]

    if img_tensor.ndim != 4:
        raise RuntimeError(
            f"Unexpected MapAnything image shape: "
            f"{img_tensor.shape}"
        )

    _, _, map_height, map_width = (
        img_tensor.shape
    )

    # --------------------------------------------------------
    # CALIBRATED CAM2 INTRINSICS
    # --------------------------------------------------------

    sx = (
        float(map_width)
        / float(CALIBRATION_WIDTH)
    )

    sy = (
        float(map_height)
        / float(CALIBRATION_HEIGHT)
    )

    K_mapanything = P_RECTIFIED.copy()

    K_mapanything[0, 0] *= sx
    K_mapanything[0, 2] *= sx
    K_mapanything[1, 1] *= sy
    K_mapanything[1, 2] *= sy

    K_torch = (
        torch.from_numpy(
            K_mapanything
        )
        .float()
        .unsqueeze(0)
    )

    view["intrinsics"] = K_torch.clone()

    if "data_norm_type" not in view:
        view["data_norm_type"] = ["dinov2"]

    # --------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------

    with torch.inference_mode():
        prediction = model.infer(
            views,
            memory_efficient_inference=True,
            minibatch_size=1,
            use_amp=True,
            amp_dtype="fp16",
            apply_mask=True,
            mask_edges=True,
        )

    if isinstance(prediction, list):
        prediction = prediction[0]
    elif isinstance(prediction, tuple):
        prediction = prediction[0]

    pts3d = extract_pts3d(prediction)
    ph, pw = pts3d.shape[:2]

    # --------------------------------------------------------
    # RF-DETR HAZARD MASK AT MAPANYTHING RESOLUTION
    # --------------------------------------------------------

    hazard_mask = load_mask(mask_path)

    hazard_mask = cv2.resize(
        hazard_mask.astype(np.uint8),
        (pw, ph),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)

    map_valid = extract_prediction_mask(
        prediction,
        ph,
        pw,
    )

    rgb_image = resize_rgb_to_shape(
        source_path,
        pw,
        ph,
    )

    finite = np.isfinite(
        pts3d
    ).all(axis=2)

    # COMPLETE LOCAL SCENE.
    # This is what the user wants to interact with.
    scene_valid = (
        map_valid
        &
        finite
    )

    scene_points = pts3d[
        scene_valid
    ].astype(np.float32)

    scene_rgb = rgb_image[
        scene_valid
    ].astype(np.uint8)

    # HAZARD SUBSET.
    hazard_valid = (
        hazard_mask
        &
        map_valid
        &
        finite
    )

    hazard_points = pts3d[
        hazard_valid
    ].astype(np.float32)

    hazard_rgb = rgb_image[
        hazard_valid
    ].astype(np.uint8)

    print(
        f"MapAnything output : "
        f"{pw} x {ph}"
    )

    print(
        f"Full scene points  : "
        f"{len(scene_points):,}"
    )

    print(
        f"Hazard pixels      : "
        f"{int(hazard_mask.sum()):,}"
    )

    print(
        f"Hazard 3D points   : "
        f"{len(hazard_points):,}"
    )

    if len(scene_points) == 0:
        raise RuntimeError(
            "MapAnything returned zero valid scene points."
        )

    if len(hazard_points) == 0:
        raise RuntimeError(
            "RF-DETR hazard mask produced zero valid "
            "MapAnything 3D points."
        )

    # --------------------------------------------------------
    # SAVE COMPLETE LOCAL SCENE
    #
    # No downsampling here. The full valid MapAnything
    # reconstruction is retained.
    # --------------------------------------------------------

    save_ply(
        scene_ply_path,
        scene_points,
        scene_rgb,
    )

    np.savez_compressed(
        scene_npy_path,
        points=scene_points,
        rgb=scene_rgb,
    )

    # Hazard-only PLY retained as a useful separate output.
    save_ply(
        hazard_ply_path,
        hazard_points,
        hazard_rgb,
    )

    np.savez_compressed(
        hazard_npy_path,
        points=hazard_points,
        rgb=hazard_rgb,
    )

    # --------------------------------------------------------
    # STATIC PREVIEW
    # --------------------------------------------------------

    title = (
        f"Hazard #{cluster['cluster_id']:04d} | "
        f"{cluster['class']} | "
        f"Full local scene + hazard | "
        f"{central['image']}"
    )

    save_3d_preview(
        preview_path,
        hazard_points,
        hazard_rgb,
        title,
    )

    # --------------------------------------------------------
    # INTERACTIVE VIEWER
    #
    # IMPORTANT:
    # We no longer generate a Plotly fragment for a Folium
    # srcdoc iframe. That was the reason the previous viewer
    # appeared blank: the fragment had no Plotly runtime inside
    # its iframe.
    #
    # Instead, we generate a REAL standalone HTML document,
    # with Plotly loaded locally, and the Folium popup embeds
    # that file using an iframe src.
    # --------------------------------------------------------

    save_full_scene_viewer(
        interactive_path,
        scene_points,
        scene_rgb,
        hazard_points,
        hazard_rgb,
        title,
    )

    # Store the actual point data for direct embedding into the main
    # Folium HTML. No iframe/external viewer is used by the final map.
    embedded_payload = build_embedded_3d_payload(
        scene_points,
        scene_rgb,
        hazard_points,
        hazard_rgb,
    )

    with open(
        metadata_path,
        "w",
        newline="",
    ) as f:
        writer = csv.writer(f)

        writer.writerow([
            "cluster_id",
            "class",
            "central_image",
            "central_timestamp_ns",
            "scene_point_count",
            "hazard_point_count",
            "x_extent_m",
            "y_extent_m",
            "z_extent_m",
            "scene_ply_path",
            "hazard_ply_path",
            "interactive_path",
        ])

        writer.writerow([
            cluster["cluster_id"],
            cluster["class"],
            central["image"],
            central["timestamp_ns"],
            len(scene_points),
            len(hazard_points),
            float(np.ptp(scene_points[:, 0])),
            float(np.ptp(scene_points[:, 1])),
            float(np.ptp(scene_points[:, 2])),
            str(scene_ply_path),
            str(hazard_ply_path),
            str(interactive_path),
        ])

    cluster["mapanything"] = {
        "cache_key": cache_key,
        "ply_path": str(hazard_ply_path),
        "scene_ply_path": str(scene_ply_path),
        "preview_path": str(preview_path),
        "interactive_path": str(interactive_path),
        "embedded_3d": embedded_payload,
        "point_count": int(len(hazard_points)),
        "scene_point_count": int(len(scene_points)),
        "x_extent_m": float(np.ptp(scene_points[:, 0])),
        "y_extent_m": float(np.ptp(scene_points[:, 1])),
        "z_extent_m": float(np.ptp(scene_points[:, 2])),
    }

    del prediction
    del views
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# RUN MAPANYTHING
# ============================================================

mapanything_start = time.perf_counter()

if args.skip_mapanything:

    print()
    print("=" * 80)
    print("MAPANYTHING SKIPPED")
    print("=" * 80)

else:

    print()
    print("=" * 80)
    print("LOADING MAPANYTHING")
    print("=" * 80)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    map_model = MapAnything.from_pretrained(
        "facebook/map-anything"
    ).to(device)

    map_model.eval()

    print(
        "MapAnything loaded."
    )

    successful = 0
    failed = 0

    for cluster in clusters:

        try:

            reconstruct_hazard(
                cluster,
                map_model,
            )

            successful += 1

        except Exception as exc:

            failed += 1

            cluster["mapanything"] = {
                "error": str(exc)
            }

            print()
            print(
                f"WARNING: MapAnything failed for "
                f"hazard #{cluster['cluster_id']:04d}"
            )

            print(
                repr(exc)
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    del map_model

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    print()
    print(
        f"MapAnything successful : {successful}"
    )
    print(
        f"MapAnything failed     : {failed}"
    )

mapanything_time = (
    time.perf_counter()
    - mapanything_start
)


# ============================================================
# SAVE MAPANYTHING SUMMARY CSV
# ============================================================

with open(
    MAPANYTHING_SUMMARY_CSV,
    "w",
    newline="",
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "cluster_id",
        "hazard_class",
        "central_image",
        "central_timestamp_ns",
        "central_latitude",
        "central_longitude",
        "central_confidence",
        "point_count",
        "x_extent_m",
        "y_extent_m",
        "z_extent_m",
        "scene_ply_path",
        "ply_path",
        "preview_path",
        "interactive_path",
        "scene_point_count",
        "error",
    ])

    for cluster in clusters:

        central = cluster[
            "central_detection"
        ]

        result = cluster.get(
            "mapanything",
            {},
        )

        writer.writerow([
            cluster["cluster_id"],
            cluster["class"],
            central["image"],
            central["timestamp_ns"],
            central["latitude"],
            central["longitude"],
            central["confidence"],
            result.get(
                "point_count",
                "",
            ),
            result.get(
                "x_extent_m",
                "",
            ),
            result.get(
                "y_extent_m",
                "",
            ),
            result.get(
                "z_extent_m",
                "",
            ),
            result.get(
                "scene_ply_path",
                "",
            ),
            result.get(
                "ply_path",
                "",
            ),
            result.get(
                "preview_path",
                "",
            ),
            result.get(
                "interactive_path",
                "",
            ),
            result.get(
                "scene_point_count",
                "",
            ),
            result.get(
                "error",
                "",
            ),
        ])


# ============================================================
# SAVE CLUSTER CSV
# ============================================================

print()
print("=" * 80)
print("WRITING CLUSTER CSV")
print("=" * 80)

with open(
    CLUSTER_CSV,
    "w",
    newline="",
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "cluster_id",
        "hazard_class",
        "anchor_latitude",
        "anchor_longitude",
        "anchor_altitude",
        "anchor_image",
        "best_image",
        "best_confidence",
        "detection_count",
        "max_distance_from_anchor_m",
        "first_image",
        "last_image",
        "central_image",
        "central_timestamp_ns",
        "central_latitude",
        "central_longitude",
        "central_confidence",
        "representative_image",
        "mapanything_scene_ply",
        "mapanything_ply",
        "mapanything_preview",
        "mapanything_interactive",
        "mapanything_scene_point_count",
        "mapanything_hazard_point_count",
    ])

    for cluster in clusters:

        detections = cluster[
            "detections"
        ]

        central = cluster[
            "central_detection"
        ]

        result = cluster.get(
            "mapanything",
            {},
        )

        writer.writerow([
            cluster["cluster_id"],
            cluster["class"],
            cluster["anchor_latitude"],
            cluster["anchor_longitude"],
            cluster["anchor_altitude"],
            cluster["anchor_image"],
            cluster["best_detection"]["image"],
            cluster["best_confidence"],
            cluster["detection_count"],
            cluster["max_distance_m"],
            detections[0]["image"],
            detections[-1]["image"],
            central["image"],
            central["timestamp_ns"],
            central["latitude"],
            central["longitude"],
            central["confidence"],
            cluster.get(
                "representative_image",
                "",
            ),
            result.get(
                "scene_ply_path",
                "",
            ),
            result.get(
                "ply_path",
                "",
            ),
            result.get(
                "preview_path",
                "",
            ),
            result.get(
                "interactive_path",
                "",
            ),
            result.get(
                "scene_point_count",
                "",
            ),
            result.get(
                "point_count",
                "",
            ),
        ])


# ============================================================
# HTML HELPERS
# ============================================================

def image_to_base64(
    image_path
):
    image_path = Path(
        image_path
    )

    if not image_path.exists():
        return None

    with open(
        image_path,
        "rb",
    ) as f:

        encoded = (
            base64.b64encode(
                f.read()
            )
            .decode("ascii")
        )

    return (
        "data:image/jpeg;base64,"
        + encoded
    )


def png_to_base64(
    image_path
):
    image_path = Path(
        image_path
    )

    if not image_path.exists():
        return None

    with open(
        image_path,
        "rb",
    ) as f:

        encoded = (
            base64.b64encode(
                f.read()
            )
            .decode("ascii")
        )

    return (
        "data:image/png;base64,"
        + encoded
    )


def relative_to_output(
    path
):
    try:
        return Path(
            path
        ).relative_to(
            OUTPUT_DIR
        ).as_posix()
    except ValueError:
        return str(path)


# ============================================================
# CREATE FOLIUM MAP
# ============================================================

print()
print("=" * 80)
print("CREATING HTML MAP")
print("=" * 80)

map_start = time.perf_counter()

center_lat = gps_df[
    "latitude_deg"
].mean()

center_lon = gps_df[
    "longitude_deg"
].mean()

m = folium.Map(
    location=[
        center_lat,
        center_lon,
    ],
    zoom_start=18,
    tiles="OpenStreetMap",
    control_scale=True,
)


# ============================================================
# VEHICLE TRAJECTORY
# ============================================================

trajectory = list(
    zip(
        gps_df[
            "latitude_deg"
        ].iloc[
            ::TRAJECTORY_STRIDE
        ],
        gps_df[
            "longitude_deg"
        ].iloc[
            ::TRAJECTORY_STRIDE
        ],
    )
)

if len(trajectory) >= 2:

    folium.PolyLine(
        trajectory,
        weight=3,
        opacity=0.7,
        tooltip="Vehicle GPS trajectory",
    ).add_to(m)


# ============================================================
# START / END
# ============================================================

folium.Marker(
    [
        gps_df[
            "latitude_deg"
        ].iloc[0],
        gps_df[
            "longitude_deg"
        ].iloc[0],
    ],
    tooltip="START",
    icon=folium.Icon(
        color="green",
        icon="play",
    ),
).add_to(m)

folium.Marker(
    [
        gps_df[
            "latitude_deg"
        ].iloc[-1],
        gps_df[
            "longitude_deg"
        ].iloc[-1],
    ],
    tooltip="END",
    icon=folium.Icon(
        color="red",
        icon="stop",
    ),
).add_to(m)


# ============================================================
# HAZARD MARKERS
# ============================================================

for cluster in clusters:

    hazard_class = cluster[
        "class"
    ]

    latitude = cluster[
        "anchor_latitude"
    ]

    longitude = cluster[
        "anchor_longitude"
    ]

    detection_count = cluster[
        "detection_count"
    ]

    central = cluster[
        "central_detection"
    ]

    # --------------------------------------------------------
    # Marker appearance
    # --------------------------------------------------------

    if hazard_class == "loose_trash":

        marker_color = "orange"
        icon_name = "trash"
        title = "LOOSE TRASH"

    elif hazard_class == "mud_spill":

        marker_color = "red"
        icon_name = "warning-sign"
        title = "MUD SPILL"

    else:

        marker_color = "purple"
        icon_name = "warning-sign"
        title = hazard_class.upper()

    # --------------------------------------------------------
    # Central segmentation image
    # --------------------------------------------------------

    representative_name = cluster.get(
        "representative_image",
        "",
    )

    embedded_image = None

    if representative_name:

        embedded_image = image_to_base64(
            REPRESENTATIVE_DIR
            / representative_name
        )

    if embedded_image:

        image_html = f"""
        <img
            src="{embedded_image}"
            style="
                width:100%;
                max-width:100%;
                height:auto;
                display:block;
                margin-top:10px;
                border-radius:6px;
            "
        >
        """

    else:

        image_html = """
        <p>Central segmentation image unavailable.</p>
        """

    # --------------------------------------------------------
    # 3D RESULT
    # --------------------------------------------------------

    result = cluster.get(
        "mapanything",
        {},
    )

    preview_path = result.get("preview_path")
    interactive_path = result.get("interactive_path")
    ply_path = result.get("ply_path")
    scene_ply_path = result.get("scene_ply_path")

    preview_base64 = None

    if preview_path:
        preview_base64 = png_to_base64(
            preview_path
        )

    if preview_base64:
        preview_html = f"""
        <img
            src="{preview_base64}"
            style="
                width:100%;
                max-width:100%;
                height:auto;
                display:block;
                margin-top:10px;
                border-radius:6px;
            "
        >
        """
    else:
        preview_html = """
        <p>3D preview unavailable.</p>
        """

    # --------------------------------------------------------
    # EMBEDDED 3D VIEWER
    #
    # The point data is embedded into the FINAL MAP HTML itself.
    # There is NO iframe and NO external localized_3d_viewer.html
    # dependency.
    # --------------------------------------------------------

    embedded_3d = result.get("embedded_3d")

    if embedded_3d:
        viewer_key = f"hazard{cluster['cluster_id']:04d}"
        interactive_viewer_html = f"""
        <div
            id="hazard3d_{viewer_key}"
            data-scene-points="{int(embedded_3d['scene_count']):,}"
            style="
                width:100%;
                height:760px;
                margin-top:10px;
                border:1px solid #ddd;
                border-radius:6px;
                overflow:hidden;
                background:#fff;
            "
        >
            <div style="padding:20px;font-family:Arial">
                Embedded 3D viewer will load when this popup is opened.
            </div>
        </div>
        """
    else:
        interactive_viewer_html = """
        <p>Embedded 3D viewer unavailable.</p>
        """

    interactive_link_html = ""

    ply_link_html = ""

    if (
        scene_ply_path
        and Path(scene_ply_path).exists()
    ):
        rel_scene_ply = relative_to_output(
            scene_ply_path
        )

        ply_link_html += f"""
        <p>
            <a
                href="{html.escape(rel_scene_ply)}"
                target="_blank"
            >
                Download complete localised point cloud (.PLY)
            </a>
        </p>
        """

    if (
        ply_path
        and Path(ply_path).exists()
    ):
        rel_ply = relative_to_output(
            ply_path
        )

        ply_link_html += f"""
        <p>
            <a
                href="{html.escape(rel_ply)}"
                target="_blank"
            >
                Download hazard-only point cloud (.PLY)
            </a>
        </p>
        """

    point_count = result.get(
        "point_count",
        "",
    )

    scene_point_count = result.get(
        "scene_point_count",
        "",
    )

    x_extent = result.get(
        "x_extent_m",
        "",
    )

    y_extent = result.get(
        "y_extent_m",
        "",
    )

    z_extent = result.get(
        "z_extent_m",
        "",
    )

    mapanything_error = result.get(
        "error",
        "",
    )

    if mapanything_error:
        three_d_status = f"""
        <p style="color:#b00020;">
            MapAnything failed:
            {html.escape(str(mapanything_error))}
        </p>
        """
    elif scene_point_count != "":
        three_d_status = f"""
        <table style="
            width:100%;
            border-collapse:collapse;
            font-size:14px;
            margin-bottom:8px;
        ">
            <tr>
                <td><b>Complete scene points</b></td>
                <td>{int(scene_point_count):,}</td>
            </tr>
            <tr>
                <td><b>Hazard points</b></td>
                <td>{int(point_count):,}</td>
            </tr>
            <tr>
                <td><b>X extent</b></td>
                <td>{float(x_extent):.3f} m</td>
            </tr>
            <tr>
                <td><b>Y extent</b></td>
                <td>{float(y_extent):.3f} m</td>
            </tr>
            <tr>
                <td><b>Z extent</b></td>
                <td>{float(z_extent):.3f} m</td>
            </tr>
        </table>
        """
    else:
        three_d_status = """
        <p>MapAnything result unavailable.</p>
        """

    # --------------------------------------------------------
    # POPUP
    # --------------------------------------------------------

    popup_html = f"""
    <div style="
        width:1100px;
        max-width:92vw;
        font-family:Arial,sans-serif;
        font-size:16px;
        line-height:1.45;
    ">

        <h2 style="
            margin-bottom:8px;
            font-size:24px;
            line-height:1.2;
        ">
            {html.escape(title)}
        </h2>

        <hr>

        <table style="
            width:100%;
            border-collapse:collapse;
            font-size:15px;
        ">

            <tr>
                <td><b>Hazard ID</b></td>
                <td>#{cluster["cluster_id"]:04d}</td>
            </tr>

            <tr>
                <td><b>Type</b></td>
                <td>{html.escape(hazard_class)}</td>
            </tr>

            <tr>
                <td><b>Detections in cluster</b></td>
                <td>{detection_count}</td>
            </tr>

            <tr>
                <td><b>Best confidence</b></td>
                <td>{cluster["best_confidence"]:.3f}</td>
            </tr>

            <tr>
                <td><b>Maximum anchor distance</b></td>
                <td>{cluster["max_distance_m"]:.2f} m</td>
            </tr>

            <tr>
                <td><b>Central frame</b></td>
                <td>{html.escape(central["image"])}</td>
            </tr>

            <tr>
                <td><b>Central confidence</b></td>
                <td>{central["confidence"]:.3f}</td>
            </tr>

            <tr>
                <td><b>Central latitude</b></td>
                <td>{central["latitude"]:.8f}</td>
            </tr>

            <tr>
                <td><b>Central longitude</b></td>
                <td>{central["longitude"]:.8f}</td>
            </tr>

        </table>

        <h3 style="
            margin-top:18px;
            font-size:20px;
        ">
            RF-DETR central-frame segmentation
        </h3>

        {image_html}

        <h3 style="
            margin-top:22px;
            font-size:20px;
        ">
            MapAnything localised 3D point cloud
        </h3>

        {three_d_status}

        <div style="
            width:100%;
            min-height:760px;
            margin-top:10px;
            border:1px solid #ddd;
            border-radius:6px;
            overflow:hidden;
            background:#fff;
        ">
            {interactive_viewer_html}
        </div>

        {interactive_link_html}

        {ply_link_html}

        <p style="
            font-size:12px;
            color:#666;
            margin-top:12px;
        ">
            The viewer shows the COMPLETE valid MapAnything
            point cloud from the central frame. RF-DETR hazard
            points are overlaid as a separate trace. The
            reconstruction itself is NOT cropped to the hazard.
            Drag to rotate, scroll to zoom, and shift-drag to pan.
        </p>

    </div>
    """

    # IMPORTANT:
    # Do NOT wrap this popup in folium.IFrame.
    #
    # The previous design created a srcdoc iframe. Relative links
    # inside that iframe were not reliably resolved to the local
    # MapAnything viewer. The popup is now inserted directly into
    # the Leaflet page, so the viewer iframe can resolve:
    #
    #   mapanything_hazards/<cache_key>/localized_3d_viewer.html
    #
    # relative to hazard_map_clustered_mapanything.html.
    popup = folium.Popup(
        folium.Html(
            popup_html,
            script=True,
        ),
        max_width=950,
    )

    folium.Marker(
        [
            latitude,
            longitude,
        ],
        popup=popup,
        tooltip=(
            f"{title} | "
            f"{detection_count} detections | "
            f"central {central['image']}"
        ),
        icon=folium.Icon(
            color=marker_color,
            icon=icon_name,
        ),
    ).add_to(m)


# ============================================================
# EMBED ALL 3D POINT-CLOUD PAYLOADS INTO THE MAIN HTML
# ============================================================

embedded_payloads = {}

for cluster in clusters:
    result = cluster.get("mapanything", {})
    payload = result.get("embedded_3d")
    if payload:
        embedded_payloads[f"hazard{cluster['cluster_id']:04d}"] = payload

print(
    f"Embedding {len(embedded_payloads):,} complete 3D point clouds directly into HTML."
)


# ============================================================
# FIT MAP
# ============================================================

m.fit_bounds([
    [
        gps_df[
            "latitude_deg"
        ].min(),
        gps_df[
            "longitude_deg"
        ].min(),
    ],
    [
        gps_df[
            "latitude_deg"
        ].max(),
        gps_df[
            "longitude_deg"
        ].max(),
    ],
])


# ============================================================
# SAVE MAP
# ============================================================

m.save(
    MAP_OUTPUT
)

# Plotly + compressed XYZ/RGB point data are now physically embedded
# inside MAP_OUTPUT. The browser does not fetch a viewer HTML/NPZ/PLY.
inject_embedded_3d_runtime(
    MAP_OUTPUT,
    embedded_payloads,
)

map_time = (
    time.perf_counter()
    - map_start
)


# ============================================================
# FINAL
# ============================================================

html_size_mb = (
    MAP_OUTPUT.stat().st_size
    / (1024 ** 2)
)

print()
print("=" * 80)
print("DONE")
print("=" * 80)

print()
print(
    f"Original detections   : "
    f"{len(detection_df):,}"
)

print(
    f"Physical hazards      : "
    f"{len(clusters):,}"
)

print(
    f"Cluster radius        : "
    f"{CLUSTER_RADIUS_METERS:.2f} m"
)

print()
print(
    f"MapAnything time      : "
    f"{mapanything_time / 60:.2f} min"
)

print(
    f"Map generation time   : "
    f"{map_time:.2f} sec"
)

print(
    f"HTML size             : "
    f"{html_size_mb:.2f} MB"
)

print()
print("OUTPUTS")
print("-" * 80)

print()
print("HTML map:")
print(MAP_OUTPUT)

print()
print("Detailed RF-DETR detections:")
print(DETAILED_DETECTIONS_CSV)

print()
print("Segmentation masks:")
print(MASK_DIR)

print()
print("Cluster CSV:")
print(CLUSTER_CSV)

print()
print("MapAnything hazard reconstructions:")
print(MAPANYTHING_DIR)

print()
print("MapAnything summary:")
print(MAPANYTHING_SUMMARY_CSV)

print()
print("=" * 80)
