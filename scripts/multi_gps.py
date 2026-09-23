#!/usr/bin/env python3

from pathlib import Path
import gc

import numpy as np
import pandas as pd
import torch

from PIL import Image, ImageDraw, ImageFont

import folium
from folium.plugins import MeasureControl

from mapanything.models import MapAnything
from mapanything.utils.image import load_images

from rfdetr import RFDETRSegSmall


# ============================================================
# CONFIG
# ============================================================

IMAGE_DIR = Path(
    "/home/lcha0115/bt60_scratch/lcha_data/map-anything/"
    "personal_data/camera_lidar_gps_test_101/extracted/"
    "CAM2/images_rect"
)

OUTPUT_DIR = Path(
    "/home/lcha0115/bt60_scratch/lcha_data/map-anything/"
    "personal_data/camera_lidar_gps_test_101/"
    "mapanything_9000_9600"
)

LABELLED_DIR = OUTPUT_DIR / "labelled_images"


# ============================================================
# GPS
# ============================================================

GPS_FILE = Path(
    "/home/lcha0115/bt60_scratch/lcha_data/map-anything/"
    "personal_data/camera_lidar_gps_test_101/extracted/"
    "gps/position.csv"
)

MAX_GPS_TIME_DIFF_NS = 1_000_000_000


# ============================================================
# FRAME RANGE
# ============================================================

START_IMAGE = 9000
END_IMAGE = 9600

# Process every 5th frame
STRIDE = 3


# ============================================================
# CAM2 CALIBRATION
# ============================================================
#
# Original rectified camera calibration:
#
# Image size = 1920 x 1200
#
# Since images are rectified, use P[:3,:3].
#
# ============================================================

CALIBRATION_WIDTH = 1920
CALIBRATION_HEIGHT = 1200

P_RECTIFIED = np.array(
    [
        [510.651676055731, 0.0, 949.416735545071],
        [0.0, 510.456423883940, 607.626749742175],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)


# ============================================================
# RF-DETR
# ============================================================

RFDETR_MODEL_PATH = Path(
    "/home/lcha0115/bt60_scratch/lcha_data/map-anything/"
    "personal_data/2D-Seg-Road-Nissan.v6i.coco-segmentation/"
    "outputs/checkpoint_best_total.pth"
)

RFDETR_CONFIDENCE_THRESHOLD = 0.25
RFDETR_MASK_THRESHOLD = 0.50


# ============================================================
# SEMANTIC VISUALIZATION
# ============================================================

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


# ============================================================
# MAP MARKER COLORS
# ============================================================

HAZARD_MARKER_COLORS = {
    "fallen_fence": "purple",
    "loose_trash": "orange",
    "mud_spill": "red",
    "public_greenway": "green",
    "sidewalk": "blue",
}


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

LABELLED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# FAST BINARY PLY WRITER
# ============================================================

def save_ply(path, points, rgb):

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

        f.write(b"ply\n")
        f.write(
            b"format binary_little_endian 1.0\n"
        )

        f.write(
            f"element vertex {len(data)}\n".encode()
        )

        f.write(b"property float x\n")
        f.write(b"property float y\n")
        f.write(b"property float z\n")

        f.write(b"property uchar red\n")
        f.write(b"property uchar green\n")
        f.write(b"property uchar blue\n")

        f.write(b"end_header\n")

        data.tofile(f)

    print(
        f"Saved: {path}"
    )


# ============================================================
# LOAD GPS
# ============================================================

print("=" * 70)
print("LOADING GPS")
print("=" * 70)


gps_df = pd.read_csv(
    GPS_FILE
)


required_columns = [
    "ros_timestamp_ns",
    "latitude_deg",
    "longitude_deg",
]


for column in required_columns:

    if column not in gps_df.columns:

        raise RuntimeError(
            f"Missing GPS column: {column}"
        )


gps_df = gps_df.dropna(
    subset=[
        "ros_timestamp_ns",
        "latitude_deg",
        "longitude_deg",
    ]
).copy()


gps_df["ros_timestamp_ns"] = (
    gps_df["ros_timestamp_ns"]
    .astype(np.int64)
)


gps_df = (
    gps_df
    .sort_values("ros_timestamp_ns")
    .reset_index(drop=True)
)


gps_timestamps = (
    gps_df["ros_timestamp_ns"]
    .to_numpy(
        dtype=np.int64
    )
)


print(
    f"GPS points: {len(gps_df):,}"
)


print(
    f"Latitude range: "
    f"{gps_df['latitude_deg'].min()} "
    f"-> "
    f"{gps_df['latitude_deg'].max()}"
)


print(
    f"Longitude range: "
    f"{gps_df['longitude_deg'].min()} "
    f"-> "
    f"{gps_df['longitude_deg'].max()}"
)


# ============================================================
# GPS MATCH FUNCTION
# ============================================================

def match_gps(
    image_timestamp_ns
):

    idx = np.searchsorted(
        gps_timestamps,
        image_timestamp_ns,
    )

    candidates = []

    if idx > 0:

        candidates.append(
            idx - 1
        )

    if idx < len(gps_timestamps):

        candidates.append(
            idx
        )

    if not candidates:

        return None

    best_idx = min(
        candidates,
        key=lambda i:
        abs(
            int(
                gps_timestamps[i]
            )
            -
            int(
                image_timestamp_ns
            )
        ),
    )

    gps_timestamp = int(
        gps_timestamps[
            best_idx
        ]
    )

    delta_ns = abs(
        gps_timestamp
        -
        int(image_timestamp_ns)
    )

    if (
        MAX_GPS_TIME_DIFF_NS
        is not None
        and
        delta_ns
        >
        MAX_GPS_TIME_DIFF_NS
    ):

        return None

    row = gps_df.iloc[
        best_idx
    ]

    return {
        "gps_timestamp_ns":
            gps_timestamp,

        "gps_delta_ns":
            delta_ns,

        "gps_delta_ms":
            delta_ns / 1_000_000.0,

        "latitude_deg":
            float(
                row["latitude_deg"]
            ),

        "longitude_deg":
            float(
                row["longitude_deg"]
            ),

        "altitude_m":
            float(
                row["altitude_m"]
            )
            if "altitude_m" in row
            else np.nan,
    }


# ============================================================
# FIND IMAGES
# ============================================================

all_images = sorted(
    [
        p
        for p in IMAGE_DIR.iterdir()
        if p.suffix.lower()
        in [
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
        ]
    ]
)


print(
    "\n" + "=" * 70
)

print(
    "IMAGE SELECTION"
)

print(
    "=" * 70
)


print(
    f"Total images: "
    f"{len(all_images):,}"
)


selected_frames = list(
    range(
        START_IMAGE,
        END_IMAGE + 1,
        STRIDE,
    )
)


selected_images = [
    all_images[
        frame - 1
    ]
    for frame in selected_frames
]


if len(selected_images) != len(
    selected_frames
):

    raise RuntimeError(
        "Number of selected images "
        "does not match frame range."
    )


print(
    f"Frame range: "
    f"{START_IMAGE} -> {END_IMAGE}"
)


print(
    f"Stride: {STRIDE}"
)


print(
    f"Selected frames: "
    f"{len(selected_frames)}"
)


print(
    f"First image: "
    f"{selected_images[0].name}"
)


print(
    f"Last image: "
    f"{selected_images[-1].name}"
)


# ============================================================
# MATCH IMAGE TIMESTAMPS TO GPS
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "SYNCHRONIZING IMAGES WITH GPS"
)

print(
    "=" * 70
)


image_gps = {}


for frame, image_path in zip(
    selected_frames,
    selected_images,
):

    try:

        image_timestamp_ns = int(
            image_path.stem
        )

    except ValueError:

        raise RuntimeError(
            f"Image filename is not a ROS "
            f"timestamp: {image_path.name}"
        )


    gps_match = match_gps(
        image_timestamp_ns
    )


    image_gps[frame] = {

        "image_timestamp_ns":
            image_timestamp_ns,

        "gps":
            gps_match,
    }


    if gps_match is None:

        print(
            f"Frame {frame}: "
            f"NO GPS MATCH"
        )

    else:

        print(
            f"Frame {frame}: "
            f"Δ = "
            f"{gps_match['gps_delta_ms']:.2f} ms | "
            f"lat = "
            f"{gps_match['latitude_deg']:.9f} | "
            f"lon = "
            f"{gps_match['longitude_deg']:.9f}"
        )


matched_count = sum(
    x["gps"] is not None
    for x in image_gps.values()
)


print(
    "\nGPS synchronization summary:"
)


print(
    f"Matched: "
    f"{matched_count}/"
    f"{len(selected_frames)}"
)


if matched_count > 0:

    gps_deltas = np.array(
        [
            x["gps"]["gps_delta_ms"]
            for x in image_gps.values()
            if x["gps"] is not None
        ]
    )

    print(
        f"Mean Δ: "
        f"{gps_deltas.mean():.3f} ms"
    )

    print(
        f"Median Δ: "
        f"{np.median(gps_deltas):.3f} ms"
    )

    print(
        f"Max Δ: "
        f"{gps_deltas.max():.3f} ms"
    )


# ============================================================
# LOAD MAPANYTHING
# ============================================================

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print(
    "\n" + "=" * 70
)

print(
    "LOADING MAPANYTHING"
)

print(
    "=" * 70
)


print(
    f"Device: {device}"
)


if torch.cuda.is_available():

    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        f"GPU memory: "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )


model = MapAnything.from_pretrained(
    "facebook/map-anything"
).to(device)


model.eval()


print(
    "MapAnything loaded."
)


# ============================================================
# LOAD ORIGINAL IMAGES
# ============================================================
#
# IMPORTANT:
#
# No manual resizing.
#
# load_images() performs MapAnything's normal preprocessing.
#
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "LOADING ORIGINAL IMAGES"
)

print(
    "=" * 70
)


views = load_images(
    [
        str(p)
        for p in selected_images
    ]
)


print(
    f"Loaded {len(views)} views."
)


# ============================================================
# DETERMINE ACTUAL MAPANYTHING IMAGE SIZE
# ============================================================

example_img = views[0]["img"]


print(
    "\n" + "=" * 70
)

print(
    "MAPANYTHING IMAGE INPUT"
)

print(
    "=" * 70
)


print(
    f"Image tensor shape: "
    f"{example_img.shape}"
)


print(
    f"Image dtype: "
    f"{example_img.dtype}"
)


if example_img.ndim != 4:

    raise RuntimeError(
        f"Unexpected image tensor shape: "
        f"{example_img.shape}"
    )


_, _, map_height, map_width = (
    example_img.shape
)


print(
    f"Actual MapAnything resolution: "
    f"{map_width} x {map_height}"
)


# ============================================================
# SCALE RECTIFIED INTRINSICS
# ============================================================
#
# We are NOT resizing the image ourselves.
#
# We simply scale the calibrated P matrix to the resolution
# actually returned by load_images().
#
# ============================================================

sx = (
    float(map_width)
    /
    float(CALIBRATION_WIDTH)
)


sy = (
    float(map_height)
    /
    float(CALIBRATION_HEIGHT)
)


K_mapanything = (
    P_RECTIFIED.copy()
)


K_mapanything[0, 0] *= sx
K_mapanything[0, 2] *= sx

K_mapanything[1, 1] *= sy
K_mapanything[1, 2] *= sy


print(
    "\nOriginal rectified P[:3,:3]:"
)

print(
    P_RECTIFIED
)


print(
    "\nScale factors:"
)

print(
    f"  sx = {sx:.8f}"
)

print(
    f"  sy = {sy:.8f}"
)


print(
    "\nIntrinsics passed to MapAnything:"
)

print(
    K_mapanything
)


# ============================================================
# BATCHED INTRINSICS
# ============================================================

K_torch = torch.from_numpy(
    K_mapanything
).float().unsqueeze(0)


print(
    "\nIntrinsics tensor:"
)

print(
    f"  shape : {K_torch.shape}"
)

print(
    f"  dtype : {K_torch.dtype}"
)


# ============================================================
# ADD INTRINSICS DIRECTLY TO EVERY VIEW
# ============================================================
#
# THIS IS THE ONLY MAPANYTHING INPUT CHANGE.
#
# No ray_directions.
#
# No manually generated rays.
#
# ============================================================

for view in views:

    view["intrinsics"] = (
        K_torch.clone()
    )


# ============================================================
# ENSURE NORMALIZATION FIELD
# ============================================================

for view in views:

    if "data_norm_type" not in view:

        view["data_norm_type"] = [
            "dinov2"
        ]


# ============================================================
# SANITY CHECK
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "MAPANYTHING INPUT SANITY CHECK"
)

print(
    "=" * 70
)


print(
    "Image:"
)

print(
    f"  shape : "
    f"{views[0]['img'].shape}"
)

print(
    f"  dtype : "
    f"{views[0]['img'].dtype}"
)

print(
    f"  min   : "
    f"{views[0]['img'].min().item():.4f}"
)

print(
    f"  max   : "
    f"{views[0]['img'].max().item():.4f}"
)


print(
    "\nIntrinsics:"
)

print(
    f"  shape : "
    f"{views[0]['intrinsics'].shape}"
)

print(
    f"  dtype : "
    f"{views[0]['intrinsics'].dtype}"
)

print(
    views[0]["intrinsics"]
)


print(
    "\nData normalization:"
)

print(
    views[0]["data_norm_type"]
)


print(
    "\nRay directions:"
)

print(
    "  NOT PROVIDED"
)

print(
    "  MapAnything derives them from "
    "the supplied intrinsics."
)


# ============================================================
# MAPANYTHING INFERENCE
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "RUNNING MAPANYTHING"
)

print(
    "=" * 70
)


print(
    f"Reconstructing "
    f"{len(views)} frames jointly."
)


print(
    f"Resolution: "
    f"{map_width} x {map_height}"
)


print(
    "Calibration: DIRECT INTRINSICS"
)


print(
    "Manual resizing: NO"
)


print(
    "Manual ray generation: NO"
)


print(
    f"Intrinsics shape: "
    f"{views[0]['intrinsics'].shape}"
)


if torch.cuda.is_available():

    torch.cuda.empty_cache()


with torch.inference_mode():

    predictions = model.infer(

        views,

        memory_efficient_inference=True,

        minibatch_size=1,

        use_amp=True,

        amp_dtype="fp16",

        apply_mask=True,

        mask_edges=True,
    )


print(
    "\nMapAnything inference complete."
)


print(
    f"Prediction entries: "
    f"{len(predictions)}"
)


# ============================================================
# SAVE RAW MAPANYTHING PREDICTIONS
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "SAVING RAW MAPANYTHING PREDICTIONS"
)

print(
    "=" * 70
)


prediction_file = (
    OUTPUT_DIR
    /
    "predictions.pt"
)


print(
    "Moving predictions to CPU..."
)


def move_to_cpu(obj):

    if torch.is_tensor(obj):

        return (
            obj
            .detach()
            .cpu()
        )


    elif isinstance(
        obj,
        dict
    ):

        return {
            key:
                move_to_cpu(value)
            for key, value
            in obj.items()
        }


    elif isinstance(
        obj,
        list
    ):

        return [
            move_to_cpu(value)
            for value in obj
        ]


    elif isinstance(
        obj,
        tuple
    ):

        return tuple(
            move_to_cpu(value)
            for value
            in obj
        )


    else:

        return obj


predictions_cpu = (
    move_to_cpu(
        predictions
    )
)


print(
    "Saving:"
)

print(
    prediction_file
)


torch.save(
    predictions_cpu,
    prediction_file
)


print(
    "Raw predictions saved."
)


# ============================================================
# EXTRACT 3D DATA FOR SEMANTIC OVERLAY
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "PREPARING 3D DATA"
)

print(
    "=" * 70
)


frame_points = []
frame_rgb = []
frame_shapes = []
frame_valid_masks = []


for i, prediction in enumerate(
    predictions_cpu
):

    frame = selected_frames[i]


    pts3d = prediction[
        "pts3d"
    ]


    if torch.is_tensor(
        pts3d
    ):

        pts3d = (
            pts3d
            .detach()
            .cpu()
            .numpy()
        )


    pts3d = np.asarray(
        pts3d
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
            f"Unexpected pts3d shape "
            f"for frame {frame}: "
            f"{pts3d.shape}"
        )


    ph, pw = pts3d.shape[:2]


    img = Image.open(
        selected_images[i]
    ).convert("RGB")


    img = img.resize(
        (
            pw,
            ph,
        ),
        Image.Resampling.BILINEAR,
    )


    img = np.asarray(
        img
    )


    valid = np.isfinite(
        pts3d
    ).all(axis=2)


    points = pts3d[
        valid
    ].astype(
        np.float32
    )


    rgb = img[
        valid
    ].astype(
        np.uint8
    )


    frame_points.append(
        points
    )


    frame_rgb.append(
        rgb
    )


    frame_shapes.append(
        (ph, pw)
    )


    frame_valid_masks.append(
        valid
    )


    print(
        f"Frame {frame}: "
        f"{ph}x{pw}, "
        f"{len(points):,} valid points"
    )


# ============================================================
# FREE MAPANYTHING MODEL
# ============================================================

del model
del views

gc.collect()


if torch.cuda.is_available():

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


print(
    "\nMapAnything memory released."
)


# ============================================================
# LOAD RF-DETR
# ============================================================
#
# KEEPING YOUR ORIGINAL WORKING RF-DETR LOADER.
#
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "LOADING RF-DETR"
)

print(
    "=" * 70
)


print(
    f"Checkpoint:"
)

print(
    RFDETR_MODEL_PATH
)


if not RFDETR_MODEL_PATH.exists():

    raise FileNotFoundError(
        "RF-DETR checkpoint does not exist:\n"
        f"{RFDETR_MODEL_PATH}"
    )


# ============================================================
# ORIGINAL WORKING RF-DETR LOADER
# ============================================================

rfdetr_model = (
    RFDETRSegSmall.from_checkpoint(
        str(
            RFDETR_MODEL_PATH
        )
    )
)


try:

    rfdetr_model.model.to(
        device
    )

except Exception:

    pass


try:

    rfdetr_model.optimize_for_inference()

except Exception:

    pass


print(
    "RF-DETR loaded."
)


# ============================================================
# FONT
# ============================================================

try:

    font = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/"
        "DejaVuSans.ttf",
        30,
    )

except Exception:

    font = ImageFont.load_default()


# ============================================================
# SEMANTIC OVERLAY STORAGE
# ============================================================

overlay_points = []
overlay_rgb = []


# ============================================================
# HAZARD GPS RECORDS
# ============================================================

hazard_records = []


# ============================================================
# RF-DETR
# ============================================================

for i, image_path in enumerate(
    selected_images
):

    frame = selected_frames[i]


    print(
        "\n" + "-" * 70
    )


    print(
        f"RF-DETR frame {frame} "
        f"({i + 1}/"
        f"{len(selected_images)})"
    )


    # ========================================================
    # GPS
    # ========================================================

    image_timestamp_ns = (
        image_gps[frame]
        ["image_timestamp_ns"]
    )


    gps = (
        image_gps[frame]
        ["gps"]
    )


    # ========================================================
    # ORIGINAL IMAGE
    # ========================================================

    image = Image.open(
        image_path
    ).convert("RGB")


    image_np = np.asarray(
        image
    )


    height, width = (
        image_np.shape[:2]
    )


    # ========================================================
    # RF-DETR PREDICTION
    # ========================================================

    detections = (
        rfdetr_model.predict(
            image_np,
            threshold=
                RFDETR_CONFIDENCE_THRESHOLD,
            include_source_image=False,
        )
    )


    # ========================================================
    # LABEL MAP
    # ========================================================

    labels = np.full(
        (height, width),
        -1,
        dtype=np.int16,
    )


    confidence_map = np.zeros(
        (height, width),
        dtype=np.float32,
    )


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


    detected_instances = []


    if (
        masks is not None
        and class_ids is not None
    ):

        masks = np.asarray(
            masks
        )


        class_ids = np.asarray(
            class_ids
        )


        if confidences is None:

            confidences = np.ones(
                len(class_ids),
                dtype=np.float32,
            )

        else:

            confidences = np.asarray(
                confidences
            )


        for j in range(
            len(class_ids)
        ):

            cls = int(
                class_ids[j]
            )


            score = float(
                confidences[j]
            )


            if (
                score
                <
                RFDETR_CONFIDENCE_THRESHOLD
            ):

                continue


            mask = np.squeeze(
                masks[j]
            ).astype(
                np.float32
            )


            if mask.shape != (
                height,
                width,
            ):

                mask = np.asarray(
                    Image.fromarray(
                        (
                            mask * 255
                        ).astype(
                            np.uint8
                        )
                    ).resize(
                        (
                            width,
                            height,
                        ),
                        Image.Resampling.NEAREST,
                    )
                ) / 255.0


            mask = (
                mask
                >= RFDETR_MASK_THRESHOLD
            )


            update = (
                mask
                &
                (
                    score
                    >
                    confidence_map
                )
            )


            labels[
                update
            ] = cls


            confidence_map[
                update
            ] = score


            detected_instances.append(
                {
                    "class_id":
                        cls,

                    "class_name":
                        CLASS_NAMES.get(
                            cls,
                            f"class_{cls}",
                        ),

                    "confidence":
                        score,

                    "mask":
                        mask,
                }
            )


    # ========================================================
    # SAVE LABELLED IMAGE
    # ========================================================

    labelled = (
        image_np.copy()
    )


    detected_classes = []


    for cls, colour in (
        CLASS_COLORS.items()
    ):

        mask = (
            labels == cls
        )


        if not np.any(mask):

            continue


        detected_classes.append(
            cls
        )


        colour = np.asarray(
            colour,
            dtype=np.float32,
        )


        labelled[
            mask
        ] = (
            SEMANTIC_ALPHA
            * colour
            +
            (1 - SEMANTIC_ALPHA)
            * labelled[mask]
        ).astype(
            np.uint8
        )


    labelled_img = (
        Image.fromarray(
            labelled
        )
    )


    draw = ImageDraw.Draw(
        labelled_img
    )


    # ========================================================
    # LEGEND
    # ========================================================

    if detected_classes:

        x = 25
        y = 25

        line_height = 45

        box_width = 350

        box_height = (
            20
            +
            len(detected_classes)
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


        for j, cls in enumerate(
            detected_classes
        ):

            yy = (
                y
                +
                12
                +
                j * line_height
            )


            draw.rectangle(
                [
                    x + 10,
                    yy,
                    x + 35,
                    yy + 25,
                ],
                fill=CLASS_COLORS[cls],
            )


            draw.text(
                (
                    x + 45,
                    yy,
                ),
                CLASS_NAMES[cls],
                fill=(255, 255, 255),
                font=font,
            )


    # ========================================================
    # FRAME NUMBER
    # ========================================================

    draw.text(
        (
            25,
            height - 90,
        ),
        f"Frame {frame}",
        fill=(255, 255, 255),
        font=font,
    )


    # ========================================================
    # GPS ON LABELLED IMAGE
    # ========================================================

    if gps is not None:

        gps_text = (
            f"GPS: "
            f"{gps['latitude_deg']:.7f}, "
            f"{gps['longitude_deg']:.7f}"
        )


        draw.text(
            (
                25,
                height - 50,
            ),
            gps_text,
            fill=(255, 255, 255),
            font=font,
        )


    labelled_img.save(
        LABELLED_DIR
        /
        f"frame_{frame}.png"
    )


    # ========================================================
    # RECORD HAZARDS + GPS
    # ========================================================

    for detection in (
        detected_instances
    ):

        cls = detection[
            "class_id"
        ]


        class_name = detection[
            "class_name"
        ]


        confidence = detection[
            "confidence"
        ]


        mask = detection[
            "mask"
        ]


        pixel_count = int(
            np.sum(mask)
        )


        if gps is None:

            print(
                f"  WARNING: "
                f"{class_name} detected "
                f"but GPS unavailable."
            )

            continue


        hazard_records.append(
            {
                "frame":
                    frame,

                "image_filename":
                    image_path.name,

                "image_timestamp_ns":
                    image_timestamp_ns,

                "gps_timestamp_ns":
                    gps[
                        "gps_timestamp_ns"
                    ],

                "gps_delta_ms":
                    gps[
                        "gps_delta_ms"
                    ],

                "latitude_deg":
                    gps[
                        "latitude_deg"
                    ],

                "longitude_deg":
                    gps[
                        "longitude_deg"
                    ],

                "altitude_m":
                    gps[
                        "altitude_m"
                    ],

                "class_id":
                    cls,

                "hazard":
                    class_name,

                "confidence":
                    confidence,

                "mask_pixels":
                    pixel_count,
            }
        )


        print(
            f"  HAZARD: "
            f"{class_name} "
            f"| confidence="
            f"{confidence:.3f} "
            f"| GPS="
            f"{gps['latitude_deg']:.8f}, "
            f"{gps['longitude_deg']:.8f} "
            f"| Δ="
            f"{gps['gps_delta_ms']:.2f} ms"
        )


    # ========================================================
    # 3D SEMANTIC OVERLAY
    # ========================================================

    pts = frame_points[i]

    original_rgb = frame_rgb[i]

    ph, pw = frame_shapes[i]

    valid = frame_valid_masks[i]


    labels_small = np.asarray(
        Image.fromarray(
            labels
        ).resize(
            (
                pw,
                ph,
            ),
            Image.Resampling.NEAREST,
        )
    )


    labels_flat = (
        labels_small
        .reshape(-1)
    )


    labels_flat = (
        labels_flat[
            valid.reshape(-1)
        ]
    )


    overlay = (
        original_rgb.copy()
    )


    for cls, colour in (
        CLASS_COLORS.items()
    ):

        mask = (
            labels_flat == cls
        )


        if not np.any(mask):

            continue


        colour = np.asarray(
            colour,
            dtype=np.float32,
        )


        overlay[
            mask
        ] = (
            SEMANTIC_ALPHA
            * colour
            +
            (1 - SEMANTIC_ALPHA)
            * overlay[mask]
        ).astype(
            np.uint8
        )


    overlay_points.append(
        pts
    )


    overlay_rgb.append(
        overlay
    )


    print(
        f"  Semantic 3D points: "
        f"{np.sum(labels_flat >= 0):,}"
    )


# ============================================================
# SAVE SEMANTIC OVERLAY
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "SAVING SEMANTIC 3D OVERLAY"
)

print(
    "=" * 70
)


overlay_points = np.concatenate(
    overlay_points,
    axis=0
)


overlay_rgb = np.concatenate(
    overlay_rgb,
    axis=0
)


overlay_path = (
    OUTPUT_DIR
    /
    "semantic_overlay_3d.ply"
)


save_ply(
    overlay_path,
    overlay_points,
    overlay_rgb,
)


# ============================================================
# SAVE HAZARD GPS CSV
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "SAVING HAZARD GPS CSV"
)

print(
    "=" * 70
)


hazard_csv_path = (
    OUTPUT_DIR
    /
    "hazards_gps.csv"
)


hazard_df = pd.DataFrame(
    hazard_records
)


hazard_df.to_csv(
    hazard_csv_path,
    index=False,
)


print(
    f"Hazard records: "
    f"{len(hazard_df)}"
)


print(
    f"Saved: "
    f"{hazard_csv_path}"
)


# ============================================================
# CREATE INTERACTIVE GPS MAP
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "CREATING GPS HAZARD MAP"
)

print(
    "=" * 70
)


map_center_lat = (
    gps_df["latitude_deg"]
    .mean()
)


map_center_lon = (
    gps_df["longitude_deg"]
    .mean()
)


m = folium.Map(
    location=[
        map_center_lat,
        map_center_lon,
    ],
    zoom_start=18,
    tiles="OpenStreetMap",
    control_scale=True,
)


# ============================================================
# VEHICLE GPS PATH
# ============================================================

gps_trajectory = list(
    zip(
        gps_df["latitude_deg"],
        gps_df["longitude_deg"],
    )
)


folium.PolyLine(
    gps_trajectory,
    color="blue",
    weight=4,
    opacity=0.7,
    tooltip="Vehicle GPS trajectory",
).add_to(m)


# ============================================================
# START
# ============================================================

start_lat = (
    gps_df["latitude_deg"]
    .iloc[0]
)


start_lon = (
    gps_df["longitude_deg"]
    .iloc[0]
)


folium.Marker(
    [
        start_lat,
        start_lon,
    ],
    popup=folium.Popup(
        f"""
        <b>START</b><br>
        Latitude: {start_lat:.9f}<br>
        Longitude: {start_lon:.9f}
        """,
        max_width=350,
    ),
    tooltip="START",
    icon=folium.Icon(
        color="green",
        icon="play",
    ),
).add_to(m)


# ============================================================
# END
# ============================================================

end_lat = (
    gps_df["latitude_deg"]
    .iloc[-1]
)


end_lon = (
    gps_df["longitude_deg"]
    .iloc[-1]
)


folium.Marker(
    [
        end_lat,
        end_lon,
    ],
    popup=folium.Popup(
        f"""
        <b>END</b><br>
        Latitude: {end_lat:.9f}<br>
        Longitude: {end_lon:.9f}
        """,
        max_width=350,
    ),
    tooltip="END",
    icon=folium.Icon(
        color="red",
        icon="stop",
    ),
).add_to(m)


# ============================================================
# HAZARD LAYERS
# ============================================================

feature_groups = {}


if len(hazard_df) > 0:

    for hazard_name in (
        hazard_df["hazard"]
        .unique()
    ):

        feature_groups[
            hazard_name
        ] = folium.FeatureGroup(
            name=hazard_name
        )

        feature_groups[
            hazard_name
        ].add_to(m)


    # ========================================================
    # ADD HAZARD MARKERS
    # ========================================================

    for _, row in (
        hazard_df.iterrows()
    ):

        hazard = row[
            "hazard"
        ]


        marker_color = (
            HAZARD_MARKER_COLORS
            .get(
                hazard,
                "black",
            )
        )


        popup_html = f"""
        <div style="width:350px">

        <h4>{hazard}</h4>

        <b>Frame:</b>
        {int(row["frame"])}
        <br>

        <b>Confidence:</b>
        {row["confidence"]:.3f}
        <br><br>

        <b>Latitude:</b>
        {row["latitude_deg"]:.9f}
        <br>

        <b>Longitude:</b>
        {row["longitude_deg"]:.9f}
        <br>

        <b>Altitude:</b>
        {row["altitude_m"]:.3f} m
        <br><br>

        <b>Image timestamp:</b>
        {int(row["image_timestamp_ns"])}
        <br>

        <b>GPS timestamp:</b>
        {int(row["gps_timestamp_ns"])}
        <br>

        <b>GPS Δ:</b>
        {row["gps_delta_ms"]:.3f} ms
        <br><br>

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
                f"Frame "
                f"{int(row['frame'])} | "
                f"Conf "
                f"{row['confidence']:.2f}"
            ),

            icon=folium.Icon(
                color=marker_color,
                icon="warning-sign",
            ),

        ).add_to(
            feature_groups[
                hazard
            ]
        )


# ============================================================
# MEASURE CONTROL
# ============================================================

m.add_child(
    MeasureControl()
)


# ============================================================
# FIT MAP
# ============================================================

m.fit_bounds(
    [
        [
            gps_df["latitude_deg"].min(),
            gps_df["longitude_deg"].min(),
        ],
        [
            gps_df["latitude_deg"].max(),
            gps_df["longitude_deg"].max(),
        ],
    ]
)


# ============================================================
# LAYER CONTROL
# ============================================================

folium.LayerControl(
    collapsed=False
).add_to(m)


# ============================================================
# SAVE MAP
# ============================================================

map_path = (
    OUTPUT_DIR
    /
    "hazards_map.html"
)


m.save(
    map_path
)


print(
    f"Saved map:"
)


print(
    map_path
)


# ============================================================
# CLEANUP
# ============================================================

del predictions
del predictions_cpu

del frame_points
del frame_rgb
del frame_shapes
del frame_valid_masks

del overlay_points
del overlay_rgb

gc.collect()


if torch.cuda.is_available():

    torch.cuda.empty_cache()


# ============================================================
# FINAL SUMMARY
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "DONE"
)

print(
    "=" * 70
)


print(
    f"Frames processed: "
    f"{len(selected_frames)}"
)


print(
    f"Stride: "
    f"{STRIDE}"
)


print(
    "\nRAW MAPANYTHING:"
)

print(
    prediction_file
)


print(
    "\nSEMANTIC 3D:"
)

print(
    overlay_path
)


print(
    "\nHAZARD GPS CSV:"
)

print(
    hazard_csv_path
)


print(
    "\nINTERACTIVE MAP:"
)

print(
    map_path
)


print(
    "\nLABELLED IMAGES:"
)

print(
    LABELLED_DIR
)


print(
    "\n" + "=" * 70
)