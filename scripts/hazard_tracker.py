# ============================================================
# RF-DETR SEG SMALL + GPS HAZARD TRACKER
#
# FEATURES
# ============================================================
#
# 1. RF-DETR Seg Small inference
# 2. ACTUAL INSTANCE SEGMENTATION MASKS
# 3. GPS matching
# 4. Anchor-based spatial hazard clustering
# 5. Configurable clustering radius
# 6. Highest-confidence detection = representative image
# 7. Segmentation overlay instead of bounding box
# 8. Masks saved separately so inference doesn't need to
#    be rerun when changing clustering radius
# 9. Base64 embedded representative images in HTML
# 10. Lightweight HTML: ONE representative image per hazard
#
#
# MODES
# ============================================================
#
# --mode inference
#
#     ALWAYS reruns RF-DETR.
#
#     Use this when:
#       - changing the model
#       - changing confidence
#       - changing the input images
#       - you need to regenerate masks
#
#
# --mode map
#
#     NEVER runs RF-DETR.
#
#     Uses:
#         rfdetr_detections_detailed.csv
#         mask_cache/
#
#     Useful for changing clustering radius.
#
#
# --mode auto
#
#     If saved inference results exist:
#         -> skip RF-DETR
#
#     Otherwise:
#         -> run RF-DETR
#
#
# EXAMPLES
# ============================================================
#
# Fresh inference:
#
# python hazard_tracker.py --mode inference
#
#
# Fresh inference with 20 m clustering:
#
# python hazard_tracker.py --mode inference --cluster-radius 20
#
#
# Rebuild map only:
#
# python hazard_tracker.py --mode map --cluster-radius 20
#
#
# Automatically decide:
#
# python hazard_tracker.py --mode auto --cluster-radius 10
#
# ============================================================


import argparse
import base64
import csv
import html
import math
import time

import cv2
import torch
import numpy as np
import pandas as pd
import folium

from pathlib import Path
from rfdetr import RFDETRSegSmall


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="RF-DETR GPS hazard detection and segmentation"
)

parser.add_argument(
    "--mode",
    choices=[
        "auto",
        "inference",
        "map",
    ],
    default="auto",
    help=(
        "auto = run inference only if results do not exist; "
        "inference = always rerun RF-DETR; "
        "map = never run RF-DETR"
    ),
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

args = parser.parse_args()


# ============================================================
# SETTINGS
# ============================================================

CONFIDENCE_THRESHOLD = args.confidence

CLUSTER_RADIUS_METERS = args.cluster_radius

BATCH_SIZE = 128

TRAJECTORY_STRIDE = 20


# ============================================================
# REPRESENTATIVE IMAGE SETTINGS
# ============================================================

EMBED_IMAGE_MAX_WIDTH = 900

EMBED_JPEG_QUALITY = 70


# ============================================================
# SEGMENTATION SETTINGS
# ============================================================

# How transparent the segmentation overlay is.
#
# 0.0 = invisible
# 1.0 = completely solid
#
# 0.35-0.50 generally looks good.
#
MASK_ALPHA = 0.40


# Thickness of segmentation boundary.
MASK_CONTOUR_THICKNESS = 3


# ============================================================
# GPU
# ============================================================

if args.mode != "map":

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA GPU not available!"
        )


print()
print("=" * 80)
print("RF-DETR GPS HAZARD TRACKER")
print("=" * 80)

print(
    f"Mode                 : {args.mode}"
)

print(
    f"Cluster radius       : "
    f"{CLUSTER_RADIUS_METERS:.1f} m"
)

print(
    f"Confidence threshold : "
    f"{CONFIDENCE_THRESHOLD:.2f}"
)

print(
    f"Batch size           : "
    f"{BATCH_SIZE}"
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
    exist_ok=True
)


# ============================================================
# OUTPUT FILES
# ============================================================

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
    / "hazard_map_clustered.html"
)


# ============================================================
# MASK CACHE
# ============================================================
#
# Masks are NOT stored inside the CSV.
#
# Each segmentation mask gets its own PNG:
#
# mask_cache/
#
#     1788829505906519421_0.png
#     1788829505906519421_1.png
#     ...
#
# The CSV simply stores:
#
#     mask_path
#
# This keeps the CSV manageable.
# ============================================================

MASK_DIR = (
    OUTPUT_DIR
    / "mask_cache"
)


MASK_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# REPRESENTATIVE IMAGES
# ============================================================

REPRESENTATIVE_DIR = (
    OUTPUT_DIR
    / "hazard_clusters"
)


REPRESENTATIVE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CHECKPOINT
# ============================================================

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


# ============================================================
# SEGMENTATION COLOURS
# ============================================================

# OpenCV uses BGR.
#
# These are used ONLY for drawing the segmentation overlay.
# ============================================================

CLASS_COLORS = {

    0: (255, 0, 255),

    1: (0, 255, 255),

    2: (255, 80, 0),

    3: (0, 255, 0),

    4: (0, 120, 255),

}


# ============================================================
# HAZARD CLASSES
# ============================================================

HAZARD_CLASSES = {

    "loose_trash",

    "mud_spill",

}


# ============================================================
# IMAGE EXTENSIONS
# ============================================================

IMAGE_EXTENSIONS = {

    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",

}


# ============================================================
# HAVERSINE
# ============================================================

def haversine_meters(
    lat1,
    lon1,
    lat2,
    lon2,
):

    """
    Calculate GPS distance in metres.
    """

    R = 6371000.0

    lat1 = math.radians(
        lat1
    )

    lat2 = math.radians(
        lat2
    )

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
            math.sqrt(1.0 - a)
        )

    )

    return R * c


# ============================================================
# CLASS ID LOOKUP
# ============================================================

def get_class_id(
    class_name
):

    for class_id, name in CLASS_NAMES.items():

        if name == class_name:

            return class_id

    return -1


# ============================================================
# MASK NORMALISATION
# ============================================================

def normalise_mask(
    mask,
    image_height,
    image_width,
):
    """
    Convert an RF-DETR mask into a clean boolean numpy array
    matching the original image dimensions.

    RF-DETR/supervision versions can return masks with slightly
    different shapes, so this function handles common cases.
    """

    # --------------------------------------------------------
    # Convert torch tensor
    # --------------------------------------------------------

    if torch.is_tensor(mask):

        mask = mask.detach().cpu().numpy()


    mask = np.asarray(
        mask
    )


    # --------------------------------------------------------
    # Remove singleton dimensions
    # --------------------------------------------------------

    mask = np.squeeze(
        mask
    )


    # --------------------------------------------------------
    # Convert to binary
    # --------------------------------------------------------

    if mask.dtype != np.bool_:

        mask = (
            mask > 0.5
        )


    # --------------------------------------------------------
    # Ensure 2D
    # --------------------------------------------------------

    if mask.ndim != 2:

        raise ValueError(

            f"Unexpected mask shape: "
            f"{mask.shape}"

        )


    # --------------------------------------------------------
    # Resize if necessary
    # --------------------------------------------------------

    if (

        mask.shape[0]
        !=
        image_height

        or

        mask.shape[1]
        !=
        image_width

    ):

        mask = cv2.resize(

            mask.astype(
                np.uint8
            ),

            (
                image_width,
                image_height
            ),

            interpolation=cv2.INTER_NEAREST

        ).astype(
            bool
        )


    return mask


# ============================================================
# SAVE MASK
# ============================================================

def save_mask(
    mask,
    output_path,
):
    """
    Save boolean segmentation mask as a compressed PNG.

    White  = mask
    Black  = background
    """

    mask_uint8 = (

        mask.astype(
            np.uint8
        )
        *
        255

    )


    success = cv2.imwrite(

        str(output_path),

        mask_uint8,

        [

            cv2.IMWRITE_PNG_COMPRESSION,
            9

        ]

    )


    if not success:

        raise RuntimeError(

            f"Failed to save mask: "
            f"{output_path}"

        )


# ============================================================
# LOAD MASK
# ============================================================

def load_mask(
    mask_path,
):

    mask = cv2.imread(

        str(mask_path),

        cv2.IMREAD_GRAYSCALE

    )


    if mask is None:

        raise RuntimeError(

            f"Could not load mask:\n"
            f"{mask_path}"

        )


    return (
        mask > 127
    )


# ============================================================
# DRAW SEGMENTATION
# ============================================================

def draw_segmentation(
    image,
    mask,
    class_id,
    confidence,
):
    """
    Draw ACTUAL segmentation mask.

    No bounding box is drawn.
    """

    class_name = CLASS_NAMES.get(

        int(class_id),

        str(class_id)

    )


    color = CLASS_COLORS.get(

        int(class_id),

        (255, 255, 255)

    )


    # ========================================================
    # MASK OVERLAY
    # ========================================================

    overlay = image.copy()


    overlay[
        mask
    ] = color


    image[
        :
    ] = cv2.addWeighted(

        overlay,

        MASK_ALPHA,

        image,

        1.0 - MASK_ALPHA,

        0

    )[

        :

    ]


    # ========================================================
    # MASK CONTOUR
    # ========================================================

    mask_uint8 = (

        mask.astype(
            np.uint8
        )
        *
        255

    )


    contours, _ = cv2.findContours(

        mask_uint8,

        cv2.RETR_EXTERNAL,

        cv2.CHAIN_APPROX_SIMPLE

    )


    cv2.drawContours(

        image,

        contours,

        -1,

        color,

        MASK_CONTOUR_THICKNESS

    )


    # ========================================================
    # LABEL
    # ========================================================
    #
    # Place label near the top-left of the mask rather than
    # drawing a bounding box.
    # ========================================================

    ys, xs = np.where(
        mask
    )


    if len(xs) == 0:

        return


    x = int(
        xs.min()
    )

    y = int(
        ys.min()
    )


    label = (

        f"{class_name} "
        f"{confidence:.2f}"

    )


    (
        tw,
        th
    ), baseline = cv2.getTextSize(

        label,

        cv2.FONT_HERSHEY_SIMPLEX,

        0.75,

        2

    )


    label_y1 = max(

        0,

        y - th - baseline - 8

    )


    cv2.rectangle(

        image,

        (
            x,
            label_y1
        ),

        (
            x + tw + 10,
            y
        ),

        color,

        -1

    )


    cv2.putText(

        image,

        label,

        (
            x + 5,
            y - 6
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.75,

        (0, 0, 0),

        2,

        cv2.LINE_AA

    )


# ============================================================
# LOAD GPS
# ============================================================

def load_gps():

    print()
    print("=" * 80)
    print("LOADING GPS")
    print("=" * 80)


    gps_df = pd.read_csv(
        GPS_FILE
    )


    gps_df[
        "ros_timestamp_ns"
    ] = pd.to_numeric(

        gps_df[
            "ros_timestamp_ns"
        ],

        errors="coerce"

    )


    gps_df[
        "latitude_deg"
    ] = pd.to_numeric(

        gps_df[
            "latitude_deg"
        ],

        errors="coerce"

    )


    gps_df[
        "longitude_deg"
    ] = pd.to_numeric(

        gps_df[
            "longitude_deg"
        ],

        errors="coerce"

    )


    gps_df[
        "altitude_m"
    ] = pd.to_numeric(

        gps_df[
            "altitude_m"
        ],

        errors="coerce"

    )


    gps_df = gps_df.dropna(

        subset=[

            "ros_timestamp_ns",

            "latitude_deg",

            "longitude_deg",

        ]

    ).copy()


    gps_df = gps_df[

        gps_df[
            "latitude_deg"
        ].between(
            -90,
            90
        )

    ]


    gps_df = gps_df[

        gps_df[
            "longitude_deg"
        ].between(
            -180,
            180
        )

    ]


    gps_df = gps_df.sort_values(

        "ros_timestamp_ns"

    ).reset_index(
        drop=True
    )


    print(
        f"GPS points: "
        f"{len(gps_df):,}"
    )


    return gps_df


# ============================================================
# GPS ARRAYS
# ============================================================

gps_df = load_gps()


gps_timestamps = gps_df[
    "ros_timestamp_ns"
].to_numpy(
    dtype=np.int64
)


gps_latitudes = gps_df[
    "latitude_deg"
].to_numpy(
    dtype=np.float64
)


gps_longitudes = gps_df[
    "longitude_deg"
].to_numpy(
    dtype=np.float64
)


gps_altitudes = gps_df[
    "altitude_m"
].to_numpy(
    dtype=np.float64
)


# ============================================================
# GPS LOOKUP
# ============================================================

def get_closest_gps(
    timestamp_ns
):

    idx = np.searchsorted(

        gps_timestamps,

        timestamp_ns

    )


    if idx <= 0:

        idx = 0


    elif idx >= len(
        gps_timestamps
    ):

        idx = (
            len(gps_timestamps)
            - 1
        )


    else:

        before = idx - 1

        after = idx


        before_diff = abs(

            gps_timestamps[
                before
            ]
            -
            timestamp_ns

        )


        after_diff = abs(

            gps_timestamps[
                after
            ]
            -
            timestamp_ns

        )


        if after_diff < before_diff:

            idx = after

        else:

            idx = before


    return {

        "gps_timestamp_ns":
            int(
                gps_timestamps[idx]
            ),

        "latitude":
            float(
                gps_latitudes[idx]
            ),

        "longitude":
            float(
                gps_longitudes[idx]
            ),

        "altitude":
            float(
                gps_altitudes[idx]
            ),

        "time_difference_ms":
            abs(

                int(
                    gps_timestamps[idx]
                )
                -
                int(timestamp_ns)

            )
            /
            1e6,

    }


# ============================================================
# DETERMINE INFERENCE MODE
# ============================================================

if args.mode == "inference":

    RUN_INFERENCE = True


elif args.mode == "map":

    RUN_INFERENCE = False


else:

    # --------------------------------------------------------
    # AUTO
    # --------------------------------------------------------

    RUN_INFERENCE = not (

        DETAILED_DETECTIONS_CSV.exists()

        and

        MASK_DIR.exists()

    )


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

if RUN_INFERENCE:

    print()
    print("=" * 80)
    print("RF-DETR INFERENCE WILL RUN")
    print("=" * 80)


else:

    print()
    print("=" * 80)
    print("RF-DETR INFERENCE WILL BE SKIPPED")
    print("=" * 80)


    if not DETAILED_DETECTIONS_CSV.exists():

        raise FileNotFoundError(

            "Inference CSV does not exist:\n\n"

            f"{DETAILED_DETECTIONS_CSV}\n\n"

            "Run:\n"

            "python hazard_tracker.py "
            "--mode inference"

        )


    # --------------------------------------------------------
    # Check masks exist
    # --------------------------------------------------------

    mask_files = list(

        MASK_DIR.glob(
            "*.png"
        )

    )


    if len(mask_files) == 0:

        raise FileNotFoundError(

            "No segmentation masks were found in:\n\n"

            f"{MASK_DIR}\n\n"

            "The previous inference run only saved "
            "bounding boxes.\n\n"

            "Run a fresh inference:\n\n"

            "python hazard_tracker.py "
            "--mode inference"

        )


    # --------------------------------------------------------
    # Verify CSV contains mask_path
    # --------------------------------------------------------

    test_df = pd.read_csv(

        DETAILED_DETECTIONS_CSV,

        nrows=1

    )


    if "mask_path" not in test_df.columns:

        raise RuntimeError(

            "The existing inference CSV does not contain "
            "'mask_path'.\n\n"

            "Run fresh inference with:\n\n"

            "python hazard_tracker.py "
            "--mode inference"

        )


# ============================================================
# RUN RF-DETR
# ============================================================

if RUN_INFERENCE:

    # ========================================================
    # LOAD MODEL
    # ========================================================

    print()
    print("=" * 80)
    print("LOADING RF-DETR SEG SMALL")
    print("=" * 80)


    t_model = time.perf_counter()


    rfdetr_model = RFDETRSegSmall.from_checkpoint(

        str(
            RFDETR_MODEL_PATH
        )

    )


    model_time = (

        time.perf_counter()
        -
        t_model

    )


    print(

        f"Checkpoint loaded in "
        f"{model_time:.2f} sec"

    )


    # ========================================================
    # OPTIMIZED FP16
    # ========================================================

    print()
    print("=" * 80)
    print("COMPILING OPTIMIZED FP16 MODEL")
    print("=" * 80)


    t_compile = time.perf_counter()


    rfdetr_model.inference(

        dtype=torch.float16,

        batch_size=BATCH_SIZE

    )


    compile_time = (

        time.perf_counter()
        -
        t_compile

    )


    print(

        f"Optimized model ready in "
        f"{compile_time:.2f} sec"

    )


    # ========================================================
    # FIND IMAGES
    # ========================================================

    print()
    print("=" * 80)
    print("SCANNING IMAGES")
    print("=" * 80)


    images = [

        p

        for p in IMAGE_DIR.iterdir()

        if (

            p.is_file()

            and

            p.suffix.lower()
            in IMAGE_EXTENSIONS

        )

    ]


    images.sort(

        key=lambda p: (

            int(p.stem)

            if p.stem.isdigit()

            else p.stem

        )

    )


    total_images = len(
        images
    )


    print(

        f"Images found: "
        f"{total_images:,}"

    )


    # ========================================================
    # CLEAR OLD MASKS
    # ========================================================

    print()
    print(
        "Clearing old segmentation masks..."
    )


    for old_mask in MASK_DIR.glob(
        "*.png"
    ):

        old_mask.unlink()


    # ========================================================
    # DETAILED CSV
    # ========================================================

    detailed_file = open(

        DETAILED_DETECTIONS_CSV,

        "w",

        newline=""

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


    # ========================================================
    # FRAME CSV
    # ========================================================

    frame_file = open(

        FRAME_SUMMARY_CSV,

        "w",

        newline=""

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


    # ========================================================
    # TIMING
    # ========================================================

    inference_start = time.perf_counter()


    image_load_time = 0.0

    gpu_inference_time = 0.0

    postprocess_time = 0.0

    mask_save_time = 0.0


    processed = 0

    hazard_frames = 0

    hazard_detections = 0


    last_print = time.perf_counter()


    # ========================================================
    # MAIN LOOP
    # ========================================================

    for batch_start in range(

        0,

        total_images,

        BATCH_SIZE

    ):


        # ====================================================
        # LOAD BATCH
        # ====================================================

        t_load = time.perf_counter()


        batch_paths = images[

            batch_start:
            batch_start + BATCH_SIZE

        ]


        batch_images = []

        valid_paths = []


        for image_path in batch_paths:


            image = cv2.imread(

                str(image_path),

                cv2.IMREAD_COLOR

            )


            if image is None:

                print(

                    f"\nWARNING: failed to read "
                    f"{image_path.name}"

                )

                continue


            batch_images.append(
                image
            )

            valid_paths.append(
                image_path
            )


        if not batch_images:

            continue


        real_batch_size = len(
            batch_images
        )


        image_load_time += (

            time.perf_counter()
            -
            t_load

        )


        # ====================================================
        # PAD FINAL BATCH
        # ====================================================

        if real_batch_size < BATCH_SIZE:


            padding_needed = (

                BATCH_SIZE
                -
                real_batch_size

            )


            last_image = (

                batch_images[-1]

            )


            for _ in range(
                padding_needed
            ):

                batch_images.append(
                    last_image
                )


        # ====================================================
        # RF-DETR INFERENCE
        # ====================================================

        torch.cuda.synchronize()


        t_gpu = time.perf_counter()


        with torch.inference_mode():

            batch_detections = (

                rfdetr_model.predict(

                    batch_images

                )

            )


        torch.cuda.synchronize()


        gpu_inference_time += (

            time.perf_counter()
            -
            t_gpu

        )


        # ====================================================
        # REMOVE PADDING
        # ====================================================

        batch_detections = (

            batch_detections[
                :real_batch_size
            ]

        )


        # ====================================================
        # PROCESS EACH IMAGE
        # ====================================================

        for (

            image_path,
            image,
            detections

        ) in zip(

            valid_paths,

            batch_images[
                :real_batch_size
            ],

            batch_detections

        ):


            t_process = time.perf_counter()


            processed += 1


            # ------------------------------------------------
            # TIMESTAMP
            # ------------------------------------------------

            try:

                image_timestamp_ns = int(

                    image_path.stem

                )

            except ValueError:

                continue


            # ------------------------------------------------
            # IMAGE DIMENSIONS
            # ------------------------------------------------

            image_height, image_width = (

                image.shape[:2]

            )


            # ------------------------------------------------
            # GPS
            # ------------------------------------------------

            gps = get_closest_gps(

                image_timestamp_ns

            )


            # ------------------------------------------------
            # DETECTIONS
            # ------------------------------------------------

            boxes = detections.xyxy

            class_ids = detections.class_id

            confidences = detections.confidence


            # =================================================
            # GET MASKS
            # =================================================

            if not hasattr(
                detections,
                "mask"
            ):

                raise RuntimeError(

                    "RF-DETR returned no segmentation masks.\n\n"

                    "Make sure RFDETRSegSmall is being used "
                    "and that your installed RF-DETR version "
                    "supports segmentation predictions."

                )


            masks = detections.mask


            hazard_classes = set()

            all_detections = []

            loose_trash_count = 0

            mud_spill_count = 0

            hazard_count_this_frame = 0


            # =================================================
            # EACH DETECTION
            # =================================================

            for detection_index, (

                box,
                class_id,
                confidence

            ) in enumerate(

                zip(

                    boxes,

                    class_ids,

                    confidences

                )

            ):


                confidence = float(
                    confidence
                )


                class_id = int(
                    class_id
                )


                # ------------------------------------------------
                # CONFIDENCE FILTER
                # ------------------------------------------------

                if (

                    confidence
                    <
                    CONFIDENCE_THRESHOLD

                ):

                    continue


                class_name = CLASS_NAMES.get(

                    class_id,

                    str(class_id)

                )


                all_detections.append(

                    f"{class_name}:"
                    f"{confidence:.3f}"

                )


                # ------------------------------------------------
                # NOT A HAZARD
                # ------------------------------------------------

                if class_name not in HAZARD_CLASSES:

                    continue


                hazard_count_this_frame += 1


                hazard_classes.add(
                    class_name
                )


                # ------------------------------------------------
                # BOX
                # ------------------------------------------------

                box = np.asarray(

                    box,

                    dtype=float

                )


                # ------------------------------------------------
                # MASK
                # ------------------------------------------------
                #
                # IMPORTANT:
                #
                # We use the actual segmentation mask returned
                # by RF-DETR.
                #
                # We do NOT recreate it from the bounding box.
                # ------------------------------------------------

                raw_mask = masks[
                    detection_index
                ]


                mask = normalise_mask(

                    raw_mask,

                    image_height,

                    image_width

                )


                # ------------------------------------------------
                # MASK FILE
                # ------------------------------------------------

                mask_filename = (

                    f"{image_timestamp_ns}_"
                    f"{detection_index}_"
                    f"{class_name}.png"

                )


                mask_path = (

                    MASK_DIR
                    /
                    mask_filename

                )


                # ------------------------------------------------
                # SAVE MASK
                # ------------------------------------------------

                t_mask = time.perf_counter()


                save_mask(

                    mask,

                    mask_path

                )


                mask_save_time += (

                    time.perf_counter()
                    -
                    t_mask

                )


                # ------------------------------------------------
                # SAVE DETAILED DETECTION
                # ------------------------------------------------

                detailed_writer.writerow([

                    image_path.name,

                    image_timestamp_ns,

                    gps[
                        "gps_timestamp_ns"
                    ],

                    gps[
                        "time_difference_ms"
                    ],

                    gps[
                        "latitude"
                    ],

                    gps[
                        "longitude"
                    ],

                    gps[
                        "altitude"
                    ],

                    class_name,

                    class_id,

                    confidence,

                    float(
                        box[0]
                    ),

                    float(
                        box[1]
                    ),

                    float(
                        box[2]
                    ),

                    float(
                        box[3]
                    ),

                    str(
                        mask_path
                    ),

                ])


                if class_name == "loose_trash":

                    loose_trash_count += 1


                elif class_name == "mud_spill":

                    mud_spill_count += 1


            # ------------------------------------------------
            # FRAME SUMMARY
            # ------------------------------------------------

            hazard_detected = (

                hazard_count_this_frame
                >
                0

            )


            if hazard_detected:

                hazard_frames += 1

                hazard_detections += (

                    hazard_count_this_frame

                )


            # ------------------------------------------------
            # FRAME CSV
            # ------------------------------------------------

            frame_writer.writerow([

                image_path.name,

                image_timestamp_ns,

                gps[
                    "gps_timestamp_ns"
                ],

                gps[
                    "time_difference_ms"
                ],

                gps[
                    "latitude"
                ],

                gps[
                    "longitude"
                ],

                gps[
                    "altitude"
                ],

                hazard_detected,

                ";".join(

                    sorted(
                        hazard_classes
                    )

                ),

                loose_trash_count,

                mud_spill_count,

                ";".join(
                    all_detections
                ),

            ])


            postprocess_time += (

                time.perf_counter()
                -
                t_process

            )


        # ====================================================
        # PROGRESS
        # ====================================================

        now = time.perf_counter()


        if (

            now - last_print
            >=
            2.0

        ):


            elapsed = (

                now
                -
                inference_start

            )


            fps = (

                processed / elapsed

                if elapsed > 0

                else 0

            )


            remaining = (

                total_images
                -
                processed

            )


            eta = (

                remaining / fps

                if fps > 0

                else 0

            )


            print(

                f"\r"
                f"{processed:,}/"
                f"{total_images:,} "
                f"("
                f"{100 * processed / total_images:.1f}%"
                f") | "
                f"{fps:.2f} img/s | "
                f"ETA {eta / 60:.1f} min | "
                f"hazard frames "
                f"{hazard_frames:,} | "
                f"hazard detections "
                f"{hazard_detections:,}",

                end="",

                flush=True

            )


            last_print = now


    # ========================================================
    # CLOSE FILES
    # ========================================================

    detailed_file.close()

    frame_file.close()


    total_inference_time = (

        time.perf_counter()
        -
        inference_start

    )


    print()
    print()
    print("=" * 80)
    print("RF-DETR INFERENCE COMPLETE")
    print("=" * 80)


    print()

    print(

        f"Images processed      : "
        f"{processed:,}"

    )


    print(

        f"Hazard frames         : "
        f"{hazard_frames:,}"

    )


    print(

        f"Hazard detections     : "
        f"{hazard_detections:,}"

    )


    print()

    print(
        "TIMING"
    )

    print("-" * 80)


    print(

        f"Image loading         : "
        f"{image_load_time / 60:.2f} min"

    )


    print(

        f"RF-DETR inference     : "
        f"{gpu_inference_time / 60:.2f} min"

    )


    print(

        f"Post-processing       : "
        f"{postprocess_time / 60:.2f} min"

    )


    print(

        f"Mask saving           : "
        f"{mask_save_time / 60:.2f} min"

    )


    print(

        f"Total inference       : "
        f"{total_inference_time / 60:.2f} min"

    )


    if total_inference_time > 0:

        print()

        print(

            f"Average throughput    : "
            f"{processed / total_inference_time:.2f} img/s"

        )


# ============================================================
# LOAD DETECTIONS
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


if "mask_path" not in detection_df.columns:

    raise RuntimeError(

        "Saved detections do not contain segmentation "
        "mask paths.\n\n"

        "Run fresh inference:\n\n"

        "python hazard_tracker.py "
        "--mode inference"

    )


print(

    f"Hazard detections loaded: "
    f"{len(detection_df):,}"

)


# ============================================================
# SORT CHRONOLOGICALLY
# ============================================================

detection_df = detection_df.sort_values(

    "image_ros_timestamp_ns"

).reset_index(
    drop=True
)


# ============================================================
# CLUSTER HAZARDS
# ============================================================

print()
print("=" * 80)
print("CLUSTERING HAZARDS")
print("=" * 80)


print(

    f"Clustering radius: "
    f"{CLUSTER_RADIUS_METERS:.2f} m"

)


print()

print(

    "Rule:"
)

print(

    "FIRST detection = permanent anchor."

)


print(

    "Same class + within radius of anchor "
    "= same physical hazard."

)


print()


clusters = []


cluster_start = time.perf_counter()


# ============================================================
# CLUSTER LOOP
# ============================================================

for _, row in detection_df.iterrows():


    hazard_class = str(

        row[
            "class"
        ]

    )


    latitude = float(

        row[
            "latitude"
        ]

    )


    longitude = float(

        row[
            "longitude"
        ]

    )


    altitude = float(

        row[
            "altitude"
        ]

    )


    confidence = float(

        row[
            "confidence"
        ]

    )


    image_name = str(

        row[
            "image"
        ]

    )


    timestamp_ns = int(

        row[
            "image_ros_timestamp_ns"
        ]

    )


    mask_path = str(

        row[
            "mask_path"
        ]

    )


    box = np.array([

        float(
            row["x1"]
        ),

        float(
            row["y1"]
        ),

        float(
            row["x2"]
        ),

        float(
            row["y2"]
        ),

    ])


    # ========================================================
    # FIND CLOSEST SAME-CLASS CLUSTER
    # ========================================================

    closest_cluster = None

    closest_distance = float(
        "inf"
    )


    for cluster in clusters:


        if (

            cluster[
                "class"
            ]
            !=
            hazard_class

        ):

            continue


        distance = haversine_meters(

            latitude,

            longitude,

            cluster[
                "anchor_latitude"
            ],

            cluster[
                "anchor_longitude"
            ],

        )


        if distance < closest_distance:

            closest_distance = distance

            closest_cluster = cluster


    # ========================================================
    # EXISTING CLUSTER
    # ========================================================

    if (

        closest_cluster is not None

        and

        closest_distance
        <=
        CLUSTER_RADIUS_METERS

    ):


        cluster = closest_cluster


        cluster[
            "detection_count"
        ] += 1


        cluster[
            "max_distance_m"
        ] = max(

            cluster[
                "max_distance_m"
            ],

            closest_distance

        )


        cluster[
            "detections"
        ].append({

            "image":
                image_name,

            "timestamp_ns":
                timestamp_ns,

            "latitude":
                latitude,

            "longitude":
                longitude,

            "altitude":
                altitude,

            "confidence":
                confidence,

            "box":
                box,

            "mask_path":
                mask_path,

        })


        # ----------------------------------------------------
        # Highest confidence representative
        # ----------------------------------------------------

        if (

            confidence
            >
            cluster[
                "best_confidence"
            ]

        ):


            cluster[
                "best_confidence"
            ] = confidence


            cluster[
                "best_detection"
            ] = {

                "image":
                    image_name,

                "timestamp_ns":
                    timestamp_ns,

                "latitude":
                    latitude,

                "longitude":
                    longitude,

                "altitude":
                    altitude,

                "confidence":
                    confidence,

                "box":
                    box,

                "mask_path":
                    mask_path,

            }


    # ========================================================
    # NEW CLUSTER
    # ========================================================

    else:


        cluster_id = (

            len(clusters)
            +
            1

        )


        detection_record = {

            "image":
                image_name,

            "timestamp_ns":
                timestamp_ns,

            "latitude":
                latitude,

            "longitude":
                longitude,

            "altitude":
                altitude,

            "confidence":
                confidence,

            "box":
                box,

            "mask_path":
                mask_path,

        }


        cluster = {

            "cluster_id":
                cluster_id,

            "class":
                hazard_class,

            # ------------------------------------------------
            # PERMANENT ANCHOR
            # ------------------------------------------------

            "anchor_latitude":
                latitude,

            "anchor_longitude":
                longitude,

            "anchor_altitude":
                altitude,

            "anchor_image":
                image_name,

            "anchor_timestamp_ns":
                timestamp_ns,

            # ------------------------------------------------
            # STATS
            # ------------------------------------------------

            "detection_count":
                1,

            "max_distance_m":
                0.0,

            # ------------------------------------------------
            # REPRESENTATIVE
            # ------------------------------------------------

            "best_confidence":
                confidence,

            "best_detection":
                detection_record,

            "detections":
                [
                    detection_record
                ],

        }


        clusters.append(
            cluster
        )


cluster_time = (

    time.perf_counter()
    -
    cluster_start

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
# PREPARE REPRESENTATIVE IMAGES
# ============================================================

print()
print("=" * 80)
print("PREPARING SEGMENTATION REPRESENTATIVE IMAGES")
print("=" * 80)


image_start = time.perf_counter()


# ============================================================
# DELETE OLD REPRESENTATIVE IMAGES
# ============================================================

for old_file in REPRESENTATIVE_DIR.iterdir():

    if old_file.is_file():

        old_file.unlink()


# ============================================================
# CREATE REPRESENTATIVE IMAGE FOR EACH CLUSTER
# ============================================================

for cluster in clusters:


    best = cluster[
        "best_detection"
    ]


    # --------------------------------------------------------
    # Source image
    # --------------------------------------------------------

    source_path = (

        IMAGE_DIR
        /
        best[
            "image"
        ]

    )


    if not source_path.exists():

        print()

        print(

            f"WARNING: missing image:"
            f" {source_path}"

        )

        continue


    # --------------------------------------------------------
    # Mask
    # --------------------------------------------------------

    mask_path = Path(

        best[
            "mask_path"
        ]

    )


    if not mask_path.exists():

        print()

        print(

            f"WARNING: missing mask:"
            f" {mask_path}"

        )

        continue


    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    image = cv2.imread(

        str(source_path),

        cv2.IMREAD_COLOR

    )


    if image is None:

        continue


    # --------------------------------------------------------
    # Read mask
    # --------------------------------------------------------

    mask = load_mask(

        mask_path

    )


    # --------------------------------------------------------
    # Resize mask if necessary
    # --------------------------------------------------------

    image_height, image_width = (

        image.shape[:2]

    )


    mask = normalise_mask(

        mask,

        image_height,

        image_width

    )


    # --------------------------------------------------------
    # CLASS
    # --------------------------------------------------------

    class_id = get_class_id(

        cluster[
            "class"
        ]

    )


    # ========================================================
    # ACTUAL SEGMENTATION
    # ========================================================

    draw_segmentation(

        image,

        mask,

        class_id,

        best[
            "confidence"
        ]

    )


    # ========================================================
    # HEADER
    # ========================================================

    header = (

        f"HAZARD #{cluster['cluster_id']} | "
        f"{cluster['class']}"

    )


    cv2.putText(

        image,

        header,

        (
            30,
            50
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        1.0,

        (0, 0, 255),

        3,

        cv2.LINE_AA

    )


    # ========================================================
    # GPS
    # ========================================================

    gps_text = (

        f"GPS: "
        f"{cluster['anchor_latitude']:.8f}, "
        f"{cluster['anchor_longitude']:.8f}"

    )


    cv2.putText(

        image,

        gps_text,

        (
            30,
            90
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        (255, 255, 255),

        2,

        cv2.LINE_AA

    )


    # ========================================================
    # DETECTION COUNT
    # ========================================================

    count_text = (

        f"Detections: "
        f"{cluster['detection_count']}"

    )


    cv2.putText(

        image,

        count_text,

        (
            30,
            125
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        (255, 255, 255),

        2,

        cv2.LINE_AA

    )


    # ========================================================
    # RESIZE
    # ========================================================

    height, width = image.shape[:2]


    if width > EMBED_IMAGE_MAX_WIDTH:


        scale = (

            EMBED_IMAGE_MAX_WIDTH
            /
            width

        )


        new_width = (

            EMBED_IMAGE_MAX_WIDTH

        )


        new_height = int(

            height
            *
            scale

        )


        image = cv2.resize(

            image,

            (
                new_width,
                new_height
            ),

            interpolation=cv2.INTER_AREA

        )


    # ========================================================
    # SAVE REPRESENTATIVE JPEG
    # ========================================================

    output_name = (

        f"hazard_"
        f"{cluster['cluster_id']:04d}_"
        f"{cluster['class']}.jpg"

    )


    output_path = (

        REPRESENTATIVE_DIR
        /
        output_name

    )


    cv2.imwrite(

        str(output_path),

        image,

        [

            cv2.IMWRITE_JPEG_QUALITY,

            EMBED_JPEG_QUALITY

        ]

    )


    cluster[
        "representative_image"
    ] = output_name


image_time = (

    time.perf_counter()
    -
    image_start

)


print(

    f"Representative images prepared in "
    f"{image_time:.2f} sec"

)


# ============================================================
# SAVE CLUSTER CSV
# ============================================================

print()
print(
    "Writing cluster CSV..."
)


with open(

    CLUSTER_CSV,

    "w",

    newline=""

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

        "representative_image",

    ])


    for cluster in clusters:


        detections = cluster[
            "detections"
        ]


        writer.writerow([

            cluster[
                "cluster_id"
            ],

            cluster[
                "class"
            ],

            cluster[
                "anchor_latitude"
            ],

            cluster[
                "anchor_longitude"
            ],

            cluster[
                "anchor_altitude"
            ],

            cluster[
                "anchor_image"
            ],

            cluster[
                "best_detection"
            ][
                "image"
            ],

            cluster[
                "best_confidence"
            ],

            cluster[
                "detection_count"
            ],

            cluster[
                "max_distance_m"
            ],

            detections[
                0
            ][
                "image"
            ],

            detections[
                -1
            ][
                "image"
            ],

            cluster.get(

                "representative_image",

                ""

            ),

        ])


# ============================================================
# BASE64 IMAGE
# ============================================================

def image_to_base64(
    image_path
):

    """
    Convert representative JPEG to Base64 so the HTML
    is self-contained.

    The HTML therefore does NOT depend on the JPEG files
    being beside the HTML.
    """

    if not image_path.exists():

        return None


    with open(

        image_path,

        "rb"

    ) as f:

        encoded = base64.b64encode(

            f.read()

        ).decode(
            "ascii"
        )


    return (

        "data:image/jpeg;base64,"
        +
        encoded

    )


# ============================================================
# CREATE MAP
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

        center_lon

    ],

    zoom_start=18,

    tiles="OpenStreetMap",

    control_scale=True,

)


# ============================================================
# HAZARD TOGGLE LAYERS
# ============================================================
#
# Each hazard class gets its own toggleable Folium layer.
#
# This means you can switch hazards on/off in the HTML map
# without regenerating the map or rerunning RF-DETR.
#
# Example:
#
#   ☑ Loose Trash
#   ☑ Mud Spill
#
# Unchecking a layer removes that hazard type from the map.
# ============================================================

hazard_layers = {}

for hazard_class in sorted(HAZARD_CLASSES):

    layer_name = hazard_class.replace("_", " ").title()

    hazard_layers[hazard_class] = folium.FeatureGroup(
        name=layer_name,
        overlay=True,
        show=True,
    )

    hazard_layers[hazard_class].add_to(m)



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
# START
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


# ============================================================
# END
# ============================================================

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


    # ========================================================
    # MARKER STYLE
    # ========================================================

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


    # ========================================================
    # REPRESENTATIVE IMAGE
    # ========================================================

    representative_name = (

        cluster.get(
            "representative_image"
        )

    )


    embedded_image = None


    if representative_name:


        image_path = (

            REPRESENTATIVE_DIR
            /
            representative_name

        )


        embedded_image = image_to_base64(

            image_path

        )


    if embedded_image:


        image_html = f"""

        <img
            src="{embedded_image}"
            style="
                width:100%;
                max-width:900px;
                height:auto;
                display:block;
                margin-top:10px;
                border-radius:6px;
            "
        >

        """


    else:


        image_html = """

        <p>
            Representative image unavailable.
        </p>

        """


    # ========================================================
    # POPUP
    # ========================================================

    popup_html = f"""

    <div style="
        width:900px;
        max-width:90vw;
        font-family:Arial,sans-serif;
    ">

        <h2 style="
            margin-bottom:8px;
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
                <td>
                    #{cluster["cluster_id"]}
                </td>
            </tr>


            <tr>
                <td><b>Type</b></td>
                <td>
                    {html.escape(hazard_class)}
                </td>
            </tr>


            <tr>
                <td><b>Detections</b></td>
                <td>
                    {detection_count}
                </td>
            </tr>


            <tr>
                <td><b>Anchor latitude</b></td>
                <td>
                    {latitude:.8f}
                </td>
            </tr>


            <tr>
                <td><b>Anchor longitude</b></td>
                <td>
                    {longitude:.8f}
                </td>
            </tr>


            <tr>
                <td><b>Maximum distance</b></td>
                <td>
                    {cluster["max_distance_m"]:.2f} m
                </td>
            </tr>


            <tr>
                <td><b>Best confidence</b></td>
                <td>
                    {cluster["best_confidence"]:.3f}
                </td>
            </tr>


            <tr>
                <td><b>First frame</b></td>
                <td>
                    {html.escape(
                        cluster["detections"][0]["image"]
                    )}
                </td>
            </tr>


            <tr>
                <td><b>Last frame</b></td>
                <td>
                    {html.escape(
                        cluster["detections"][-1]["image"]
                    )}
                </td>
            </tr>

        </table>


        <br>


        <h3>
            Segmentation
        </h3>


        {image_html}


    </div>

    """


    # ========================================================
    # POPUP
    # ========================================================

    popup = folium.Popup(

        folium.IFrame(

            popup_html,

            width=930,

            height=850,

        ),

        max_width=950,

    )


    # ========================================================
    # MARKER
    # ========================================================

    # ========================================================
    # ADD MARKER TO TOGGLEABLE HAZARD LAYER
    # ========================================================

    target_layer = hazard_layers.get(
        hazard_class
    )

    if target_layer is not None:

        folium.Marker(

            [

                latitude,

                longitude

            ],

            popup=popup,

            tooltip=(

                f"{title} | "
                f"{detection_count} detections"

            ),

            icon=folium.Icon(

                color=marker_color,

                icon=icon_name,

            ),

        ).add_to(target_layer)


# ============================================================
# HAZARD LAYER CONTROL
# ============================================================
#
# Adds the checkbox menu in the top-right corner of the map.
#
# You can toggle:
#   - Loose Trash
#   - Mud Spill
#
# independently while viewing the HTML.
# ============================================================

folium.LayerControl(
    position="topright",
    collapsed=False,
).add_to(m)


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
# SAVE HTML
# ============================================================

print()
print(
    "Embedding representative segmentation images..."
)


m.save(
    MAP_OUTPUT
)


map_time = (

    time.perf_counter()
    -
    map_start

)


# ============================================================
# FILE SIZE
# ============================================================

html_size_mb = (

    MAP_OUTPUT.stat().st_size
    /
    (1024 ** 2)

)


mask_cache_size_mb = sum(

    p.stat().st_size

    for p in MASK_DIR.glob(
        "*.png"
    )

) / (1024 ** 2)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 80)
print("DONE")
print("=" * 80)


print()

print(

    f"Cluster radius        : "
    f"{CLUSTER_RADIUS_METERS:.1f} m"

)


print(

    f"Original detections   : "
    f"{len(detection_df):,}"

)


print(

    f"Physical hazards      : "
    f"{len(clusters):,}"

)


print()

print(
    "Clusters by class:"
)


for class_name in sorted(

    set(

        cluster[
            "class"
        ]

        for cluster in clusters

    )

):


    count = sum(

        1

        for cluster in clusters

        if cluster[
            "class"
        ] == class_name

    )


    detections_count = sum(

        cluster[
            "detection_count"
        ]

        for cluster in clusters

        if cluster[
            "class"
        ] == class_name

    )


    print(

        f"  {class_name:<20}"
        f"{detections_count:>6} detections "
        f"-> "
        f"{count:>5} locations"

    )


print()

print(

    f"Clustering time       : "
    f"{cluster_time:.2f} sec"

)


print(

    f"Image preparation     : "
    f"{image_time:.2f} sec"

)


print(

    f"Map generation        : "
    f"{map_time:.2f} sec"

)


print()

print(

    f"Mask cache size       : "
    f"{mask_cache_size_mb:.2f} MB"

)


print(

    f"HTML size             : "
    f"{html_size_mb:.2f} MB"

)


print()

print(
    "HTML:"
)


print(
    MAP_OUTPUT
)


print()

print(
    "Detailed detections:"
)


print(
    DETAILED_DETECTIONS_CSV
)


print()

print(
    "Segmentation masks:"
)


print(
    MASK_DIR
)


print()

print(
    "Cluster CSV:"
)


print(
    CLUSTER_CSV
)


print()

print("=" * 80)