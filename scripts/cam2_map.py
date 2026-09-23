
import csv
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from mapanything.models import MapAnything
from mapanything.utils.image import preprocess_inputs
from rfdetr import RFDETRSegSmall


# ============================================================
# CONFIGURATION
# ============================================================

CAMERA = "CAM2"


# ============================================================
# SCAL3R BLOCK SETTINGS
# ============================================================

BLOCK_SIZE = 60
OVERLAP_SIZE = 30
BLOCK_STEP = BLOCK_SIZE - OVERLAP_SIZE


# ============================================================
# REPRESENTATIVE MAPANYTHING VIEWS
# ============================================================
#
# Each block is given ALL 60 images.
#
# Only these two views are kept as final geometry:
#
#     offset 29
#     offset 30
#
# These are the two central views of each block.
#
# Example:
#
# Block 10:
#
#     270 -> 329
#
#     offset 29 = frame 299
#     offset 30 = frame 300
#
# ============================================================

REPRESENTATIVE_OFFSETS = [
    29,
    30,
]


# ============================================================
# ORIGINAL CAM2 OFFSET
# ============================================================
#
# Scal3R frame 0 corresponds to original CAM2 image index 250.
#
# Therefore:
#
#     Scal3R 0    -> CAM2 250
#     Scal3R 270  -> CAM2 520
#
# ============================================================

SCAL3R_ORIGINAL_START_INDEX = 250


# ============================================================
# MAPANYTHING DATA
# ============================================================

EXTRACTED = Path(
    "/home/lcha0115/bt60_scratch/lcha_data/"
    "map-anything/personal_data/"
    "camera_lidar_test_101/extracted"
)

IMAGE_DIR = (
    EXTRACTED /
    CAMERA /
    "images_rect"
)

CAMERA_INFO_PATH = (
    EXTRACTED /
    CAMERA /
    "camera_info.csv"
)


# ============================================================
# SCAL3R INPUT
# ============================================================

SCAL3R_IMAGE_DIR = Path(
    "/home/lcha0115/bt60_scratch/lcha_data/"
    "map-anything/Scal3R/personal_data/"
    "sequential_250_7250/"
    "CAM2"
)


# ============================================================
# SCAL3R POSES
# ============================================================

SCAL3R_MAT_PATH = Path(
    "/home/lcha0115/bt60_scratch/lcha_data/"
    "map-anything/Scal3R/results/"
    "cam132/mat.txt"
)


# ============================================================
# OUTPUT
# ============================================================
#
# All blocks go into this one directory.
#
# global_rgb/
#     block_000.ply
#     block_001.ply
#     ...
#
# global_semantic/
#     block_000.ply
#     block_001.ply
#     ...
#
# Final:
#
#     merged_global_rgb.ply
#     merged_global_semantic.ply
#
# ============================================================

OUTPUT_DIR = Path(
    "/home/lcha0115/bt60_scratch/lcha_data/"
    "map-anything/personal_data/"
    "camera_lidar_test_101/"
    "map_cam2_scal3r_all_blocks_2views"
)

RAW_DIR = (
    OUTPUT_DIR /
    "global_rgb"
)

SEMANTIC_DIR = (
    OUTPUT_DIR /
    "global_semantic"
)

OVERLAY_DIR = (
    OUTPUT_DIR /
    "overlays"
)


# ============================================================
# RESUME
# ============================================================
#
# True:
#     If block PLY already exists, skip that block.
#
# This is VERY useful on HPC if the job gets killed.
#
# ============================================================

RESUME_EXISTING_BLOCKS = True


# ============================================================
# SAVE OVERLAYS
# ============================================================
#
# Saving only the representative-frame overlays.
#
# Two overlays per block.
#
# ============================================================

SAVE_OVERLAYS = True


# ============================================================
# RF-DETR
# ============================================================

RFDETR_MODEL_PATH = Path(
    "/home/lcha0115/bt60_scratch/lcha_data/"
    "map-anything/personal_data/"
    "2D-Seg-Road-Nissan.v6i.coco-segmentation/"
    "outputs/checkpoint_best_total.pth"
)

RFDETR_CONFIDENCE_THRESHOLD = 0.25
RFDETR_MASK_THRESHOLD = 0.50

SEMANTIC_OVERLAY_ALPHA = 0.65


# ============================================================
# CLASS COLOURS
# ============================================================

CLASS_COLORS = {

    0: (255, 0, 255),      # fallen_fence
    1: (0, 255, 255),      # loose_trash
    2: (255, 80, 0),       # mud_spill
    3: (0, 255, 0),       # public_greenway
    4: (0, 120, 255),      # sidewalk

}


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():

    device = torch.device(
        "cuda"
    )

else:

    device = torch.device(
        "cpu"
    )


# ============================================================
# PARSE ARRAY
# ============================================================

def parse_array(value):

    value = str(
        value
    ).strip()

    replacements = [

        "np.float64(",
        "np.float32(",
        "np.int64(",
        "np.int32(",

    ]

    for replacement in replacements:

        value = value.replace(
            replacement,
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

            f"Could not parse array: {value}"

        )

    return values


# ============================================================
# LOAD RECTIFIED INTRINSICS
# ============================================================

def load_rectified_intrinsics(path):

    print(
        "\nLoading camera intrinsics:"
    )

    print(
        path
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

            f"No camera info rows found:\n{path}"

        )

    row = rows[0]

    P = parse_array(
        row["P"]
    )

    if P.size != 12:

        raise ValueError(

            f"P contains {P.size} values; "
            "expected 12."

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
# LOAD ORIGINAL IMAGES
# ============================================================

def load_images(path):

    print(
        "\nLoading original CAM2 images:"
    )

    print(
        path
    )

    files = sorted(

        path.glob("*.png"),

        key=lambda p:
        int(p.stem)

    )

    records = []

    for p in files:

        try:

            timestamp = int(
                p.stem
            )

        except ValueError:

            continue

        records.append(
            (
                timestamp,
                p
            )
        )

    if not records:

        raise RuntimeError(

            f"No timestamped PNG images found:\n{path}"

        )

    return records


# ============================================================
# LOAD SCAL3R INPUT IMAGES
# ============================================================

def load_scal3r_input_images(path):

    print(
        "\nLoading Scal3R input images:"
    )

    print(
        path
    )

    files = sorted(

        path.glob("*.png"),

        key=lambda p:
        int(p.stem)

    )

    if not files:

        raise RuntimeError(

            f"No PNG images found:\n{path}"

        )

    return files


# ============================================================
# LOAD SCAL3R POSES
# ============================================================

def load_scal3r_poses(path):

    print(
        "\n"
        + "=" * 80
    )

    print(
        "LOADING SCAL3R POSES"
    )

    print(
        "=" * 80
    )

    print(
        "\nmat.txt:"
    )

    print(
        path
    )

    data = np.loadtxt(
        path
    )

    if data.ndim == 1:

        data = data.reshape(
            1,
            -1
        )

    if data.shape[1] != 16:

        raise ValueError(

            "Expected 16 values per pose. "

            f"Got shape {data.shape}"

        )

    poses = data.reshape(
        -1,
        4,
        4
    )

    print(
        "\nLoaded:",
        len(poses),
        "poses"
    )

    print(
        "Pose shape:",
        poses.shape
    )

    return poses


# ============================================================
# VALIDATE POSES
# ============================================================

def validate_poses(
    poses
):

    if not np.isfinite(
        poses
    ).all():

        raise ValueError(

            "Scal3R poses contain NaN/Inf."

        )

    translations = (
        poses[:, :3, 3]
    )

    rotations = (
        poses[:, :3, :3]
    )

    determinants = np.linalg.det(
        rotations
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "SCAL3R POSE VALIDATION"
    )

    print(
        "=" * 80
    )

    print(
        "\nFirst translation:"
    )

    print(
        translations[0]
    )

    print(
        "\nLast translation:"
    )

    print(
        translations[-1]
    )

    print(
        "\nOverall translation range:"
    )

    print(
        "X:",
        translations[:, 0].min(),
        "->",
        translations[:, 0].max()
    )

    print(
        "Y:",
        translations[:, 1].min(),
        "->",
        translations[:, 1].max()
    )

    print(
        "Z:",
        translations[:, 2].min(),
        "->",
        translations[:, 2].max()
    )

    print(
        "\nRotation determinant:"
    )

    print(
        "first:",
        determinants[0]
    )

    print(
        "min:",
        determinants.min()
    )

    print(
        "max:",
        determinants.max()
    )


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

    original_shape = (
        points.shape
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
        original_shape
    )


# ============================================================
# SAVE RGB PLY
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

    print(
        "\nWriting:"
    )

    print(
        path
    )

    print(
        "Points:",
        len(points)
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

            f"element vertex "
            f"{len(points)}\n"

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
# SAVE SEMANTIC PLY
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
        np.isfinite(
            points
        ).all(
            axis=1
        )
    )

    points = points[
        valid
    ]

    rgb_colors = rgb_colors[
        valid
    ]

    labels = labels[
        valid
    ]

    semantic_colors = semantic_colors[
        valid
    ]

    path.parent.mkdir(

        parents=True,

        exist_ok=True

    )

    print(
        "\nWriting semantic:"
    )

    print(
        path
    )

    print(
        "Points:",
        len(points)
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

            f"element vertex "
            f"{len(points)}\n"

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
            "property int label\n"
        )

        f.write(
            "property uchar semantic_red\n"
        )

        f.write(
            "property uchar semantic_green\n"
        )

        f.write(
            "property uchar semantic_blue\n"
        )

        f.write(
            "end_header\n"
        )

        for (
            p,
            rgb,
            label,
            sem
        ) in zip(

            points,
            rgb_colors,
            labels,
            semantic_colors

        ):

            f.write(

                f"{p[0]:.6f} "
                f"{p[1]:.6f} "
                f"{p[2]:.6f} "
                f"{int(rgb[0])} "
                f"{int(rgb[1])} "
                f"{int(rgb[2])} "
                f"{int(label)} "
                f"{int(sem[0])} "
                f"{int(sem[1])} "
                f"{int(sem[2])}\n"

            )


# ============================================================
# LOAD RF-DETR
# ============================================================

def load_rfdetr_model():

    print(
        "\n"
        + "=" * 80
    )

    print(
        "LOADING RF-DETR"
    )

    print(
        "=" * 80
    )

    if not RFDETR_MODEL_PATH.exists():

        raise FileNotFoundError(

            f"RF-DETR checkpoint not found:\n"
            f"{RFDETR_MODEL_PATH}"

        )

    model = (

        RFDETRSegSmall
        .from_checkpoint(

            str(
                RFDETR_MODEL_PATH
            )

        )

    )

    try:

        model.model.to(
            device
        )

    except AttributeError:

        pass

    try:

        model.optimize_for_inference()

    except Exception as exc:

        print(

            "RF-DETR optimisation unavailable:",
            exc

        )

    print(
        "RF-DETR loaded."
    )

    return model


# ============================================================
# NUMPY
# ============================================================

def as_numpy(
    value
):

    if torch.is_tensor(
        value
    ):

        return (

            value
            .detach()
            .cpu()
            .numpy()

        )

    return np.asarray(
        value
    )


# ============================================================
# RESIZE INSTANCE MASKS
# ============================================================

def resize_instance_masks(
    masks,
    output_size
):

    W_out, H_out = (
        output_size
    )

    masks = as_numpy(
        masks
    )

    if masks.ndim == 2:

        masks = masks[
            None
        ]

    if masks.ndim != 3:

        raise ValueError(

            "Unexpected RF-DETR mask shape: "
            f"{masks.shape}"

        )

    masks_t = torch.from_numpy(

        masks.astype(
            np.float32
        )

    ).unsqueeze(
        1
    )

    masks_t = torch.nn.functional.interpolate(

        masks_t,

        size=(
            H_out,
            W_out
        ),

        mode="bilinear",

        align_corners=False

    ).squeeze(
        1
    )

    return masks_t.numpy()


# ============================================================
# SEGMENT IMAGE
# ============================================================

def segment_image(
    image,
    model
):

    H, W = (
        image.shape[:2]
    )

    labels = np.full(

        (
            H,
            W
        ),

        -1,

        dtype=np.int32

    )

    confidence = np.zeros(

        (
            H,
            W
        ),

        dtype=np.float32

    )

    detections = model.predict(

        image,

        threshold=
        RFDETR_CONFIDENCE_THRESHOLD

    )

    masks = getattr(

        detections,

        "mask",

        None

    )

    if masks is None:

        return (
            labels,
            confidence
        )

    masks = resize_instance_masks(

        masks,

        (
            W,
            H
        )

    )

    class_ids = as_numpy(

        detections.class_id

    ).astype(

        np.int32

    ).reshape(
        -1
    )

    confidences = as_numpy(

        detections.confidence

    ).astype(

        np.float32

    ).reshape(
        -1
    )

    n = min(

        masks.shape[0],

        len(class_ids),

        len(confidences)

    )

    masks = masks[
        :n
    ]

    class_ids = class_ids[
        :n
    ]

    confidences = confidences[
        :n
    ]

    if n == 0:

        return (
            labels,
            confidence
        )

    binary = (

        masks >=
        RFDETR_MASK_THRESHOLD

    )

    score_maps = np.where(

        binary,

        confidences[
            :,
            None,
            None
        ],

        -1.0

    )

    best = np.argmax(

        score_maps,

        axis=0

    )

    best_score = np.max(

        score_maps,

        axis=0

    )

    valid = (

        best_score >=
        RFDETR_CONFIDENCE_THRESHOLD

    )

    labels[
        valid
    ] = class_ids[
        best[
            valid
        ]
    ]

    confidence[
        valid
    ] = best_score[
        valid
    ]

    return (
        labels,
        confidence
    )


# ============================================================
# SAVE OVERLAY
# ============================================================

def save_overlay(
    path,
    image,
    labels
):

    image = np.asarray(

        image,

        dtype=np.uint8

    )

    result = image.astype(
        np.float32
    ).copy()

    for class_id, rgb in (

        CLASS_COLORS.items()

    ):

        mask = (
            labels ==
            class_id
        )

        if not np.any(
            mask
        ):

            continue

        rgb = np.asarray(

            rgb,

            dtype=np.float32

        )

        result[
            mask
        ] = (

            (
                1.0 -
                SEMANTIC_OVERLAY_ALPHA
            )
            *
            result[
                mask
            ]
            +
            SEMANTIC_OVERLAY_ALPHA
            *
            rgb

        )

    result = np.clip(

        result,

        0,

        255

    ).astype(
        np.uint8
    )

    path.parent.mkdir(

        parents=True,

        exist_ok=True

    )

    Image.fromarray(
        result
    ).save(
        path
    )


# ============================================================
# DETERMINE BLOCKS
# ============================================================

def make_blocks(
    num_frames
):

    blocks = []

    start = 0

    block_number = 0

    while start < num_frames:

        end = min(

            start +
            BLOCK_SIZE,

            num_frames

        )

        blocks.append({

            "number":
                block_number,

            "start":
                start,

            "end":
                end,

        })

        block_number += 1

        if end >= num_frames:

            break

        start += BLOCK_STEP

    return blocks


# ============================================================
# PLY HEADER INFORMATION
# ============================================================

def get_ply_vertex_count(
    path
):

    with open(
        path,
        "r"
    ) as f:

        for line in f:

            line = line.strip()

            if line.startswith(
                "element vertex"
            ):

                return int(

                    line.split()[
                        2
                    ]

                )

            if line == "end_header":

                break

    raise RuntimeError(

        f"Could not find vertex count in:\n{path}"

    )


# ============================================================
# FIND END OF PLY HEADER
# ============================================================

def read_ply_header(
    path
):

    header = []

    with open(
        path,
        "r"
    ) as f:

        while True:

            line = f.readline()

            if not line:

                raise RuntimeError(

                    f"Unexpected EOF in PLY:\n{path}"

                )

            header.append(
                line
            )

            if line.strip() == "end_header":

                break

    return header


# ============================================================
# MERGE RGB PLY FILES
# ============================================================
#
# IMPORTANT:
#
# This does NOT alter the geometry.
#
# It simply creates one PLY containing the vertices from
# every block PLY.
#
# ============================================================

def merge_rgb_plys(
    block_files,
    output_path
):

    print(
        "\n"
        + "=" * 80
    )

    print(
        "MERGING RGB BLOCKS"
    )

    print(
        "=" * 80
    )

    block_files = [

        p for p in block_files
        if p.exists()

    ]

    if not block_files:

        print(
            "No RGB block files found."
        )

        return None

    total_points = 0

    counts = []

    for path in block_files:

        count = get_ply_vertex_count(
            path
        )

        counts.append(
            count
        )

        total_points += count

        print(

            f"{path.name}: "
            f"{count:,} points"

        )

    print(
        "\nTotal:",
        f"{total_points:,}",
        "points"
    )

    header = read_ply_header(
        block_files[0]
    )

    new_header = []

    for line in header:

        if line.startswith(
            "element vertex"
        ):

            new_header.append(

                f"element vertex "
                f"{total_points}\n"

            )

        else:

            new_header.append(
                line
            )

    output_path.parent.mkdir(

        parents=True,

        exist_ok=True

    )

    with open(
        output_path,
        "w"
    ) as out:

        for line in new_header:

            out.write(
                line
            )

        for block_number, path in enumerate(

            block_files

        ):

            print(

                f"Merging "
                f"{block_number + 1}/"
                f"{len(block_files)}: "
                f"{path.name}"

            )

            with open(
                path,
                "r"
            ) as f:

                header_done = False

                for line in f:

                    if not header_done:

                        if line.strip() == "end_header":

                            header_done = True

                        continue

                    out.write(
                        line
                    )

    print(
        "\nMerged RGB:"
    )

    print(
        output_path
    )

    print(
        "Points:",
        f"{total_points:,}"
    )

    return output_path


# ============================================================
# MERGE SEMANTIC PLY FILES
# ============================================================

def merge_semantic_plys(
    block_files,
    output_path
):

    print(
        "\n"
        + "=" * 80
    )

    print(
        "MERGING SEMANTIC BLOCKS"
    )

    print(
        "=" * 80
    )

    block_files = [

        p for p in block_files
        if p.exists()

    ]

    if not block_files:

        print(
            "No semantic block files found."
        )

        return None

    total_points = 0

    for path in block_files:

        count = get_ply_vertex_count(
            path
        )

        total_points += count

        print(

            f"{path.name}: "
            f"{count:,} points"

        )

    print(
        "\nTotal:",
        f"{total_points:,}",
        "semantic points"
    )

    header = read_ply_header(
        block_files[0]
    )

    new_header = []

    for line in header:

        if line.startswith(
            "element vertex"
        ):

            new_header.append(

                f"element vertex "
                f"{total_points}\n"

            )

        else:

            new_header.append(
                line
            )

    output_path.parent.mkdir(

        parents=True,

        exist_ok=True

    )

    with open(
        output_path,
        "w"
    ) as out:

        for line in new_header:

            out.write(
                line
            )

        for block_number, path in enumerate(

            block_files

        ):

            print(

                f"Merging semantic "
                f"{block_number + 1}/"
                f"{len(block_files)}: "
                f"{path.name}"

            )

            with open(
                path,
                "r"
            ) as f:

                header_done = False

                for line in f:

                    if not header_done:

                        if line.strip() == "end_header":

                            header_done = True

                        continue

                    out.write(
                        line
                    )

    print(
        "\nMerged semantic:"
    )

    print(
        output_path
    )

    print(
        "Points:",
        f"{total_points:,}"
    )

    return output_path


# ============================================================
# PROCESS ONE BLOCK
# ============================================================

def process_block(
    block,
    original_records,
    scal3r_input_images,
    scal3r_poses,
    K_rect,
    mapanything,
    rfdetr,
    total_blocks
):

    block_number = block[
        "number"
    ]

    block_start = block[
        "start"
    ]

    block_end = block[
        "end"
    ]

    block_size_actual = (

        block_end -
        block_start

    )

    rgb_output = (

        RAW_DIR /

        f"block_{block_number:03d}.ply"

    )

    semantic_output = (

        SEMANTIC_DIR /

        f"block_{block_number:03d}.ply"

    )


    # ========================================================
    # RESUME
    # ========================================================

    if (

        RESUME_EXISTING_BLOCKS

        and

        rgb_output.exists()

    ):

        print(
            "\n"
            + "=" * 80
        )

        print(

            f"BLOCK "
            f"{block_number:03d} "
            f"ALREADY EXISTS — SKIPPING"

        )

        print(
            rgb_output
        )

        print(
            "=" * 80
        )

        return {

            "number":
                block_number,

            "start":
                block_start,

            "end":
                block_end,

            "rgb":
                rgb_output,

            "semantic":
                semantic_output
                if semantic_output.exists()
                else None,

            "skipped":
                True,

        }


    print(
        "\n\n"
        + "#" * 80
    )

    print(

        f"PROCESSING BLOCK "
        f"{block_number:03d} / "
        f"{total_blocks - 1:03d}"

    )

    print(
        "#" * 80
    )

    original_start = (

        SCAL3R_ORIGINAL_START_INDEX
        +
        block_start

    )

    original_end = (

        SCAL3R_ORIGINAL_START_INDEX
        +
        block_end
        -
        1

    )

    print(
        "\nScal3R frames:",
        f"{block_start} -> {block_end - 1}"
    )

    print(
        "Original CAM2:",
        f"{original_start} -> {original_end}"
    )

    print(
        "Number of input views:",
        block_size_actual
    )


    # ========================================================
    # REPRESENTATIVE INDICES
    # ========================================================
    #
    # Normal blocks:
    #
    #     [29, 30]
    #
    # For the final partial block, if it has fewer than 31
    # frames, automatically choose its centre.
    #
    # ========================================================

    if block_size_actual >= 31:

        representative_offsets = (

            REPRESENTATIVE_OFFSETS.copy()

        )

    else:

        centre = (

            block_size_actual // 2

        )

        representative_offsets = [
            centre
        ]

    representative_indices = [

        block_start + offset

        for offset in representative_offsets

    ]

    print(
        "\nRepresentative offsets:",
        representative_offsets
    )

    print(
        "Representative Scal3R frames:",
        representative_indices
    )


    # ========================================================
    # BUILD MAPANYTHING VIEWS
    # ========================================================

    views = []

    block_images = []

    block_original_indices = []

    block_original_paths = []

    block_scal3r_paths = []

    block_timestamps = []


    for scal3r_index in range(

        block_start,
        block_end

    ):

        original_index = (

            SCAL3R_ORIGINAL_START_INDEX
            +
            scal3r_index

        )

        timestamp, original_path = (

            original_records[
                original_index
            ]

        )

        scal3r_path = (

            scal3r_input_images[
                scal3r_index
            ]

        )

        image = np.asarray(

            Image.open(
                scal3r_path
            ).convert(
                "RGB"
            )

        )

        views.append({

            "img":
                image,

            "intrinsics":
                K_rect.astype(
                    np.float32
                ),

        })

        block_images.append(
            image
        )

        block_original_indices.append(
            original_index
        )

        block_original_paths.append(
            original_path
        )

        block_scal3r_paths.append(
            scal3r_path
        )

        block_timestamps.append(
            timestamp
        )


    print(
        "\nFirst image:",
        block_scal3r_paths[0].name
    )

    print(
        "Last image:",
        block_scal3r_paths[-1].name
    )


    # ========================================================
    # PREPROCESS
    # ========================================================

    preprocess_start = (
        time.perf_counter()
    )

    processed_views = (
        preprocess_inputs(
            views
        )
    )

    preprocess_time = (

        time.perf_counter()
        -
        preprocess_start

    )

    print(

        f"\nPreprocessing: "
        f"{preprocess_time:.3f} s"

    )


    # ========================================================
    # ONE MAPANYTHING INFERENCE
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        f"MAPANYTHING — BLOCK "
        f"{block_number:03d}"
    )

    print(
        f"Input views: "
        f"{block_size_actual}"
    )

    print(
        "MapAnything calls for this block: 1"
    )

    print(
        "=" * 80
    )

    inference_start = (
        time.perf_counter()
    )

    with torch.inference_mode():

        predictions = (

            mapanything.infer(

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

        )

    if device.type == "cuda":

        torch.cuda.synchronize()

    inference_time = (

        time.perf_counter()
        -
        inference_start

    )

    print(

        f"\nMapAnything inference: "
        f"{inference_time:.3f} s"

    )

    print(
        "Predictions:",
        len(predictions)
    )


    # ========================================================
    # BLOCK ACCUMULATORS
    # ========================================================

    block_global_points = []

    block_global_colors = []

    block_semantic_points = []

    block_semantic_rgb = []

    block_semantic_labels = []

    block_semantic_colors = []


    # ========================================================
    # REPRESENTATIVE VIEWS ONLY
    # ========================================================

    for representative_number, scal3r_index in enumerate(

        representative_indices

    ):

        block_offset = (

            scal3r_index -
            block_start

        )

        image = (

            block_images[
                block_offset
            ]

        )

        original_index = (

            block_original_indices[
                block_offset
            ]

        )

        scal3r_path = (

            block_scal3r_paths[
                block_offset
            ]

        )

        original_path = (

            block_original_paths[
                block_offset
            ]

        )

        timestamp = (

            block_timestamps[
                block_offset
            ]

        )


        print(
            "\n"
            + "-" * 80
        )

        print(

            f"REPRESENTATIVE "
            f"{representative_number + 1}/"
            f"{len(representative_indices)}"

        )

        print(
            "Block offset:",
            block_offset
        )

        print(
            "Scal3R frame:",
            scal3r_index
        )

        print(
            "Original CAM2:",
            original_index
        )

        print(
            "Image:",
            scal3r_path.name
        )


        # ====================================================
        # MAPANYTHING OUTPUT
        # ====================================================

        prediction = (

            predictions[
                block_offset
            ]

        )

        points_grid = (

            prediction[
                "pts3d_cam"
            ]
            .detach()
            .cpu()
            .numpy()

        )

        if points_grid.ndim == 4:

            points_grid = (
                points_grid[0]
            )

        H_map, W_map = (
            points_grid.shape[:2]
        )

        print(
            "MapAnything grid:",
            points_grid.shape
        )


        # ====================================================
        # MASK
        # ====================================================

        map_mask = (

            prediction[
                "mask"
            ]
            .detach()
            .cpu()
            .numpy()

        )

        if map_mask.ndim == 4:

            map_mask = (
                map_mask[0]
            )

        if map_mask.ndim == 3:

            map_mask = (
                map_mask[..., 0]
            )

        map_mask = (
            map_mask.astype(
                bool
            )
        )


        # ====================================================
        # IMAGE TO MAP GRID
        # ====================================================

        image_grid = np.asarray(

            Image.fromarray(
                image
            ).resize(

                (
                    W_map,
                    H_map
                ),

                Image.Resampling.BILINEAR

            )

        )


        # ====================================================
        # VALID POINTS
        # ====================================================

        valid = (

            map_mask

            &

            np.isfinite(
                points_grid
            ).all(
                axis=-1
            )

        )

        local_points = (

            points_grid[
                valid
            ]

        )

        local_colors = (

            image_grid[
                valid
            ]

        )

        print(
            "Valid MapAnything points:",
            len(local_points)
        )


        # ====================================================
        # RF-DETR
        # ====================================================

        segmentation_start = (
            time.perf_counter()
        )

        labels_image, confidence_image = (

            segment_image(

                image,

                rfdetr

            )

        )

        if device.type == "cuda":

            torch.cuda.synchronize()

        segmentation_time = (

            time.perf_counter()
            -
            segmentation_start

        )

        print(

            f"RF-DETR: "
            f"{segmentation_time:.3f} s"

        )


        # ====================================================
        # SAVE OVERLAY
        # ====================================================

        if SAVE_OVERLAYS:

            overlay_path = (

                OVERLAY_DIR /

                f"block_{block_number:03d}_"
                f"frame_{scal3r_index:06d}.png"

            )

            save_overlay(

                overlay_path,

                image,

                labels_image

            )


        # ====================================================
        # LABELS TO MAP GRID
        # ====================================================

        labels_grid = np.asarray(

            Image.fromarray(

                labels_image.astype(
                    np.int32
                ),

                mode="I"

            ).resize(

                (
                    W_map,
                    H_map
                ),

                Image.Resampling.NEAREST

            ),

            dtype=np.int32

        )

        confidence_grid = np.asarray(

            Image.fromarray(

                confidence_image.astype(
                    np.float32
                ),

                mode="F"

            ).resize(

                (
                    W_map,
                    H_map
                ),

                Image.Resampling.BILINEAR

            ),

            dtype=np.float32

        )


        # ====================================================
        # SCAL3R POSE
        # ====================================================

        T_scal3r = (

            scal3r_poses[
                scal3r_index
            ]

        )

        print(
            "Scal3R translation:",
            T_scal3r[:3, 3]
        )


        # ====================================================
        # TRANSFORM RGB
        # ====================================================

        transform_start = (
            time.perf_counter()
        )

        global_points = (

            transform_points(

                local_points,

                T_scal3r

            )

        )

        transform_time = (

            time.perf_counter()
            -
            transform_start

        )

        print(

            f"Transform: "
            f"{transform_time:.5f} s"

        )


        block_global_points.append(

            global_points.astype(
                np.float32
            )

        )

        block_global_colors.append(

            local_colors.astype(
                np.uint8
            )

        )


        # ====================================================
        # SEMANTIC
        # ====================================================

        semantic_valid = (

            valid

            &

            (labels_grid >= 0)

            &

            (
                confidence_grid
                >=
                RFDETR_CONFIDENCE_THRESHOLD
            )

        )

        semantic_local_points = (

            points_grid[
                semantic_valid
            ]

        )

        semantic_rgb = (

            image_grid[
                semantic_valid
            ]

        )

        semantic_labels = (

            labels_grid[
                semantic_valid
            ]

        )


        semantic_global_points = (

            transform_points(

                semantic_local_points,

                T_scal3r

            )

        )


        # ====================================================
        # SEMANTIC COLOURS
        # ====================================================

        semantic_colours = np.zeros(

            (
                len(
                    semantic_labels
                ),
                3
            ),

            dtype=np.uint8

        )

        for class_id, colour in (

            CLASS_COLORS.items()

        ):

            class_mask = (

                semantic_labels ==
                class_id

            )

            semantic_colours[
                class_mask
            ] = colour


        if len(
            semantic_global_points
        ) > 0:

            block_semantic_points.append(

                semantic_global_points.astype(
                    np.float32
                )

            )

            block_semantic_rgb.append(

                semantic_rgb.astype(
                    np.uint8
                )

            )

            block_semantic_labels.append(

                semantic_labels.astype(
                    np.int32
                )

            )

            block_semantic_colors.append(

                semantic_colours

            )

        print(

            "Semantic points:",
            len(semantic_global_points)

        )


    # ========================================================
    # MERGE REPRESENTATIVE RGB VIEWS WITHIN BLOCK
    # ========================================================

    block_global_points = np.concatenate(

        block_global_points,

        axis=0

    )

    block_global_colors = np.concatenate(

        block_global_colors,

        axis=0

    )


    # ========================================================
    # SAVE BLOCK RGB
    # ========================================================

    save_ply(

        rgb_output,

        block_global_points,

        block_global_colors

    )


    # ========================================================
    # MERGE SEMANTIC WITHIN BLOCK
    # ========================================================

    if len(
        block_semantic_points
    ) > 0:

        block_semantic_points = np.concatenate(

            block_semantic_points,

            axis=0

        )

        block_semantic_rgb = np.concatenate(

            block_semantic_rgb,

            axis=0

        )

        block_semantic_labels = np.concatenate(

            block_semantic_labels,

            axis=0

        )

        block_semantic_colors = np.concatenate(

            block_semantic_colors,

            axis=0

        )

        save_semantic_ply(

            semantic_output,

            block_semantic_points,

            block_semantic_rgb,

            block_semantic_labels,

            block_semantic_colors

        )

    else:

        semantic_output = None

        print(
            "\nNo semantic points in this block."
        )


    # ========================================================
    # SAVE BLOCK METADATA
    # ========================================================

    poses_path = (

        OUTPUT_DIR /

        f"block_{block_number:03d}_"
        "representative_poses.npy"

    )

    np.save(

        poses_path,

        scal3r_poses[
            representative_indices
        ]

    )


    mapping_path = (

        OUTPUT_DIR /

        f"block_{block_number:03d}_"
        "representative_frames.csv"

    )

    with open(

        mapping_path,

        "w",

        newline=""

    ) as f:

        writer = csv.writer(
            f
        )

        writer.writerow([

            "block",

            "representative",

            "block_offset",

            "scal3r_index",

            "original_cam2_index",

            "timestamp",

            "original_image",

            "scal3r_input_image",

        ])

        for representative_number, scal3r_index in enumerate(

            representative_indices

        ):

            block_offset = (

                scal3r_index -
                block_start

            )

            writer.writerow([

                block_number,

                representative_number,

                block_offset,

                scal3r_index,

                block_original_indices[
                    block_offset
                ],

                block_timestamps[
                    block_offset
                ],

                block_original_paths[
                    block_offset
                ].name,

                block_scal3r_paths[
                    block_offset
                ].name,

            ])


    # ========================================================
    # SUMMARY
    # ========================================================

    summary_path = (

        OUTPUT_DIR /

        f"block_{block_number:03d}_summary.txt"

    )

    with open(

        summary_path,

        "w"

    ) as f:

        f.write(

            f"Block: "
            f"{block_number}\n"

        )

        f.write(

            f"Scal3R range: "
            f"{block_start} - "
            f"{block_end - 1}\n"

        )

        f.write(

            f"Original CAM2 range: "
            f"{original_start} - "
            f"{original_end}\n"

        )

        f.write(

            f"Input views: "
            f"{block_size_actual}\n"

        )

        f.write(

            f"Representative offsets: "
            f"{representative_offsets}\n"

        )

        f.write(

            f"Representative Scal3R frames: "
            f"{representative_indices}\n"

        )

        f.write(

            f"RGB points: "
            f"{len(block_global_points)}\n"

        )

        if semantic_output is not None:

            f.write(

                f"Semantic points: "
                f"{len(block_semantic_points)}\n"

            )

        f.write(

            f"MapAnything inference time: "
            f"{inference_time:.3f} s\n"

        )


    # ========================================================
    # CLEAN GPU
    # ========================================================

    del predictions

    del processed_views

    del views

    if device.type == "cuda":

        torch.cuda.empty_cache()

    print(
        "\n"
        + "-" * 80
    )

    print(
        f"BLOCK {block_number:03d} COMPLETE"
    )

    print(
        "RGB:",
        rgb_output
    )

    if semantic_output is not None:

        print(
            "Semantic:",
            semantic_output
        )

    print(
        "-" * 80
    )


    return {

        "number":
            block_number,

        "start":
            block_start,

        "end":
            block_end,

        "rgb":
            rgb_output,

        "semantic":
            semantic_output,

        "skipped":
            False,

    }


# ============================================================
# MAIN
# ============================================================

def main():

    total_start = (
        time.perf_counter()
    )

    print(
        "\n"
        + "#" * 80
    )

    print(
        "CAM2 MAPANYTHING + SCAL3R"
    )

    print(
        "ALL BLOCKS + FINAL MERGE"
    )

    print(
        "#" * 80
    )


    # ========================================================
    # PATH CHECK
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "PATH CHECK"
    )

    print(
        "=" * 80
    )

    required_paths = [

        IMAGE_DIR,

        CAMERA_INFO_PATH,

        SCAL3R_IMAGE_DIR,

        SCAL3R_MAT_PATH,

        RFDETR_MODEL_PATH,

    ]

    for path in required_paths:

        print(
            "\n",
            path
        )

        print(
            "  exists:",
            path.exists()
        )

        if not path.exists():

            raise FileNotFoundError(

                f"Missing required path:\n{path}"

            )


    # ========================================================
    # LOAD DATA
    # ========================================================

    original_records = (
        load_images(
            IMAGE_DIR
        )
    )

    scal3r_input_images = (

        load_scal3r_input_images(

            SCAL3R_IMAGE_DIR

        )

    )

    scal3r_poses = (

        load_scal3r_poses(

            SCAL3R_MAT_PATH

        )

    )

    (
        K_rect,
        width,
        height
    ) = load_rectified_intrinsics(

        CAMERA_INFO_PATH

    )


    # ========================================================
    # VALIDATE
    # ========================================================

    validate_poses(
        scal3r_poses
    )

    num_scal3r_frames = min(

        len(
            scal3r_input_images
        ),

        len(
            scal3r_poses
        )

    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "DATASET"
    )

    print(
        "=" * 80
    )

    print(
        "Original CAM2 images:",
        len(original_records)
    )

    print(
        "Scal3R images:",
        len(scal3r_input_images)
    )

    print(
        "Scal3R poses:",
        len(scal3r_poses)
    )

    print(
        "Usable Scal3R frames:",
        num_scal3r_frames
    )

    print(
        "Camera resolution:",
        f"{width} x {height}"
    )


    # ========================================================
    # ORIGINAL FRAME VALIDATION
    # ========================================================

    required_original_frames = (

        SCAL3R_ORIGINAL_START_INDEX
        +
        num_scal3r_frames

    )

    if required_original_frames > len(
        original_records
    ):

        raise ValueError(

            "Not enough original CAM2 images.\n"
            f"Need at least "
            f"{required_original_frames}, "
            f"but found "
            f"{len(original_records)}."

        )


    # ========================================================
    # MAKE BLOCKS
    # ========================================================

    blocks = make_blocks(

        num_scal3r_frames

    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "BLOCK PLAN"
    )

    print(
        "=" * 80
    )

    print(
        "Block size:",
        BLOCK_SIZE
    )

    print(
        "Overlap:",
        OVERLAP_SIZE
    )

    print(
        "Step:",
        BLOCK_STEP
    )

    print(
        "Number of blocks:",
        len(blocks)
    )

    print(
        "\nFirst block:"
    )

    print(
        blocks[0]
    )

    print(
        "\nLast block:"
    )

    print(
        blocks[-1]
    )


    # ========================================================
    # OUTPUT DIRS
    # ========================================================

    OUTPUT_DIR.mkdir(

        parents=True,

        exist_ok=True

    )

    RAW_DIR.mkdir(

        parents=True,

        exist_ok=True

    )

    SEMANTIC_DIR.mkdir(

        parents=True,

        exist_ok=True

    )

    OVERLAY_DIR.mkdir(

        parents=True,

        exist_ok=True

    )


    # ========================================================
    # LOAD MODELS
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "LOADING MAPANYTHING"
    )

    print(
        "=" * 80
    )

    mapanything = (

        MapAnything
        .from_pretrained(
            "facebook/map-anything"
        )
        .to(device)

    )

    mapanything.eval()

    print(
        "MapAnything loaded."
    )


    print(
        "\n"
        + "=" * 80
    )

    print(
        "LOADING RF-DETR"
    )

    print(
        "=" * 80
    )

    rfdetr = (
        load_rfdetr_model()
    )


    # ========================================================
    # PROCESS ALL BLOCKS
    # ========================================================

    completed = []

    failed = []

    all_start = (
        time.perf_counter()
    )

    for block_index, block in enumerate(

        blocks

    ):

        block_start_time = (
            time.perf_counter()
        )

        try:

            result = process_block(

                block,

                original_records,

                scal3r_input_images,

                scal3r_poses,

                K_rect,

                mapanything,

                rfdetr,

                len(blocks)

            )

            completed.append(
                result
            )

        except Exception as exc:

            print(
                "\n"
                + "!" * 80
            )

            print(

                f"BLOCK "
                f"{block['number']:03d} FAILED"

            )

            print(
                repr(exc)
            )

            print(
                "!" * 80
            )

            failed.append({

                "block":
                    block,

                "error":
                    repr(exc),

            })

            # ----------------------------------------------
            # CONTINUE TO NEXT BLOCK
            # ----------------------------------------------
            #
            # This means one bad block does not destroy the
            # entire run.
            #
            # ----------------------------------------------

            if device.type == "cuda":

                torch.cuda.empty_cache()

        block_time = (

            time.perf_counter()
            -
            block_start_time

        )

        print(

            f"\nBlock "
            f"{block['number']:03d} wall time: "
            f"{block_time / 60:.2f} min"

        )

        print(

            f"Overall progress: "
            f"{block_index + 1}/"
            f"{len(blocks)}"

        )


    all_time = (

        time.perf_counter()
        -
        all_start

    )


    # ========================================================
    # COLLECT EXISTING BLOCK FILES
    # ========================================================

    rgb_block_files = sorted(

        RAW_DIR.glob(
            "block_*.ply"
        )

    )

    semantic_block_files = sorted(

        SEMANTIC_DIR.glob(
            "block_*.ply"
        )

    )


    print(
        "\n"
        + "=" * 80
    )

    print(
        "BLOCK PROCESSING COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        "Expected blocks:",
        len(blocks)
    )

    print(
        "RGB block files:",
        len(rgb_block_files)
    )

    print(
        "Semantic block files:",
        len(semantic_block_files)
    )

    print(
        "Failed blocks:",
        len(failed)
    )


    # ========================================================
    # MERGE RGB
    # ========================================================

    merged_rgb_path = (

        OUTPUT_DIR /

        "merged_global_rgb.ply"

    )

    merged_rgb = merge_rgb_plys(

        rgb_block_files,

        merged_rgb_path

    )


    # ========================================================
    # MERGE SEMANTIC
    # ========================================================

    merged_semantic_path = (

        OUTPUT_DIR /

        "merged_global_semantic.ply"

    )

    merged_semantic = merge_semantic_plys(

        semantic_block_files,

        merged_semantic_path

    )


    # ========================================================
    # SAVE FAILED BLOCK LIST
    # ========================================================

    failed_path = (

        OUTPUT_DIR /

        "failed_blocks.txt"

    )

    with open(

        failed_path,

        "w"

    ) as f:

        if not failed:

            f.write(
                "No failed blocks.\n"
            )

        else:

            for item in failed:

                block = item[
                    "block"
                ]

                f.write(

                    f"Block "
                    f"{block['number']:03d}: "
                    f"{block['start']} -> "
                    f"{block['end'] - 1}\n"

                )

                f.write(

                    f"Error: "
                    f"{item['error']}\n\n"

                )


    # ========================================================
    # SAVE RUN SUMMARY
    # ========================================================

    summary_path = (

        OUTPUT_DIR /

        "all_blocks_summary.txt"

    )

    with open(

        summary_path,

        "w"

    ) as f:

        f.write(
            "CAM2 MapAnything + Scal3R\n"
        )

        f.write(
            "All blocks + merge\n\n"
        )

        f.write(

            f"Block size: "
            f"{BLOCK_SIZE}\n"

        )

        f.write(

            f"Overlap: "
            f"{OVERLAP_SIZE}\n"

        )

        f.write(

            f"Step: "
            f"{BLOCK_STEP}\n"

        )

        f.write(

            f"Representative offsets: "
            f"{REPRESENTATIVE_OFFSETS}\n"

        )

        f.write(

            f"Scal3R frames: "
            f"{num_scal3r_frames}\n"

        )

        f.write(

            f"Expected blocks: "
            f"{len(blocks)}\n"

        )

        f.write(

            f"RGB block files: "
            f"{len(rgb_block_files)}\n"

        )

        f.write(

            f"Semantic block files: "
            f"{len(semantic_block_files)}\n"

        )

        f.write(

            f"Failed blocks: "
            f"{len(failed)}\n"

        )

        f.write(

            f"Total block processing time: "
            f"{all_time / 60:.2f} min\n"

        )

        if merged_rgb is not None:

            f.write(

                f"Final RGB: "
                f"{merged_rgb}\n"

            )

        if merged_semantic is not None:

            f.write(

                f"Final semantic: "
                f"{merged_semantic}\n"

            )


    # ========================================================
    # GPU
    # ========================================================

    if device.type == "cuda":

        allocated = (

            torch.cuda.memory_allocated()
            /
            (1024 ** 3)

        )

        reserved = (

            torch.cuda.memory_reserved()
            /
            (1024 ** 3)

        )

        print(
            "\nGPU memory:"
        )

        print(

            "Allocated:",
            f"{allocated:.2f} GB"

        )

        print(

            "Reserved:",
            f"{reserved:.2f} GB"

        )

        torch.cuda.empty_cache()


    # ========================================================
    # FINAL
    # ========================================================

    total_time = (

        time.perf_counter()
        -
        total_start

    )

    print(
        "\n"
        + "#" * 80
    )

    print(
        "ALL BLOCKS COMPLETE"
    )

    print(
        "#" * 80
    )

    print(
        "\nBlocks expected:",
        len(blocks)
    )

    print(
        "RGB blocks:",
        len(rgb_block_files)
    )

    print(
        "Semantic blocks:",
        len(semantic_block_files)
    )

    print(
        "Failed blocks:",
        len(failed)
    )

    print(

        "\nTotal runtime:",
        f"{total_time / 60:.2f} minutes"

    )

    if merged_rgb is not None:

        print(
            "\nFINAL RGB:"
        )

        print(
            merged_rgb
        )

    if merged_semantic is not None:

        print(
            "\nFINAL SEMANTIC:"
        )

        print(
            merged_semantic
        )

    print(
        "\nSummary:"
    )

    print(
        summary_path
    )

    print(
        "\nFailed blocks:"
    )

    print(
        failed_path
    )

    print(
        "\nOutput directory:"
    )

    print(
        OUTPUT_DIR
    )

    print(
        "#" * 80
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
