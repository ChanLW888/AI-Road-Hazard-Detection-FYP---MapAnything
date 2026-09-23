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

CAMERA = "CAM2"

# "raw" or "rect"
IMAGE_STREAM = "rect"

IMAGE_DIR = (
    EXTRACTED
    / CAMERA
    / (
        "images_rect"
        if IMAGE_STREAM == "rect"
        else "images_raw"
    )
)

CAMERA_INFO = (
    EXTRACTED
    / CAMERA
    / "camera_info.csv"
)

POSE_DIR = (
    EXTRACTED
    / "poses"
    / CAMERA
)

OUTPUT_DIR = (
    BAG_ROOT
    / "mapanything_output"
)


# ============================================================
# FRAME SELECTION
# ============================================================
#
# These are IMAGE INDICES.
#
# Example:
#
# START_INDEX = 0
# STRIDE = 20
# NUM_FRAMES = 5
#
# gives:
#
# 0, 20, 40, 60, 80
#
# which correspond to:
#
# 000000.txt
# 000020.txt
# 000040.txt
# 000060.txt
# 000080.txt
#
# ============================================================

START_INDEX = 200
STRIDE = 10
NUM_FRAMES = 5


# ============================================================
# MAPANYTHING SETTINGS
# ============================================================

USE_POSES = True
USE_INTRINSICS = True

USE_MULTIVIEW_CONFIDENCE = True


# ============================================================
# PRINT HEADER
# ============================================================

print("=" * 80)
print("MAPANYTHING / ROS2 BAG MULTIVIEW")
print("=" * 80)

print("Camera       :", CAMERA)
print("Image stream :", IMAGE_STREAM)
print("Image dir    :", IMAGE_DIR)
print("Camera info  :", CAMERA_INFO)
print("Pose dir     :", POSE_DIR)
print("Output       :", OUTPUT_DIR)


# ============================================================
# DEVICE
# ============================================================

device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)

print("\nDevice:", device)

if device.type == "cuda":
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# CHECK PATHS
# ============================================================

print("\n" + "=" * 80)
print("1. CHECK INPUT DATA")
print("=" * 80)

for path in [
    EXTRACTED,
    IMAGE_DIR,
    CAMERA_INFO,
    POSE_DIR,
]:

    if not path.exists():

        raise FileNotFoundError(
            f"\nRequired path does not exist:\n{path}"
        )

print("All required paths exist.")


# ============================================================
# CAMERA CALIBRATION
# ============================================================

print("\n" + "=" * 80)
print("2. CAMERA CALIBRATION")
print("=" * 80)


def parse_array(value):
    """
    Convert the strings produced by the extractor into
    numeric numpy arrays.

    Handles:

        [1, 2, 3]

        [1 2 3]

        np.float64(1.0), np.float64(2.0), ...
    """

    value = str(value).strip()

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
            "Could not parse calibration array:\n"
            + str(value)
        )

    return values


def load_calibration(path):

    with open(
        path,
        "r"
    ) as f:

        rows = list(
            csv.DictReader(f)
        )

    if not rows:

        raise RuntimeError(
            f"No calibration rows found in {path}"
        )

    row = rows[0]

    K = parse_array(
        row["K"]
    )

    if K.size != 9:

        raise ValueError(
            f"K contains {K.size} values; "
            "expected 9."
        )

    K = K.reshape(
        3,
        3
    )

    width = int(
        row["width"]
    )

    height = int(
        row["height"]
    )

    distortion_model = row.get(
        "distortion_model",
        ""
    )

    return (
        K,
        width,
        height,
        distortion_model
    )


(
    K_np,
    image_width,
    image_height,
    distortion_model
) = load_calibration(
    CAMERA_INFO
)


print("\nK =")
print(K_np)

print("\nfx:", K_np[0, 0])
print("fy:", K_np[1, 1])
print("cx:", K_np[0, 2])
print("cy:", K_np[1, 2])

print(
    "\nResolution:",
    image_width,
    "x",
    image_height
)

print(
    "Distortion:",
    distortion_model
)


# ============================================================
# FIND IMAGES
# ============================================================

print("\n" + "=" * 80)
print("3. FIND IMAGES")
print("=" * 80)


image_files = sorted(
    IMAGE_DIR.glob("*.png")
)


image_records = []


for path in image_files:

    try:

        timestamp = int(
            path.stem
        )

    except ValueError:

        continue

    image_records.append(
        (
            timestamp,
            path
        )
    )


image_records.sort(
    key=lambda x: x[0]
)


if not image_records:

    raise RuntimeError(
        f"No timestamped PNG images found in:\n"
        f"{IMAGE_DIR}"
    )


print(
    "Images found:",
    len(image_records)
)


# ============================================================
# SELECT FRAMES
# ============================================================

print("\n" + "=" * 80)
print("4. SELECT FRAMES")
print("=" * 80)


indices = [
    START_INDEX + i * STRIDE
    for i in range(NUM_FRAMES)
]


for index in indices:

    if index < 0:

        raise ValueError(
            f"Invalid negative image index: {index}"
        )

    if index >= len(image_records):

        raise IndexError(
            f"Requested image index {index}, "
            f"but only {len(image_records)} images exist."
        )


selected = [
    image_records[index]
    for index in indices
]


for view_id, (
    index,
    (timestamp, image_path)
) in enumerate(
    zip(indices, selected)
):

    print(
        f"\nView {view_id}"
    )

    print(
        "  image index :",
        index
    )

    print(
        "  timestamp   :",
        timestamp
    )

    print(
        "  image       :",
        image_path.name
    )


# ============================================================
# LOAD POSES
# ============================================================

print("\n" + "=" * 80)
print("5. LOAD CAMERA POSES")
print("=" * 80)

print(
    "Pose convention:"
)

print(
    "image index N -> pose NNNNNN.txt"
)


def load_pose(path):

    T = np.loadtxt(
        path
    )

    if T.shape != (4, 4):

        raise ValueError(
            f"\nInvalid pose:\n"
            f"{path}\n"
            f"Shape = {T.shape}\n"
            f"Expected = (4,4)"
        )

    return T


poses = []


for view_id, index in enumerate(
    indices
):

    pose_path = (
        POSE_DIR
        / f"{index:06d}.txt"
    )


    if not pose_path.exists():

        raise FileNotFoundError(
            f"\nPose missing for image index "
            f"{index}:\n"
            f"{pose_path}"
        )


    T = load_pose(
        pose_path
    )


    poses.append(
        T
    )


    timestamp = (
        selected[view_id][0]
    )


    print(
        f"\nView {view_id}"
    )

    print(
        "  image index:",
        index
    )

    print(
        "  timestamp:",
        timestamp
    )

    print(
        "  pose:",
        pose_path.name
    )

    print(
        "  position:",
        T[:3, 3]
    )


# ============================================================
# TRAJECTORY CHECK
# ============================================================

print("\n" + "=" * 80)
print("6. TRAJECTORY CHECK")
print("=" * 80)


camera_positions = np.array(
    [
        T[:3, 3]
        for T in poses
    ],
    dtype=np.float64
)


for i in range(
    1,
    len(poses)
):

    translation = (
        camera_positions[i]
        -
        camera_positions[i - 1]
    )

    distance = np.linalg.norm(
        translation
    )


    R0 = poses[i - 1][:3, :3]
    R1 = poses[i][:3, :3]

    R_relative = (
        R0.T @ R1
    )


    angle = np.arccos(
        np.clip(
            (
                np.trace(
                    R_relative
                )
                - 1.0
            )
            / 2.0,
            -1.0,
            1.0
        )
    )


    print(
        f"\n{indices[i-1]} -> {indices[i]}"
    )

    print(
        "  translation:",
        translation
    )

    print(
        "  distance:",
        distance,
        "m"
    )

    print(
        "  rotation:",
        np.degrees(angle),
        "degrees"
    )


# ============================================================
# LOAD MAPANYTHING
# ============================================================

print("\n" + "=" * 80)
print("7. LOAD MAPANYTHING")
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
print("8. BUILD MAPANYTHING VIEWS")
print("=" * 80)


views = []


for view_id, (
    timestamp,
    image_path
) in enumerate(
    selected
):

    image = np.asarray(
        Image.open(
            image_path
        ).convert(
            "RGB"
        )
    )


    if image.ndim != 3:

        raise ValueError(
            f"Image {image_path} "
            f"has shape {image.shape}"
        )


    if image.shape[2] != 3:

        raise ValueError(
            f"Image {image_path} "
            "is not RGB."
        )


    view = {
        "img": image
    }


    if USE_INTRINSICS:

        view["intrinsics"] = (
            K_np.astype(
                np.float32
            )
        )


    if USE_POSES:

        view["camera_poses"] = (
            poses[view_id]
            .astype(
                np.float32
            )
        )


    views.append(
        view
    )


    print(
        f"\nView {view_id}:"
    )

    print(
        "  image:",
        image.shape
    )

    if USE_INTRINSICS:

        print(
            "  intrinsics:",
            view["intrinsics"].shape
        )

    if USE_POSES:

        print(
            "  pose:",
            view["camera_poses"].shape
        )


# ============================================================
# PREPROCESS
# ============================================================

print("\n" + "=" * 80)
print("9. PREPROCESS")
print("=" * 80)


processed_views = preprocess_inputs(
    views
)


print(
    "Preprocessing successful."
)


# ============================================================
# INFERENCE
# ============================================================

print("\n" + "=" * 80)
print("10. MAPANYTHING INFERENCE")
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

    use_multiview_confidence=(
        USE_MULTIVIEW_CONFIDENCE
    ),

    ignore_calibration_inputs=(
        not USE_INTRINSICS
    ),

    ignore_depth_inputs=True,

    ignore_pose_inputs=(
        not USE_POSES
    ),

    ignore_depth_scale_inputs=True,

    ignore_pose_scale_inputs=True,
)


print(
    "Inference complete."
)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
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
        ).all(axis=1)
    )


    points = points[
        valid
    ]

    colors = colors[
        valid
    ]


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


        for point, color in zip(
            points,
            colors
        ):

            f.write(
                f"{point[0]:.6f} "
                f"{point[1]:.6f} "
                f"{point[2]:.6f} "
                f"{int(color[0])} "
                f"{int(color[1])} "
                f"{int(color[2])}\n"
            )


# ============================================================
# EXTRACT PTS3D_CAM
# ============================================================

print("\n" + "=" * 80)
print("11. SAVE POINT CLOUDS")
print("=" * 80)


all_points = []
all_colors = []


for view_id, prediction in enumerate(
    predictions
):

    timestamp = (
        selected[view_id][0]
    )

    image_path = (
        selected[view_id][1]
    )


    # --------------------------------------------------------
    # pts3d_cam
    # --------------------------------------------------------

    points = (
        prediction["pts3d_cam"]
        .detach()
        .cpu()
        .numpy()
    )


    # --------------------------------------------------------
    # Remove batch dimension if present
    # --------------------------------------------------------

    if points.ndim == 4:

        points = points[0]


    if points.ndim != 3:

        raise ValueError(
            f"Unexpected pts3d_cam shape: "
            f"{points.shape}"
        )


    # --------------------------------------------------------
    # Mask
    # --------------------------------------------------------

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


    if mask.shape != points.shape[:2]:

        raise ValueError(
            "\nMask / point shape mismatch:\n"
            f"points = {points.shape}\n"
            f"mask   = {mask.shape}"
        )


    # --------------------------------------------------------
    # Image colours
    # --------------------------------------------------------

    image = np.asarray(
        Image.open(
            image_path
        ).convert(
            "RGB"
        )
    )


    # MapAnything can resize internally.
    if (
        image.shape[:2]
        !=
        points.shape[:2]
    ):

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


    # --------------------------------------------------------
    # Valid points
    # --------------------------------------------------------

    valid = (
        mask
        &
        np.isfinite(
            points
        ).all(
            axis=-1
        )
    )


    points_valid = points[
        valid
    ]

    colors_valid = image[
        valid
    ]


    print(
        f"\nView {view_id}"
    )

    print(
        "  index:",
        indices[view_id]
    )

    print(
        "  timestamp:",
        timestamp
    )

    print(
        "  pts3d_cam:",
        points.shape
    )

    print(
        "  valid points:",
        len(points_valid)
    )


    # --------------------------------------------------------
    # Individual PLY
    # --------------------------------------------------------

    ply_path = (
        OUTPUT_DIR
        / f"{timestamp}_cam.ply"
    )


    save_ply(
        ply_path,
        points_valid,
        colors_valid
    )


    print(
        "  saved:",
        ply_path
    )


    all_points.append(
        points_valid
    )

    all_colors.append(
        colors_valid
    )


# ============================================================
# MERGED PLY
# ============================================================

print("\n" + "=" * 80)
print("12. MERGE POINT CLOUDS")
print("=" * 80)


merged_points = np.concatenate(
    all_points,
    axis=0
)


merged_colors = np.concatenate(
    all_colors,
    axis=0
)


merged_path = (
    OUTPUT_DIR
    / "multiview_raw_cam.ply"
)


save_ply(
    merged_path,
    merged_points,
    merged_colors
)


print(
    "Merged points:",
    len(merged_points)
)

print(
    "Saved:",
    merged_path
)


# ============================================================
# CAMERA TRAJECTORY PLY
# ============================================================

trajectory_path = (
    OUTPUT_DIR
    / "camera_trajectory.ply"
)


# Give every trajectory point a red colour.
trajectory_colors = np.tile(
    np.array(
        [255, 0, 0],
        dtype=np.uint8
    ),
    (
        len(camera_positions),
        1
    )
)


save_ply(
    trajectory_path,
    camera_positions,
    trajectory_colors
)


print(
    "Saved:",
    trajectory_path
)


# ============================================================
# DEBUG DATA
# ============================================================

debug_path = (
    OUTPUT_DIR
    / "debug_results.npz"
)


np.savez(
    debug_path,

    image_indices=np.asarray(
        indices,
        dtype=np.int64
    ),

    timestamps=np.asarray(
        [
            x[0]
            for x in selected
        ],
        dtype=np.int64
    ),

    poses=np.asarray(
        poses
    ),

    camera_positions=(
        camera_positions
    ),

    K=K_np
)


print(
    "Saved:",
    debug_path
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("COMPLETE")
print("=" * 80)

print(
    "\nCamera:",
    CAMERA
)

print(
    "Image stream:",
    IMAGE_STREAM
)

print(
    "Selected indices:",
    indices
)

print(
    "Number of views:",
    len(views)
)


print("\nPLY FILES:")

for timestamp, _ in selected:

    print(
        " ",
        OUTPUT_DIR
        / f"{timestamp}_cam.ply"
    )

print(
    " ",
    merged_path
)

print(
    " ",
    trajectory_path
)


print("\n" + "=" * 80)
print("TRANSFORMATION STATUS")
print("=" * 80)

print(
    "CameraInfo K -> MapAnything : YES"
)

print(
    "Generated camera pose -> MapAnything : YES"
)

print(
    "Pose applied to pts3d_cam : NO"
)

print(
    "Pose inverted : NO"
)

print(
    "Manual rotation : NO"
)

print(
    "Manual translation : NO"
)

print(
    "LiDAR : NO"
)

print(
    "LiDAR -> camera transform : NO"
)

print(
    "Post-inference transformation : NO"
)

print(
    "pts3d_cam -> PLY : DIRECT"
)

print(
    "Merged PLY : RAW CONCATENATION"
)

print("=" * 80)