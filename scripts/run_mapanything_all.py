import os
import numpy as np
import torch

from PIL import Image

from mapanything.models import MapAnything
from mapanything.utils.image import preprocess_inputs


# ============================================================
# MAPANYTHING / A2D2
# RAW MULTIVIEW BASELINE
# ============================================================
#
# IMPORTANT:
#
# This script deliberately does NOT transform pts3d_cam.
#
# A2D2:
#     image
#       +
#     calibration K
#       +
#     generated camera pose
#       |
#       v
# MapAnything
#       |
#       v
#   pts3d_cam
#       |
#       +----------------------+
#       |                      |
#       v                      v
# individual PLY          merged PLY
#
# NO:
#   - LiDAR
#   - lidar_to_camera
#   - Tr
#   - manual rotation
#   - manual translation
#   - pose inversion
#   - coordinate conversion
#   - post-inference transformation
#
# ============================================================


# ============================================================
# 1. CONFIGURATION
# ============================================================

A2D2_ROOT = "a2d2"

CALIB_FILE = os.path.join(
    A2D2_ROOT,
    "calib.txt"
)

CAMERA_DIR = os.path.join(
    A2D2_ROOT,
    "camera_lidar",
    "20180810_150607",
    "camera",
    "cam_front_center"
)

POSE_DIR = os.path.join(
    A2D2_ROOT,
    "generated_poses"
)

OUTPUT_DIR = "outputs_debug"


# ------------------------------------------------------------
# FRAME SETTINGS
# ------------------------------------------------------------

START_FRAME = 200
STRIDE = 20
NUM_FRAMES = 5

FRAMES = [
    START_FRAME + i * STRIDE
    for i in range(NUM_FRAMES)
]


# ============================================================
# A2D2 FILE NAMING
# ============================================================

IMAGE_PREFIX = (
    "20180810150607_camera_frontcenter_"
)

IMAGE_SUFFIX = ".png"


# Generated pose files are expected to be:
#
# generated_poses/
#     000000600.txt
#     000000620.txt
#     ...
#
# The script also supports the long A2D2-style name below.

POSE_PREFIX = (
    "20180810150607_camera_frontcenter_"
)

POSE_SUFFIX = ".txt"


# ============================================================
# START
# ============================================================

print("=" * 70)
print("MAPANYTHING / A2D2 RAW MULTIVIEW BASELINE")
print("=" * 70)

print("\nFrames:")
print(FRAMES)

print("\nStart frame:")
print(START_FRAME)

print("\nStride:")
print(STRIDE)

print("\nNumber of views:")
print(NUM_FRAMES)


# ============================================================
# 2. DEVICE
# ============================================================

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("\nDevice:", device)

if device == "cuda":
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# 3. READ CALIBRATION
# ============================================================

print("\n" + "=" * 70)
print("1. CAMERA CALIBRATION")
print("=" * 70)

print(
    "Calibration file:",
    CALIB_FILE
)


if not os.path.isfile(CALIB_FILE):

    raise FileNotFoundError(
        f"\nCalibration file not found:\n"
        f"{CALIB_FILE}"
    )


def read_calib(path):

    P2 = None
    Tr = None

    with open(path, "r") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            if line.startswith("P2:"):

                values = np.fromstring(
                    line.split(":", 1)[1],
                    sep=" "
                )

                if len(values) != 12:

                    raise ValueError(
                        "P2 must contain 12 values."
                    )

                P2 = values.reshape(3, 4)


            elif line.startswith("Tr:"):

                values = np.fromstring(
                    line.split(":", 1)[1],
                    sep=" "
                )

                if len(values) != 12:

                    raise ValueError(
                        "Tr must contain 12 values."
                    )

                Tr = values.reshape(3, 4)

    return P2, Tr


P2, Tr = read_calib(
    CALIB_FILE
)


if P2 is None:

    raise RuntimeError(
        "P2 was not found in calib.txt"
    )


print("\nP2:")
print(P2)

print("\nTr:")
print(Tr)


# ------------------------------------------------------------
# Extract camera intrinsic matrix
# ------------------------------------------------------------

K_np = P2[:, :3]


print("\nK:")
print(K_np)

print("\nCamera parameters:")

print(
    "  fx =",
    K_np[0, 0]
)

print(
    "  fy =",
    K_np[1, 1]
)

print(
    "  cx =",
    K_np[0, 2]
)

print(
    "  cy =",
    K_np[1, 2]
)


print("\nIMPORTANT:")
print("Using P2[:,:3] as K.")
print("Tr is NOT used.")
print("LiDAR is NOT used.")


K = torch.from_numpy(
    K_np
).float().to(device)


# ============================================================
# 4. LOAD MAPANYTHING
# ============================================================

print("\n" + "=" * 70)
print("2. LOAD MAPANYTHING")
print("=" * 70)

print("Loading model...")


model = MapAnything.from_pretrained(
    "facebook/map-anything"
).to(device)


model.eval()


print("Model loaded.")


# ============================================================
# 5. FIND FRONT-CENTRE IMAGES
# ============================================================

print("\n" + "=" * 70)
print("3. LOAD A2D2 FRONT-CENTRE IMAGES")
print("=" * 70)

print(
    "Camera directory:",
    CAMERA_DIR
)


if not os.path.isdir(CAMERA_DIR):

    raise FileNotFoundError(
        f"\nCamera directory not found:\n"
        f"{CAMERA_DIR}"
    )


image_paths = []


for frame in FRAMES:

    # --------------------------------------------------------
    # Main filename
    # --------------------------------------------------------

    filename = (
        IMAGE_PREFIX
        + f"{frame:09d}"
        + IMAGE_SUFFIX
    )

    image_path = os.path.join(
        CAMERA_DIR,
        filename
    )


    # --------------------------------------------------------
    # Fallback filename
    # --------------------------------------------------------

    if not os.path.isfile(image_path):

        fallback = os.path.join(
            CAMERA_DIR,
            f"{frame:09d}{IMAGE_SUFFIX}"
        )

        if os.path.isfile(fallback):

            image_path = fallback

        else:

            raise FileNotFoundError(
                f"\nCould not find image for frame "
                f"{frame}.\n\n"
                f"Tried:\n"
                f"  {image_path}\n"
                f"  {fallback}"
            )


    print(
        f"\nFrame {frame}:"
    )

    print(
        " ",
        image_path
    )


    image_paths.append(
        image_path
    )


# ============================================================
# 6. LOAD IMAGES AS HWC
# ============================================================

print("\n" + "=" * 70)
print("4. IMAGE FORMAT CHECK")
print("=" * 70)

print(
    "preprocess_inputs() expects:"
)

print(
    "    (H, W, 3)"
)

print(
    "\nWe therefore load the PNGs directly"
)

print(
    "as HWC NumPy arrays."
)


views = []


for image_path in image_paths:

    image = np.array(
        Image.open(
            image_path
        ).convert("RGB")
    )


    print(
        "\nImage:",
        image_path
    )

    print(
        "  shape:",
        image.shape
    )

    print(
        "  dtype:",
        image.dtype
    )


    if image.ndim != 3:

        raise ValueError(
            f"Expected 3D image, "
            f"got {image.shape}"
        )


    if image.shape[2] != 3:

        raise ValueError(
            f"Expected RGB image "
            f"(H,W,3), "
            f"got {image.shape}"
        )


    views.append(
        {
            "img": image
        }
    )


print(
    "\nLoaded:",
    len(views),
    "images"
)


# ============================================================
# 7. LOAD GENERATED POSES
# ============================================================

print("\n" + "=" * 70)
print("5. LOAD GENERATED CAMERA POSES")
print("=" * 70)

print(
    "Pose directory:",
    POSE_DIR
)


if not os.path.isdir(POSE_DIR):

    raise FileNotFoundError(
        f"\nPose directory not found:\n"
        f"{POSE_DIR}\n\n"
        "Generate the A2D2 poses first."
    )


poses = []


for frame in FRAMES:

    # --------------------------------------------------------
    # Try short filename first
    # --------------------------------------------------------

    short_pose = os.path.join(
        POSE_DIR,
        f"{frame:09d}.txt"
    )


    # --------------------------------------------------------
    # Try long filename
    # --------------------------------------------------------

    long_pose = os.path.join(
        POSE_DIR,
        POSE_PREFIX
        + f"{frame:09d}"
        + POSE_SUFFIX
    )


    if os.path.isfile(short_pose):

        pose_path = short_pose

    elif os.path.isfile(long_pose):

        pose_path = long_pose

    else:

        raise FileNotFoundError(
            f"\nCould not find pose for frame "
            f"{frame}.\n\n"
            f"Tried:\n"
            f"  {short_pose}\n"
            f"  {long_pose}"
        )


    print(
        f"\nFrame {frame}:"
    )

    print(
        "Pose:",
        pose_path
    )


    T = np.loadtxt(
        pose_path
    )


    if T.shape != (4, 4):

        raise ValueError(
            f"Pose {pose_path} has shape "
            f"{T.shape}, expected (4,4)"
        )


    print("\nT:")
    print(T)


    R = T[:3, :3]

    t = T[:3, 3]


    print(
        "\nPosition:",
        t
    )


    print(
        "det(R):",
        np.linalg.det(R)
    )


    print(
        "orthogonality:",
        np.linalg.norm(
            R.T @ R - np.eye(3)
        )
    )


    print(
        "local X:",
        R[:, 0]
    )

    print(
        "local Y:",
        R[:, 1]
    )

    print(
        "local Z:",
        R[:, 2]
    )


    poses.append(
        T
    )


# ============================================================
# 8. CAMERA TRAJECTORY DIAGNOSTIC
# ============================================================

print("\n" + "=" * 70)
print("6. CAMERA TRAJECTORY")
print("=" * 70)


camera_positions = np.array(
    [
        T[:3, 3]
        for T in poses
    ],
    dtype=np.float64
)


for i, frame in enumerate(FRAMES):

    print(
        f"\nFrame {frame}"
    )

    print(
        "position:",
        camera_positions[i]
    )


print(
    "\nConsecutive camera motion:"
)


for i in range(
    1,
    len(FRAMES)
):

    delta = (
        camera_positions[i]
        -
        camera_positions[i - 1]
    )


    distance = np.linalg.norm(
        delta
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
        f"\n{FRAMES[i-1]} -> "
        f"{FRAMES[i]}"
    )

    print(
        "  delta:",
        delta
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
# 9. BUILD MAPANYTHING VIEWS
# ============================================================

print("\n" + "=" * 70)
print("7. BUILD MAPANYTHING VIEWS")
print("=" * 70)


for i in range(
    len(views)
):

    # --------------------------------------------------------
    # Camera intrinsics
    # --------------------------------------------------------

    views[i]["intrinsics"] = (
        K.clone()
    )


    # --------------------------------------------------------
    # Camera pose
    #
    # PASS DIRECTLY.
    #
    # No:
    #   inverse
    #   transpose
    #   rotation
    #   translation
    # --------------------------------------------------------

    views[i]["camera_poses"] = (
        torch.from_numpy(
            poses[i]
        )
        .float()
        .to(device)
    )


    # --------------------------------------------------------
    # Metric scale
    # --------------------------------------------------------

    views[i]["is_metric_scale"] = (
        torch.tensor(
            [True],
            dtype=torch.bool,
            device=device
        )
    )


    print(
        f"\nView {i}"
    )

    print(
        "Frame:",
        FRAMES[i]
    )

    print(
        "image:",
        views[i]["img"].shape
    )

    print(
        "intrinsics:",
        views[i]["intrinsics"].shape
    )

    print(
        "camera pose:",
        views[i]["camera_poses"].shape
    )


# ============================================================
# 10. PREPROCESS
# ============================================================

print("\n" + "=" * 70)
print("8. PREPROCESS INPUTS")
print("=" * 70)


print(
    "Input image shape before preprocessing:"
)


for i, view in enumerate(
    views
):

    print(
        f"  view {i}:",
        view["img"].shape,
        type(view["img"])
    )


processed_views = preprocess_inputs(
    views
)


print(
    "\nPreprocessing successful."
)


# ============================================================
# 11. MAPANYTHING INFERENCE
# ============================================================

print("\n" + "=" * 70)
print("9. MAPANYTHING MULTIVIEW INFERENCE")
print("=" * 70)


print(
    "\nNumber of views:",
    len(processed_views)
)


print(
    "\nAll selected views are passed"
)

print(
    "to ONE MapAnything inference call."
)


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

    ignore_pose_inputs=False,

    ignore_depth_scale_inputs=False,

    ignore_pose_scale_inputs=True,
)


print(
    "\nInference completed."
)


print(
    "Predictions:",
    len(predictions)
)


# ============================================================
# 12. CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 13. PLY WRITER
# ============================================================

def save_colored_ply(
    filename,
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


    if points.ndim != 2:

        raise ValueError(
            "points must be Nx3"
        )


    if points.shape[1] != 3:

        raise ValueError(
            "points must be Nx3"
        )


    if len(points) != len(colors):

        raise ValueError(
            "points and colors "
            "must have equal length"
        )


    with open(
        filename,
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
# 14. EXTRACT RAW pts3d_cam
# ============================================================

print("\n" + "=" * 70)
print("10. RAW pts3d_cam")
print("=" * 70)


all_points = []

all_colors = []


for i, pred in enumerate(
    predictions
):

    frame = FRAMES[i]


    # --------------------------------------------------------
    # RAW MapAnything output
    # --------------------------------------------------------

    pts3d_cam = (
        pred["pts3d_cam"]
        .detach()
        .cpu()
        .numpy()
    )


    print(
        f"\nFrame {frame}"
    )

    print(
        "Raw pts3d_cam shape:",
        pts3d_cam.shape
    )


    # --------------------------------------------------------
    # Remove batch dimension
    #
    # (1,H,W,3)
    # ->
    # (H,W,3)
    # --------------------------------------------------------

    if pts3d_cam.ndim == 4:

        pts3d_cam = pts3d_cam[0]


    # --------------------------------------------------------
    # Mask
    # --------------------------------------------------------

    mask = (
        pred["mask"]
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


    # --------------------------------------------------------
    # Image colours
    # --------------------------------------------------------

    img = (
        pred["img_no_norm"]
        .detach()
        .cpu()
        .numpy()
    )


    if img.ndim == 4:

        img = img[0]


    # CHW -> HWC

    if (
        img.ndim == 3
        and img.shape[0] == 3
    ):

        img = np.transpose(
            img,
            (1, 2, 0)
        )


    if img.max() <= 1.0:

        img = img * 255.0


    img = np.clip(
        img,
        0,
        255
    ).astype(
        np.uint8
    )


    # --------------------------------------------------------
    # VALID POINTS
    # --------------------------------------------------------

    valid = (
        mask
        &
        np.isfinite(
            pts3d_cam
        ).all(axis=-1)
    )


    points = pts3d_cam[
        valid
    ]

    colors = img[
        valid
    ]


    print(
        "Valid points:",
        len(points)
    )


    if len(points) > 0:

        print(
            "X:",
            points[:, 0].min(),
            "to",
            points[:, 0].max()
        )

        print(
            "Y:",
            points[:, 1].min(),
            "to",
            points[:, 1].max()
        )

        print(
            "Z:",
            points[:, 2].min(),
            "to",
            points[:, 2].max()
        )


    # ========================================================
    # SAVE RAW INDIVIDUAL PLY
    # ========================================================
    #
    # NO POSE.
    # NO ROTATION.
    # NO TRANSLATION.
    #
    # EXACTLY pts3d_cam.
    #
    # ========================================================

    individual_file = os.path.join(
        OUTPUT_DIR,
        f"frame_{frame:06d}_cam.ply"
    )


    save_colored_ply(
        individual_file,
        points,
        colors
    )


    print(
        "Saved:",
        individual_file
    )


    all_points.append(
        points
    )

    all_colors.append(
        colors
    )


# ============================================================
# 15. MERGE RAW POINT CLOUDS
# ============================================================

print("\n" + "=" * 70)
print("11. MERGE RAW POINT CLOUDS")
print("=" * 70)


merged_points = np.concatenate(
    all_points,
    axis=0
)


merged_colors = np.concatenate(
    all_colors,
    axis=0
)


print(
    "Total points:",
    len(merged_points)
)


print(
    "\nIMPORTANT:"
)

print(
    "This is ONLY concatenation."
)

print(
    "No transformation has been applied."
)

print(
    "No pose has been applied."
)

print(
    "No coordinate conversion has been applied."
)


merged_file = os.path.join(
    OUTPUT_DIR,
    "multiview_raw_cam.ply"
)


save_colored_ply(
    merged_file,
    merged_points,
    merged_colors
)


print(
    "\nSaved:",
    merged_file
)


# ============================================================
# 16. CAMERA TRAJECTORY PLY
# ============================================================

print("\n" + "=" * 70)
print("12. CAMERA TRAJECTORY PLY")
print("=" * 70)


trajectory_colors = np.zeros(
    (
        len(camera_positions),
        3
    ),
    dtype=np.uint8
)


for i in range(
    len(camera_positions)
):

    if len(camera_positions) == 1:

        value = 255

    else:

        value = int(
            255
            * i
            /
            (len(camera_positions) - 1)
        )


    trajectory_colors[i] = [
        value,
        255 - value,
        0
    ]


trajectory_file = os.path.join(
    OUTPUT_DIR,
    "camera_trajectory.ply"
)


save_colored_ply(
    trajectory_file,
    camera_positions,
    trajectory_colors
)


print(
    "Saved:",
    trajectory_file
)


# ============================================================
# 17. SAVE DEBUG DATA
# ============================================================

print("\n" + "=" * 70)
print("13. SAVE DEBUG DATA")
print("=" * 70)


debug_file = os.path.join(
    OUTPUT_DIR,
    "raw_multiview_debug.npz"
)


np.savez(
    debug_file,

    frames=np.array(
        FRAMES
    ),

    poses=np.array(
        poses
    ),

    camera_positions=np.array(
        camera_positions
    ),

    K=np.array(
        K_np
    )
)


print(
    "Saved:",
    debug_file
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("RAW MULTIVIEW TEST COMPLETE")
print("=" * 70)


print("\nFrames:")
print(FRAMES)


print(
    "\nStride:",
    STRIDE
)


print(
    "Number of views:",
    NUM_FRAMES
)


print("\nPLY FILES:")


for frame in FRAMES:

    print(
        "  ",
        os.path.join(
            OUTPUT_DIR,
            f"frame_{frame:06d}_cam.ply"
        )
    )


print(
    "  ",
    merged_file
)


print(
    "  ",
    trajectory_file
)


print("\n" + "=" * 70)

print("FINAL TRANSFORMATION STATUS")

print("=" * 70)

print(
    "pts3d_cam -> PLY: DIRECT"
)

print(
    "Pose applied to points: NO"
)

print(
    "Pose inverted: NO"
)

print(
    "Rotation applied: NO"
)

print(
    "Translation applied: NO"
)

print(
    "LiDAR used: NO"
)

print(
    "Tr used: NO"
)

print(
    "Coordinate conversion: NO"
)

print(
    "Merged PLY: RAW CONCATENATION"
)

print("=" * 70)