import os
import glob
import torch
import numpy as np

from mapanything.models import MapAnything
from mapanything.utils.image import load_images, preprocess_inputs


# ============================================================
# Configuration
# ============================================================

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

IMAGE_DIR = "data_test"

USE_INTRINSICS = True
USE_DEPTH = False
USE_POSES = False

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)

if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# Load MapAnything
# ============================================================

print("Loading MapAnything model...")

model = MapAnything.from_pretrained(
    "facebook/map-anything"
).to(device)

model.eval()

print("Model loaded.")


# ============================================================
# Load images
# ============================================================

print("Loading images from:", IMAGE_DIR)

image_paths = sorted(
    glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))
    + glob.glob(os.path.join(IMAGE_DIR, "*.png"))
)

print("Found", len(image_paths), "images")

# load_images returns the normal image-only views
image_views = load_images(IMAGE_DIR)

print("Images loaded.")


# ============================================================
# Construct MapAnything input views
# ============================================================

views = []

for i, image_view in enumerate(image_views):

    view = {
        "img": image_view["img"]
    }

    # --------------------------------------------------------
    # Camera intrinsics
    # --------------------------------------------------------

    if USE_INTRINSICS:

        # TODO:
        # Replace this with the actual A2D2 intrinsic matrix
        K = torch.tensor(
            [
                [1000.0, 0.0, 960.0],
                [0.0, 1000.0, 604.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
            device=device,
        )

        view["intrinsics"] = K


    # --------------------------------------------------------
    # Depth
    # --------------------------------------------------------

    if USE_DEPTH:

        # TODO:
        # Load your depth map for this image.
        #
        # Expected shape:
        # (H, W)

        depth = np.load(
            f"depth/{i:06d}.npy"
        )

        depth = torch.from_numpy(depth).float().to(device)

        view["depth_z"] = depth

        # Tell MapAnything the depth is metric
        view["is_metric_scale"] = torch.tensor(
            [True],
            device=device
        )


    # --------------------------------------------------------
    # Camera pose
    # --------------------------------------------------------

    if USE_POSES:

        # TODO:
        # Replace with your actual A2D2 camera pose.
        #
        # Expected:
        # 4 x 4 camera-to-world matrix
        #
        # Convention:
        # +X = right
        # +Y = down
        # +Z = forward

        pose = torch.eye(
            4,
            dtype=torch.float32,
            device=device
        )

        view["camera_poses"] = pose

        view["is_metric_scale"] = torch.tensor(
            [True],
            device=device
        )


    views.append(view)


# ============================================================
# Preprocess inputs
# ============================================================

print("Preprocessing inputs...")

views = preprocess_inputs(views)

print("Inputs ready.")


# ============================================================
# Run inference
# ============================================================

print("Running inference...")

predictions = model.infer(
    views,

    memory_efficient_inference=True,

    # T4: keep this at 1
    minibatch_size=1,

    use_amp=True,
    amp_dtype="bf16",

    apply_mask=True,
    mask_edges=True,

    apply_confidence_mask=False,
    confidence_percentile=10,

    use_multiview_confidence=False,

    # Explicitly tell MapAnything which inputs to use
    ignore_calibration_inputs=not USE_INTRINSICS,
    ignore_depth_inputs=not USE_DEPTH,
    ignore_pose_inputs=not USE_POSES,
)

print("Inference complete.")


# ============================================================
# Inspect results
# ============================================================

for i, pred in enumerate(predictions):

    print(f"\n========== View {i} ==========")

    pts3d = pred["pts3d"]
    pts3d_cam = pred["pts3d_cam"]
    depth_z = pred["depth_z"]

    intrinsics = pred["intrinsics"]
    camera_poses = pred["camera_poses"]

    confidence = pred["conf"]
    mask = pred["mask"]

    metric_scaling_factor = pred[
        "metric_scaling_factor"
    ]

    print("pts3d:", pts3d.shape)
    print("pts3d_cam:", pts3d_cam.shape)
    print("depth_z:", depth_z.shape)
    print("intrinsics:", intrinsics.shape)
    print("camera_poses:", camera_poses.shape)
    print("confidence:", confidence.shape)
    print("mask:", mask.shape)
    print(
        "metric scaling:",
        metric_scaling_factor
    )