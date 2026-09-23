#!/usr/bin/env python3

from pathlib import Path
import tempfile

import torch
from PIL import Image

from mapanything.models import MapAnything
from mapanything.utils.image import load_images


# ============================================================
# CONFIG
# ============================================================

IMAGE_DIR = Path(
    "/home/lcha0115/bt60_scratch/lcha_data/map-anything/"
    "personal_data/camera_lidar_gps_test_101/extracted/CAM2/images_rect"
)

OUTPUT_DIR = Path(
    "/home/lcha0115/bt60_scratch/lcha_data/map-anything/"
    "personal_data/camera_lidar_gps_test_101/mapanything_ptonly_4000_4100"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START_IMAGE = 4000
END_IMAGE = 4100

# ============================================================
# RESOLUTION
# ============================================================
# Original:
#     1920 x 1200
#
# New:
#     960 x 600
#
# Same aspect ratio, half width and half height.
#
# This reduces pixels by 4x per image.

RESIZE_WIDTH = 960
RESIZE_HEIGHT = 600

# ============================================================
# GPU / MEMORY SETTINGS
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 70)
print("MapAnything multi-view reconstruction")
print("=" * 70)

print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"GPU memory: "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )

# ============================================================
# FIND IMAGES
# ============================================================

all_images = sorted(
    [
        p for p in IMAGE_DIR.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
    ]
)

print(f"\nTotal images in directory: {len(all_images)}")

if len(all_images) < END_IMAGE:
    raise RuntimeError(
        f"Only {len(all_images)} images found, "
        f"but image {END_IMAGE} was requested."
    )

# ============================================================
# SELECT IMAGES 600 -> 1000
# ============================================================

# Python is 0-indexed:
# image 600  -> index 599
# image 1000 -> index 999

selected_images = all_images[START_IMAGE - 1:END_IMAGE]

print(
    f"Selected images: "
    f"{START_IMAGE} -> {END_IMAGE}"
)

print(f"Number of selected images: {len(selected_images)}")
print(f"First image: {selected_images[0].name}")
print(f"Last image:  {selected_images[-1].name}")

# ============================================================
# CREATE TEMPORARY RESIZED IMAGES
# ============================================================

print("\nCreating resized images...")

temp_dir = OUTPUT_DIR / "resized_images"
temp_dir.mkdir(parents=True, exist_ok=True)

resized_images = []

for i, image_path in enumerate(selected_images):

    output_path = temp_dir / image_path.name

    if not output_path.exists():

        img = Image.open(image_path).convert("RGB")

        original_size = img.size

        img = img.resize(
            (RESIZE_WIDTH, RESIZE_HEIGHT),
            Image.Resampling.BILINEAR,
        )

        img.save(output_path)

    resized_images.append(output_path)

    if (i + 1) % 50 == 0 or i == 0 or i == len(selected_images) - 1:
        print(
            f"  Resized {i + 1}/{len(selected_images)}"
        )

print("\nResize complete.")

print(
    f"Original resolution: "
    f"{original_size[0]} x {original_size[1]}"
)

print(
    f"MapAnything resolution: "
    f"{RESIZE_WIDTH} x {RESIZE_HEIGHT}"
)

# ============================================================
# LOAD MAPANYTHING
# ============================================================

print("\nLoading MapAnything...")

model = MapAnything.from_pretrained(
    "facebook/map-anything"
).to(device)

model.eval()

print("MapAnything loaded.")

# ============================================================
# LOAD RESIZED IMAGES
# ============================================================

print("\nLoading resized images...")

views = load_images(
    [str(p) for p in resized_images]
)

print(f"Loaded {len(views)} views.")

# Print actual tensor size
print(
    f"Input tensor shape: "
    f"{views[0]['img'].shape}"
)

# ============================================================
# RUN MULTI-VIEW INFERENCE
# ============================================================

print("\nRunning MapAnything inference...")
print(
    f"Reconstructing {len(views)} frames jointly."
)
print(
    f"Resolution: {RESIZE_WIDTH} x {RESIZE_HEIGHT}"
)
print(
    "Precision: FP16"
)
print()

# Clear anything left in CUDA cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()

with torch.no_grad():

    predictions = model.infer(
        views,

        # NOTE:
        # This only helps the downstream dense prediction
        # head, not the multi-view attention stage.
        memory_efficient_inference=True,

        minibatch_size=1,

        # FP16 is preferable for the T4
        use_amp=True,
        amp_dtype="fp16",

        # Mask unreliable regions
        apply_mask=True,
        mask_edges=True,
    )

# ============================================================
# SAVE RAW PREDICTIONS
# ============================================================

prediction_file = OUTPUT_DIR / "predictions.pt"

print("\nSaving predictions to:")
print(prediction_file)

torch.save(
    predictions,
    prediction_file
)

print("\nDONE.")
print("=" * 70)