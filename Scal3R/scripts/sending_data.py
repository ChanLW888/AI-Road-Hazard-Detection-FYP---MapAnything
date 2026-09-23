#!/usr/bin/env python3

from pathlib import Path
import shutil
import re


# ============================================================
# CONFIG
# ============================================================

BASE = Path(
    "/home/lcha0115/bt60_scratch/lcha_data/map-anything/"
    "personal_data/camera_lidar_test_101/extracted"
)

CAMERAS = ["CAM2"]

# 4000th image through 4150th image, inclusive.
START_IMAGE = 250
END_IMAGE = 7250

OUTPUT = (
    Path(
        "/home/lcha0115/bt60_scratch/lcha_data/Scal3R/"
        "personal_data"
    )
    / f"sequential_{START_IMAGE}_{END_IMAGE}"
)

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


# ============================================================
# HELPERS
# ============================================================

def timestamp_from_filename(path: Path) -> int:
    """
    Extract the timestamp from the filename.

    Example:
        1787879901611251758.jpg
        ->
        1787879901611251758
    """

    match = re.search(
        r"(\d+)$",
        path.stem,
    )

    if not match:
        raise ValueError(
            f"Could not extract timestamp from: {path}"
        )

    return int(match.group(1))


def get_sorted_images(folder: Path):
    """
    Find all supported images and sort them
    chronologically by their timestamp.
    """

    if not folder.exists():
        raise FileNotFoundError(
            f"Folder does not exist:\n{folder}"
        )

    files = [
        f
        for f in folder.iterdir()
        if (
            f.is_file()
            and f.suffix.lower() in IMAGE_EXTENSIONS
        )
    ]

    files.sort(
        key=timestamp_from_filename
    )

    if not files:
        raise RuntimeError(
            f"No images found in:\n{folder}"
        )

    return files


# ============================================================
# START
# ============================================================

print("=" * 70)
print("Sequential multi-camera image extraction")
print("=" * 70)

print(
    f"Base directory : {BASE}"
)

print(
    f"Image range    : "
    f"{START_IMAGE} -> {END_IMAGE}"
)

print(
    f"Cameras        : "
    f"{', '.join(CAMERAS)}"
)

print(
    f"Output         : {OUTPUT}"
)

print()


# ============================================================
# PREPARE OUTPUT
# ============================================================

OUTPUT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# PROCESS EACH CAMERA
# ============================================================

total_copied = 0

for cam in CAMERAS:

    print("-" * 70)
    print(f"Processing {cam}")
    print("-" * 70)

    source_dir = (
        BASE
        / cam
        / "images_rect"
    )

    output_dir = (
        OUTPUT
        / cam
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Get chronologically sorted images
    # --------------------------------------------------------

    images = get_sorted_images(
        source_dir
    )

    print(
        f"Found {len(images)} images"
    )

    # --------------------------------------------------------
    # Check requested range
    # --------------------------------------------------------

    if len(images) < END_IMAGE:

        raise RuntimeError(
            f"{cam} only has {len(images)} images, "
            f"but image {END_IMAGE} was requested."
        )

    # --------------------------------------------------------
    # Copy requested images
    # --------------------------------------------------------

    copied = 0

    for image_number in range(
        START_IMAGE,
        END_IMAGE + 1,
    ):

        # Human numbering:
        #
        # 4000th image -> Python index 3999
        #
        source_index = (
            image_number - 1
        )

        source = images[
            source_index
        ]

        # Keep the ORIGINAL filename.
        destination = (
            output_dir
            / source.name
        )

        shutil.copy2(
            source,
            destination,
        )

        copied += 1
        total_copied += 1

        print(
            f"{image_number:5d} -> "
            f"{source.name}"
        )

    print()
    print(
        f"{cam}: copied {copied} images"
    )

    print(
        f"Output: {output_dir}"
    )

    print()


# ============================================================
# SUMMARY
# ============================================================

expected_per_camera = (
    END_IMAGE
    - START_IMAGE
    + 1
)

expected_total = (
    expected_per_camera
    * len(CAMERAS)
)


print("=" * 70)
print("DONE")
print("=" * 70)

print(
    f"Output directory : {OUTPUT}"
)

print(
    f"Images per camera: {expected_per_camera}"
)

print(
    f"Total images     : {total_copied}"
)

print(
    f"Expected total   : {expected_total}"
)

print()

for cam in CAMERAS:

    camera_output = (
        OUTPUT / cam
    )

    count = len(
        [
            f
            for f in camera_output.iterdir()
            if (
                f.is_file()
                and f.suffix.lower()
                in IMAGE_EXTENSIONS
            )
        ]
    )

    print(
        f"{cam}: {count} images"
    )

print()

print("Final structure:")
print()

print(
    f"{OUTPUT}/"
)

for cam in CAMERAS:

    print(
        f"├── {cam}/"
    )

    print(
        f"│   ├── <original timestamp>.jpg"
    )

    print(
        f"│   └── ..."
    )