#!/usr/bin/env python3

"""
Robust YOLO segmentation inference for road-hazard images.

Processes:
    CAM1/images_rect
    CAM2/images_rect
    CAM6/images_rect

Hazard classes:
    0: blocked_footpath_trip_hazard
    1: fallen_fence
    3: loose_trash
    4: mud
    5: paint
    8: skip_bin

Ignored classes:
    2: grass
    6: road
    7: sidewalk

Features:
    - Validates images before inference.
    - Skips corrupted/unreadable images instead of crashing.
    - Processes in small batches.
    - Records bad images in bad_images.txt.
    - Copies an image into every detected hazard class folder.
    - Generates hazard_detections.csv.
    - Prints progress and final statistics.

Run:
    python yolo_inference.py
"""

from pathlib import Path
import csv
import shutil
import time

from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

WEIGHTS = Path(
    "/home/lcha0115/bt60_scratch/lcha_data/map-anything/weights/best.pt"
)

DATASET_ROOT = Path(
    "/home/lcha0115/bt60_scratch/lcha_data/map-anything/"
    "personal_data/camera_lidar_test_101/extracted"
)

CAMERAS = {
    "CAM1": DATASET_ROOT / "CAM1" / "images_rect",
    "CAM2": DATASET_ROOT / "CAM2" / "images_rect",
    "CAM6": DATASET_ROOT / "CAM6" / "images_rect",
}

OUTPUT_ROOT = DATASET_ROOT / "hazard_inference"

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

# Number of images sent to YOLO at once.
# Increase if GPU memory allows.
BATCH_SIZE = 16

# First NVIDIA GPU.
DEVICE = "0"

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

HAZARD_CLASSES = {
    0: "blocked_footpath_trip_hazard",
    1: "fallen_fence",
    3: "loose_trash",
    4: "mud",
    5: "paint",
    8: "skip_bin",
}


# ============================================================
# IMAGE VALIDATION
# ============================================================

def validate_image(image_path: Path):
    """
    Check whether an image can actually be opened.

    Returns:
        True, None
        or
        False, error message
    """

    try:
        with Image.open(image_path) as img:
            # Verify the file structure.
            img.verify()

        # Open again because verify() invalidates the image object.
        with Image.open(image_path) as img:
            img.load()

        return True, None

    except (UnidentifiedImageError, OSError, ValueError) as e:
        return False, str(e)

    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def get_images(folder: Path):
    """Return supported image files recursively."""
    return sorted(
        p
        for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


# ============================================================
# OUTPUT
# ============================================================

def make_output_directories():

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for camera in CAMERAS:

        for class_name in HAZARD_CLASSES.values():

            (
                OUTPUT_ROOT
                / camera
                / class_name
            ).mkdir(
                parents=True,
                exist_ok=True,
            )


def copy_hazard_image(
    image_path: Path,
    camera: str,
    hazard_class: str,
):
    """
    Copy an image to the hazard folder.

    Handles duplicate filenames safely.
    """

    destination_dir = (
        OUTPUT_ROOT
        / camera
        / hazard_class
    )

    destination = destination_dir / image_path.name

    if not destination.exists():

        shutil.copy2(
            image_path,
            destination,
        )

        return

    # Duplicate filename protection.
    stem = image_path.stem
    suffix = image_path.suffix

    counter = 1

    while True:

        destination = (
            destination_dir
            / f"{stem}_{counter}{suffix}"
        )

        if not destination.exists():

            shutil.copy2(
                image_path,
                destination,
            )

            return

        counter += 1


# ============================================================
# MODEL
# ============================================================

def print_model_info(model):

    print("\n" + "=" * 70)
    print("MODEL")
    print("=" * 70)

    print(f"Weights : {WEIGHTS}")
    print(f"Task    : {model.task}")

    print("Classes :")

    for class_id, class_name in model.names.items():

        status = (
            "HAZARD"
            if class_id in HAZARD_CLASSES
            else "ignored"
        )

        print(
            f"  {class_id}: "
            f"{class_name:<35} "
            f"[{status}]"
        )

    print("=" * 70)


def verify_model_classes(model):

    print("\nVerifying hazard classes...")

    for class_id, expected_name in HAZARD_CLASSES.items():

        actual_name = model.names.get(class_id)

        if actual_name != expected_name:

            raise RuntimeError(
                f"Model class mismatch!\n"
                f"Class {class_id}: "
                f"expected '{expected_name}', "
                f"got '{actual_name}'"
            )

    print("All hazard classes verified.")


# ============================================================
# INFERENCE
# ============================================================

def process_batch(
    model,
    batch_paths,
    camera,
    csv_writer,
    stats,
):
    """
    Run YOLO inference on one validated batch.
    """

    results = model.predict(
        source=[str(p) for p in batch_paths],
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        device=DEVICE,
        batch=len(batch_paths),
        task="segment",
        verbose=False,
    )

    for image_path, result in zip(
        batch_paths,
        results,
    ):

        detected_hazards = {}

        if (
            result.boxes is not None
            and len(result.boxes) > 0
        ):

            class_ids = (
                result.boxes.cls
                .cpu()
                .tolist()
            )

            confidences = (
                result.boxes.conf
                .cpu()
                .tolist()
            )

            for class_id, confidence in zip(
                class_ids,
                confidences,
            ):

                class_id = int(class_id)

                if class_id not in HAZARD_CLASSES:
                    continue

                class_name = HAZARD_CLASSES[class_id]

                # Keep highest confidence for this
                # class within this image.
                if (
                    class_name not in detected_hazards
                    or confidence
                    > detected_hazards[class_name]
                ):
                    detected_hazards[class_name] = confidence

        # ----------------------------------------------------
        # Save hazard image(s)
        # ----------------------------------------------------

        for class_name, confidence in detected_hazards.items():

            class_id = next(
                cid
                for cid, name in HAZARD_CLASSES.items()
                if name == class_name
            )

            copy_hazard_image(
                image_path,
                camera,
                class_name,
            )

            stats[camera][class_name] += 1

            csv_writer.writerow([
                camera,
                str(image_path.relative_to(
                    CAMERAS[camera]
                )),
                class_name,
                class_id,
                f"{confidence:.5f}",
            ])

        yield bool(detected_hazards)


# ============================================================
# MAIN
# ============================================================

def main():

    start_time = time.time()

    print("\n" + "=" * 70)
    print("ROBUST YOLO ROAD HAZARD INFERENCE")
    print("=" * 70)

    # --------------------------------------------------------
    # Check weights
    # --------------------------------------------------------

    if not WEIGHTS.exists():

        raise FileNotFoundError(
            f"YOLO weights not found:\n{WEIGHTS}"
        )

    # --------------------------------------------------------
    # Discover images
    # --------------------------------------------------------

    print("\nChecking camera directories...")

    all_images = {}

    total_images = 0

    for camera, image_dir in CAMERAS.items():

        if not image_dir.exists():

            raise FileNotFoundError(
                f"{camera} image directory not found:\n"
                f"{image_dir}"
            )

        images = get_images(image_dir)

        all_images[camera] = images

        print(
            f"{camera}: {len(images):,} images"
        )

        total_images += len(images)

    print("-" * 70)
    print(f"TOTAL: {total_images:,} images")
    print("-" * 70)

    if total_images == 0:

        print("No images found. Exiting.")

        return

    # --------------------------------------------------------
    # Output folders
    # --------------------------------------------------------

    print("\nCreating output folders...")

    make_output_directories()

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading YOLO model...")

    model = YOLO(str(WEIGHTS))

    print_model_info(model)

    verify_model_classes(model)

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    csv_path = (
        OUTPUT_ROOT
        / "hazard_detections.csv"
    )

    bad_images_path = (
        OUTPUT_ROOT
        / "bad_images.txt"
    )

    csv_file = open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    )

    csv_writer = csv.writer(csv_file)

    csv_writer.writerow([
        "camera",
        "image",
        "hazard_class",
        "class_id",
        "confidence",
    ])

    bad_file = open(
        bad_images_path,
        "w",
        encoding="utf-8",
    )

    bad_file.write(
        "# Corrupted/unreadable images skipped during inference\n"
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    stats = {
        camera: {
            class_name: 0
            for class_name in HAZARD_CLASSES.values()
        }
        for camera in CAMERAS
    }

    images_with_hazards = {
        camera: 0
        for camera in CAMERAS
    }

    images_without_hazards = {
        camera: 0
        for camera in CAMERAS
    }

    bad_images = {
        camera: 0
        for camera in CAMERAS
    }

    processed = {
        camera: 0
        for camera in CAMERAS
    }

    # --------------------------------------------------------
    # Process cameras one at a time
    # --------------------------------------------------------

    for camera, images in all_images.items():

        print("\n" + "=" * 70)
        print(f"PROCESSING {camera}")
        print("=" * 70)

        camera_start = time.time()

        # ----------------------------------------------------
        # Validate images first
        # ----------------------------------------------------

        print(
            f"\nValidating {len(images):,} images..."
        )

        valid_images = []

        validation_start = time.time()

        for index, image_path in enumerate(images, 1):

            valid, error = validate_image(
                image_path
            )

            if valid:

                valid_images.append(image_path)

            else:

                bad_images[camera] += 1

                bad_file.write(
                    f"{camera}\t"
                    f"{image_path}\t"
                    f"{error}\n"
                )

            if (
                index % 500 == 0
                or index == len(images)
            ):

                print(
                    f"\rValidation: "
                    f"{index:,}/{len(images):,}",
                    end="",
                    flush=True,
                )

        print()

        validation_time = (
            time.time() - validation_start
        )

        print(
            f"Valid images: "
            f"{len(valid_images):,}"
        )

        print(
            f"Bad images  : "
            f"{bad_images[camera]:,}"
        )

        print(
            f"Validation time: "
            f"{validation_time / 60:.1f} min"
        )

        # ----------------------------------------------------
        # Batch inference
        # ----------------------------------------------------

        print("\nRunning inference...")

        for batch_start in range(
            0,
            len(valid_images),
            BATCH_SIZE,
        ):

            batch_paths = valid_images[
                batch_start:
                batch_start + BATCH_SIZE
            ]

            # Normally this should not fail because
            # images were already validated.
            #
            # But if a file changes/is removed between
            # validation and inference, retry each image
            # individually so one bad file cannot stop
            # the entire run.

            try:

                hazard_flags = list(
                    process_batch(
                        model,
                        batch_paths,
                        camera,
                        csv_writer,
                        stats,
                    )
                )

                for has_hazard in hazard_flags:

                    processed[camera] += 1

                    if has_hazard:
                        images_with_hazards[camera] += 1
                    else:
                        images_without_hazards[camera] += 1

            except Exception as batch_error:

                print(
                    "\nWARNING: Batch failed."
                )

                print(
                    f"Batch starting at image "
                    f"{batch_start:,}"
                )

                print(
                    f"Reason: {batch_error}"
                )

                print(
                    "Retrying images individually..."
                )

                for image_path in batch_paths:

                    try:

                        # Validate again.
                        valid, error = validate_image(
                            image_path
                        )

                        if not valid:

                            bad_images[camera] += 1

                            bad_file.write(
                                f"{camera}\t"
                                f"{image_path}\t"
                                f"{error}\n"
                            )

                            continue

                        flags = list(
                            process_batch(
                                model,
                                [image_path],
                                camera,
                                csv_writer,
                                stats,
                            )
                        )

                        has_hazard = (
                            flags[0]
                            if flags
                            else False
                        )

                        processed[camera] += 1

                        if has_hazard:
                            images_with_hazards[camera] += 1
                        else:
                            images_without_hazards[camera] += 1

                    except Exception as image_error:

                        bad_images[camera] += 1

                        bad_file.write(
                            f"{camera}\t"
                            f"{image_path}\t"
                            f"{type(image_error).__name__}: "
                            f"{image_error}\n"
                        )

                        print(
                            f"\nSkipped problematic image:"
                            f"\n  {image_path}"
                            f"\n  {image_error}"
                        )

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            completed = min(
                batch_start + len(batch_paths),
                len(valid_images),
            )

            elapsed = (
                time.time() - camera_start
            )

            rate = (
                completed / elapsed
                if elapsed > 0
                else 0
            )

            percent = (
                completed / len(valid_images) * 100
                if valid_images
                else 100
            )

            print(
                f"\rInference: "
                f"{completed:,}/{len(valid_images):,} "
                f"({percent:6.2f}%) | "
                f"{rate:6.1f} img/s | "
                f"hazard images: "
                f"{images_with_hazards[camera]:,}",
                end="",
                flush=True,
            )

            # Flush CSV periodically.
            if completed % 500 < BATCH_SIZE:
                csv_file.flush()
                bad_file.flush()

        print()

        camera_elapsed = (
            time.time() - camera_start
        )

        print(
            f"\n{camera} complete in "
            f"{camera_elapsed / 60:.1f} minutes"
        )

        print(
            f"  Processed              : "
            f"{processed[camera]:,}"
        )

        print(
            f"  Hazard images          : "
            f"{images_with_hazards[camera]:,}"
        )

        print(
            f"  No hazard              : "
            f"{images_without_hazards[camera]:,}"
        )

        print(
            f"  Bad/unreadable         : "
            f"{bad_images[camera]:,}"
        )

        print("\n  Hazard detections:")

        for class_name, count in stats[camera].items():

            print(
                f"    {class_name:<35}: "
                f"{count:,}"
            )

        csv_file.flush()
        bad_file.flush()

    # --------------------------------------------------------
    # Close files
    # --------------------------------------------------------

    csv_file.close()
    bad_file.close()

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    total_processed = sum(
        processed.values()
    )

    total_hazard_images = sum(
        images_with_hazards.values()
    )

    total_no_hazard_images = sum(
        images_without_hazards.values()
    )

    total_bad_images = sum(
        bad_images.values()
    )

    total_detections = sum(
        sum(camera_stats.values())
        for camera_stats in stats.values()
    )

    total_time = (
        time.time() - start_time
    )

    print("\n\n" + "=" * 70)
    print("INFERENCE COMPLETE")
    print("=" * 70)

    print(
        f"Images found           : "
        f"{total_images:,}"
    )

    print(
        f"Images processed       : "
        f"{total_processed:,}"
    )

    print(
        f"Images with hazards    : "
        f"{total_hazard_images:,}"
    )

    print(
        f"Images without hazards : "
        f"{total_no_hazard_images:,}"
    )

    print(
        f"Bad/unreadable images  : "
        f"{total_bad_images:,}"
    )

    print(
        f"Total hazard detections: "
        f"{total_detections:,}"
    )

    print("\nHazard counts:")

    for class_id, class_name in HAZARD_CLASSES.items():

        count = sum(
            stats[camera][class_name]
            for camera in CAMERAS
        )

        print(
            f"  {class_name:<35}: "
            f"{count:,}"
        )

    print("\nPer-camera summary:")

    for camera in CAMERAS:

        print(
            f"\n{camera}:"
        )

        print(
            f"  Found       : "
            f"{len(all_images[camera]):,}"
        )

        print(
            f"  Processed   : "
            f"{processed[camera]:,}"
        )

        print(
            f"  Hazards     : "
            f"{images_with_hazards[camera]:,}"
        )

        print(
            f"  No hazard   : "
            f"{images_without_hazards[camera]:,}"
        )

        print(
            f"  Bad images  : "
            f"{bad_images[camera]:,}"
        )

    print("\nOutput directory:")
    print(
        f"  {OUTPUT_ROOT}"
    )

    print("\nDetection CSV:")
    print(
        f"  {csv_path}"
    )

    print("\nBad images:")
    print(
        f"  {bad_images_path}"
    )

    print(
        f"\nTotal runtime: "
        f"{total_time / 60:.1f} minutes"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
