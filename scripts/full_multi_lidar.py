#!/usr/bin/env python3

"""Separate per-camera multiview MapAnything.

CAM2 and CAM6 are reconstructed independently.

For EACH camera:
  - 20 frames are passed together to MapAnything.
  - RGB-only multiview PLY is saved separately.
  - RGB + sparse LiDAR multiview PLY is saved separately.
  - LiDAR extrinsics are used ONLY to project LiDAR into that camera.
  - No CAM2 <-> CAM6 transformation.
  - No PLY merging.
  - No post-inference coordinate transforms.
  - No Poisson / mesh / voxel processing.
"""

import csv
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from PIL import Image, ImageDraw

from mapanything.models import MapAnything
from mapanything.utils.image import preprocess_inputs


# ============================================================
# CONFIG
# ============================================================
BAG_ROOT = Path(
    "/home/lcha0115/bt60_scratch/lcha_data/"
    "map-anything/personal_data/hazard_test_101"
)
EXTRACTED = BAG_ROOT / "extracted"

CAMERAS = ["CAM2", "CAM6"]

REFERENCE_CAMERA = "CAM2"
REFERENCE_INDEX = 9000
NUM_FRAMES = 20

SYNC_TOLERANCE_NS = 50_000_000

LIDAR_MAX_RANGE = 50.0
LIDAR_MIN_CAMERA_DEPTH = 0.05
BORDER_MARGIN_PIXELS = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR = BAG_ROOT / (
    f"mapanything_separate_multiview_{REFERENCE_INDEX}_"
    f"{REFERENCE_INDEX + NUM_FRAMES - 1}_"
    f"{int(LIDAR_MAX_RANGE)}m"
)


# ============================================================
# LiDAR -> CAMERA EXTRINSICS
#
# IMPORTANT:
# These are ONLY used for LiDAR projection into each camera.
# They are NOT used to transform MapAnything output.
# CAM2 and CAM6 PLYs remain completely separate.
# ============================================================
LIDAR_TO_CAMERA = {
    "CAM2": np.array([
        [0.64901045, 0.76012775, -0.03148390, -0.18778682],
        [0.09876220, -0.12521404, -0.98720184, -0.85765461],
        [-0.75434174, 0.63759489, -0.15633711, -0.93159365],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64),

    "CAM6": np.array([
        [-0.67094235, -0.74109244, -0.02486673, 0.26797529],
        [-0.11488898, 0.13702719, -0.98388214, -0.69426488],
        [0.73255504, -0.65727128, -0.17708070, -0.91874840],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64),
}


# ============================================================
# CALIBRATION / TIMESTAMPS
# ============================================================
def parse_array(value):
    value = str(value).strip()

    for wrapper in [
        "np.float64(",
        "np.float32(",
        "np.int64(",
        "np.int32(",
    ]:
        value = value.replace(wrapper, "")

    value = (
        value.replace(")", "")
        .replace("[", "")
        .replace("]", "")
        .replace(",", " ")
    )

    values = np.fromstring(value, sep=" ", dtype=np.float64)

    if values.size == 0:
        raise RuntimeError(f"Could not parse array: {value}")

    return values


def load_intrinsics(camera):
    path = EXTRACTED / camera / "camera_info.csv"

    with open(path, "r") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise RuntimeError(f"No camera info found: {path}")

    row = rows[0]
    P = parse_array(row["P"])

    if P.size != 12:
        raise RuntimeError(f"{camera}: invalid P size {P.size}")

    K = P.reshape(3, 4)[:, :3]

    return K, int(row["width"]), int(row["height"])


def load_images(camera):
    directory = EXTRACTED / camera / "images_rect"

    records = []

    for path in directory.glob("*.png"):
        try:
            timestamp = int(path.stem)
        except ValueError:
            continue

        records.append((timestamp, path))

    records.sort(key=lambda x: x[0])

    if not records:
        raise RuntimeError(f"No images found: {directory}")

    return records


def load_lidar_files():
    directory = EXTRACTED / "lidar"

    records = []

    for path in directory.glob("*.ply"):
        try:
            timestamp = int(path.stem)
        except ValueError:
            continue

        records.append((timestamp, path))

    records.sort(key=lambda x: x[0])

    if not records:
        raise RuntimeError(f"No LiDAR files found: {directory}")

    return records


def nearest_timestamp(target, records):
    timestamps = np.asarray(
        [x[0] for x in records],
        dtype=np.int64,
    )

    index = int(np.searchsorted(timestamps, target))

    candidates = []

    if index > 0:
        candidates.append(index - 1)

    if index < len(records):
        candidates.append(index)

    if not candidates:
        return None, None, None

    best = min(
        candidates,
        key=lambda i: abs(int(records[i][0]) - int(target)),
    )

    timestamp = int(records[best][0])

    return (
        timestamp,
        records[best][1],
        abs(timestamp - int(target)),
    )


# ============================================================
# LiDAR PLY
# ============================================================
def load_ascii_ply(path):
    header = []

    with open(path, "r") as f:
        while True:
            line = f.readline()

            if not line:
                raise RuntimeError(f"Unexpected EOF: {path}")

            line = line.strip()
            header.append(line)

            if line == "end_header":
                break

    vertex_count = None

    for line in header:
        parts = line.split()

        if (
            len(parts) == 3
            and parts[0] == "element"
            and parts[1] == "vertex"
        ):
            vertex_count = int(parts[2])
            break

    if vertex_count is None:
        raise RuntimeError(f"No vertex count found: {path}")

    properties = []
    in_vertex = False

    for line in header:
        parts = line.split()

        if (
            len(parts) >= 3
            and parts[0] == "element"
            and parts[1] == "vertex"
        ):
            in_vertex = True
            continue

        if in_vertex and len(parts) >= 3 and parts[0] == "property":
            properties.append(parts[-1])

        elif in_vertex and len(parts) >= 2 and parts[0] == "element":
            break

    data = np.loadtxt(
        path,
        skiprows=len(header),
        max_rows=vertex_count,
    )

    if data.ndim == 1:
        data = data.reshape(1, -1)

    return data, properties


def load_lidar_ply(path):
    data, properties = load_ascii_ply(path)

    xyz = data[
        :,
        [
            properties.index("x"),
            properties.index("y"),
            properties.index("z"),
        ],
    ].astype(np.float64)

    ranges = np.linalg.norm(xyz, axis=1)

    return xyz[ranges <= LIDAR_MAX_RANGE]


# ============================================================
# LiDAR PROJECTION
#
# Extrinsics are used ONLY here.
# ============================================================
def transform_points(points, T):
    homogeneous = np.concatenate(
        [points, np.ones((len(points), 1))],
        axis=1,
    )

    return (homogeneous @ T.T)[:, :3]


def project_lidar(points, camera, K, width, height):
    camera_points = transform_points(
        points,
        LIDAR_TO_CAMERA[camera],
    )

    X = camera_points[:, 0]
    Y = camera_points[:, 1]
    Z = camera_points[:, 2]

    valid = (
        np.isfinite(camera_points).all(axis=1)
        & (Z > LIDAR_MIN_CAMERA_DEPTH)
    )

    X = X[valid]
    Y = Y[valid]
    Z = Z[valid]

    if len(Z) == 0:
        return (
            np.empty(0, dtype=np.intp),
            np.empty(0, dtype=np.intp),
            np.empty(0, dtype=np.float64),
        )

    u = np.rint(
        K[0, 0] * X / Z + K[0, 2]
    ).astype(np.intp)

    v = np.rint(
        K[1, 1] * Y / Z + K[1, 2]
    ).astype(np.intp)

    inside = (
        (u >= BORDER_MARGIN_PIXELS)
        & (u < width - BORDER_MARGIN_PIXELS)
        & (v >= BORDER_MARGIN_PIXELS)
        & (v < height - BORDER_MARGIN_PIXELS)
    )

    return u[inside], v[inside], Z[inside]


def lidar_depth_image(
    points,
    camera,
    K,
    width,
    height,
):
    u, v, depths = project_lidar(
        points,
        camera,
        K,
        width,
        height,
    )

    depth = np.zeros(
        (height, width),
        dtype=np.float32,
    )

    if len(u) == 0:
        return depth

    pixel_ids = (
        v.astype(np.int64) * width
        + u.astype(np.int64)
    )

    # Nearest LiDAR point wins if multiple points
    # project onto the same pixel.
    order = np.lexsort((depths, pixel_ids))

    sorted_pixels = pixel_ids[order]

    keep = np.ones(
        len(order),
        dtype=bool,
    )

    if len(order) > 1:
        keep[1:] = (
            sorted_pixels[1:]
            != sorted_pixels[:-1]
        )

    selected = order[keep]

    depth[
        v[selected],
        u[selected],
    ] = depths[selected].astype(np.float32)

    return depth


def save_lidar_overlay(
    image,
    points,
    camera,
    K,
    width,
    height,
    path,
):
    u, v, depths = project_lidar(
        points,
        camera,
        K,
        width,
        height,
    )

    overlay = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(overlay)

    if len(u) == 0:
        overlay.save(path)
        return 0

    dmin = float(depths.min())
    dmax = float(depths.max())

    if dmax > dmin:
        normalized = (
            (depths - dmin)
            / (dmax - dmin)
        )
    else:
        normalized = np.zeros_like(depths)

    for px, py, d in zip(
        u,
        v,
        normalized,
    ):
        r = int(255 * d)
        b = int(255 * (1.0 - d))

        draw.rectangle(
            [
                max(0, int(px) - 1),
                max(0, int(py) - 1),
                min(width - 1, int(px) + 1),
                min(height - 1, int(py) + 1),
            ],
            fill=(r, 0, b),
        )

    overlay.save(path)

    return len(u)


# ============================================================
# LOAD ONE CAMERA'S FRAMES
# ============================================================
def load_camera_frame_data(
    camera,
    camera_records,
    reference_records,
    lidar_records,
    calibration,
):
    frame_data = []

    for frame_number in range(NUM_FRAMES):
        reference_index = (
            REFERENCE_INDEX + frame_number
        )

        reference_timestamp = int(
            reference_records[reference_index][0]
        )

        image_timestamp, image_path, image_delta = (
            nearest_timestamp(
                reference_timestamp,
                camera_records,
            )
        )

        if image_path is None:
            raise RuntimeError(
                f"Could not synchronize {camera}"
            )

        if image_delta > SYNC_TOLERANCE_NS:
            print(
                f"  WARNING {camera} frame "
                f"{frame_number:03d}: image delta "
                f"{image_delta / 1e6:.2f} ms"
            )

        lidar_timestamp, lidar_path, lidar_delta = (
            nearest_timestamp(
                reference_timestamp,
                lidar_records,
            )
        )

        if lidar_path is None:
            raise RuntimeError(
                f"Could not synchronize LiDAR "
                f"for {camera} frame {frame_number}"
            )

        if lidar_delta > SYNC_TOLERANCE_NS:
            print(
                f"  WARNING {camera} frame "
                f"{frame_number:03d}: LiDAR delta "
                f"{lidar_delta / 1e6:.2f} ms"
            )

        image = np.asarray(
            Image.open(image_path).convert("RGB")
        )

        lidar = load_lidar_ply(lidar_path)

        frame_data.append(
            {
                "image": image,
                "lidar": lidar,
                "timestamp": image_timestamp,
                "image_path": image_path,
            }
        )

    return frame_data


# ============================================================
# MAPANYTHING
# ============================================================
def build_views(
    frame_data,
    calibration,
    camera,
    use_lidar,
):
    K = calibration[camera]["K"]
    width = calibration[camera]["width"]
    height = calibration[camera]["height"]

    views = []

    for item in frame_data:
        view = {
            "img": item["image"],
            "intrinsics": K.astype(np.float32),
        }

        if use_lidar:
            depth = lidar_depth_image(
                item["lidar"],
                camera,
                K,
                width,
                height,
            )

            view["depth_z"] = (
                torch.from_numpy(depth).float()
            )

            view["is_metric_scale"] = torch.tensor(
                [True],
                device=DEVICE,
            )

        views.append(view)

    return views


def run_mapanything(
    model,
    frame_data,
    calibration,
    camera,
    use_lidar,
):
    views = build_views(
        frame_data,
        calibration,
        camera,
        use_lidar,
    )

    print(
        f"{camera}: MapAnything inputs = "
        f"{len(views)} frames"
    )

    processed = preprocess_inputs(views)

    with torch.no_grad():
        predictions = model.infer(
            processed,
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
            ignore_depth_inputs=not use_lidar,
            ignore_pose_inputs=True,
            ignore_depth_scale_inputs=not use_lidar,
            ignore_pose_scale_inputs=True,
        )

    return predictions


# ============================================================
# SAVE MAPANYTHING PLY
#
# IMPORTANT:
# No transform is applied here.
# No CAM2/CAM6 merging happens here.
# Each PLY remains in MapAnything's own output frame.
# ============================================================
def collect_points(
    predictions,
    frame_data,
):
    if len(predictions) != len(frame_data):
        raise RuntimeError(
            f"Prediction count {len(predictions)} "
            f"!= frame count {len(frame_data)}"
        )

    all_points = []
    all_colors = []

    for index, (
        prediction,
        item,
    ) in enumerate(
        zip(predictions, frame_data)
    ):
        points = (
            prediction["pts3d"]
            .detach()
            .cpu()
            .numpy()
        )

        while (
            points.ndim > 3
            and points.shape[0] == 1
        ):
            points = points[0]

        if (
            points.ndim != 3
            or points.shape[-1] != 3
        ):
            raise RuntimeError(
                f"Unexpected pts3d shape: "
                f"{points.shape}"
            )

        h, w = points.shape[:2]

        mask = prediction.get("mask")

        if mask is None:
            valid_mask = np.ones(
                (h, w),
                dtype=bool,
            )
        else:
            valid_mask = np.squeeze(
                mask.detach()
                .cpu()
                .numpy()
            ).astype(bool)

            if valid_mask.shape != (h, w):
                valid_mask = np.ones(
                    (h, w),
                    dtype=bool,
                )

        rgb = np.asarray(
            Image.fromarray(
                item["image"]
            )
            .convert("RGB")
            .resize((w, h))
        )

        valid = (
            valid_mask
            & np.isfinite(points).all(axis=-1)
        )

        valid_points = (
            points[valid]
            .astype(np.float32)
        )

        valid_colors = (
            rgb[valid]
            .astype(np.uint8)
        )

        all_points.append(valid_points)
        all_colors.append(valid_colors)

        print(
            f"  View {index + 1:03d}: "
            f"{len(valid_points):,} points"
        )

    return (
        np.concatenate(all_points),
        np.concatenate(all_colors),
    )


def save_ply(
    path,
    points,
    colors,
):
    cloud = o3d.geometry.PointCloud()

    cloud.points = (
        o3d.utility.Vector3dVector(
            points.astype(np.float64)
        )
    )

    cloud.colors = (
        o3d.utility.Vector3dVector(
            colors.astype(np.float64)
            / 255.0
        )
    )

    ok = o3d.io.write_point_cloud(
        str(path),
        cloud,
        write_ascii=False,
        compressed=False,
    )

    if not ok:
        raise RuntimeError(
            f"Failed to save: {path}"
        )

    print(
        f"Saved {len(points):,} points -> {path}"
    )


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 80)
    print("SEPARATE PER-CAMERA MULTIVIEW MAPANYTHING")
    print("=" * 80)
    print(f"Device: {DEVICE}")

    if DEVICE.type == "cuda":
        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(f"Cameras: {CAMERAS}")
    print(
        f"Frames: {REFERENCE_INDEX} - "
        f"{REFERENCE_INDEX + NUM_FRAMES - 1}"
    )
    print(
        f"LiDAR range: "
        f"{LIDAR_MAX_RANGE:.1f} m"
    )

    print()
    print("CAM2 and CAM6 are COMPLETELY SEPARATE.")
    print("No cross-camera transforms.")
    print("No PLY merging.")
    print(
        "Extrinsics are used ONLY for LiDAR "
        "projection into each camera."
    )
    print("Poisson: DISABLED")
    print("Mesh processing: DISABLED")
    print("Voxel fusion: DISABLED")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Calibration and image records
    # --------------------------------------------------------
    calibration = {}
    camera_records = {}

    for camera in CAMERAS:
        K, width, height = (
            load_intrinsics(camera)
        )

        calibration[camera] = {
            "K": K,
            "width": width,
            "height": height,
        }

        camera_records[camera] = (
            load_images(camera)
        )

        print(
            f"{camera}: "
            f"{width} x {height}, "
            f"{len(camera_records[camera]):,} images"
        )

    lidar_records = load_lidar_files()

    reference_records = camera_records[
        REFERENCE_CAMERA
    ]

    end_index = (
        REFERENCE_INDEX + NUM_FRAMES
    )

    if end_index > len(reference_records):
        raise RuntimeError(
            "Requested frame range exceeds "
            "available reference frames."
        )

    # --------------------------------------------------------
    # Load MapAnything once
    # --------------------------------------------------------
    print("\nLoading MapAnything...")

    model = (
        MapAnything
        .from_pretrained(
            "facebook/map-anything"
        )
        .to(DEVICE)
    )

    model.eval()

    print("MapAnything loaded.")

    # --------------------------------------------------------
    # Each camera is processed independently
    # --------------------------------------------------------
    for camera in CAMERAS:
        print("\n" + "=" * 80)
        print(
            f"PROCESSING {camera} "
            f"AS AN INDEPENDENT MULTIVIEW"
        )
        print("=" * 80)

        frame_data = load_camera_frame_data(
            camera,
            camera_records[camera],
            reference_records,
            lidar_records,
            calibration,
        )

        # Save first frame and LiDAR projection.
        first = frame_data[0]

        Image.fromarray(
            first["image"]
        ).save(
            OUTPUT_DIR
            / f"frame_000_{camera}.png"
        )

        count = save_lidar_overlay(
            first["image"],
            first["lidar"],
            camera,
            calibration[camera]["K"],
            calibration[camera]["width"],
            calibration[camera]["height"],
            OUTPUT_DIR
            / f"frame_000_{camera}_lidar_projected.png",
        )

        print(
            f"{camera}: "
            f"{count:,} LiDAR points projected"
        )

        # ----------------------------------------------------
        # RGB-only multiview
        # ----------------------------------------------------
        print("\n" + "-" * 80)
        print(f"{camera}: MULTIVIEW RGB ONLY")
        print("-" * 80)

        predictions = run_mapanything(
            model,
            frame_data,
            calibration,
            camera,
            use_lidar=False,
        )

        rgb_points, rgb_colors = collect_points(
            predictions,
            frame_data,
        )

        save_ply(
            OUTPUT_DIR
            / f"mapanything_{camera}_multiview_rgb.ply",
            rgb_points,
            rgb_colors,
        )

        del predictions

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

        # ----------------------------------------------------
        # RGB + LiDAR multiview
        # ----------------------------------------------------
        print("\n" + "-" * 80)
        print(f"{camera}: MULTIVIEW RGB + LiDAR")
        print("-" * 80)

        predictions = run_mapanything(
            model,
            frame_data,
            calibration,
            camera,
            use_lidar=True,
        )

        lidar_points, lidar_colors = collect_points(
            predictions,
            frame_data,
        )

        save_ply(
            OUTPUT_DIR
            / f"mapanything_{camera}_multiview_lidar.ply",
            lidar_points,
            lidar_colors,
        )

        del predictions

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

        print(
            f"\n{camera} COMPLETE."
        )
        print(
            "Its PLYs have NOT been transformed "
            "or merged with the other camera."
        )

    # --------------------------------------------------------
    # Done
    # --------------------------------------------------------
    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Output: {OUTPUT_DIR}")
    print()
    print("CAM2:")
    print("  mapanything_CAM2_multiview_rgb.ply")
    print("  mapanything_CAM2_multiview_lidar.ply")
    print()
    print("CAM6:")
    print("  mapanything_CAM6_multiview_rgb.ply")
    print("  mapanything_CAM6_multiview_lidar.ply")
    print()
    print("No CAM2/CAM6 PLY merging was performed.")
    print("No MapAnything point-cloud transform was performed.")


if __name__ == "__main__":
    main()
