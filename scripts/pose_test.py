#!/usr/bin/env python3

from pathlib import Path
import re
import gc

import numpy as np
import torch
from PIL import Image

from mapanything.models import MapAnything
from mapanything.utils.image import load_images


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(
    "/home/lcha0115/bt60_scratch/lcha_data/map-anything/"
    "personal_data/camera_lidar_gps_test_101/extracted"
)

IMAGE_DIR = BASE_DIR / "CAM2" / "images_rect"

POSE_DIR = (
    BASE_DIR.parent
    / "corrected_poses"
    / "CAM2"
    / "poses"
)

CALIB_PATH = (
    BASE_DIR
    / "calibration"
    / "CAM2_calib.txt"
)

OUTPUT_DIR = (
    BASE_DIR.parent
    / "corrected_pose_diagnostic"
)

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# FRAMES
# ============================================================

TEST_FRAMES = [
    5500,
    5502,
    5504,
    5506,
    5508,
    5510,
]


# ============================================================
# CALIBRATION
# ============================================================

def read_calibration(path):

    text = Path(path).read_text()

    width = int(
        re.search(
            r"width:\s*(\d+)",
            text
        ).group(1)
    )

    height = int(
        re.search(
            r"height:\s*(\d+)",
            text
        ).group(1)
    )

    def read_matrix(name, rows, cols):

        pattern = (
            rf"{name}:\s*\n"
            rf"((?:[^\n]+\n?){{{rows}}})"
        )

        match = re.search(
            pattern,
            text
        )

        if match is None:
            raise RuntimeError(
                f"Could not find {name}"
            )

        values = []

        for line in (
            match.group(1)
            .strip()
            .splitlines()
        ):
            values.extend(
                float(x)
                for x in line.split()
            )

        return np.asarray(
            values,
            dtype=np.float32
        ).reshape(
            rows,
            cols
        )

    K = read_matrix("K", 3, 3)
    D = read_matrix("D", 1, 4)
    R = read_matrix("R", 3, 3)
    P = read_matrix("P", 3, 4)

    return width, height, K, D, R, P


# ============================================================
# LOAD POSE
# ============================================================

def load_pose(frame):

    path = POSE_DIR / f"{frame:06d}.txt"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing pose:\n{path}"
        )

    T = np.loadtxt(path).astype(np.float32)

    if T.shape != (4, 4):
        raise RuntimeError(
            f"Invalid pose shape {T.shape} "
            f"for {path}"
        )

    return T


# ============================================================
# EULER
# ============================================================

def rotation_to_euler(R):

    sy = np.sqrt(
        R[0, 0] ** 2 +
        R[1, 0] ** 2
    )

    if sy >= 1e-6:

        x = np.arctan2(
            R[2, 1],
            R[2, 2]
        )

        y = np.arctan2(
            -R[2, 0],
            sy
        )

        z = np.arctan2(
            R[1, 0],
            R[0, 0]
        )

    else:

        x = np.arctan2(
            -R[1, 2],
            R[1, 1]
        )

        y = np.arctan2(
            -R[2, 0],
            sy
        )

        z = 0.0

    return np.degrees(
        [x, y, z]
    )


# ============================================================
# PLY WRITER
# ============================================================

def write_ply(path, points, colors):

    points = np.asarray(
        points,
        dtype=np.float32
    )

    colors = np.asarray(
        colors,
        dtype=np.uint8
    )

    if len(points) == 0:

        print(
            f"WARNING: No points to save: {path}"
        )

        return

    if len(points) != len(colors):

        raise RuntimeError(
            f"PLY point/color mismatch: "
            f"{len(points)} points vs "
            f"{len(colors)} colors"
        )

    valid = (
        np.isfinite(points).all(axis=1)
        &
        np.isfinite(colors).all(axis=1)
    )

    points = points[valid]
    colors = colors[valid]

    data = np.empty(
        len(points),
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
    )

    data["x"] = points[:, 0]
    data["y"] = points[:, 1]
    data["z"] = points[:, 2]

    data["red"] = colors[:, 0]
    data["green"] = colors[:, 1]
    data["blue"] = colors[:, 2]

    with open(path, "wb") as f:

        f.write(
            b"ply\n"
        )

        f.write(
            b"format binary_little_endian 1.0\n"
        )

        f.write(
            f"element vertex {len(data)}\n".encode()
        )

        f.write(
            b"property float x\n"
        )

        f.write(
            b"property float y\n"
        )

        f.write(
            b"property float z\n"
        )

        f.write(
            b"property uchar red\n"
        )

        f.write(
            b"property uchar green\n"
        )

        f.write(
            b"property uchar blue\n"
        )

        f.write(
            b"end_header\n"
        )

        data.tofile(f)

    print(
        f"Saved {len(points):,} points:"
    )

    print(
        f"  {path}"
    )


# ============================================================
# EXTRACT RETURNED POSE
# ============================================================

def extract_pose(prediction):

    if "camera_poses" not in prediction:

        raise RuntimeError(
            "No camera_poses in prediction.\n"
            f"Keys: {list(prediction.keys())}"
        )

    pose = prediction["camera_poses"]

    if torch.is_tensor(pose):

        pose = (
            pose
            .detach()
            .float()
            .cpu()
            .numpy()
        )

    else:

        pose = np.asarray(
            pose,
            dtype=np.float32
        )

    if pose.ndim == 3:

        pose = pose[0]

    if pose.shape != (4, 4):

        raise RuntimeError(
            f"Unexpected returned pose shape: "
            f"{pose.shape}"
        )

    return pose.astype(
        np.float32
    )


# ============================================================
# EXTRACT PTS3D
# ============================================================

def extract_pts3d(prediction):

    if "pts3d" not in prediction:

        raise RuntimeError(
            "MapAnything prediction does not contain "
            "'pts3d'.\n\n"
            f"Available keys:\n"
            f"{list(prediction.keys())}"
        )

    pts = prediction["pts3d"]

    if torch.is_tensor(pts):

        pts = (
            pts
            .detach()
            .float()
            .cpu()
            .numpy()
        )

    else:

        pts = np.asarray(
            pts,
            dtype=np.float32
        )

    print(
        "    Raw pts3d shape:",
        pts.shape
    )

    # --------------------------------------------------------
    # [B,H,W,3]
    # --------------------------------------------------------

    if pts.ndim == 4:

        if pts.shape[0] != 1:

            raise RuntimeError(
                f"Unexpected pts3d batch shape: "
                f"{pts.shape}"
            )

        pts = pts[0]

    # --------------------------------------------------------
    # [H,W,3]
    # --------------------------------------------------------

    if pts.ndim == 3:

        if pts.shape[-1] != 3:

            raise RuntimeError(
                f"Unexpected pts3d shape: "
                f"{pts.shape}"
            )

        pts = pts.reshape(
            -1,
            3
        )

    # --------------------------------------------------------
    # [N,3]
    # --------------------------------------------------------

    elif pts.ndim == 2:

        if pts.shape[-1] != 3:

            raise RuntimeError(
                f"Unexpected pts3d shape: "
                f"{pts.shape}"
            )

    else:

        raise RuntimeError(
            f"Unsupported pts3d dimensions: "
            f"{pts.shape}"
        )

    return pts.astype(
        np.float32
    )


# ============================================================
# LOAD RGB
# ============================================================

def load_rgb_for_points(
    image_path,
    width,
    height
):

    image = Image.open(
        image_path
    ).convert("RGB")

    image = image.resize(
        (width, height),
        Image.Resampling.BILINEAR
    )

    rgb = np.asarray(
        image,
        dtype=np.uint8
    )

    return rgb.reshape(
        -1,
        3
    )


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print("=" * 80)
    print("MAPANYTHING — SUPPLIED POSES + DIRECT PTS3D")
    print("=" * 80)

    print()
    print("Device:", DEVICE)
    print("Frames:", TEST_FRAMES)


    # ========================================================
    # 1. FIND IMAGES
    # ========================================================

    print()
    print("=" * 80)
    print("1. FIND IMAGES")
    print("=" * 80)

    image_paths = sorted(
        [
            p
            for p in IMAGE_DIR.iterdir()
            if (
                p.is_file()
                and p.suffix.lower()
                in [
                    ".jpg",
                    ".jpeg",
                    ".png"
                ]
            )
        ],
        key=lambda p: p.stem
    )

    print(
        f"Found {len(image_paths):,} images"
    )

    selected_images = []

    for frame in TEST_FRAMES:

        if frame >= len(image_paths):

            raise IndexError(
                f"Frame {frame} exceeds "
                f"available images "
                f"({len(image_paths)})"
            )

        path = image_paths[frame]

        selected_images.append(
            path
        )

        print(
            f"{frame:06d} -> {path.name}"
        )


    # ========================================================
    # 2. LOAD SUPPLIED POSES
    # ========================================================

    print()
    print("=" * 80)
    print("2. LOAD SUPPLIED CORRECTED POSES")
    print("=" * 80)

    supplied_poses = []

    for frame in TEST_FRAMES:

        T = load_pose(
            frame
        )

        supplied_poses.append(
            T
        )

        print()
        print(
            f"FRAME {frame:06d}"
        )

        print(
            "Position:",
            T[:3, 3]
        )

        print(
            "Rotation:"
        )

        print(
            T[:3, :3]
        )

    supplied_poses = np.stack(
        supplied_poses
    )


    # ========================================================
    # 3. POSE MOTION
    # ========================================================

    print()
    print("=" * 80)
    print("3. SUPPLIED POSE MOTION")
    print("=" * 80)

    for i in range(
        1,
        len(TEST_FRAMES)
    ):

        T0 = supplied_poses[i - 1]
        T1 = supplied_poses[i]

        delta = (
            np.linalg.inv(T0)
            @
            T1
        )

        distance = np.linalg.norm(
            delta[:3, 3]
        )

        print(
            f"{TEST_FRAMES[i-1]:06d}"
            f" -> "
            f"{TEST_FRAMES[i]:06d}"
            f" : "
            f"{distance:.4f} m"
        )


    # ========================================================
    # 4. CALIBRATION
    # ========================================================

    print()
    print("=" * 80)
    print("4. CALIBRATION")
    print("=" * 80)

    (
        width,
        height,
        K,
        D,
        R_cal,
        P
    ) = read_calibration(
        CALIB_PATH
    )

    print(
        f"Original resolution: "
        f"{width} x {height}"
    )


    # ========================================================
    # 5. LOAD VIEWS
    # ========================================================

    print()
    print("=" * 80)
    print("5. LOAD VIEWS")
    print("=" * 80)

    views = load_images(
        [
            str(p)
            for p in selected_images
        ],
        resolution_set=518,
        norm_type="dinov2",
        patch_size=14,
    )

    for i, view in enumerate(
        views
    ):

        print()
        print(
            f"View {i}:"
        )

        print(
            "  img:",
            view["img"].shape
        )


    H = views[0]["img"].shape[-2]
    W = views[0]["img"].shape[-1]

    print()
    print(
        f"MapAnything resolution: "
        f"{W} x {H}"
    )


    # ========================================================
    # 6. SCALE INTRINSICS
    # ========================================================

    K_scaled = (
        P[:3, :3]
        .copy()
    )

    sx = W / width
    sy = H / height

    K_scaled[0, 0] *= sx
    K_scaled[0, 2] *= sx

    K_scaled[1, 1] *= sy
    K_scaled[1, 2] *= sy

    print()
    print(
        "Scaled K:"
    )

    print(
        K_scaled
    )


    # ========================================================
    # 7. ATTACH INTRINSICS + SUPPLIED POSES
    # ========================================================

    print()
    print("=" * 80)
    print("7. ATTACH SUPPLIED POSES")
    print("=" * 80)

    K_tensor = (
        torch.from_numpy(
            K_scaled
        )
        .float()
        .to(DEVICE)
    )

    for i, (
        frame,
        view,
        T
    ) in enumerate(
        zip(
            TEST_FRAMES,
            views,
            supplied_poses
        )
    ):

        # ----------------------------------------------------
        # Intrinsics
        # ----------------------------------------------------

        view["intrinsics"] = (
            K_tensor
            .clone()
            .unsqueeze(0)
        )

        # ----------------------------------------------------
        # SUPPLIED CAMERA POSE
        #
        # [4,4] -> [1,4,4]
        # ----------------------------------------------------

        pose_tensor = (
            torch.from_numpy(
                T
            )
            .float()
            .to(DEVICE)
            .unsqueeze(0)
        )

        view["camera_poses"] = (
            pose_tensor
        )

        print()
        print(
            f"Frame {frame:06d}"
        )

        print(
            "  img:",
            view["img"].shape
        )

        print(
            "  intrinsics:",
            view["intrinsics"].shape
        )

        print(
            "  camera_poses:",
            view["camera_poses"].shape
        )

        assert (
            view["img"].ndim == 4
        )

        assert (
            view["intrinsics"].shape
            ==
            (1, 3, 3)
        )

        assert (
            view["camera_poses"].shape
            ==
            (1, 4, 4)
        )


    # ========================================================
    # 8. FINAL INPUT CHECK
    # ========================================================

    print()
    print("=" * 80)
    print("8. FINAL INPUT CHECK")
    print("=" * 80)

    print()
    print(
        "img          = [1,3,H,W]"
    )

    print(
        "intrinsics   = [1,3,3]"
    )

    print(
        "camera_poses = [1,4,4]"
    )

    print()
    print(
        "Supplied poses WILL be given to MapAnything."
    )

    print(
        "pts3d WILL NOT be transformed afterwards."
    )


    # ========================================================
    # 9. LOAD MODEL
    # ========================================================

    print()
    print("=" * 80)
    print("9. LOAD MAPANYTHING")
    print("=" * 80)

    model = (
        MapAnything
        .from_pretrained(
            "facebook/map-anything"
        )
        .to(DEVICE)
    )

    model.eval()

    print(
        "Model loaded."
    )


    # ========================================================
    # 10. INFERENCE
    # ========================================================

    print()
    print("=" * 80)
    print("10. MAPANYTHING INFERENCE")
    print("=" * 80)

    print()
    print(
        "Running MapAnything..."
    )

    with torch.no_grad():

        predictions = model.infer(

            views,

            memory_efficient_inference=True,

            minibatch_size=1,

            use_amp=True,

            amp_dtype="bf16",

            apply_mask=True,

            mask_edges=True,

            apply_confidence_mask=False,

            use_multiview_confidence=False,

            ignore_calibration_inputs=False,

            ignore_depth_inputs=True,

            ignore_pose_inputs=False,
        )

    print()
    print(
        "Inference complete."
    )


    # ========================================================
    # 11. PRINT PREDICTION KEYS
    # ========================================================

    print()
    print("=" * 80)
    print("11. PREDICTION CONTENT")
    print("=" * 80)

    for i, prediction in enumerate(
        predictions
    ):

        print()
        print(
            f"FRAME {TEST_FRAMES[i]:06d}"
        )

        for key, value in prediction.items():

            if torch.is_tensor(value):

                print(
                    f"  {key:30s}"
                    f" {tuple(value.shape)}"
                    f" {value.dtype}"
                )

            else:

                print(
                    f"  {key:30s}"
                    f" {type(value)}"
                )


    # ========================================================
    # 12. DIRECT PTS3D RECONSTRUCTION
    #
    # IMPORTANT:
    #
    # THE SUPPLIED POSES ARE NOT USED HERE.
    #
    # We simply take:
    #
    #     prediction["pts3d"]
    #
    # exactly as returned by MapAnything.
    #
    # ========================================================

    print()
    print("=" * 80)
    print("12. DIRECT PTS3D RECONSTRUCTION")
    print("=" * 80)

    all_points = []
    all_colors = []

    per_frame_stats = []


    for i, (
        frame,
        image_path,
        prediction
    ) in enumerate(
        zip(
            TEST_FRAMES,
            selected_images,
            predictions
        )
    ):

        print()
        print(
            "-" * 80
        )

        print(
            f"FRAME {frame:06d}"
        )

        # ----------------------------------------------------
        # DIRECT MapAnything pts3d
        # ----------------------------------------------------

        points = extract_pts3d(
            prediction
        )

        print(
            f"Raw pts3d: "
            f"{len(points):,}"
        )

        # ----------------------------------------------------
        # Only remove NaN / Inf.
        #
        # NO transformation.
        # ----------------------------------------------------

        valid = np.isfinite(
            points
        ).all(axis=1)

        points = points[
            valid
        ]

        print(
            f"Valid pts3d: "
            f"{len(points):,}"
        )

        # ----------------------------------------------------
        # RGB
        # ----------------------------------------------------

        colors = load_rgb_for_points(
            image_path,
            W,
            H
        )

        colors = colors[
            valid
        ]

        if len(points) != len(colors):

            raise RuntimeError(
                f"Point/color mismatch "
                f"for frame {frame:06d}: "
                f"{len(points)} points vs "
                f"{len(colors)} colors"
            )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # NO pose transform.
        #
        # The points below are exactly the
        # MapAnything pts3d output.
        # ----------------------------------------------------

        print()
        print(
            "Using MapAnything pts3d DIRECTLY."
        )

        print(
            "NO supplied pose transformation."
        )

        print()
        print(
            "pts3d min:",
            np.min(
                points,
                axis=0
            )
        )

        print(
            "pts3d max:",
            np.max(
                points,
                axis=0
            )
        )

        print(
            "pts3d mean:",
            np.mean(
                points,
                axis=0
            )
        )

        # ----------------------------------------------------
        # Save individual reconstruction
        # ----------------------------------------------------

        frame_path = (
            OUTPUT_DIR
            /
            f"reconstruction_frame_{frame:06d}.ply"
        )

        write_ply(
            frame_path,
            points,
            colors
        )

        # ----------------------------------------------------
        # Add directly to merged reconstruction
        # ----------------------------------------------------

        all_points.append(
            points
        )

        all_colors.append(
            colors
        )

        per_frame_stats.append(
            [
                frame,
                len(points)
            ]
        )


    # ========================================================
    # 13. MERGED DIRECT PTS3D
    # ========================================================

    print()
    print("=" * 80)
    print("13. MERGING DIRECT PTS3D")
    print("=" * 80)

    merged_points = np.concatenate(
        all_points,
        axis=0
    )

    merged_colors = np.concatenate(
        all_colors,
        axis=0
    )

    print()
    print(
        f"Total points: "
        f"{len(merged_points):,}"
    )

    print()
    print(
        "NO POSE TRANSFORMATION HAS BEEN APPLIED."
    )

    merged_path = (
        OUTPUT_DIR
        /
        "reconstruction_pts3d_supplied_pose_input.ply"
    )

    write_ply(
        merged_path,
        merged_points,
        merged_colors
    )


    # ========================================================
    # 14. MAPANYTHING RETURNED POSES
    # ========================================================

    print()
    print("=" * 80)
    print("14. MAPANYTHING RETURNED POSES")
    print("=" * 80)

    returned_poses = []

    for frame, prediction in zip(
        TEST_FRAMES,
        predictions
    ):

        T = extract_pose(
            prediction
        )

        returned_poses.append(
            T
        )

        print()
        print(
            f"FRAME {frame:06d}"
        )

        print(
            "Translation:",
            T[:3, 3]
        )

        print(
            "Euler:",
            rotation_to_euler(
                T[:3, :3]
            )
        )

    returned_poses = np.stack(
        returned_poses
    )


    # ========================================================
    # 15. MAPANYTHING RELATIVE TRAJECTORY
    # ========================================================

    print()
    print("=" * 80)
    print("15. RELATIVE MAPANYTHING TRAJECTORY")
    print("=" * 80)

    T0 = returned_poses[0]

    T0_inv = np.linalg.inv(
        T0
    )

    relative_poses = []

    for i, frame in enumerate(
        TEST_FRAMES
    ):

        Trel = (
            T0_inv
            @
            returned_poses[i]
        )

        relative_poses.append(
            Trel
        )

        print()
        print(
            f"{frame:06d}: "
            f"{Trel[:3,3]}"
        )

        if i > 0:

            delta = (
                np.linalg.inv(
                    relative_poses[i - 1]
                )
                @
                Trel
            )

            print(
                f"  step: "
                f"{np.linalg.norm(delta[:3,3]):.4f} m"
            )

    relative_poses = np.asarray(
        relative_poses
    )


    # ========================================================
    # 16. SUPPLIED RELATIVE TRAJECTORY
    # ========================================================

    print()
    print("=" * 80)
    print("16. SUPPLIED CORRECTED TRAJECTORY")
    print("=" * 80)

    supplied_T0 = supplied_poses[0]

    supplied_T0_inv = np.linalg.inv(
        supplied_T0
    )

    supplied_relative = []

    for i, frame in enumerate(
        TEST_FRAMES
    ):

        Trel = (
            supplied_T0_inv
            @
            supplied_poses[i]
        )

        supplied_relative.append(
            Trel
        )

        print(
            f"{frame:06d}: "
            f"{Trel[:3,3]}"
        )

    supplied_relative = np.asarray(
        supplied_relative
    )


    # ========================================================
    # 17. TRAJECTORY COMPARISON
    # ========================================================

    print()
    print("=" * 80)
    print("17. TRAJECTORY COMPARISON")
    print("=" * 80)

    for i, frame in enumerate(
        TEST_FRAMES
    ):

        supplied_xyz = (
            supplied_relative[
                i,
                :3,
                3
            ]
        )

        map_xyz = (
            relative_poses[
                i,
                :3,
                3
            ]
        )

        error = np.linalg.norm(
            map_xyz
            -
            supplied_xyz
        )

        print()
        print(
            f"{frame:06d}"
        )

        print(
            "  Supplied:",
            supplied_xyz
        )

        print(
            "  MapAnything:",
            map_xyz
        )

        print(
            f"  Position difference: "
            f"{error:.4f} m"
        )


    # ========================================================
    # 18. MAPANYTHING TRAJECTORY PLY
    # ========================================================

    map_points = (
        relative_poses[
            :,
            :3,
            3
        ]
        .astype(
            np.float32
        )
    )

    map_colors = np.tile(
        np.array(
            [255, 0, 0],
            dtype=np.uint8
        ),
        (
            len(map_points),
            1
        )
    )

    map_path = (
        OUTPUT_DIR
        /
        "mapanything_corrected_pose_trajectory.ply"
    )

    write_ply(
        map_path,
        map_points,
        map_colors
    )


    # ========================================================
    # 19. SUPPLIED TRAJECTORY PLY
    # ========================================================

    supplied_points = (
        supplied_relative[
            :,
            :3,
            3
        ]
        .astype(
            np.float32
        )
    )

    supplied_colors = np.tile(
        np.array(
            [0, 0, 255],
            dtype=np.uint8
        ),
        (
            len(supplied_points),
            1
        )
    )

    supplied_path = (
        OUTPUT_DIR
        /
        "supplied_corrected_pose_trajectory.ply"
    )

    write_ply(
        supplied_path,
        supplied_points,
        supplied_colors
    )


    # ========================================================
    # 20. COMBINED TRAJECTORY
    # ========================================================

    combined_points = np.concatenate(
        [
            supplied_points,
            map_points
        ],
        axis=0
    )

    combined_colors = np.concatenate(
        [
            supplied_colors,
            map_colors
        ],
        axis=0
    )

    combined_path = (
        OUTPUT_DIR
        /
        "trajectory_comparison.ply"
    )

    write_ply(
        combined_path,
        combined_points,
        combined_colors
    )


    # ========================================================
    # 21. SAVE RESULTS
    # ========================================================

    results_path = (
        OUTPUT_DIR
        /
        "corrected_pose_results.npz"
    )

    np.savez(
        results_path,

        frames=np.asarray(
            TEST_FRAMES
        ),

        supplied_poses=(
            supplied_poses
        ),

        returned_poses=(
            returned_poses
        ),

        supplied_relative=(
            supplied_relative
        ),

        relative_poses=(
            relative_poses
        ),

        K_scaled=(
            K_scaled
        ),

        reconstruction_points=(
            merged_points
        ),

        reconstruction_colors=(
            merged_colors
        ),

        per_frame_stats=(
            np.asarray(
                per_frame_stats,
                dtype=np.int64
            )
        ),
    )

    print()
    print(
        "Results saved:"
    )

    print(
        results_path
    )


    # ========================================================
    # DONE
    # ========================================================

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)

    print()
    print(
        "ACTUAL RECONSTRUCTION:"
    )

    print(
        merged_path
    )

    print()
    print(
        "Individual reconstructions:"
    )

    for frame in TEST_FRAMES:

        print(
            OUTPUT_DIR
            /
            f"reconstruction_frame_{frame:06d}.ply"
        )

    print()
    print(
        "MapAnything trajectory:"
    )

    print(
        map_path
    )

    print()
    print(
        "Supplied trajectory:"
    )

    print(
        supplied_path
    )

    print()
    print(
        "Combined trajectory:"
    )

    print(
        combined_path
    )

    print()
    print(
        f"TOTAL RECONSTRUCTION POINTS: "
        f"{len(merged_points):,}"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "The corrected poses were supplied to MapAnything."
    )

    print(
        "The returned pts3d were NOT transformed."
    )


    # ========================================================
    # CLEANUP
    # ========================================================

    del predictions
    del model
    del views

    del merged_points
    del merged_colors

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()