#!/usr/bin/env python3

"""
A2D2 / MapAnything FULL CALIBRATION + POSE DIAGNOSTICS

Run:

    python scripts/check_a2d2_calibration.py

This script DOES NOT run MapAnything.

It investigates:

1. Image / pose matching
2. Camera intrinsics
3. Image dimensions
4. Pose matrix validity
5. Translation statistics
6. Frame-to-frame motion
7. Rotation statistics
8. Camera trajectory
9. Camera forward/right/up directions
10. Yaw / pitch / roll behaviour
11. Camera->world vs world->camera candidates
12. A2D2 -> OpenCV camera-axis conversion candidates
13. Relative transforms between frames
14. Optional A2D2 cams_lidars.json
15. Optional a2d2_demo.txt
16. Saves trajectory plots
17. Saves diagnostic CSV files

IMPORTANT:

We are NOT going to blindly assume a pose convention.

The goal is to determine what the pose files actually represent
before feeding them into MapAnything.
"""

import os
import glob
import json
import math
import csv
import argparse

import numpy as np


# ============================================================
# DEFAULT PATHS
# ============================================================

SEQUENCE_DIR = (
    "a2d2_data_full/"
    "dataset/sequences/00"
)

DEFAULT_IMAGE_DIR = (
    SEQUENCE_DIR + "/color"
)

DEFAULT_POSE_DIR = (
    SEQUENCE_DIR + "/pose"
)

DEFAULT_INTRINSIC_FILE = (
    SEQUENCE_DIR + "/intrinsic.txt"
)

DEFAULT_DEMO_FILE = (
    SEQUENCE_DIR + "/a2d2_demo.txt"
)

# We search for these automatically
POSSIBLE_CALIB_FILES = [
    SEQUENCE_DIR + "/cams_lidars.json",
    "a2d2_data_full/cams_lidars.json",
    "data/cams_lidars.json",
]


OUTPUT_DIR = "calibration_diagnostics"


# ============================================================
# MATRIX HELPERS
# ============================================================

def rotation_angle_deg(R):

    value = (
        np.trace(R) - 1.0
    ) / 2.0

    value = np.clip(
        value,
        -1.0,
        1.0
    )

    return np.degrees(
        np.arccos(value)
    )


def rotation_error(R):

    orth_error = np.linalg.norm(
        R.T @ R - np.eye(3),
        ord="fro"
    )

    det = np.linalg.det(R)

    return orth_error, det


def is_valid_rotation(R):

    orth_error, det = rotation_error(R)

    return (
        orth_error < 1e-3
        and
        abs(det - 1.0) < 1e-3
    )


def make_axis_conversion():

    """
    Candidate A2D2 camera -> OpenCV camera conversion.

    A2D2-style vehicle/camera axes:

        X = forward
        Y = left
        Z = up

    OpenCV camera axes:

        X = right
        Y = down
        Z = forward

    Therefore:

        X_cv = -Y_a2d2
        Y_cv = -Z_a2d2
        Z_cv =  X_a2d2
    """

    return np.array(
        [
            [0.0, -1.0,  0.0],
            [0.0,  0.0, -1.0],
            [1.0,  0.0,  0.0],
        ],
        dtype=float
    )


def print_matrix(name, M):

    print(f"\n{name}:")

    print(
        np.array2string(
            M,
            precision=6,
            suppress_small=True
        )
    )


def load_pose(path):

    M = np.loadtxt(path)

    if M.shape != (4, 4):

        raise ValueError(
            f"{path} has shape {M.shape}, "
            f"expected 4x4"
        )

    return M


# ============================================================
# FILE DISCOVERY
# ============================================================

def image_frame_ids(image_dir):

    paths = sorted(
        glob.glob(
            os.path.join(
                image_dir,
                "*.jpg"
            )
        )
        +
        glob.glob(
            os.path.join(
                image_dir,
                "*.png"
            )
        )
    )

    ids = {}

    for path in paths:

        stem = os.path.splitext(
            os.path.basename(path)
        )[0]

        try:

            frame_id = int(stem)

        except ValueError:

            continue

        ids[frame_id] = path

    return ids


def pose_frame_ids(pose_dir):

    paths = sorted(
        glob.glob(
            os.path.join(
                pose_dir,
                "*.txt"
            )
        )
    )

    ids = {}

    for path in paths:

        stem = os.path.splitext(
            os.path.basename(path)
        )[0]

        try:

            frame_id = int(stem)

        except ValueError:

            continue

        ids[frame_id] = path

    return ids


# ============================================================
# INTRINSICS
# ============================================================

def check_intrinsics(
    intrinsic_file,
    image_ids
):

    print("\n")
    print("=" * 70)
    print("1. CAMERA INTRINSICS")
    print("=" * 70)

    Kraw = np.loadtxt(
        intrinsic_file
    )

    print(
        "File:",
        intrinsic_file
    )

    print_matrix(
        "Raw intrinsic matrix",
        Kraw
    )

    if Kraw.shape == (4, 4):

        K = Kraw[:3, :3]

    elif Kraw.shape == (3, 3):

        K = Kraw

    else:

        raise ValueError(
            f"Unexpected intrinsic shape "
            f"{Kraw.shape}"
        )

    print_matrix(
        "Camera matrix K",
        K
    )

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    print("\nParameters:")

    print(f"  fx = {fx}")
    print(f"  fy = {fy}")
    print(f"  cx = {cx}")
    print(f"  cy = {cy}")

    print("\nChecks:")

    print(
        "  focal lengths positive:",
        fx > 0 and fy > 0
    )

    print(
        "  finite:",
        np.all(
            np.isfinite(K)
        )
    )

    print(
        "  K[2,2] == 1:",
        abs(K[2, 2] - 1) < 1e-6
    )

    # --------------------------------------------------------
    # Image size
    # --------------------------------------------------------

    try:

        from PIL import Image

        first_id = sorted(
            image_ids.keys()
        )[0]

        image_path = image_ids[
            first_id
        ]

        with Image.open(
            image_path
        ) as im:

            width, height = im.size

        print("\nActual image dimensions:")

        print(
            f"  width  = {width}"
        )

        print(
            f"  height = {height}"
        )

        print("\nPrincipal point:")

        print(
            f"  cx / width  = "
            f"{cx / width:.4f}"
        )

        print(
            f"  cy / height = "
            f"{cy / height:.4f}"
        )

        print(
            "  inside image:",
            (
                0 <= cx <= width
                and
                0 <= cy <= height
            )
        )

        print("\nFocal / image ratios:")

        print(
            f"  fx / width  = "
            f"{fx / width:.4f}"
        )

        print(
            f"  fy / height = "
            f"{fy / height:.4f}"
        )

    except Exception as e:

        print(
            "Could not inspect image dimensions:",
            e
        )

    return K


# ============================================================
# POSE LOADING
# ============================================================

def load_all_poses(
    pose_ids
):

    data = {}

    for frame_id in sorted(
        pose_ids.keys()
    ):

        try:

            M = load_pose(
                pose_ids[frame_id]
            )

            R = M[:3, :3]
            t = M[:3, 3]

            orth_error, det = (
                rotation_error(R)
            )

            data[frame_id] = {

                "M": M,

                "R": R,

                "t": t,

                "valid":
                    np.all(
                        np.isfinite(M)
                    )
                    and
                    is_valid_rotation(R),

                "orth_error":
                    orth_error,

                "det":
                    det,
            }

        except Exception as e:

            print(
                f"Could not load "
                f"{frame_id:06d}: {e}"
            )

    return data


# ============================================================
# POSE INTEGRITY
# ============================================================

def check_pose_integrity(
    pose_data,
    sample_ids
):

    print("\n")
    print("=" * 70)
    print("2. POSE INTEGRITY")
    print("=" * 70)

    bad = 0

    for frame_id in sample_ids:

        if frame_id not in pose_data:

            print(
                f"[MISSING] "
                f"{frame_id:06d}"
            )

            bad += 1

            continue

        p = pose_data[
            frame_id
        ]

        print(
            f"\nFrame "
            f"{frame_id:06d}"
        )

        print(
            "  orthogonality error:",
            f"{p['orth_error']:.3e}"
        )

        print(
            "  determinant:",
            f"{p['det']:.9f}"
        )

        print(
            "  valid rotation:",
            p["valid"]
        )

        print(
            "  translation:",
            p["t"]
        )

        bottom_row_ok = np.allclose(
            p["M"][3],
            [0, 0, 0, 1],
            atol=1e-5
        )

        print(
            "  bottom row:",
            bottom_row_ok
        )

        if not p["valid"]:

            bad += 1

    print(
        "\nBad sampled poses:",
        bad
    )


# ============================================================
# TRAJECTORY STATISTICS
# ============================================================

def compute_motion(
    pose_data,
    frame_ids
):

    translations = []
    rotations = []
    ids = []

    for frame_id in frame_ids:

        if frame_id not in pose_data:
            continue

        p = pose_data[
            frame_id
        ]

        if not p["valid"]:
            continue

        ids.append(
            frame_id
        )

        translations.append(
            p["t"]
        )

        rotations.append(
            p["R"]
        )

    translations = np.asarray(
        translations
    )

    rotations = np.asarray(
        rotations
    )

    ids = np.asarray(
        ids
    )

    step_distance = []
    step_angle = []
    step_ids = []

    for i in range(
        1,
        len(ids)
    ):

        # Only consecutive files
        if (
            ids[i]
            != ids[i - 1] + 1
        ):
            continue

        dt = (
            translations[i]
            -
            translations[i - 1]
        )

        distance = np.linalg.norm(
            dt
        )

        dR = (
            rotations[i - 1].T
            @
            rotations[i]
        )

        angle = rotation_angle_deg(
            dR
        )

        step_distance.append(
            distance
        )

        step_angle.append(
            angle
        )

        step_ids.append(
            ids[i]
        )

    return (
        ids,
        translations,
        rotations,
        np.asarray(step_distance),
        np.asarray(step_angle),
        np.asarray(step_ids)
    )


# ============================================================
# TRAJECTORY ANALYSIS
# ============================================================

def analyze_trajectory(
    pose_data,
    image_ids
):

    print("\n")
    print("=" * 70)
    print("3. VEHICLE TRAJECTORY")
    print("=" * 70)

    common_ids = sorted(
        set(image_ids.keys())
        &
        set(pose_data.keys())
    )

    (
        ids,
        translations,
        rotations,
        step_distances,
        step_angles,
        step_ids
    ) = compute_motion(
        pose_data,
        common_ids
    )

    print(
        "Valid trajectory frames:",
        len(ids)
    )

    print("\nGlobal translation:")

    print(
        "  X:"
        f" {translations[:,0].min():.3f}"
        f" -> {translations[:,0].max():.3f}"
    )

    print(
        "  Y:"
        f" {translations[:,1].min():.3f}"
        f" -> {translations[:,1].max():.3f}"
    )

    print(
        "  Z:"
        f" {translations[:,2].min():.3f}"
        f" -> {translations[:,2].max():.3f}"
    )

    print("\nFrame-to-frame translation:")

    if len(step_distances):

        print(
            "  mean:",
            np.mean(step_distances)
        )

        print(
            "  median:",
            np.median(step_distances)
        )

        print(
            "  max:",
            np.max(step_distances)
        )

        print("\nFrame-to-frame rotation:")

        print(
            "  mean:",
            np.mean(step_angles),
            "deg"
        )

        print(
            "  median:",
            np.median(step_angles),
            "deg"
        )

        print(
            "  max:",
            np.max(step_angles),
            "deg"
        )

    # --------------------------------------------------------
    # Estimate dominant movement direction
    # --------------------------------------------------------

    deltas = (
        translations[1:]
        -
        translations[:-1]
    )

    valid = (
        np.linalg.norm(
            deltas,
            axis=1
        ) > 1e-6
    )

    deltas = deltas[
        valid
    ]

    if len(deltas):

        mean_direction = np.mean(
            deltas,
            axis=0
        )

        mean_direction /= (
            np.linalg.norm(
                mean_direction
            )
        )

        print(
            "\nDominant translation direction:"
        )

        print(
            " ",
            mean_direction
        )

        print(
            "\nThis is useful for determining"
        )

        print(
            "whether the pose world axes make"
        )

        print(
            "sense for the actual road trajectory."
        )


# ============================================================
# CAMERA AXIS ANALYSIS
# ============================================================

def analyze_camera_axes(
    pose_data,
    frame_ids
):

    print("\n")
    print("=" * 70)
    print("4. CAMERA AXIS / ORIENTATION ANALYSIS")
    print("=" * 70)

    print(
        """
For each pose R:

    column 0 = camera X axis
    column 1 = camera Y axis
    column 2 = camera Z axis

We print all three because we need to
determine which one points forward.
"""
    )

    rows = []

    for frame_id in frame_ids:

        if frame_id not in pose_data:
            continue

        R = pose_data[
            frame_id
        ]["R"]

        t = pose_data[
            frame_id
        ]["t"]

        x_axis = R[:, 0]
        y_axis = R[:, 1]
        z_axis = R[:, 2]

        rows.append([
            frame_id,
            *t,
            *x_axis,
            *y_axis,
            *z_axis
        ])

    rows = np.asarray(
        rows
    )

    if len(rows) == 0:
        return

    print("\nRepresentative camera axes:")

    for row in rows[
        ::max(
            1,
            len(rows) // 10
        )
    ]:

        frame_id = int(
            row[0]
        )

        print(
            f"\nFrame {frame_id:06d}"
        )

        print(
            "  position:",
            row[1:4]
        )

        print(
            "  X axis:",
            row[4:7]
        )

        print(
            "  Y axis:",
            row[7:10]
        )

        print(
            "  Z axis:",
            row[10:13]
        )


# ============================================================
# EULER ANGLES
# ============================================================

def rotation_to_euler(R):

    """
    ZYX convention.

    Returns:
        yaw, pitch, roll
    in degrees.
    """

    sy = math.sqrt(
        R[0, 0] ** 2
        +
        R[1, 0] ** 2
    )

    singular = (
        sy < 1e-6
    )

    if not singular:

        yaw = math.atan2(
            R[1, 0],
            R[0, 0]
        )

        pitch = math.atan2(
            -R[2, 0],
            sy
        )

        roll = math.atan2(
            R[2, 1],
            R[2, 2]
        )

    else:

        yaw = math.atan2(
            -R[1, 2],
            R[1, 1]
        )

        pitch = math.atan2(
            -R[2, 0],
            sy
        )

        roll = 0

    return np.degrees([
        yaw,
        pitch,
        roll
    ])


def analyze_euler(
    pose_data,
    frame_ids
):

    print("\n")
    print("=" * 70)
    print("5. ORIENTATION / YAW PITCH ROLL")
    print("=" * 70)

    values = []

    for frame_id in frame_ids:

        if frame_id not in pose_data:
            continue

        R = pose_data[
            frame_id
        ]["R"]

        yaw, pitch, roll = (
            rotation_to_euler(R)
        )

        values.append([
            frame_id,
            yaw,
            pitch,
            roll
        ])

    values = np.asarray(
        values
    )

    if len(values) == 0:
        return

    for name, idx in [
        ("yaw", 1),
        ("pitch", 2),
        ("roll", 3)
    ]:

        x = values[:, idx]

        print(
            f"\n{name}:"
        )

        print(
            "  min:",
            np.min(x)
        )

        print(
            "  max:",
            np.max(x)
        )

        print(
            "  range:",
            np.ptp(x)
        )


# ============================================================
# POSE CONVENTION COMPARISON
# ============================================================

def compare_pose_conventions(
    pose_data,
    frame_ids
):

    print("\n")
    print("=" * 70)
    print("6. CAMERA->WORLD vs WORLD->CAMERA")
    print("=" * 70)

    for frame_id in frame_ids:

        if frame_id not in pose_data:
            continue

        M = pose_data[
            frame_id
        ]["M"]

        R = M[:3, :3]
        t = M[:3, 3]

        # If M = camera -> world
        c2w_position = t

        # If M = world -> camera
        w2c_position = (
            -R.T @ t
        )

        print(
            f"\nFrame {frame_id:06d}"
        )

        print(
            " raw t:",
            t
        )

        print(
            " position if C2W:",
            c2w_position
        )

        print(
            " position if W2C:",
            w2c_position
        )


# ============================================================
# RELATIVE TRANSFORMS
# ============================================================

def analyze_relative_transforms(
    pose_data,
    frame_pairs
):

    print("\n")
    print("=" * 70)
    print("7. RELATIVE TRANSFORM CHECK")
    print("=" * 70)

    print(
        """
This is particularly important.

For two frames i and j:

    T_i
    T_j

we calculate:

    T_i^-1 T_j

This tells us the motion from frame i
to frame j under the assumed convention.

If the vehicle is travelling straight,
the relative transforms should show
small smooth rotations and translations.
"""
    )

    for i, j in frame_pairs:

        if (
            i not in pose_data
            or
            j not in pose_data
        ):
            continue

        Ti = pose_data[i]["M"]
        Tj = pose_data[j]["M"]

        relative = (
            np.linalg.inv(Ti)
            @
            Tj
        )

        R = relative[:3, :3]
        t = relative[:3, 3]

        angle = rotation_angle_deg(
            R
        )

        print(
            f"\nFrame {i:06d}"
            f" -> {j:06d}"
        )

        print(
            " relative translation:",
            t
        )

        print(
            " relative distance:",
            np.linalg.norm(t)
        )

        print(
            " relative rotation:",
            angle,
            "deg"
        )


# ============================================================
# AXIS CANDIDATES
# ============================================================

def print_axis_candidates(
    pose_data,
    frame_id
):

    print("\n")
    print("=" * 70)
    print("8. COORDINATE FRAME CANDIDATES")
    print("=" * 70)

    if frame_id not in pose_data:
        return

    M = pose_data[
        frame_id
    ]["M"]

    R = M[:3, :3]
    t = M[:3, 3]

    C = make_axis_conversion()

    print(
        "\nCandidate A2D2 camera-axis conversion:"
    )

    print_matrix(
        "C",
        C
    )

    # --------------------------------------------------------
    # Candidate A
    #
    # Raw matrix is C2W.
    # Preserve world frame.
    # Change camera basis only.
    # --------------------------------------------------------

    R_A = (
        R @ C.T
    )

    T_A = np.eye(4)

    T_A[:3, :3] = R_A

    T_A[:3, 3] = t

    print_matrix(
        "\nCandidate A:"
        "\nRaw = C2W"
        "\nPreserve A2D2 world"
        "\nChange camera basis",
        T_A
    )

    # --------------------------------------------------------
    # Candidate B
    #
    # Raw matrix is W2C.
    # First invert.
    # Then change camera basis.
    # --------------------------------------------------------

    T_inv = np.linalg.inv(
        M
    )

    R_B = (
        T_inv[:3, :3]
        @
        C.T
    )

    T_B = np.eye(4)

    T_B[:3, :3] = R_B

    T_B[:3, 3] = T_inv[:3, 3]

    print_matrix(
        "\nCandidate B:"
        "\nRaw = W2C"
        "\nInvert"
        "\nChange camera basis",
        T_B
    )

    # --------------------------------------------------------
    # Old wrong candidate
    # --------------------------------------------------------

    R_old = (
        C @ R @ C.T
    )

    t_old = (
        C @ t
    )

    T_old = np.eye(4)

    T_old[:3, :3] = R_old

    T_old[:3, 3] = t_old

    print_matrix(
        "\nOLD candidate:"
        "\nC @ R @ C.T"
        "\nC @ t",
        T_old
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "The OLD candidate rotates the WORLD frame."
    )

    print(
        "We should NOT use it blindly."
    )


# ============================================================
# SEARCH FOR A2D2 CALIBRATION
# ============================================================

def find_calibration_file():

    for path in POSSIBLE_CALIB_FILES:

        if os.path.exists(path):

            return path

    return None


def inspect_a2d2_calibration():

    print("\n")
    print("=" * 70)
    print("9. A2D2 SENSOR CALIBRATION")
    print("=" * 70)

    path = find_calibration_file()

    if path is None:

        print(
            "cams_lidars.json not found."
        )

        print(
            "This is okay, but if you have the"
        )

        print(
            "original A2D2 calibration file,"
        )

        print(
            "copy it into the sequence directory."
        )

        return

    print(
        "Found:",
        path
    )

    try:

        with open(
            path,
            "r"
        ) as f:

            data = json.load(f)

    except Exception as e:

        print(
            "Could not read JSON:",
            e
        )

        return

    print(
        "\nTop-level entries:"
    )

    if isinstance(
        data,
        dict
    ):

        for key in data.keys():

            print(
                " ",
                key
            )

    # --------------------------------------------------------
    # Search recursively for front-center
    # --------------------------------------------------------

    def recursive_find(
        obj,
        target,
        path=""
    ):

        results = []

        if isinstance(
            obj,
            dict
        ):

            for key, value in obj.items():

                new_path = (
                    f"{path}/{key}"
                )

                if (
                    target.lower()
                    in key.lower()
                ):

                    results.append(
                        (
                            new_path,
                            value
                        )
                    )

                results.extend(
                    recursive_find(
                        value,
                        target,
                        new_path
                    )
                )

        elif isinstance(
            obj,
            list
        ):

            for i, value in enumerate(
                obj
            ):

                results.extend(
                    recursive_find(
                        value,
                        target,
                        f"{path}/{i}"
                    )
                )

        return results

    results = recursive_find(
        data,
        "front_center"
    )

    print(
        "\nEntries containing "
        "'front_center':",
        len(results)
    )

    for path, value in results[:10]:

        print(
            "\nPATH:",
            path
        )

        print(
            value
        )


# ============================================================
# a2d2_demo.txt
# ============================================================

def inspect_demo_file():

    print("\n")
    print("=" * 70)
    print("10. A2D2 DEMO / SEQUENCE METADATA")
    print("=" * 70)

    if not os.path.exists(
        DEFAULT_DEMO_FILE
    ):

        print(
            "Not found:",
            DEFAULT_DEMO_FILE
        )

        return

    print(
        "Found:",
        DEFAULT_DEMO_FILE
    )

    try:

        with open(
            DEFAULT_DEMO_FILE,
            "r",
            errors="ignore"
        ) as f:

            lines = f.readlines()

    except Exception as e:

        print(
            "Could not read:",
            e
        )

        return

    print(
        "\nNumber of lines:",
        len(lines)
    )

    print(
        "\nFirst 30 lines:"
    )

    for line in lines[:30]:

        print(
            line.rstrip()
        )


# ============================================================
# SAVE TRAJECTORY CSV
# ============================================================

def save_trajectory_csv(
    pose_data,
    image_ids
):

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    path = os.path.join(
        OUTPUT_DIR,
        "trajectory.csv"
    )

    common_ids = sorted(
        set(image_ids.keys())
        &
        set(pose_data.keys())
    )

    with open(
        path,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "frame",
            "x",
            "y",
            "z",
            "yaw",
            "pitch",
            "roll",
            "cam_x_x",
            "cam_x_y",
            "cam_x_z",
            "cam_y_x",
            "cam_y_y",
            "cam_y_z",
            "cam_z_x",
            "cam_z_y",
            "cam_z_z",
        ])

        for frame_id in common_ids:

            p = pose_data[
                frame_id
            ]

            R = p["R"]
            t = p["t"]

            yaw, pitch, roll = (
                rotation_to_euler(R)
            )

            writer.writerow([
                frame_id,
                t[0],
                t[1],
                t[2],
                yaw,
                pitch,
                roll,
                *R[:, 0],
                *R[:, 1],
                *R[:, 2],
            ])

    print(
        "\nSaved:",
        path
    )


# ============================================================
# PLOTS
# ============================================================

def save_plots(
    pose_data,
    image_ids
):

    try:

        import matplotlib.pyplot as plt

    except ImportError:

        print(
            "\nmatplotlib not installed."
        )

        print(
            "Skipping plots."
        )

        return

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    common_ids = sorted(
        set(image_ids.keys())
        &
        set(pose_data.keys())
    )

    ids = []
    positions = []
    eulers = []

    for frame_id in common_ids:

        p = pose_data[
            frame_id
        ]

        ids.append(
            frame_id
        )

        positions.append(
            p["t"]
        )

        eulers.append(
            rotation_to_euler(
                p["R"]
            )
        )

    ids = np.asarray(
        ids
    )

    positions = np.asarray(
        positions
    )

    eulers = np.asarray(
        eulers
    )

    # --------------------------------------------------------
    # XY trajectory
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 8)
    )

    plt.plot(
        positions[:, 0],
        positions[:, 1]
    )

    plt.xlabel(
        "Pose X"
    )

    plt.ylabel(
        "Pose Y"
    )

    plt.title(
        "A2D2 Pose Trajectory"
    )

    plt.axis(
        "equal"
    )

    plt.grid()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "trajectory_xy.png"
        ),
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # XZ
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 8)
    )

    plt.plot(
        positions[:, 0],
        positions[:, 2]
    )

    plt.xlabel(
        "Pose X"
    )

    plt.ylabel(
        "Pose Z"
    )

    plt.title(
        "Pose X-Z"
    )

    plt.grid()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "trajectory_xz.png"
        ),
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # Coordinates over time
    # --------------------------------------------------------

    plt.figure(
        figsize=(12, 8)
    )

    plt.plot(
        ids,
        positions[:, 0],
        label="X"
    )

    plt.plot(
        ids,
        positions[:, 1],
        label="Y"
    )

    plt.plot(
        ids,
        positions[:, 2],
        label="Z"
    )

    plt.xlabel(
        "Frame"
    )

    plt.ylabel(
        "Position"
    )

    plt.title(
        "Pose Coordinates vs Frame"
    )

    plt.legend()

    plt.grid()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "position_vs_frame.png"
        ),
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # Euler angles
    # --------------------------------------------------------

    plt.figure(
        figsize=(12, 8)
    )

    plt.plot(
        ids,
        eulers[:, 0],
        label="Yaw"
    )

    plt.plot(
        ids,
        eulers[:, 1],
        label="Pitch"
    )

    plt.plot(
        ids,
        eulers[:, 2],
        label="Roll"
    )

    plt.xlabel(
        "Frame"
    )

    plt.ylabel(
        "Degrees"
    )

    plt.title(
        "Pose Orientation"
    )

    plt.legend()

    plt.grid()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "orientation_vs_frame.png"
        ),
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "\nSaved plots to:",
        OUTPUT_DIR
    )


# ============================================================
# FINAL DIAGNOSTIC SUMMARY
# ============================================================

def final_summary(
    pose_data,
    image_ids
):

    print("\n")
    print("=" * 70)
    print("11. FINAL DIAGNOSTIC SUMMARY")
    print("=" * 70)

    common = sorted(
        set(image_ids.keys())
        &
        set(pose_data.keys())
    )

    print(
        "\nImage count:",
        len(image_ids)
    )

    print(
        "Pose count:",
        len(pose_data)
    )

    print(
        "Matched:",
        len(common)
    )

    if not common:

        return

    translations = np.asarray([
        pose_data[i]["t"]
        for i in common
        if pose_data[i]["valid"]
    ])

    if len(translations):

        ranges = np.ptp(
            translations,
            axis=0
        )

        print(
            "\nTranslation ranges:"
        )

        print(
            "  X:",
            ranges[0]
        )

        print(
            "  Y:",
            ranges[1]
        )

        print(
            "  Z:",
            ranges[2]
        )

        dominant_axis = np.argmax(
            ranges
        )

        names = [
            "X",
            "Y",
            "Z"
        ]

        print(
            "\nDominant trajectory axis:",
            names[dominant_axis]
        )

        print(
            "\nThis is a key diagnostic."
        )

        print(
            "If the vehicle is travelling mostly"
        )

        print(
            "straight, the dominant trajectory"
        )

        print(
            "axis should correspond to the"
        )

        print(
            "vehicle-forward/world-forward axis."
        )

    print(
        "\nDO NOT change coordinate conventions"
    )

    print(
        "based only on the previous PLY."
    )

    print(
        "We now have enough information to"
    )

    print(
        "compare the actual A2D2 trajectory"
    )

    print(
        "against the camera orientation."
    )

    print(
        "\nNext step:"
    )

    print(
        "Run this script and send me:"
    )

    print(
        "  1. the terminal output"
    )

    print(
        "  2. trajectory_xy.png"
    )

    print(
        "  3. orientation_vs_frame.png"
    )

    print(
        "  4. position_vs_frame.png"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--images",
        default=DEFAULT_IMAGE_DIR
    )

    parser.add_argument(
        "--poses",
        default=DEFAULT_POSE_DIR
    )

    parser.add_argument(
        "--intrinsics",
        default=DEFAULT_INTRINSIC_FILE
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=10
    )

    args = parser.parse_args()

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print("\n")
    print("=" * 70)
    print(
        "A2D2 / MAPANYTHING FULL DIAGNOSTICS"
    )
    print("=" * 70)

    print(
        "\nImages:",
        args.images
    )

    print(
        "Poses:",
        args.poses
    )

    print(
        "Intrinsics:",
        args.intrinsics
    )

    # --------------------------------------------------------
    # Discover
    # --------------------------------------------------------

    image_ids = image_frame_ids(
        args.images
    )

    pose_ids = pose_frame_ids(
        args.poses
    )

    print(
        "\nFound images:",
        len(image_ids)
    )

    print(
        "Found poses:",
        len(pose_ids)
    )

    # --------------------------------------------------------
    # Matching
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("IMAGE / POSE MATCHING")
    print("=" * 70)

    common = (
        set(image_ids.keys())
        &
        set(pose_ids.keys())
    )

    print(
        "Matching:",
        len(common)
    )

    print(
        "Images without pose:",
        len(
            set(image_ids.keys())
            -
            set(pose_ids.keys())
        )
    )

    print(
        "Poses without image:",
        len(
            set(pose_ids.keys())
            -
            set(image_ids.keys())
        )
    )

    # --------------------------------------------------------
    # Intrinsics
    # --------------------------------------------------------

    check_intrinsics(
        args.intrinsics,
        image_ids
    )

    # --------------------------------------------------------
    # Load poses
    # --------------------------------------------------------

    print(
        "\nLoading all pose matrices..."
    )

    pose_data = load_all_poses(
        pose_ids
    )

    print(
        "Loaded:",
        len(pose_data)
    )

    # --------------------------------------------------------
    # Representative samples
    # --------------------------------------------------------

    common_ids = sorted(
        common
    )

    n = min(
        args.samples,
        len(common_ids)
    )

    indices = np.linspace(
        0,
        len(common_ids) - 1,
        n,
        dtype=int
    )

    sample_ids = [
        common_ids[i]
        for i in indices
    ]

    print(
        "\nRepresentative frames:"
    )

    print(
        sample_ids
    )

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    check_pose_integrity(
        pose_data,
        sample_ids
    )

    analyze_trajectory(
        pose_data,
        image_ids
    )

    analyze_camera_axes(
        pose_data,
        sample_ids
    )

    analyze_euler(
        pose_data,
        common_ids
    )

    compare_pose_conventions(
        pose_data,
        sample_ids[:5]
    )

    # --------------------------------------------------------
    # Relative transforms
    # --------------------------------------------------------

    if len(common_ids) >= 4:

        frame_pairs = [
            (
                common_ids[0],
                common_ids[1]
            ),
            (
                common_ids[0],
                common_ids[min(
                    10,
                    len(common_ids) - 1
                )]
            ),
            (
                common_ids[0],
                common_ids[min(
                    100,
                    len(common_ids) - 1
                )]
            ),
        ]

        analyze_relative_transforms(
            pose_data,
            frame_pairs
        )

    # --------------------------------------------------------
    # Axis candidates
    # --------------------------------------------------------

    print_axis_candidates(
        pose_data,
        common_ids[0]
    )

    # --------------------------------------------------------
    # A2D2 metadata
    # --------------------------------------------------------

    inspect_a2d2_calibration()

    inspect_demo_file()

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    save_trajectory_csv(
        pose_data,
        image_ids
    )

    # --------------------------------------------------------
    # Save plots
    # --------------------------------------------------------

    save_plots(
        pose_data,
        image_ids
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    final_summary(
        pose_data,
        image_ids
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "DIAGNOSTICS COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()