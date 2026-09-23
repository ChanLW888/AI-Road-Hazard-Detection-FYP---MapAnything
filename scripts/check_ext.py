import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

SEQUENCE_DIR = (
    "a2d2_data_full/dataset/sequences/00"
)

POSE_DIR = os.path.join(
    SEQUENCE_DIR,
    "pose"
)

EXTRINSIC_FILE = os.path.join(
    SEQUENCE_DIR,
    "lidar_to_camera.txt"
)


# ============================================================
# Helpers
# ============================================================

def frame_id(path):

    name = os.path.basename(path)

    nums = re.findall(
        r"\d+",
        os.path.splitext(name)[0]
    )

    return int(nums[-1])


def load_pose(path):

    T = np.loadtxt(path)

    if T.shape != (4, 4):

        raise ValueError(
            f"Bad pose shape: {path}"
        )

    return T


def camera_position_from_cam2world(T):

    """
    If T is camera -> world:

        p_world = R p_camera + t

    then camera position in world is simply:

        t
    """

    return T[:3, 3]


def camera_position_from_world2cam(T):

    """
    If T is world -> camera:

        p_camera = R p_world + t

    then camera position in world is:

        -R.T @ t
    """

    R = T[:3, :3]

    t = T[:3, 3]

    return -R.T @ t


# ============================================================
# Load extrinsic
# ============================================================

print("=" * 70)
print("A2D2 POSE + LIDAR/CAMERA EXTRINSIC CHECK")
print("=" * 70)

print()
print(
    "Extrinsic:",
    EXTRINSIC_FILE
)

T_lidar_camera = np.loadtxt(
    EXTRINSIC_FILE
)

print()
print("T_lidar_camera:")
print(T_lidar_camera)

R_ext = T_lidar_camera[:3, :3]

print()
print(
    "Extrinsic translation:",
    T_lidar_camera[:3, 3]
)

print(
    "Extrinsic translation magnitude:",
    np.linalg.norm(
        T_lidar_camera[:3, 3]
    )
)

print(
    "Extrinsic det(R):",
    np.linalg.det(R_ext)
)


# ============================================================
# Load poses
# ============================================================

pose_files = sorted(
    glob.glob(
        os.path.join(
            POSE_DIR,
            "*.txt"
        )
    ),
    key=frame_id
)

print()
print(
    "Pose files:",
    len(pose_files)
)


# ============================================================
# Storage
# ============================================================

frames = []

raw_translation = []

candidate_A = []
candidate_B = []
candidate_C = []
candidate_D = []


# ============================================================
# Interpretations
# ============================================================

for pose_file in pose_files:

    fid = frame_id(
        pose_file
    )

    T = load_pose(
        pose_file
    )

    # --------------------------------------------------------
    # A
    #
    # Assume pose is LiDAR -> World.
    #
    # Camera position before extrinsic:
    # translation of T
    # --------------------------------------------------------

    pos_A = camera_position_from_cam2world(
        T
    )


    # --------------------------------------------------------
    # B
    #
    # Assume pose is World -> LiDAR.
    #
    # Camera/LiDAR position:
    # -R.T t
    # --------------------------------------------------------

    pos_B = camera_position_from_world2cam(
        T
    )


    # --------------------------------------------------------
    # C
    #
    # Assume:
    #
    # T = LiDAR -> World
    #
    # and compose:
    #
    # T_world_camera =
    #       T_world_lidar
    #       @
    #       T_lidar_camera
    # --------------------------------------------------------

    T_world_camera_C = (
        T
        @
        T_lidar_camera
    )

    pos_C = (
        T_world_camera_C[
            :3,
            3
        ]
    )


    # --------------------------------------------------------
    # D
    #
    # Assume pose is World -> LiDAR.
    #
    # First invert:
    #
    # T_world_lidar =
    #       inv(T_world_lidar_inverse)
    #
    # then compose extrinsic.
    # --------------------------------------------------------

    T_world_lidar_D = np.linalg.inv(
        T
    )

    T_world_camera_D = (
        T_world_lidar_D
        @
        T_lidar_camera
    )

    pos_D = (
        T_world_camera_D[
            :3,
            3
        ]
    )


    frames.append(
        fid
    )

    raw_translation.append(
        pos_A
    )

    candidate_A.append(
        pos_A
    )

    candidate_B.append(
        pos_B
    )

    candidate_C.append(
        pos_C
    )

    candidate_D.append(
        pos_D
    )


# ============================================================
# Convert arrays
# ============================================================

frames = np.array(
    frames
)

candidate_A = np.array(
    candidate_A
)

candidate_B = np.array(
    candidate_B
)

candidate_C = np.array(
    candidate_C
)

candidate_D = np.array(
    candidate_D
)


# ============================================================
# Print representative frames
# ============================================================

sample_indices = np.linspace(
    0,
    len(frames) - 1,
    10,
    dtype=int
)

print()
print("=" * 70)
print("REPRESENTATIVE CAMERA POSITIONS")
print("=" * 70)

for idx in sample_indices:

    print()
    print(
        f"Frame {frames[idx]:06d}"
    )

    print(
        "A: pose = LiDAR->world:"
    )

    print(
        candidate_A[idx]
    )

    print(
        "B: pose = world->LiDAR:"
    )

    print(
        candidate_B[idx]
    )

    print(
        "C: LiDAR->world @ lidar->camera:"
    )

    print(
        candidate_C[idx]
    )

    print(
        "D: inv(pose) @ lidar->camera:"
    )

    print(
        candidate_D[idx]
    )


# ============================================================
# Plot XY trajectories
# ============================================================

plt.figure(
    figsize=(10, 8)
)

plt.plot(
    candidate_A[:, 0],
    candidate_A[:, 1],
    label="A: raw pose"
)

plt.plot(
    candidate_B[:, 0],
    candidate_B[:, 1],
    label="B: inverse raw pose"
)

plt.plot(
    candidate_C[:, 0],
    candidate_C[:, 1],
    label="C: pose @ lidar_to_camera"
)

plt.plot(
    candidate_D[:, 0],
    candidate_D[:, 1],
    label="D: inverse(pose) @ lidar_to_camera"
)

plt.xlabel(
    "World X"
)

plt.ylabel(
    "World Y"
)

plt.title(
    "A2D2 Camera Trajectory Candidates"
)

plt.axis(
    "equal"
)

plt.grid(
    True
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "pose_extrinsic_candidates.png",
    dpi=200
)

plt.close()


# ============================================================
# Plot Z
# ============================================================

plt.figure(
    figsize=(12, 6)
)

plt.plot(
    frames,
    candidate_A[:, 2],
    label="A: raw pose"
)

plt.plot(
    frames,
    candidate_B[:, 2],
    label="B: inverse raw pose"
)

plt.plot(
    frames,
    candidate_C[:, 2],
    label="C: pose @ extrinsic"
)

plt.plot(
    frames,
    candidate_D[:, 2],
    label="D: inverse(pose) @ extrinsic"
)

plt.xlabel(
    "Frame"
)

plt.ylabel(
    "World Z"
)

plt.title(
    "Camera Z Position Candidates"
)

plt.grid(
    True
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "pose_extrinsic_z.png",
    dpi=200
)

plt.close()


# ============================================================
# Consecutive motion statistics
# ============================================================

print()
print("=" * 70)
print("MOTION STATISTICS")
print("=" * 70)


def print_motion(
    name,
    trajectory
):

    diffs = np.diff(
        trajectory,
        axis=0
    )

    distances = np.linalg.norm(
        diffs,
        axis=1
    )

    print()
    print(
        name
    )

    print(
        "  mean frame motion:",
        np.mean(distances)
    )

    print(
        "  median frame motion:",
        np.median(distances)
    )

    print(
        "  max frame motion:",
        np.max(distances)
    )


print_motion(
    "A: raw pose",
    candidate_A
)

print_motion(
    "B: inverse raw pose",
    candidate_B
)

print_motion(
    "C: pose @ extrinsic",
    candidate_C
)

print_motion(
    "D: inverse(pose) @ extrinsic",
    candidate_D
)


# ============================================================
# Final
# ============================================================

print()
print("=" * 70)
print("DONE")
print("=" * 70)

print()
print(
    "Generated:"
)

print(
    "  pose_extrinsic_candidates.png"
)

print(
    "  pose_extrinsic_z.png"
)