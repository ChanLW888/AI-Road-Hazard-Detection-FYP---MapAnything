from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData


# ============================================================
# PATHS
# ============================================================

DATA_ROOT = Path(
    "/home/lcha0115/bt60_scratch/lcha_data/map-anything/"
    "personal_data/camera_lidar_test_101/extracted"
)

LIDAR_DIR = DATA_ROOT / "lidar"


# ============================================================
# CAMERA CALIBRATION
# ============================================================

CAMERA_INTRINSICS = {
    "CAM1": np.array([
        [517.681288252460, 0.0, 948.547972574546],
        [0.0, 517.117613949968, 620.752826554990],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32),

    "CAM2": np.array([
        [510.651676055731, 0.0, 949.416735545071],
        [0.0, 510.456423883940, 607.626749742175],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32),

    "CAM6": np.array([
        [502.623384784943, 0.0, 963.262380398809],
        [0.0, 502.141680171936, 602.114058539104],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32),
}


# Assumption:
# These transforms convert LiDAR-frame XYZ -> camera-frame XYZ.
#
# If your calibration matrices instead represent camera -> LiDAR,
# these MUST be inverted before projection.
CAMERA_EXTRINSICS = {
    "CAM1": np.array([
        [-0.74075182,  0.67164194,  0.01355925, -0.02493503],
        [ 0.24699625,  0.29107072, -0.92426765, -0.10583372],
        [-0.62472362, -0.68130386, -0.38150420, -1.42902479],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float32),

    "CAM2": np.array([
        [ 0.63311790,  0.77355266, -0.02789272, -0.41198233],
        [ 0.12521772, -0.13791188, -0.98249724, -0.67023320],
        [-0.76386009,  0.61854393, -0.18417698, -0.66565741],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float32),

    "CAM6": np.array([
        [-0.70679283, -0.70736252, 0.00906393,  0.88808945],
        [-0.15894889,  0.14630976, -0.97638553, -0.71249618],
        [ 0.68933239, -0.69154300, -0.21584518, -0.60088861],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float32),
}


# ============================================================
# LIDAR INDEX
# ============================================================

_LIDAR_TIMESTAMPS = None


def _build_lidar_index():
    global _LIDAR_TIMESTAMPS

    if _LIDAR_TIMESTAMPS is None:

        records = []

        for path in LIDAR_DIR.glob("*.ply"):

            try:
                timestamp = int(path.stem)
            except ValueError:
                continue

            records.append(
                (timestamp, path)
            )

        records.sort(
            key=lambda x: x[0]
        )

        _LIDAR_TIMESTAMPS = records

    return _LIDAR_TIMESTAMPS


# ============================================================
# TIMESTAMP MATCHING
# ============================================================

def _nearest_lidar(
    timestamp,
    max_timestamp_delta=None,
):
    records = _build_lidar_index()

    if not records:
        return None, None

    stamps = np.asarray(
        [x[0] for x in records],
        dtype=np.int64,
    )

    timestamp = int(timestamp)

    i = int(
        np.searchsorted(
            stamps,
            timestamp,
        )
    )

    candidates = []

    if i > 0:
        candidates.append(i - 1)

    if i < len(records):
        candidates.append(i)

    if not candidates:
        return None, None

    best = min(
        candidates,
        key=lambda j: abs(
            int(stamps[j]) - timestamp
        ),
    )

    delta = abs(
        int(stamps[best]) - timestamp
    )

    if (
        max_timestamp_delta is not None
        and max_timestamp_delta >= 0
        and delta > max_timestamp_delta
    ):
        return None, delta

    return records[best][1], delta


# ============================================================
# LOAD LIDAR
# ============================================================

def load_lidar(
    timestamp,
    max_timestamp_delta=None,
    return_intensity=False,
):
    path, delta = _nearest_lidar(
        timestamp,
        max_timestamp_delta=max_timestamp_delta,
    )

    if path is None:
        return None

    ply = PlyData.read(
        str(path)
    )

    vertex = ply["vertex"].data

    xyz = np.stack(
        [
            vertex["x"],
            vertex["y"],
            vertex["z"],
        ],
        axis=1,
    ).astype(
        np.float32
    )

    if return_intensity:
        intensity = (
            np.asarray(vertex["intensity"], dtype=np.float32)
            if "intensity" in vertex.dtype.names
            else np.zeros(len(xyz), dtype=np.float32)
        )
        return xyz, intensity

    return xyz


# ============================================================
# PROJECT LIDAR INTO CAMERA
# ============================================================

def project_lidar(
    lidar_xyz,
    camera,
    height=1200,
    width=1920,
    min_depth=1.0,
    max_depth=80.0,
    intensity=None,
    return_intensity=False,
):
    K = CAMERA_INTRINSICS[camera]
    T = CAMERA_EXTRINSICS[camera]

    if lidar_xyz is None:
        return (None, None, None) if return_intensity else (None, None)

    if len(lidar_xyz) == 0:
        result = (
            np.zeros(
                (height, width),
                dtype=np.float32,
            ),
            np.zeros(
                (height, width),
                dtype=bool,
            ),
        )
        return (*result, np.zeros((height, width), dtype=np.float32)) if return_intensity else result

    points_h = np.concatenate(
        [
            lidar_xyz,
            np.ones(
                (len(lidar_xyz), 1),
                dtype=np.float32,
            ),
        ],
        axis=1,
    )

    # LiDAR -> camera
    cam = points_h @ T.T

    X = cam[:, 0]
    Y = cam[:, 1]
    Z = cam[:, 2]

    valid = (
        np.isfinite(X)
        & np.isfinite(Y)
        & np.isfinite(Z)
        & (Z > min_depth)
        & (Z < max_depth)
    )

    X = X[valid]
    Y = Y[valid]
    Z = Z[valid]
    if intensity is not None:
        intensity = np.asarray(intensity, dtype=np.float32)[valid]

    if len(Z) == 0:
        result = (
            np.zeros(
                (height, width),
                dtype=np.float32,
            ),
            np.zeros(
                (height, width),
                dtype=bool,
            ),
        )
        return (*result, np.zeros((height, width), dtype=np.float32)) if return_intensity else result

    u = np.rint(
        K[0, 0] * X / Z
        + K[0, 2]
    ).astype(
        np.int32
    )

    v = np.rint(
        K[1, 1] * Y / Z
        + K[1, 2]
    ).astype(
        np.int32
    )

    valid = (
        (u >= 0)
        & (u < width)
        & (v >= 0)
        & (v < height)
    )

    u = u[valid]
    v = v[valid]
    Z = Z[valid]
    if intensity is not None:
        intensity = intensity[valid]

    depth = np.zeros(
        (height, width),
        dtype=np.float32,
    )

    mask = np.zeros(
        (height, width),
        dtype=bool,
    )

    if len(Z) == 0:
        return depth, mask

    # Keep closest LiDAR point when multiple
    # points land on the same pixel.
    order = np.argsort(Z)

    u = u[order]
    v = v[order]
    Z = Z[order]
    if intensity is not None:
        intensity = intensity[order]

    flat = (
        v * width
        + u
    )

    keep = np.ones(
        len(flat),
        dtype=bool,
    )

    if len(flat) > 1:
        keep[1:] = (
            flat[1:]
            != flat[:-1]
        )

    u = u[keep]
    v = v[keep]
    Z = Z[keep]
    if intensity is not None:
        intensity = intensity[keep]

    depth[v, u] = Z
    mask[v, u] = True

    if return_intensity:
        intensity_map = np.zeros((height, width), dtype=np.float32)
        if intensity is not None:
            intensity_map[v, u] = intensity
        return depth, mask, intensity_map
    return depth, mask


# ============================================================
# PUBLIC DEPTH API
# ============================================================

def get_lidar_depth(
    timestamp,
    camera,
    device,
    max_timestamp_delta=None,
    return_info=False,
):
    path, timestamp_delta = _nearest_lidar(
        timestamp,
        max_timestamp_delta=max_timestamp_delta,
    )

    info = {
        "timestamp": int(timestamp),
        "camera": str(camera),
        "timestamp_delta": (
            int(timestamp_delta)
            if timestamp_delta is not None
            else None
        ),
        "matched": path is not None,
        "lidar_path": str(path) if path is not None else "",
    }

    xyz = load_lidar(
        timestamp,
        max_timestamp_delta=max_timestamp_delta,
    )

    if xyz is None:
        if return_info:
            return None, None, info
        return None, None

    depth, mask = project_lidar(
        xyz,
        camera,
    )

    result = (
        torch.from_numpy(
            depth
        ).to(device),
        torch.from_numpy(
            mask
        ).to(device),
    )

    if return_info:
        return *result, info

    return result


def get_lidar_projection_with_intensity(timestamp, camera, max_timestamp_delta=None):
    """Return native projection and reflectance image for visual calibration checks."""
    loaded = load_lidar(timestamp, max_timestamp_delta, return_intensity=True)
    if loaded is None:
        return None, None, None
    xyz, intensity = loaded
    return project_lidar(xyz, camera, intensity=intensity, return_intensity=True)


# ============================================================
# LIDAR DEPTH LOSS
# ============================================================

def lidar_depth_loss(
    predicted_depth,
    lidar_depth,
    lidar_mask,
):
    if predicted_depth.ndim == 3:
        predicted_depth = (
            predicted_depth[0]
        )

    valid = (
        lidar_mask.bool()
        & torch.isfinite(
            predicted_depth
        )
        & torch.isfinite(
            lidar_depth
        )
        & (predicted_depth > 0)
        & (lidar_depth > 0)
    )

    if not valid.any():
        return (
            predicted_depth.sum()
            * 0.0
        )

    return F.smooth_l1_loss(
        torch.log(
            predicted_depth[
                valid
            ].clamp_min(1e-3)
        ),
        torch.log(
            lidar_depth[
                valid
            ].clamp_min(1e-3)
        ),
    )
