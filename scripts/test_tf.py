import numpy as np
import open3d as o3d
from pathlib import Path
from scipy.spatial.transform import Rotation


# ============================================================
# CONFIG
# ============================================================

LIDAR_PLY = Path(
    "mapanything_3camera_tf_output/lidar_reference.ply"
)

CAMERA_PLY = Path(
    "mapanything_3camera_tf_output/CAM2_raw_camera.ply"
)

OUTPUT_DIR = Path(
    "tf_translation_test"
)

OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# CAM2 TF
# velodyne -> cam2_optical_frame
# ============================================================

translation = np.array([
    -0.35812314453904737,
     0.7418722578139009,
    -0.9622312541336254,
])


quaternion_xyzw = np.array([
    -0.696591590508534,
    -0.30318163432606277,
     0.27662795664738776,
     0.5884879151191293,
])


R = Rotation.from_quat(
    quaternion_xyzw
).as_matrix()


print("=" * 80)
print("CAM2 TF TRANSLATION TEST")
print("=" * 80)

print("\nRotation:")
print(R)

print("\nTranslation:")
print(translation)


# ============================================================
# LOAD CAMERA CLOUD
# ============================================================

print("\nLoading camera cloud:")
print(CAMERA_PLY)

cam = o3d.io.read_point_cloud(
    str(CAMERA_PLY)
)

points = np.asarray(
    cam.points
)

print(
    "Camera points:",
    len(points)
)


if len(points) == 0:
    raise RuntimeError(
        "Camera point cloud is empty."
    )


# ============================================================
# DIRECT TRANSFORMATION
# ============================================================

direct = (
    points @ R.T
    + translation
)


# ============================================================
# ROTATION ONLY
# ============================================================

rotation_only = (
    points @ R.T
)


# ============================================================
# SAVE DIRECT
# ============================================================

direct_pcd = o3d.geometry.PointCloud()

direct_pcd.points = (
    o3d.utility.Vector3dVector(
        direct
    )
)

if cam.has_colors():

    direct_pcd.colors = (
        cam.colors
    )

o3d.io.write_point_cloud(
    str(
        OUTPUT_DIR /
        "cam2_direct.ply"
    ),
    direct_pcd
)


# ============================================================
# SAVE ROTATION ONLY
# ============================================================

rotation_pcd = o3d.geometry.PointCloud()

rotation_pcd.points = (
    o3d.utility.Vector3dVector(
        rotation_only
    )
)

if cam.has_colors():

    rotation_pcd.colors = (
        cam.colors
    )

o3d.io.write_point_cloud(
    str(
        OUTPUT_DIR /
        "cam2_rotation_only.ply"
    ),
    rotation_pcd
)


# ============================================================
# PRINT CENTROIDS
# ============================================================

print("\n" + "=" * 80)
print("CENTROID TEST")
print("=" * 80)

print(
    "\nRaw camera centroid:"
)

print(
    points.mean(axis=0)
)

print(
    "\nRotation-only centroid:"
)

print(
    rotation_only.mean(axis=0)
)

print(
    "\nDirect centroid:"
)

print(
    direct.mean(axis=0)
)

print(
    "\nDifference:"
)

print(
    direct.mean(axis=0)
    -
    rotation_only.mean(axis=0)
)

print(
    "\nExpected translation:"
)

print(
    translation
)


# ============================================================
# LOAD LIDAR
# ============================================================

print("\n" + "=" * 80)
print("LOADING LIDAR")
print("=" * 80)

print(
    LIDAR_PLY
)

lidar = o3d.io.read_point_cloud(
    str(LIDAR_PLY)
)

lidar_points = np.asarray(
    lidar.points
)

print(
    "LiDAR points:",
    len(lidar_points)
)


if len(lidar_points) == 0:

    raise RuntimeError(
        "LiDAR point cloud is empty."
    )


# ============================================================
# BUILD KD TREE
# ============================================================

lidar_tree = o3d.geometry.KDTreeFlann(
    lidar
)


# ============================================================
# NEAREST NEIGHBOUR TEST
# ============================================================

def nearest_distances(
    points,
    lidar_tree,
):

    distances = []

    for p in points:

        k, idx, d2 = (
            lidar_tree.search_knn_vector_3d(
                p,
                1
            )
        )

        if k > 0:

            distances.append(
                np.sqrt(d2[0])
            )

    return np.asarray(
        distances
    )


print("\n" + "=" * 80)
print("NEAREST NEIGHBOUR TEST")
print("=" * 80)


print(
    "\nTesting DIRECT..."
)

direct_dist = nearest_distances(
    direct,
    lidar_tree
)


print(
    "Testing ROTATION ONLY..."
)

rotation_dist = nearest_distances(
    rotation_only,
    lidar_tree
)


# ============================================================
# STATISTICS
# ============================================================

def report(
    name,
    d,
):

    print("\n" + "-" * 60)

    print(name)

    print("-" * 60)

    print(
        "Mean:",
        np.mean(d),
        "m"
    )

    print(
        "Median:",
        np.median(d),
        "m"
    )

    print(
        "90th percentile:",
        np.percentile(d, 90),
        "m"
    )

    print(
        "95th percentile:",
        np.percentile(d, 95),
        "m"
    )

    print(
        "< 0.10 m:",
        np.mean(d < 0.10) * 100,
        "%"
    )

    print(
        "< 0.20 m:",
        np.mean(d < 0.20) * 100,
        "%"
    )

    print(
        "< 0.50 m:",
        np.mean(d < 0.50) * 100,
        "%"
    )


report(
    "DIRECT",
    direct_dist
)

report(
    "ROTATION ONLY",
    rotation_dist
)


# ============================================================
# FINAL COMPARISON
# ============================================================

print("\n" + "=" * 80)
print("FINAL RESULT")
print("=" * 80)

direct_median = np.median(
    direct_dist
)

rotation_median = np.median(
    rotation_dist
)


if direct_median < rotation_median:

    print(
        "\nDIRECT TRANSFORMATION "
        "fits LiDAR better."
    )

else:

    print(
        "\nROTATION-ONLY transformation "
        "fits LiDAR better."
    )


print(
    "\nDirect median:",
    direct_median
)

print(
    "Rotation-only median:",
    rotation_median
)

print(
    "\nOutputs:"
)

print(
    OUTPUT_DIR /
    "cam2_direct.ply"
)

print(
    OUTPUT_DIR /
    "cam2_rotation_only.ply"
)