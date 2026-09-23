import os
import glob
import numpy as np
import open3d as o3d

INPUT_DIR = "outputs_ply"
OUTPUT_DIR = "outputs/ply"

os.makedirs(OUTPUT_DIR, exist_ok=True)

files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.npy")))

print(f"Found {len(files)} point files")

for npy_file in files:

    points = np.load(npy_file)

    # Remove invalid points
    valid = np.isfinite(points).all(axis=1)
    points = points[valid]

    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Output filename
    name = os.path.splitext(os.path.basename(npy_file))[0]
    ply_file = os.path.join(OUTPUT_DIR, name + ".ply")

    o3d.io.write_point_cloud(
        ply_file,
        pcd
    )

    print(f"{npy_file} -> {ply_file} ({len(points):,} points)")

print("Done.")