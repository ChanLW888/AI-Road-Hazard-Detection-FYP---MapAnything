#!/usr/bin/env python3

"""
Poisson surface reconstruction for existing MapAnything outputs.

This script does NOT rerun MapAnything.

It takes the PLY files produced by the separate-camera
multiview MapAnything script and performs Poisson reconstruction
independently on each PLY.

IMPORTANT:
- CAM2 and CAM6 remain completely separate.
- RGB and LiDAR MapAnything outputs remain separate.
- No coordinate transforms.
- No registration.
- No merging.
- No voxel fusion.
- No MapAnything inference.

Expected input directory:

mapanything_separate_multiview_9000_9019_50m/
    mapanything_CAM2_multiview_rgb.ply
    mapanything_CAM2_multiview_lidar.ply
    mapanything_CAM6_multiview_rgb.ply
    mapanything_CAM6_multiview_lidar.ply

Outputs:

mapanything_separate_multiview_9000_9019_50m/
    poisson/
        mapanything_CAM2_multiview_rgb_poisson.ply
        mapanything_CAM2_multiview_lidar_poisson.ply
        mapanything_CAM6_multiview_rgb_poisson.ply
        mapanything_CAM6_multiview_lidar_poisson.ply
"""


import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


# ============================================================
# CONFIG
# ============================================================

# Change this if your MapAnything output directory is different.
INPUT_DIR = Path(
    "/home/lcha0115/bt60_scratch/lcha_data/"
    "map-anything/personal_data/hazard_test_101/"
    "mapanything_separate_multiview_9000_9019_50m"
)

OUTPUT_DIR = INPUT_DIR / "poisson"


# ------------------------------------------------------------
# Poisson parameters
# ------------------------------------------------------------

# Smaller voxel = more geometric detail but more memory/time.
VOXEL_SIZE = 0.05

# Poisson octree depth.
# 8-9 is a reasonable starting range for these point clouds.
POISSON_DEPTH = 9

# Search radius used when estimating normals.
NORMAL_RADIUS = 0.10

# Number of neighbours used for normal estimation.
NORMAL_MAX_NN = 100

# Number of nearest neighbours used to orient normals.
# IMPORTANT:
# Open3D versions differ in the exact API. This script uses
# the positional argument form for compatibility.
NORMAL_ORIENTATION_K = 100

# Remove the lowest-density Poisson vertices.
#
# 0.00 = keep everything
# 0.02 = remove lowest 2%
# 0.05 = remove lowest 5%
# 0.10 = remove lowest 10%
DENSITY_QUANTILE = 0.05

# Whether to voxel-downsample before Poisson.
USE_VOXEL_DOWNSAMPLE = True


# ============================================================
# INPUT FILES
# ============================================================

INPUT_FILES = [
    "mapanything_CAM2_multiview_rgb.ply",
    "mapanything_CAM2_multiview_lidar.ply",
    "mapanything_CAM6_multiview_rgb.ply",
    "mapanything_CAM6_multiview_lidar.ply",
]


# ============================================================
# POINT CLOUD PREPARATION
# ============================================================

def prepare_point_cloud(path):
    print()
    print("-" * 80)
    print(f"Loading: {path.name}")
    print("-" * 80)

    pcd = o3d.io.read_point_cloud(str(path))

    if pcd.is_empty():
        raise RuntimeError(f"Point cloud is empty: {path}")

    print(f"Input points: {len(pcd.points):,}")

    # --------------------------------------------------------
    # Remove invalid points
    # --------------------------------------------------------
    points = np.asarray(pcd.points)

    valid = np.isfinite(points).all(axis=1)

    if not np.all(valid):
        print(
            f"Removing "
            f"{np.count_nonzero(~valid):,} invalid points"
        )

        pcd = pcd.select_by_index(
            np.flatnonzero(valid)
        )

    if pcd.is_empty():
        raise RuntimeError(
            f"No valid points remain: {path}"
        )

    # --------------------------------------------------------
    # Voxel downsample
    # --------------------------------------------------------
    if USE_VOXEL_DOWNSAMPLE:
        print(
            f"Voxel downsampling: "
            f"{VOXEL_SIZE:.3f} m"
        )

        pcd = pcd.voxel_down_sample(
            voxel_size=VOXEL_SIZE
        )

        print(
            f"After downsampling: "
            f"{len(pcd.points):,}"
        )

    # --------------------------------------------------------
    # Normal estimation
    # --------------------------------------------------------
    print(
        f"Estimating normals "
        f"(radius={NORMAL_RADIUS:.3f}, "
        f"max_nn={NORMAL_MAX_NN})..."
    )

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=NORMAL_RADIUS,
            max_nn=NORMAL_MAX_NN,
        )
    )

    # --------------------------------------------------------
    # Normal orientation
    # --------------------------------------------------------
    print(
        f"Orienting normals "
        f"(k={NORMAL_ORIENTATION_K})..."
    )

    pcd.orient_normals_consistent_tangent_plane(
        NORMAL_ORIENTATION_K
    )

    # --------------------------------------------------------
    # Make sure normals are finite
    # --------------------------------------------------------
    normals = np.asarray(pcd.normals)

    valid_normals = np.isfinite(normals).all(axis=1)

    if not np.all(valid_normals):
        print(
            f"Removing "
            f"{np.count_nonzero(~valid_normals):,} "
            f"points with invalid normals"
        )

        pcd = pcd.select_by_index(
            np.flatnonzero(valid_normals)
        )

    return pcd


# ============================================================
# POISSON
# ============================================================

def run_poisson(pcd, input_name):
    print()
    print("-" * 80)
    print(f"Running Poisson: {input_name}")
    print("-" * 80)

    print(
        f"Poisson depth: {POISSON_DEPTH}"
    )

    mesh, densities = (
        o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd,
            depth=POISSON_DEPTH,
            linear_fit=False,
        )
    )

    print(
        f"Initial mesh vertices: "
        f"{len(mesh.vertices):,}"
    )

    print(
        f"Initial mesh triangles: "
        f"{len(mesh.triangles):,}"
    )

    # --------------------------------------------------------
    # Density filtering
    # --------------------------------------------------------
    densities = np.asarray(densities)

    if len(densities) > 0 and DENSITY_QUANTILE > 0:
        threshold = np.quantile(
            densities,
            DENSITY_QUANTILE,
        )

        print(
            f"Density threshold "
            f"(quantile={DENSITY_QUANTILE:.3f}): "
            f"{threshold:.6f}"
        )

        vertices_to_remove = (
            densities < threshold
        )

        mesh.remove_vertices_by_mask(
            vertices_to_remove
        )

        print(
            f"After density filtering:"
        )
        print(
            f"  vertices: "
            f"{len(mesh.vertices):,}"
        )
        print(
            f"  triangles: "
            f"{len(mesh.triangles):,}"
        )

    # --------------------------------------------------------
    # Keep vertex colours if available
    # --------------------------------------------------------
    if pcd.has_colors():
        print("Transferring point-cloud colours...")

        # Poisson normally preserves colours inconsistently
        # depending on Open3D version, so use nearest-neighbour
        # colour assignment from the original input cloud.
        source_points = np.asarray(pcd.points)
        source_colors = np.asarray(pcd.colors)

        mesh_points = np.asarray(mesh.vertices)

        if (
            len(source_points) > 0
            and len(source_colors) == len(source_points)
            and len(mesh_points) > 0
        ):
            source_pcd = o3d.geometry.PointCloud()
            source_pcd.points = (
                o3d.utility.Vector3dVector(
                    source_points
                )
            )
            source_pcd.colors = (
                o3d.utility.Vector3dVector(
                    source_colors
                )
            )

            kdtree = o3d.geometry.KDTreeFlann(
                source_pcd
            )

            mesh_colors = np.zeros(
                (len(mesh_points), 3),
                dtype=np.float64,
            )

            for i, point in enumerate(mesh_points):
                _, indices, _ = kdtree.search_knn_vector_3d(
                    point,
                    1,
                )

                if indices:
                    mesh_colors[i] = source_colors[
                        indices[0]
                    ]

            mesh.vertex_colors = (
                o3d.utility.Vector3dVector(
                    mesh_colors
                )
            )

    return mesh


# ============================================================
# PROCESS ONE FILE
# ============================================================

def process_file(input_path):
    output_name = (
        input_path.stem
        + "_poisson.ply"
    )

    output_path = (
        OUTPUT_DIR
        / output_name
    )

    print()
    print("=" * 80)
    print(f"PROCESSING {input_path.name}")
    print("=" * 80)

    if output_path.exists():
        print(
            f"Output already exists: "
            f"{output_path}"
        )
        print("Overwriting.")

    pcd = prepare_point_cloud(
        input_path
    )

    mesh = run_poisson(
        pcd,
        input_path.name,
    )

    # --------------------------------------------------------
    # Save mesh as PLY
    # --------------------------------------------------------
    print()
    print(
        f"Saving Poisson mesh -> "
        f"{output_path}"
    )

    ok = o3d.io.write_triangle_mesh(
        str(output_path),
        mesh,
        write_ascii=False,
        compressed=False,
        write_vertex_normals=True,
        write_vertex_colors=mesh.has_vertex_colors(),
    )

    if not ok:
        raise RuntimeError(
            f"Failed to save mesh: "
            f"{output_path}"
        )

    print(
        f"Saved: "
        f"{len(mesh.vertices):,} vertices, "
        f"{len(mesh.triangles):,} triangles"
    )

    return output_path


# ============================================================
# MAIN
# ============================================================

def main():
    global POISSON_DEPTH
    global VOXEL_SIZE
    global NORMAL_RADIUS
    global NORMAL_MAX_NN
    global NORMAL_ORIENTATION_K
    global DENSITY_QUANTILE
    global USE_VOXEL_DOWNSAMPLE

    parser = argparse.ArgumentParser(
        description=(
            "Run Poisson surface reconstruction "
            "independently on existing "
            "MapAnything PLY outputs."
        )
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=INPUT_DIR,
        help="Directory containing MapAnything PLY files.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <input-dir>/poisson.",
    )

    parser.add_argument(
        "--depth",
        type=int,
        default=POISSON_DEPTH,
        help="Poisson octree depth.",
    )

    parser.add_argument(
        "--voxel",
        type=float,
        default=VOXEL_SIZE,
        help="Voxel size before Poisson.",
    )

    parser.add_argument(
        "--normal-radius",
        type=float,
        default=NORMAL_RADIUS,
        help="Normal estimation radius.",
    )

    parser.add_argument(
        "--normal-nn",
        type=int,
        default=NORMAL_MAX_NN,
        help="Maximum neighbours for normal estimation.",
    )

    parser.add_argument(
        "--orientation-k",
        type=int,
        default=NORMAL_ORIENTATION_K,
        help="Neighbour count for normal orientation.",
    )

    parser.add_argument(
        "--density",
        type=float,
        default=DENSITY_QUANTILE,
        help="Poisson density quantile to remove.",
    )

    parser.add_argument(
        "--no-downsample",
        action="store_true",
        help="Disable voxel downsampling before Poisson.",
    )

    args = parser.parse_args()

    input_dir = args.input_dir

    if args.output_dir is None:
        output_dir = input_dir / "poisson"
    else:
        output_dir = args.output_dir

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Apply CLI settings.
    POISSON_DEPTH = args.depth
    VOXEL_SIZE = args.voxel
    NORMAL_RADIUS = args.normal_radius
    NORMAL_MAX_NN = args.normal_nn
    NORMAL_ORIENTATION_K = args.orientation_k
    DENSITY_QUANTILE = args.density
    USE_VOXEL_DOWNSAMPLE = not args.no_downsample

    print("=" * 80)
    print("MAPANYTHING -> POISSON SURFACE RECONSTRUCTION")
    print("=" * 80)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print()
    print("CAM2/CAM6: SEPARATE")
    print("RGB/LiDAR: SEPARATE")
    print("Coordinate transforms: NONE")
    print("PLY merging: NONE")
    print()
    print(f"Voxel size: {VOXEL_SIZE}")
    print(f"Poisson depth: {POISSON_DEPTH}")
    print(f"Normal radius: {NORMAL_RADIUS}")
    print(f"Normal max NN: {NORMAL_MAX_NN}")
    print(
        f"Normal orientation k: "
        f"{NORMAL_ORIENTATION_K}"
    )
    print(
        f"Density quantile: "
        f"{DENSITY_QUANTILE}"
    )
    print(
        f"Voxel downsampling: "
        f"{USE_VOXEL_DOWNSAMPLE}"
    )

    # --------------------------------------------------------
    # Find files
    # --------------------------------------------------------
    files = []

    for filename in INPUT_FILES:
        path = input_dir / filename

        if path.exists():
            files.append(path)
        else:
            print()
            print(
                f"WARNING: missing input: "
                f"{path}"
            )

    if not files:
        raise RuntimeError(
            "None of the expected MapAnything "
            "PLY files were found."
        )

    print()
    print(
        f"Found {len(files)} "
        f"MapAnything PLY files."
    )

    # --------------------------------------------------------
    # Process independently
    # --------------------------------------------------------
    results = []

    for path in files:
        results.append(
            process_file(path)
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    print()
    print("=" * 80)
    print("ALL POISSON RECONSTRUCTIONS COMPLETE")
    print("=" * 80)

    for path in results:
        print(path)

    print()
    print(
        "CAM2 and CAM6 were never merged "
        "or transformed."
    )


if __name__ == "__main__":
    main()
