#!/usr/bin/env python3

"""
Extended Poisson parameter sweep for existing MapAnything PLYs.

IMPORTANT:
- MapAnything is NOT run.
- Images and LiDAR are NOT loaded.
- Existing MapAnything PLYs are loaded once and reused.
- Each parameter combination is written to its own folder.
- One failed test does not stop the entire sweep.
- A CSV summary is written at the end.

Expected input files:
    mapanything_raw_CAM2.ply
    mapanything_lidar_CAM2.ply
    mapanything_raw_CAM6.ply
    mapanything_lidar_CAM6.ply

Output:
    <OUTPUT_DIR>/poisson_tests_extended/
        test_001_...
        test_002_...
        ...
        poisson_test_summary.csv
"""

import os
import csv
import time
import traceback
from pathlib import Path

import numpy as np
import open3d as o3d


# ============================================================
# CONFIGURATION
# ============================================================

REFERENCE_INDEX = 9000
NUM_FRAMES = 20
LIDAR_RANGE = 50

OUTPUT_DIR = Path(
    f"/home/lcha0115/bt60_scratch/lcha_data/map-anything/"
    f"personal_data/hazard_test_101/"
    f"mapanything_ablation_{REFERENCE_INDEX}_"
    f"{REFERENCE_INDEX + NUM_FRAMES - 1}_{LIDAR_RANGE}m"
)

POISSON_TEST_DIR = OUTPUT_DIR / "poisson_tests_extended"

# Existing MapAnything PLYs.
PLY_FILES = {
    "raw_CAM2": OUTPUT_DIR / "mapanything_raw_CAM2.ply",
    "lidar_CAM2": OUTPUT_DIR / "mapanything_lidar_CAM2.ply",
    "raw_CAM6": OUTPUT_DIR / "mapanything_raw_CAM6.ply",
    "lidar_CAM6": OUTPUT_DIR / "mapanything_lidar_CAM6.ply",
}


# ============================================================
# PARAMETER SWEEP
# ============================================================
#
# The sweep is deliberately curated rather than doing every
# possible Cartesian combination.
#
# Main things being tested:
#
#   VOXEL_SIZE
#       Controls point-cloud downsampling.
#
#   DEPTH
#       Controls Poisson reconstruction resolution.
#
#   NORMAL_RADIUS
#       Controls the neighbourhood used to estimate normals.
#
#   NORMAL_MAX_NN
#       Maximum number of neighbours used for normal estimation.
#
#   DENSITY_QUANTILE
#       Removes low-density Poisson vertices.
#
# Total: 48 tests
#
# These are designed around the current problem:
#   - sidewalk should remain reasonably detailed
#   - houses/trees should become less blobbed
#   - avoid excessively smoothing everything
# ============================================================

POISSON_TESTS = [

    # --------------------------------------------------------
    # GROUP A: Current / baseline
    # --------------------------------------------------------

    {
        "name": "current_003_d11_r003_n30_q010",
        "voxel": 0.03,
        "depth": 11,
        "radius": 0.03,
        "max_nn": 30,
        "density": 0.10,
    },

    {
        "name": "baseline_005_d09_r006_n30_q005",
        "voxel": 0.05,
        "depth": 9,
        "radius": 0.06,
        "max_nn": 30,
        "density": 0.05,
    },

    {
        "name": "baseline_003_d09_r005_n30_q005",
        "voxel": 0.03,
        "depth": 9,
        "radius": 0.05,
        "max_nn": 30,
        "density": 0.05,
    },

    {
        "name": "baseline_004_d09_r005_n30_q003",
        "voxel": 0.04,
        "depth": 9,
        "radius": 0.05,
        "max_nn": 30,
        "density": 0.03,
    },


    # --------------------------------------------------------
    # GROUP B: Normal radius sweep
    # --------------------------------------------------------
    #
    # Same basic reconstruction, changing only normal radius.
    # Useful for finding whether large neighbourhoods are
    # causing the trees/houses to become overly smooth.
    # --------------------------------------------------------

    {
        "name": "radius_003",
        "voxel": 0.05,
        "depth": 9,
        "radius": 0.03,
        "max_nn": 30,
        "density": 0.05,
    },

    {
        "name": "radius_004",
        "voxel": 0.05,
        "depth": 9,
        "radius": 0.04,
        "max_nn": 30,
        "density": 0.05,
    },

    {
        "name": "radius_005",
        "voxel": 0.05,
        "depth": 9,
        "radius": 0.05,
        "max_nn": 30,
        "density": 0.05,
    },

    {
        "name": "radius_006",
        "voxel": 0.05,
        "depth": 9,
        "radius": 0.06,
        "max_nn": 30,
        "density": 0.05,
    },

    {
        "name": "radius_008",
        "voxel": 0.05,
        "depth": 9,
        "radius": 0.08,
        "max_nn": 30,
        "density": 0.05,
    },

    {
        "name": "radius_010",
        "voxel": 0.05,
        "depth": 9,
        "radius": 0.10,
        "max_nn": 30,
        "density": 0.05,
    },


    # --------------------------------------------------------
    # GROUP C: Very local normals
    # --------------------------------------------------------
    #
    # Designed specifically to preserve local structure.
    # --------------------------------------------------------

    {
        "name": "verylocal_003_n20",
        "voxel": 0.05,
        "depth": 9,
        "radius": 0.03,
        "max_nn": 20,
        "density": 0.05,
    },

    {
        "name": "verylocal_004_n20",
        "voxel": 0.05,
        "depth": 9,
        "radius": 0.04,
        "max_nn": 20,
        "density": 0.05,
    },

    {
        "name": "verylocal_005_n20",
        "voxel": 0.05,
        "depth": 9,
        "radius": 0.05,
        "max_nn": 20,
        "density": 0.05,
    },

    {
        "name": "verylocal_003_n30",
        "voxel": 0.05,
        "depth": 9,
        "radius": 0.03,
        "max_nn": 30,
        "density": 0.05,
    },

    {
        "name": "verylocal_004_n30",
        "voxel": 0.05,
        "depth": 9,
        "radius": 0.04,
        "max_nn": 30,
        "density": 0.05,
    },


    # --------------------------------------------------------
    # GROUP D: Max-NN sweep
    # --------------------------------------------------------

    {
        "name": "maxnn_020",
        "voxel": 0.05,
        "depth": 9,
        "radius": 0.06,
        "max_nn": 20,
        "density": 0.05,
    },

    {
        "name": "maxnn_030",
        "voxel": 0.05,
        "depth": 9,
        "radius": 0.06,
        "max_nn": 30,
        "density": 0.05,
    },

    {
        "name": "maxnn_050",
        "voxel": 0.05,
        "depth": 9,
        "radius": 0.06,
        "max_nn": 50,
        "density": 0.05,
    },

    {
        "name": "maxnn_075",
        "voxel": 0.05,
        "depth": 9,
        "radius": 0.06,
        "max_nn": 75,
        "density": 0.05,
    },

    {
        "name": "maxnn_100",
        "voxel": 0.05,
        "depth": 9,
        "radius": 0.06,
        "max_nn": 100,
        "density": 0.05,
    },


    # --------------------------------------------------------
    # GROUP E: Poisson depth sweep
    # --------------------------------------------------------

    {
        "name": "depth_008",
        "voxel": 0.05,
        "depth": 8,
        "radius": 0.06,
        "max_nn": 30,
        "density": 0.05,
    },

    {
        "name": "depth_009",
        "voxel": 0.05,
        "depth": 9,
        "radius": 0.06,
        "max_nn": 30,
        "density": 0.05,
    },

    {
        "name": "depth_010",
        "voxel": 0.05,
        "depth": 10,
        "radius": 0.06,
        "max_nn": 30,
        "density": 0.05,
    },

    {
        "name": "depth_011",
        "voxel": 0.05,
        "depth": 11,
        "radius": 0.06,
        "max_nn": 30,
        "density": 0.05,
    },


    # --------------------------------------------------------
    # GROUP F: Fine voxel sizes
    # --------------------------------------------------------

    {
        "name": "fine_003_d09",
        "voxel": 0.03,
        "depth": 9,
        "radius": 0.05,
        "max_nn": 30,
        "density": 0.05,
    },

    {
        "name": "fine_003_d10",
        "voxel": 0.03,
        "depth": 10,
        "radius": 0.05,
        "max_nn": 30,
        "density": 0.05,
    },

    {
        "name": "fine_003_d11",
        "voxel": 0.03,
        "depth": 11,
        "radius": 0.05,
        "max_nn": 30,
        "density": 0.05,
    },

    {
        "name": "fine_004_d09",
        "voxel": 0.04,
        "depth": 9,
        "radius": 0.05,
        "max_nn": 30,
        "density": 0.05,
    },

    {
        "name": "fine_004_d10",
        "voxel": 0.04,
        "depth": 10,
        "radius": 0.05,
        "max_nn": 30,
        "density": 0.05,
    },

    {
        "name": "fine_004_d11",
        "voxel": 0.04,
        "depth": 11,
        "radius": 0.05,
        "max_nn": 30,
        "density": 0.05,
    },


    # --------------------------------------------------------
    # GROUP G: Coarser voxel sizes
    # --------------------------------------------------------

    {
        "name": "coarse_007_d08",
        "voxel": 0.07,
        "depth": 8,
        "radius": 0.07,
        "max_nn": 30,
        "density": 0.05,
    },

    {
        "name": "coarse_007_d09",
        "voxel": 0.07,
        "depth": 9,
        "radius": 0.07,
        "max_nn": 30,
        "density": 0.05,
    },

    {
        "name": "coarse_007_d10",
        "voxel": 0.07,
        "depth": 10,
        "radius": 0.07,
        "max_nn": 30,
        "density": 0.05,
    },

    {
        "name": "coarse_010_d08",
        "voxel": 0.10,
        "depth": 8,
        "radius": 0.10,
        "max_nn": 30,
        "density": 0.05,
    },

    {
        "name": "coarse_010_d09",
        "voxel": 0.10,
        "depth": 9,
        "radius": 0.10,
        "max_nn": 30,
        "density": 0.05,
    },


    # --------------------------------------------------------
    # GROUP H: Density filtering
    # --------------------------------------------------------

    {
        "name": "density_000",
        "voxel": 0.05,
        "depth": 9,
        "radius": 0.06,
        "max_nn": 30,
        "density": 0.00,
    },

    {
        "name": "density_002",
        "voxel": 0.05,
        "depth": 9,
        "radius": 0.06,
        "max_nn": 30,
        "density": 0.02,
    },

    {
        "name": "density_005",
        "voxel": 0.05,
        "depth": 9,
        "radius": 0.06,
        "max_nn": 30,
        "density": 0.05,
    },

    {
        "name": "density_010",
        "voxel": 0.05,
        "depth": 9,
        "radius": 0.06,
        "max_nn": 30,
        "density": 0.10,
    },

    {
        "name": "density_015",
        "voxel": 0.05,
        "depth": 9,
        "radius": 0.06,
        "max_nn": 30,
        "density": 0.15,
    },


    # --------------------------------------------------------
    # GROUP I: Balanced combinations
    # --------------------------------------------------------

    {
        "name": "balanced_A",
        "voxel": 0.04,
        "depth": 9,
        "radius": 0.05,
        "max_nn": 30,
        "density": 0.03,
    },

    {
        "name": "balanced_B",
        "voxel": 0.04,
        "depth": 10,
        "radius": 0.05,
        "max_nn": 30,
        "density": 0.03,
    },

    {
        "name": "balanced_C",
        "voxel": 0.04,
        "depth": 10,
        "radius": 0.04,
        "max_nn": 20,
        "density": 0.03,
    },

    {
        "name": "balanced_D",
        "voxel": 0.04,
        "depth": 9,
        "radius": 0.04,
        "max_nn": 20,
        "density": 0.02,
    },

    {
        "name": "balanced_E",
        "voxel": 0.03,
        "depth": 10,
        "radius": 0.04,
        "max_nn": 20,
        "density": 0.02,
    },

    {
        "name": "balanced_F",
        "voxel": 0.05,
        "depth": 10,
        "radius": 0.05,
        "max_nn": 30,
        "density": 0.03,
    },


    # --------------------------------------------------------
    # GROUP J: Aggressive local/detail configurations
    # --------------------------------------------------------

    {
        "name": "detail_A",
        "voxel": 0.03,
        "depth": 10,
        "radius": 0.03,
        "max_nn": 20,
        "density": 0.02,
    },

    {
        "name": "detail_B",
        "voxel": 0.03,
        "depth": 10,
        "radius": 0.04,
        "max_nn": 20,
        "density": 0.02,
    },

    {
        "name": "detail_C",
        "voxel": 0.03,
        "depth": 11,
        "radius": 0.03,
        "max_nn": 20,
        "density": 0.02,
    },

    {
        "name": "detail_D",
        "voxel": 0.04,
        "depth": 10,
        "radius": 0.04,
        "max_nn": 20,
        "density": 0.02,
    },

    {
        "name": "detail_E",
        "voxel": 0.04,
        "depth": 11,
        "radius": 0.04,
        "max_nn": 20,
        "density": 0.02,
    },

    {
        "name": "detail_F",
        "voxel": 0.05,
        "depth": 10,
        "radius": 0.04,
        "max_nn": 20,
        "density": 0.02,
    },
]


# ============================================================
# HELPERS
# ============================================================

def load_existing_ply(path):
    """Load an existing MapAnything PLY once."""

    print(f"Loading: {path}")

    if not path.exists():
        raise FileNotFoundError(f"PLY not found: {path}")

    pcd = o3d.io.read_point_cloud(str(path))

    points = np.asarray(pcd.points)

    if len(points) == 0:
        raise RuntimeError(f"PLY contains no points: {path}")

    colors = np.asarray(pcd.colors)

    if len(colors) == len(points):
        colors = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
    else:
        colors = np.full(
            (len(points), 3),
            128,
            dtype=np.uint8,
        )

    print(f"  Points: {len(points):,}")

    return points.astype(np.float64), colors


def transfer_colors_nearest(mesh, source_pcd, source_colors):
    """Transfer RGB from downsampled source points to mesh vertices."""

    mesh_points = np.asarray(mesh.vertices)

    if len(mesh_points) == 0:
        return

    kdtree = o3d.geometry.KDTreeFlann(source_pcd)

    mesh_colors = np.zeros(
        (len(mesh_points), 3),
        dtype=np.float64,
    )

    for i, point in enumerate(mesh_points):

        _, idx, _ = kdtree.search_knn_vector_3d(
            point,
            1,
        )

        if idx:
            mesh_colors[i] = source_colors[idx[0]] / 255.0

    mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_colors)


def run_poisson(
    points,
    colors,
    output_path,
    test,
    label,
):
    """
    Run one Poisson reconstruction.

    Returns statistics for the CSV.
    """

    voxel = test["voxel"]
    depth = test["depth"]
    radius = test["radius"]
    max_nn = test["max_nn"]
    density_quantile = test["density"]

    start_time = time.time()

    print()
    print("=" * 80)
    print(f"{label}")
    print("=" * 80)

    print(
        f"voxel={voxel:.3f}, "
        f"depth={depth}, "
        f"radius={radius:.3f}, "
        f"max_nn={max_nn}, "
        f"density={density_quantile:.3f}"
    )

    original_count = len(points)

    # --------------------------------------------------------
    # Create source point cloud
    # --------------------------------------------------------

    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(points)

    pcd.colors = o3d.utility.Vector3dVector(
        colors.astype(np.float64) / 255.0
    )

    # --------------------------------------------------------
    # Remove invalid points
    # --------------------------------------------------------

    finite_mask = np.isfinite(points).all(axis=1)

    if not finite_mask.all():

        print(
            f"Removing "
            f"{np.count_nonzero(~finite_mask):,} "
            f"non-finite points"
        )

        points = points[finite_mask]
        colors = colors[finite_mask]

        pcd.points = o3d.utility.Vector3dVector(points)

        pcd.colors = o3d.utility.Vector3dVector(
            colors.astype(np.float64) / 255.0
        )

    # --------------------------------------------------------
    # Voxel downsample
    # --------------------------------------------------------

    print("Voxel downsampling...")

    pcd_down = pcd.voxel_down_sample(
        voxel_size=voxel
    )

    downsampled_count = len(pcd_down.points)

    print(
        f"Downsampled: "
        f"{downsampled_count:,} points"
    )

    if downsampled_count < 10:

        raise RuntimeError(
            "Too few points after voxel downsampling"
        )

    # --------------------------------------------------------
    # Estimate normals
    # --------------------------------------------------------

    print("Estimating normals...")

    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius,
            max_nn=max_nn,
        )
    )

    # --------------------------------------------------------
    # Orient normals
    # --------------------------------------------------------

    print("Orienting normals...")

    pcd_down.orient_normals_consistent_tangent_plane(
        max_nn
    )

    # --------------------------------------------------------
    # Poisson reconstruction
    # --------------------------------------------------------

    print("Running Poisson reconstruction...")

    mesh, densities = (
        o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd_down,
            depth=depth,
            width=0,
            scale=1.1,
            linear_fit=False,
        )
    )

    poisson_vertex_count = len(mesh.vertices)
    poisson_triangle_count = len(mesh.triangles)

    print(
        f"Poisson output: "
        f"{poisson_vertex_count:,} vertices, "
        f"{poisson_triangle_count:,} triangles"
    )

    # --------------------------------------------------------
    # Density filtering
    # --------------------------------------------------------

    densities = np.asarray(densities)

    if len(densities) > 0:

        threshold = np.quantile(
            densities,
            density_quantile,
        )

        keep_mask = densities >= threshold

        removed = np.count_nonzero(~keep_mask)

        print(
            f"Density threshold: {threshold:.6f}"
        )

        print(
            f"Removing "
            f"{removed:,} low-density vertices"
        )

        mesh.remove_vertices_by_mask(
            ~keep_mask
        )

    filtered_vertex_count = len(mesh.vertices)
    filtered_triangle_count = len(mesh.triangles)

    # --------------------------------------------------------
    # Mesh normals
    # --------------------------------------------------------

    print("Computing mesh normals...")

    mesh.compute_vertex_normals()

    # --------------------------------------------------------
    # Transfer colours
    # --------------------------------------------------------

    print("Transferring colours...")

    transfer_colors_nearest(
        mesh,
        pcd_down,
        np.asarray(pcd_down.colors) * 255.0,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Saving: {output_path}")

    success = o3d.io.write_triangle_mesh(
        str(output_path),
        mesh,
        write_ascii=False,
        compressed=False,
        write_vertex_normals=True,
        write_vertex_colors=True,
    )

    if not success:
        raise RuntimeError(
            f"Failed to save mesh: {output_path}"
        )

    elapsed = time.time() - start_time

    print(
        f"Finished in {elapsed:.2f} seconds"
    )

    return {
        "status": "success",
        "original_points": original_count,
        "downsampled_points": downsampled_count,
        "poisson_vertices": poisson_vertex_count,
        "poisson_triangles": poisson_triangle_count,
        "filtered_vertices": filtered_vertex_count,
        "filtered_triangles": filtered_triangle_count,
        "runtime_seconds": elapsed,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print("EXTENDED POISSON PARAMETER SWEEP")
    print("=" * 80)

    print(f"Output directory:")
    print(f"  {POISSON_TEST_DIR}")

    print()
    print(f"Number of parameter tests: {len(POISSON_TESTS)}")
    print(f"Number of input PLYs:      {len(PLY_FILES)}")
    print(
        f"Total expected meshes:     "
        f"{len(POISSON_TESTS) * len(PLY_FILES)}"
    )

    print()
    print("MapAnything: DISABLED")
    print("Images/LiDAR: DISABLED")
    print("Existing PLYs: REUSED")

    POISSON_TEST_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load every PLY ONCE.
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("LOADING EXISTING PLYs")
    print("=" * 80)

    loaded_clouds = {}

    for name, path in PLY_FILES.items():

        try:
            loaded_clouds[name] = load_existing_ply(path)

        except Exception as exc:

            print()
            print(
                f"FAILED TO LOAD {name}: "
                f"{exc}"
            )

    if not loaded_clouds:

        raise RuntimeError(
            "No existing MapAnything PLYs could be loaded."
        )

    # --------------------------------------------------------
    # CSV result rows
    # --------------------------------------------------------

    results = []

    total_runs = (
        len(POISSON_TESTS)
        * len(loaded_clouds)
    )

    run_number = 0

    sweep_start = time.time()

    # --------------------------------------------------------
    # Run all tests
    # --------------------------------------------------------

    for test_index, test in enumerate(
        POISSON_TESTS,
        start=1,
    ):

        test_name = test["name"]

        test_dir = (
            POISSON_TEST_DIR
            / f"test_{test_index:03d}_{test_name}"
        )

        test_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        print()
        print()
        print("#" * 100)
        print(
            f"TEST {test_index}/{len(POISSON_TESTS)}"
        )
        print(
            f"Name: {test_name}"
        )
        print("#" * 100)

        for cloud_name, (
            points,
            colors,
        ) in loaded_clouds.items():

            run_number += 1

            print()
            print(
                f"OVERALL RUN "
                f"{run_number}/{total_runs}"
            )

            output_path = (
                test_dir
                / f"{cloud_name}_poisson.ply"
            )

            row = {
                "test_index": test_index,
                "test_name": test_name,
                "cloud": cloud_name,
                "voxel": test["voxel"],
                "depth": test["depth"],
                "radius": test["radius"],
                "max_nn": test["max_nn"],
                "density_quantile": test["density"],
                "status": "failed",
                "original_points": len(points),
                "downsampled_points": "",
                "poisson_vertices": "",
                "poisson_triangles": "",
                "filtered_vertices": "",
                "filtered_triangles": "",
                "runtime_seconds": "",
                "output_path": str(output_path),
                "error": "",
            }

            try:

                stats = run_poisson(
                    points=points,
                    colors=colors,
                    output_path=output_path,
                    test=test,
                    label=(
                        f"Test {test_index} | "
                        f"{cloud_name}"
                    ),
                )

                row.update(stats)

            except Exception as exc:

                print()
                print("!" * 80)
                print(
                    f"FAILED: "
                    f"{test_name} | "
                    f"{cloud_name}"
                )
                print(
                    f"Error: {exc}"
                )
                print("!" * 80)

                row["status"] = "failed"
                row["error"] = str(exc)

                traceback.print_exc()

            results.append(row)

    # --------------------------------------------------------
    # Save summary CSV
    # --------------------------------------------------------

    csv_path = (
        POISSON_TEST_DIR
        / "poisson_test_summary.csv"
    )

    fieldnames = [
        "test_index",
        "test_name",
        "cloud",
        "voxel",
        "depth",
        "radius",
        "max_nn",
        "density_quantile",
        "status",
        "original_points",
        "downsampled_points",
        "poisson_vertices",
        "poisson_triangles",
        "filtered_vertices",
        "filtered_triangles",
        "runtime_seconds",
        "output_path",
        "error",
    ]

    with open(
        csv_path,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(results)

    total_elapsed = (
        time.time()
        - sweep_start
    )

    successful = sum(
        row["status"] == "success"
        for row in results
    )

    failed = sum(
        row["status"] == "failed"
        for row in results
    )

    print()
    print()
    print("=" * 80)
    print("SWEEP COMPLETE")
    print("=" * 80)

    print(
        f"Successful: {successful}/{len(results)}"
    )

    print(
        f"Failed:     {failed}/{len(results)}"
    )

    print(
        f"Total time: {total_elapsed / 60:.2f} minutes"
    )

    print()
    print(
        f"Summary CSV:"
    )
    print(
        f"  {csv_path}"
    )

    print()
    print(
        f"Results:"
    )
    print(
        f"  {POISSON_TEST_DIR}"
    )

    print()
    print("Done.")


if __name__ == "__main__":
    main()
