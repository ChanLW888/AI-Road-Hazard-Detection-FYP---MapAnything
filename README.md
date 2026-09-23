# 3D Reconstruction and Road Hazard Detection Pipeline

## Overview

This project develops a **3D reconstruction and road hazard detection pipeline** for reconstructing road environments from multi-camera imagery and LiDAR data, while detecting and spatially localising road hazards and relevant road-environment features in 3D.

The pipeline builds upon **MapAnything**, a feed-forward metric 3D reconstruction framework capable of producing dense metric 3D geometry from images and additional geometric inputs such as camera calibration, poses, and depth. MapAnything provides outputs including 3D points, depth, camera poses, camera intrinsics, confidence values, and validity masks.

The project extends the reconstruction framework by integrating:

* Multi-camera road imagery
* Camera intrinsic and extrinsic calibration
* LiDAR point clouds
* LiDAR-to-camera projection
* Metric depth information
* MapAnything 3D reconstruction
* Multi-camera point-cloud registration
* 2D semantic segmentation
* 2D-to-3D semantic projection
* Semantic 3D point-cloud generation
* 3D visualisation and evaluation

The overall objective is to produce a **metric 3D representation of the road environment with detected hazards spatially localised within the reconstructed scene**.

---

# Pipeline

```text
                    Multi-Camera Images
                           │
                           ▼
                ┌─────────────────────┐
                │ Camera Calibration  │
                │ Intrinsics/Poses    │
                └──────────┬──────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │   MapAnything     │
                 │ 3D Reconstruction │
                 └─────────┬─────────┘
                           │
                    Dense Metric 3D
                       Geometry
                           │
                           ▼
                  ┌────────────────┐
                  │ Multi-Camera    │
                  │ Registration    │
                  └───────┬────────┘
                          │
                          │
          LiDAR ──────────┤
          Point Cloud     │
          + Calibration   │
                          ▼
                 ┌─────────────────┐
                 │ LiDAR Projection│
                 │ & Depth Fusion  │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │ Semantic        │
                 │ Segmentation    │
                 └────────┬────────┘
                          │
                    Semantic Masks
                          │
                          ▼
                 ┌─────────────────┐
                 │ 2D → 3D        │
                 │ Projection      │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │ Semantic 3D     │
                 │ Point Cloud     │
                 └────────┬────────┘
                          │
                          ▼
                  3D Road Environment
                  + Hazard Localisation
```

---

# 1. 3D Reconstruction

## MapAnything

MapAnything is used as the primary metric 3D reconstruction model.

Given a set of images, MapAnything produces dense metric 3D geometry and associated camera information.

Relevant outputs include:

| Output            | Description                     |
| ----------------- | ------------------------------- |
| `pts3d`           | 3D points in world coordinates  |
| `pts3d_cam`       | 3D points in camera coordinates |
| `depth_z`         | Z-depth in the camera frame     |
| `depth_along_ray` | Depth along the camera ray      |
| `camera_poses`    | Camera-to-world poses           |
| `intrinsics`      | Camera intrinsic parameters     |
| `conf`            | Per-pixel confidence            |
| `mask`            | Valid geometry mask             |

MapAnything supports image-only reconstruction as well as combinations of images, calibration, poses and depth information.

The resulting metric 3D geometry forms the basis of the downstream point-cloud and hazard-localisation pipeline.

---

# 2. Multi-Camera Reconstruction

The road environment is observed using multiple cameras.

The current camera configuration includes:

```text
CAM1
CAM2
CAM6
```

Each camera has its own:

* Image sequence
* Intrinsic calibration
* Extrinsic calibration
* Relationship to the LiDAR coordinate system

The individual camera reconstructions are subsequently transformed into a common coordinate system so that the different views of the road environment can be combined.

MapAnything supports camera intrinsics and camera poses as geometric inputs during reconstruction. Camera poses are represented using the OpenCV camera convention:

```text
+X → Right
+Y → Down
+Z → Forward
```

---

# 3. Camera Calibration

Camera calibration provides the geometric relationship between image pixels and the physical 3D environment.

The pipeline uses:

### Camera Intrinsics

The intrinsic matrix describes the camera's internal projection parameters:

```text
K =
┌             ┐
│ fx  0   cx  │
│ 0   fy  cy  │
│ 0   0    1  │
└             ┘
```

These parameters are used when projecting between image coordinates and camera coordinates.

### Camera Extrinsics

Extrinsic parameters describe the transformation between camera coordinate systems and the common reconstruction coordinate system.

Correct calibration is essential for:

* LiDAR projection
* Multi-camera registration
* 2D-to-3D projection
* Metric reconstruction

---

# 4. LiDAR Integration

LiDAR is incorporated into the pipeline to provide **sparse metric depth measurements** from the physical environment.

The LiDAR point cloud is transformed into the relevant camera coordinate system using the calibrated LiDAR-to-camera extrinsic transformation.

The transformed points are then projected onto the corresponding camera image.

```text
LiDAR Point Cloud
       │
       ▼
LiDAR → Camera Transform
       │
       ▼
Project 3D Points onto Image
       │
       ▼
Sparse Metric Depth
       │
       ▼
MapAnything
```

LiDAR range filtering is applied before projection.

The current experiments use a maximum LiDAR range of:

```text
50 m
```

The LiDAR information can then be used as an additional geometric input to the reconstruction process.

MapAnything supports depth inputs alongside camera calibration information, with `depth_z` requiring corresponding calibration information such as intrinsics or ray directions.

---

# 5. LiDAR Sensitivity Experiments

A series of experiments are used to investigate how LiDAR information affects the resulting 3D reconstruction.

The image input remains consistent while the amount or availability of LiDAR information is varied.

Example configurations include:

```text
No LiDAR
Reduced LiDAR
Intermediate LiDAR
Full LiDAR
```

The resulting reconstructions can be compared using factors such as:

* Point-cloud density
* Geometric completeness
* Surface consistency
* Multi-view alignment
* Depth consistency
* Reconstruction artefacts

These experiments provide a way to investigate the contribution of external metric depth information to the reconstruction.

---

# 6. Multi-Camera Registration

Each camera produces its own reconstructed geometry.

Because the cameras observe the environment from different viewpoints, the individual reconstructions must be transformed into a common coordinate system.

The registration process can be represented as:

```text
CAM1 Reconstruction ──┐
                      │
CAM2 Reconstruction ──┼──► Common Coordinate Frame
                      │
CAM6 Reconstruction ──┘
```

Camera calibration and geometric transformations are used to align the individual point clouds.

The resulting common coordinate system allows geometry and semantic information from multiple cameras to be combined.

---

# 7. Semantic Segmentation

The reconstructed environment is combined with a 2D semantic segmentation pipeline.

The segmentation model operates on the camera images and produces a pixel-level semantic mask.

The current semantic classes are:

| Class ID | Class                    |
| -------: | ------------------------ |
|        0 | Fallen fence             |
|        1 | Public Greenway          |
|        2 | Loose trash              |
|        3 | Mud Spill                |
|        4 | Sidewalk                 |
|        5 | Skip bin on the sidewalk |

These classes represent the current set of road-environment features used in the semantic detection and 3D localisation pipeline.

The genuinely hazardous classes include features such as:

* Fallen fences
* Loose trash
* Mud spills
* Skip bins obstructing the sidewalk

Public Greenway and Sidewalk provide contextual semantic information about the road environment.

---

# 8. 2D → 3D Semantic Projection

Once semantic regions have been detected in the image plane, the corresponding pixels can be associated with the reconstructed 3D geometry.

For each valid segmented pixel:

```text
Image Pixel
     │
     ├── Camera Intrinsics
     ├── Camera Pose
     └── Depth / 3D Reconstruction
     │
     ▼
3D Point
```

This converts the 2D semantic segmentation into a **semantic 3D point cloud**.

The resulting 3D points retain both their spatial coordinates and semantic class.

This enables detected hazards to be spatially localised within the reconstructed environment rather than being represented only as 2D image detections.

---

# 9. Semantic 3D Point Cloud

The reconstructed geometry from the different cameras is transformed into a common coordinate system and combined with the semantic segmentation results.

The resulting representation contains:

```text
X
Y
Z
RGB
Semantic Class
Confidence
```

This allows the road environment and detected hazards to be visualised in 3D.

Conceptually:

```text
                    3D Road Environment
                            │
          ┌─────────────────┼─────────────────┐
          │                 │                 │
       Roadside          Sidewalk          Hazards
          │                 │                 │
          └─────────────────┼─────────────────┘
                            │
                            ▼
                   Semantic Point Cloud
```

---

# 10. Road Hazard Localisation

The main purpose of integrating semantic segmentation with 3D reconstruction is to move from **2D hazard detection to 3D hazard localisation**.

Instead of only identifying that a hazard exists in an image:

```text
2D:
"There is a fallen fence in this image."
```

the pipeline aims to determine its location within the reconstructed environment:

```text
3D:
"These reconstructed 3D points correspond to the fallen fence."
```

This enables hazards to be examined relative to surrounding road features such as:

* Sidewalks
* Public greenways
* Roads
* Other environmental structures

The resulting 3D representation can therefore be used for spatial analysis and visualisation of road hazards.

---

# 11. Main Processing Stages

The complete pipeline can be summarised as follows.

### Stage 1 — Image Acquisition

Obtain synchronised images from the available road-facing cameras.

### Stage 2 — Camera Calibration

Load the intrinsic and extrinsic calibration parameters for each camera.

### Stage 3 — LiDAR Processing

Filter the LiDAR point cloud and transform LiDAR points into the relevant camera coordinate frame.

### Stage 4 — LiDAR Projection

Project the transformed LiDAR points onto the camera images to obtain sparse metric depth information.

### Stage 5 — Metric 3D Reconstruction

Run MapAnything using the available image and geometric inputs.

### Stage 6 — Multi-Camera Registration

Transform reconstructed geometry from each camera into a common coordinate system.

### Stage 7 — Semantic Segmentation

Run the semantic segmentation model on the camera images.

### Stage 8 — 2D → 3D Projection

Associate segmented pixels with reconstructed 3D points.

### Stage 9 — Semantic Point-Cloud Generation

Merge the reconstructed geometry and semantic labels into a common 3D representation.

### Stage 10 — Visualisation and Evaluation

Visualise the reconstructed road environment and analyse the spatial distribution and quality of the detected semantic classes.

---

# 12. Output

The pipeline can generate several types of output.

## 3D Point Clouds

```text
.ply
```

Containing the reconstructed 3D geometry.

## Semantic Point Clouds

```text
.ply
```

Containing reconstructed geometry together with semantic class information.

## Projected LiDAR

Visualisations showing LiDAR points projected onto the camera images.

## Segmentation Masks

2D semantic masks showing the detected road-environment classes.

## 3D Hazard Localisation

3D points corresponding to segmented regions, allowing detected hazards to be spatially located within the reconstructed environment.

---

# 13. Current Semantic Classes

The current semantic segmentation configuration contains six classes:

```text
0 - Fallen fence
1 - Public Greenway
2 - Loose trash
3 - Mud Spill
4 - Sidewalk
5 - Skip bin on the sidewalk
```

These classes are used throughout the 2D segmentation and 3D semantic projection stages.

The class set can be expanded as additional road-environment features are incorporated into the dataset and segmentation model.

---

# 14. Project Structure

The repository contains the Python scripts required for the different stages of the reconstruction and hazard-detection pipeline.

A typical structure is:

```text
FYP/
│
├── map-anything/
│
├── scripts/
│   ├── MapAnything inference
│   ├── LiDAR projection
│   ├── camera calibration
│   ├── point-cloud registration
│   ├── semantic segmentation
│   ├── 2D → 3D projection
│   └── point-cloud merging
│
├── calibration/
│   └── camera / LiDAR calibration files
│
└── README.md
```

Large datasets, generated point clouds, model checkpoints and inference outputs should generally be kept outside the source-code repository where appropriate.

---

# 15. Requirements

The reconstruction pipeline is based on Python and PyTorch.

The original MapAnything framework uses Python 3.12 and can be installed into a dedicated conda environment.

Example environment setup:

```bash
conda create -n mapanything python=3.12 -y
conda activate mapanything

pip install -e .
```

PyTorch and CUDA should be installed according to the target GPU and computing environment.

Additional dependencies may be required for:

* LiDAR processing
* Open3D
* Image processing
* Semantic segmentation
* Point-cloud registration
* Visualisation

---

# 16. MapAnything

This project uses MapAnything as the underlying metric 3D reconstruction framework.

MapAnything provides a unified interface for metric 3D reconstruction and supports combinations of images, camera calibration, camera poses and depth information.

The framework also provides a unified output format containing 3D geometry, camera information and confidence values, making it suitable as the reconstruction component of the downstream pipeline.

For more information about the original MapAnything framework:

* [MapAnything Project Page](https://map-anything.github.io/)
* [MapAnything GitHub Repository](https://github.com/facebookresearch/map-anything)
* [MapAnything Paper](https://map-anything.github.io/assets/MapAnything.pdf)
* [MapAnything arXiv](https://arxiv.org/abs/2509.13414)

---

# 17. Acknowledgements

This project builds upon the open-source **MapAnything** framework developed by researchers from Meta and Carnegie Mellon University.

The original MapAnything repository also acknowledges several related open-source projects, including DUSt3R, MASt3R, MoGe, VGGT, VGGSfM and DINOv2.

---

# 18. Citation

If MapAnything is used in this project, please cite the original work:

```bibtex
@inproceedings{keetha2026mapanything,
  title={{MapAnything}: Universal Feed-Forward Metric {3D} Reconstruction},
  author={Nikhil Keetha and Norman M\"{u}ller and Johannes Sch\"{o}nberger and Lorenzo Porzi and Yuchen Zhang and Tobias Fischer and Arno Knapitsch and Duncan Zauss and Ethan Weber and Nelson Antunes and Jonathon Luiten and Manuel Lopez-Antequera and Samuel Rota Bul\`{o} and Christian Richardt and Deva Ramanan and Sebastian Scherer and Peter Kontschieder},
  booktitle={International Conference on 3D Vision (3DV)},
  year={2026},
  organization={IEEE}
}
```

The original MapAnything code is released under the Apache 2.0 license.
