#!/usr/bin/env python3

from pathlib import Path

import cv2
import numpy as np


# ============================================================
# CONFIG
# ============================================================

CALIB_FILE = Path(
    "/home/lcha0115/bt60_scratch/lcha_data/map-anything/personal_data/camera_lidar_imu_test_04/extracted/calibration/CAM1_calib.txt"
)

IMAGE_FILE = Path(
    "/home/lcha0115/bt60_scratch/lcha_data/map-anything/personal_data/camera_lidar_imu_test_04/extracted/CAM1/images_raw/1786948267439034126.png"
)

OUTPUT_FILE = Path(
    "image_undistorted.png"
)

SHOW = False


# ============================================================
# LOAD CALIBRATION
# ============================================================

def load_calibration(calib_file):

    with open(calib_file, "r") as f:

        lines = [
            line.strip()
            for line in f
            if line.strip()
        ]

    width = None
    height = None

    K = None
    D = None
    R = None
    P = None

    i = 0

    while i < len(lines):

        line = lines[i]

        # ----------------------------------------------------
        # IMAGE SIZE
        # ----------------------------------------------------

        if line.startswith("width:"):

            width = int(
                line.split(":")[1].strip()
            )

        elif line.startswith("height:"):

            height = int(
                line.split(":")[1].strip()
            )

        # ----------------------------------------------------
        # K
        # ----------------------------------------------------

        elif line == "K:":

            K = np.array(
                [
                    list(
                        map(
                            float,
                            lines[i + 1].split()
                        )
                    ),
                    list(
                        map(
                            float,
                            lines[i + 2].split()
                        )
                    ),
                    list(
                        map(
                            float,
                            lines[i + 3].split()
                        )
                    ),
                ],
                dtype=np.float64,
            )

            i += 3

        # ----------------------------------------------------
        # D
        # ----------------------------------------------------

        elif line == "D:":

            D = np.array(
                list(
                    map(
                        float,
                        lines[i + 1].split()
                    )
                ),
                dtype=np.float64,
            ).reshape(4, 1)

            i += 1

        # ----------------------------------------------------
        # R
        # ----------------------------------------------------

        elif line == "R:":

            R = np.array(
                [
                    list(
                        map(
                            float,
                            lines[i + 1].split()
                        )
                    ),
                    list(
                        map(
                            float,
                            lines[i + 2].split()
                        )
                    ),
                    list(
                        map(
                            float,
                            lines[i + 3].split()
                        )
                    ),
                ],
                dtype=np.float64,
            )

            i += 3

        # ----------------------------------------------------
        # P
        # ----------------------------------------------------

        elif line == "P:":

            P = np.array(
                [
                    list(
                        map(
                            float,
                            lines[i + 1].split()
                        )
                    ),
                    list(
                        map(
                            float,
                            lines[i + 2].split()
                        )
                    ),
                    list(
                        map(
                            float,
                            lines[i + 3].split()
                        )
                    ),
                ],
                dtype=np.float64,
            )

            i += 3

        i += 1

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if width is None or height is None:
        raise ValueError(
            "Calibration file is missing width/height."
        )

    if K is None:
        raise ValueError(
            "Calibration file is missing K."
        )

    if D is None:
        raise ValueError(
            "Calibration file is missing D."
        )

    if R is None:
        raise ValueError(
            "Calibration file is missing R."
        )

    if P is None:
        raise ValueError(
            "Calibration file is missing P."
        )

    return width, height, K, D, R, P


# ============================================================
# LOAD CALIBRATION
# ============================================================

print("=" * 70)
print("FISHEYE IMAGE UNDISTORTION")
print("=" * 70)

print(
    "\nCalibration:",
    CALIB_FILE
)

(
    calibration_width,
    calibration_height,
    K,
    D,
    R,
    P,
) = load_calibration(
    CALIB_FILE
)


print("\nCalibration image size:")
print(
    calibration_width,
    "x",
    calibration_height
)

print("\nK:")
print(K)

print("\nD:")
print(D.ravel())

print("\nR:")
print(R)

print("\nP:")
print(P)


# ============================================================
# LOAD IMAGE
# ============================================================

print("\nLoading image:")

print(
    IMAGE_FILE
)

image = cv2.imread(
    str(IMAGE_FILE)
)

if image is None:

    raise FileNotFoundError(
        f"Could not read image:\n{IMAGE_FILE}"
    )


image_height, image_width = image.shape[:2]

print(
    "\nImage size:",
    image_width,
    "x",
    image_height
)


# ============================================================
# CHECK RESOLUTION
# ============================================================

if (
    image_width != calibration_width
    or
    image_height != calibration_height
):

    raise ValueError(
        "\nImage resolution does not match calibration.\n"
        f"Image:        {image_width} x {image_height}\n"
        f"Calibration:  {calibration_width} x "
        f"{calibration_height}"
    )


# ============================================================
# CREATE FISHEYE RECTIFICATION MAP
# ============================================================

print(
    "\nCreating fisheye rectification map..."
)

map_x, map_y = cv2.fisheye.initUndistortRectifyMap(

    K,

    D,

    R,

    P[:3, :3],

    (
        calibration_width,
        calibration_height,
    ),

    cv2.CV_32FC1,
)


# ============================================================
# UNDISTORT
# ============================================================

print(
    "Undistorting image..."
)

undistorted = cv2.remap(

    image,

    map_x,
    map_y,

    interpolation=cv2.INTER_LINEAR,

    borderMode=cv2.BORDER_CONSTANT,
)


# ============================================================
# SAVE
# ============================================================

success = cv2.imwrite(
    str(OUTPUT_FILE),
    undistorted,
)

if not success:

    raise RuntimeError(
        f"Could not save:\n{OUTPUT_FILE}"
    )


print(
    "\nSaved:"
)

print(
    OUTPUT_FILE
)


# ============================================================
# DISPLAY
# ============================================================

if SHOW:

    comparison = np.hstack(
        (
            image,
            undistorted,
        )
    )

    maximum_width = 1800

    if comparison.shape[1] > maximum_width:

        scale = (
            maximum_width /
            comparison.shape[1]
        )

        comparison = cv2.resize(
            comparison,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )

    cv2.imshow(
        "Original | Undistorted",
        comparison,
    )

    print(
        "\nPress any key to close..."
    )

    cv2.waitKey(0)

    cv2.destroyAllWindows()


print(
    "\nDone."
)