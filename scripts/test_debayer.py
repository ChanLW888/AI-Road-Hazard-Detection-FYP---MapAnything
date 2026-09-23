#!/usr/bin/env python3

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr


# ============================================================================
# CONFIG
# ============================================================================

BAG_PATH = Path(
    "/home/lcha0115/bt60_scratch/lcha_data/map-anything/"
    "personal_data/camera_lidar_imu_test_04"
)

# Test CAM1.
IMAGE_TOPIC = "/CAM1/CAM1_node/image_raw"

OUTPUT_DIR = (
    BAG_PATH
    / "debayer_test"
)

# Only test one image.
FRAME_NUMBER = 0


# ============================================================================
# BAYER PATTERNS
# ============================================================================

BAYER_PATTERNS = {

    "RGGB": cv2.COLOR_BAYER_RG2RGB,

    "BGGR": cv2.COLOR_BAYER_BG2RGB,

    "GBRG": cv2.COLOR_BAYER_GB2RGB,

    "GRBG": cv2.COLOR_BAYER_GR2RGB,

}


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 80)
    print("BAYER DEBAYER TEST")
    print("=" * 80)

    print()
    print("Bag:")
    print(BAG_PATH)

    print()
    print("Topic:")
    print(IMAGE_TOPIC)

    print()
    print("Output:")
    print(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================================
    # FIND ONE RAW IMAGE
    # ========================================================================

    print()
    print("=" * 80)
    print("READING ONE RAW IMAGE")
    print("=" * 80)

    raw_msg = None
    bag_timestamp = None

    with Reader(BAG_PATH) as reader:

        for connection, timestamp, raw in reader.messages():

            if connection.topic != IMAGE_TOPIC:
                continue

            raw_msg = deserialize_cdr(
                raw,
                connection.msgtype,
            )

            bag_timestamp = timestamp

            break

    if raw_msg is None:

        raise RuntimeError(
            f"No image found on:\n{IMAGE_TOPIC}"
        )

    # ========================================================================
    # IMAGE INFORMATION
    # ========================================================================

    width = int(raw_msg.width)
    height = int(raw_msg.height)

    encoding = "bayer_bggr8"

    print()
    print("Image found:")
    print(f"  Width:     {width}")
    print(f"  Height:    {height}")
    print(f"  Encoding:  {encoding}")
    print(f"  Timestamp: {bag_timestamp}")

    # ========================================================================
    # RAW DATA
    # ========================================================================

    data = bytes(
        raw_msg.data
    )

    raw = np.frombuffer(
        data,
        dtype=np.uint8,
    )

    expected_size = (
        height * width
    )

    if raw.size < expected_size:

        raise RuntimeError(
            f"Raw image contains {raw.size} bytes, "
            f"but expected at least {expected_size}."
        )

    raw = raw[
        :expected_size
    ].reshape(
        height,
        width,
    )

    print()
    print(
        "Raw Bayer array:",
        raw.shape
    )

    # ========================================================================
    # SAVE RAW BAYER IMAGE
    # ========================================================================

    raw_output = (
        OUTPUT_DIR
        / "00_raw_bayer.png"
    )

    # This is NOT a colour image.
    # It is simply the Bayer mosaic visualised as grayscale.
    cv2.imwrite(
        str(raw_output),
        raw,
    )

    print()
    print("Saved:")
    print(raw_output)

    # ========================================================================
    # TEST ALL FOUR PATTERNS
    # ========================================================================

    results = {}

    print()
    print("=" * 80)
    print("TESTING BAYER PATTERNS")
    print("=" * 80)

    for name, conversion in BAYER_PATTERNS.items():

        print()
        print(f"Testing {name}...")

        # ------------------------------------------------------------
        # Debayer
        # ------------------------------------------------------------

        rgb = cv2.cvtColor(
            raw,
            conversion,
        )

        results[name] = rgb

        # ------------------------------------------------------------
        # Save RGB PNG
        # ------------------------------------------------------------

        output_path = (
            OUTPUT_DIR
            / f"{name}.png"
        )

        Image.fromarray(
            rgb
        ).save(
            output_path
        )

        print(
            f"  Saved: {output_path}"
        )

    # ========================================================================
    # CREATE 2x2 COMPARISON
    # ========================================================================

    print()
    print("=" * 80)
    print("CREATING COMPARISON")
    print("=" * 80)

    # Put labels onto copies.
    labelled = {}

    for name, image in results.items():

        copy = image.copy()

        cv2.putText(
            copy,
            name,
            (40, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.0,
            (255, 255, 255),
            4,
            cv2.LINE_AA,
        )

        labelled[name] = copy

    # OpenCV arrays are RGB at this point.
    top = np.hstack(
        [
            labelled["RGGB"],
            labelled["BGGR"],
        ]
    )

    bottom = np.hstack(
        [
            labelled["GBRG"],
            labelled["GRBG"],
        ]
    )

    comparison = np.vstack(
        [
            top,
            bottom,
        ]
    )

    comparison_path = (
        OUTPUT_DIR
        / "comparison_2x2.png"
    )

    Image.fromarray(
        comparison
    ).save(
        comparison_path
    )

    print()
    print(
        "Comparison saved:"
    )

    print(
        comparison_path
    )

    # ========================================================================
    # FINISHED
    # ========================================================================

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)

    print()
    print("Files:")
    print(
        f"  {OUTPUT_DIR}/00_raw_bayer.png"
    )
    print(
        f"  {OUTPUT_DIR}/RGGB.png"
    )
    print(
        f"  {OUTPUT_DIR}/BGGR.png"
    )
    print(
        f"  {OUTPUT_DIR}/GBRG.png"
    )
    print(
        f"  {OUTPUT_DIR}/GRBG.png"
    )
    print(
        f"  {OUTPUT_DIR}/comparison_2x2.png"
    )

    print()
    print(
        "Actual ROS encoding:",
        encoding
    )


if __name__ == "__main__":
    main()