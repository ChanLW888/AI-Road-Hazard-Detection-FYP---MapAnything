import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

CSV_PATH = (
    "/home/lcha0115/bt60_scratch/lcha_data/map-anything/"
    "personal_data/camera_lidar_test_101/extracted/imu/imu.csv"
)

OUTPUT_DIR = os.path.dirname(CSV_PATH)

OUTPUT_TRAJECTORY_PNG = os.path.join(
    OUTPUT_DIR,
    "imu_heading_based_trajectory.png"
)

OUTPUT_HEADING_PNG = os.path.join(
    OUTPUT_DIR,
    "imu_heading.png"
)

OUTPUT_YAW_RATE_PNG = os.path.join(
    OUTPUT_DIR,
    "imu_yaw_rate.png"
)

OUTPUT_SPEED_PNG = os.path.join(
    OUTPUT_DIR,
    "imu_forward_speed.png"
)

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "imu_heading_trajectory.csv"
)


# ============================================================
# IMU SETTINGS
# ============================================================

INITIAL_STATIONARY_SAMPLES = 200

SBG_TIME_SCALE = 1e-6

# If gyro is already rad/s:
GYRO_SCALE = 1.0

# If gyro is deg/s, use:
# GYRO_SCALE = np.pi / 180.0

# Change to -1 if left/right are reversed.
YAW_SIGN = 1.0

GRAVITY = 9.80665

# Which IMU axis points approximately in the vehicle's
# forward direction?
#
# 0 = X
# 1 = Y
# 2 = Z
FORWARD_AXIS = 0


# ============================================================
# ACCELERATION FILTER
# ============================================================

USE_ACCEL_FILTER = True

ACCEL_FILTER_WINDOW = 5


# ============================================================
# SPEED DRIFT CONTROL
# ============================================================

# IMPORTANT:
#
# Pure accelerometer integration drifts heavily.
#
# This option forces the estimated vehicle speed back toward
# zero whenever the vehicle is detected to be stationary.
#
USE_STATIONARY_ZERO_VELOCITY = True

STATIONARY_ACCEL_THRESHOLD = 0.20
STATIONARY_GYRO_THRESHOLD = 0.05

# Number of samples required before declaring the vehicle
# stationary.
STATIONARY_WINDOW = 10


# ============================================================
# LOAD DATA
# ============================================================

print("Loading IMU data...")

df = pd.read_csv(CSV_PATH)

print(
    f"Loaded {len(df)} IMU samples"
)


required_columns = [
    "sbg_time_stamp_us",
    "accel_x",
    "accel_y",
    "accel_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
]

missing = [
    c for c in required_columns
    if c not in df.columns
]

if missing:

    raise RuntimeError(
        f"Missing required columns:\n{missing}"
    )


# ============================================================
# TIMESTAMP
# ============================================================

timestamps = (
    df["sbg_time_stamp_us"]
    .to_numpy(dtype=np.float64)
    * SBG_TIME_SCALE
)

timestamps -= timestamps[0]

raw_dt = np.diff(timestamps)

valid_dt = raw_dt[raw_dt > 0]

if len(valid_dt) == 0:

    raise RuntimeError(
        "No valid timestamps."
    )

median_dt = np.median(valid_dt)

dt = np.empty(
    len(timestamps),
    dtype=np.float64
)

dt[0] = median_dt
dt[1:] = raw_dt

dt[dt <= 0] = median_dt


print()
print("TIME")
print("----")

print(
    f"Duration: {timestamps[-1]:.2f} s"
)

print(
    f"Median dt: {median_dt:.6f} s"
)

print(
    f"IMU frequency: "
    f"{1.0 / median_dt:.2f} Hz"
)


# ============================================================
# READ ACCELERATION
# ============================================================

accel = df[
    ["accel_x", "accel_y", "accel_z"]
].to_numpy(
    dtype=np.float64
)


# ============================================================
# READ GYRO
# ============================================================

gyro = df[
    ["gyro_x", "gyro_y", "gyro_z"]
].to_numpy(
    dtype=np.float64
)

gyro *= GYRO_SCALE


# ============================================================
# INITIAL STATIONARY PERIOD
# ============================================================

n0 = min(
    INITIAL_STATIONARY_SAMPLES,
    len(df)
)

initial_accel = np.mean(
    accel[:n0],
    axis=0
)

initial_gyro = np.mean(
    gyro[:n0],
    axis=0
)


print()
print("INITIAL IMU")
print("-----------")

print(
    "Initial acceleration:",
    initial_accel
)

print(
    "Initial gyro bias:",
    initial_gyro
)

print(
    "Initial acceleration magnitude:",
    np.linalg.norm(initial_accel)
)


# ============================================================
# REMOVE GYRO BIAS
# ============================================================

gyro_corrected = (
    gyro - initial_gyro
)


# ============================================================
# FILTER ACCELERATION
# ============================================================

if USE_ACCEL_FILTER:

    print()
    print(
        f"Filtering acceleration "
        f"({ACCEL_FILTER_WINDOW} samples)..."
    )

    kernel = (
        np.ones(
            ACCEL_FILTER_WINDOW
        )
        / ACCEL_FILTER_WINDOW
    )

    accel_filtered = np.column_stack([
        np.convolve(
            accel[:, axis],
            kernel,
            mode="same"
        )
        for axis in range(3)
    ])

else:

    accel_filtered = accel.copy()


# ============================================================
# GRAVITY DIRECTION
# ============================================================

gravity_unit = (
    initial_accel
    / np.linalg.norm(initial_accel)
)

gravity_vector_body = (
    gravity_unit
    * GRAVITY
)


# ============================================================
# YAW INTEGRATION
# ============================================================

print()
print("Integrating vehicle heading...")


yaw_rate = (
    gyro_corrected[:, 2]
    * YAW_SIGN
)


yaw = np.zeros(
    len(df),
    dtype=np.float64
)


for i in range(1, len(df)):

    yaw[i] = (
        yaw[i - 1]
        +
        0.5
        *
        (
            yaw_rate[i - 1]
            +
            yaw_rate[i]
        )
        *
        dt[i]
    )


yaw_deg = np.degrees(yaw)

# Yaw-rate in degrees/sec
yaw_rate_deg = np.degrees(
    yaw_rate
)


# ============================================================
# PRINT HEADING INFORMATION
# ============================================================

print()
print("HEADING")
print("-------")

print(
    f"Final heading: "
    f"{yaw_deg[-1]:.2f} degrees"
)

print(
    f"Maximum heading: "
    f"{yaw_deg.max():.2f} degrees"
)

print(
    f"Minimum heading: "
    f"{yaw_deg.min():.2f} degrees"
)


# ============================================================
# LINEAR ACCELERATION
# ============================================================

linear_accel_body = (
    accel_filtered
    - gravity_vector_body
)


# ============================================================
# FORWARD ACCELERATION
# ============================================================

forward_accel = (
    linear_accel_body[:, FORWARD_AXIS]
)


# ============================================================
# REMOVE INITIAL FORWARD BIAS
# ============================================================

initial_forward_bias = np.mean(
    forward_accel[:n0]
)

forward_accel -= (
    initial_forward_bias
)


print()
print(
    "Initial forward acceleration bias:",
    initial_forward_bias
)


# ============================================================
# DETECT STATIONARY PERIODS
# ============================================================

print()
print("Detecting stationary periods...")


accel_magnitude = np.linalg.norm(
    linear_accel_body,
    axis=1
)

gyro_magnitude = np.linalg.norm(
    gyro_corrected,
    axis=1
)


stationary = (
    (accel_magnitude < STATIONARY_ACCEL_THRESHOLD)
    &
    (gyro_magnitude < STATIONARY_GYRO_THRESHOLD)
)


# Require several consecutive samples
# to reduce false stationary detections.

if STATIONARY_WINDOW > 1:

    stationary_filtered = np.zeros_like(
        stationary,
        dtype=bool
    )

    count = 0

    for i in range(len(stationary)):

        if stationary[i]:

            count += 1

        else:

            count = 0

        if count >= STATIONARY_WINDOW:

            stationary_filtered[
                i - STATIONARY_WINDOW + 1:i + 1
            ] = True

    stationary = stationary_filtered


print(
    f"Stationary samples detected: "
    f"{stationary.sum()} / {len(stationary)}"
)


# ============================================================
# FORWARD VELOCITY
# ============================================================

print()
print("Integrating forward velocity...")


forward_velocity = np.zeros(
    len(df),
    dtype=np.float64
)


for i in range(1, len(df)):

    forward_velocity[i] = (
        forward_velocity[i - 1]
        +
        0.5
        *
        (
            forward_accel[i - 1]
            +
            forward_accel[i]
        )
        *
        dt[i]
    )

    # --------------------------------------------------------
    # ZERO-VELOCITY UPDATE
    # --------------------------------------------------------
    #
    # If the IMU indicates the vehicle is stationary,
    # force speed back to zero.
    #
    # This dramatically reduces integration drift.
    # --------------------------------------------------------

    if (
        USE_STATIONARY_ZERO_VELOCITY
        and stationary[i]
    ):

        forward_velocity[i] = 0.0


# Prevent tiny negative speeds caused by integration noise.

forward_velocity[
    np.abs(forward_velocity) < 0.05
] = 0.0


# ============================================================
# OPTIONAL SPEED SANITY CHECK
# ============================================================

print()
print("SPEED")
print("-----")

print(
    f"Maximum estimated speed: "
    f"{forward_velocity.max():.2f} m/s"
)

print(
    f"Minimum estimated speed: "
    f"{forward_velocity.min():.2f} m/s"
)

print(
    f"Final estimated speed: "
    f"{forward_velocity[-1]:.2f} m/s"
)

print(
    f"Maximum estimated speed: "
    f"{forward_velocity.max() * 3.6:.2f} km/h"
)


# ============================================================
# HEADING-BASED POSITION
# ============================================================

print()
print(
    "Building heading-based trajectory..."
)


position = np.zeros(
    (len(df), 2),
    dtype=np.float64
)


for i in range(1, len(df)):

    # --------------------------------------------------------
    # Current vehicle heading
    # --------------------------------------------------------

    heading_x = np.cos(
        yaw[i]
    )

    heading_y = np.sin(
        yaw[i]
    )

    # --------------------------------------------------------
    # Distance travelled during this timestep
    # --------------------------------------------------------

    distance = (
        forward_velocity[i]
        * dt[i]
    )

    dx = (
        distance
        * heading_x
    )

    dy = (
        distance
        * heading_y
    )

    position[i, 0] = (
        position[i - 1, 0]
        + dx
    )

    position[i, 1] = (
        position[i - 1, 1]
        + dy
    )


# ============================================================
# SAVE CSV
# ============================================================

output_df = pd.DataFrame({

    "time_s": timestamps,

    "yaw_rad": yaw,

    "yaw_deg": yaw_deg,

    "yaw_rate_rad_s": yaw_rate,

    "yaw_rate_deg_s": yaw_rate_deg,

    "forward_accel_mps2": forward_accel,

    "forward_velocity_mps": forward_velocity,

    "stationary": stationary,

    "x_m": position[:, 0],

    "y_m": position[:, 1],

})

output_df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# TURN INFORMATION
# ============================================================

print()
print("=" * 60)
print("VEHICLE TURNING")
print("=" * 60)

print(
    f"Final heading: "
    f"{yaw_deg[-1]:.2f} degrees"
)

print(
    f"Maximum heading: "
    f"{yaw_deg.max():.2f} degrees"
)

print(
    f"Minimum heading: "
    f"{yaw_deg.min():.2f} degrees"
)

print(
    f"Final estimated speed: "
    f"{forward_velocity[-1]:.2f} m/s"
)

print(
    f"Final X: "
    f"{position[-1, 0]:.2f} m"
)

print(
    f"Final Y: "
    f"{position[-1, 1]:.2f} m"
)


# ============================================================
# PLOT — HEADING
# ============================================================

plt.figure(
    figsize=(14, 6)
)

plt.plot(
    timestamps,
    yaw_deg,
    linewidth=1.5
)

plt.axhline(
    0,
    linestyle="--",
    linewidth=0.8
)

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Heading (degrees)"
)

plt.title(
    "Vehicle Heading from IMU Gyroscope"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    OUTPUT_HEADING_PNG,
    dpi=300
)


# ============================================================
# PLOT — YAW RATE
# ============================================================

plt.figure(
    figsize=(14, 6)
)

plt.plot(
    timestamps,
    yaw_rate_deg,
    linewidth=1.2
)

plt.axhline(
    0,
    linestyle="--",
    linewidth=0.8
)

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Yaw rate (degrees/s)"
)

plt.title(
    "Vehicle Yaw Rate"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    OUTPUT_YAW_RATE_PNG,
    dpi=300
)


# ============================================================
# PLOT — SPEED
# ============================================================

plt.figure(
    figsize=(14, 6)
)

plt.plot(
    timestamps,
    forward_velocity,
    linewidth=1.5
)

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Forward speed (m/s)"
)

plt.title(
    "Estimated Vehicle Forward Speed"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    OUTPUT_SPEED_PNG,
    dpi=300
)


# ============================================================
# PLOT — VEHICLE TRAJECTORY
# ============================================================

plt.figure(
    figsize=(10, 10)
)

plt.plot(
    position[:, 0],
    position[:, 1],
    linewidth=2,
    label="IMU trajectory"
)


# ============================================================
# START / END
# ============================================================

plt.scatter(
    position[0, 0],
    position[0, 1],
    s=120,
    marker="o",
    label="Start"
)

plt.scatter(
    position[-1, 0],
    position[-1, 1],
    s=120,
    marker="X",
    label="End"
)


# ============================================================
# HEADING ARROWS
# ============================================================

arrow_count = 30

indices = np.linspace(
    0,
    len(position) - 1,
    arrow_count,
    dtype=int
)


# Automatically choose arrow size based on trajectory scale.

trajectory_scale = max(
    np.ptp(position[:, 0]),
    np.ptp(position[:, 1]),
    1.0
)

heading_length = (
    trajectory_scale * 0.02
)


for i in indices:

    dx = (
        np.cos(yaw[i])
        * heading_length
    )

    dy = (
        np.sin(yaw[i])
        * heading_length
    )

    plt.arrow(
        position[i, 0],
        position[i, 1],
        dx,
        dy,
        length_includes_head=True,
        head_width=heading_length * 0.25,
        alpha=0.7
    )


# ============================================================
# LABEL TURNING EVENTS
# ============================================================

TURN_THRESHOLD = 2.0

for i in range(
    1,
    len(yaw_rate_deg)
):

    # --------------------------------------------------------
    # Positive yaw rate
    # --------------------------------------------------------

    if (
        yaw_rate_deg[i] > TURN_THRESHOLD
        and yaw_rate_deg[i - 1] <= TURN_THRESHOLD
    ):

        plt.text(
            position[i, 0],
            position[i, 1],
            " L",
            fontsize=10
        )

    # --------------------------------------------------------
    # Negative yaw rate
    # --------------------------------------------------------

    elif (
        yaw_rate_deg[i] < -TURN_THRESHOLD
        and yaw_rate_deg[i - 1] >= -TURN_THRESHOLD
    ):

        plt.text(
            position[i, 0],
            position[i, 1],
            " R",
            fontsize=10
        )


# ============================================================
# FORMAT TRAJECTORY
# ============================================================

plt.xlabel(
    "X position (m)"
)

plt.ylabel(
    "Y position (m)"
)

plt.title(
    "Heading-Based Vehicle Trajectory from IMU"
)

plt.axis("equal")

plt.grid(True)

plt.legend()

plt.tight_layout()


# ============================================================
# SAVE TRAJECTORY
# ============================================================

plt.savefig(
    OUTPUT_TRAJECTORY_PNG,
    dpi=300
)


# ============================================================
# OUTPUT
# ============================================================

print()
print("=" * 60)
print("OUTPUT")
print("=" * 60)

print(
    f"Trajectory:\n"
    f"{OUTPUT_TRAJECTORY_PNG}"
)

print(
    f"\nHeading:\n"
    f"{OUTPUT_HEADING_PNG}"
)

print(
    f"\nYaw rate:\n"
    f"{OUTPUT_YAW_RATE_PNG}"
)

print(
    f"\nSpeed:\n"
    f"{OUTPUT_SPEED_PNG}"
)

print(
    f"\nData:\n"
    f"{OUTPUT_CSV}"
)

print()
print("Done.")


plt.show()