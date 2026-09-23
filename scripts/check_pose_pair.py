import numpy as np

POSE_DIR = "a2d2_data_full/dataset/sequences/00/pose"

FRAMES = [0, 10]


def load_pose(frame):
    path = f"{POSE_DIR}/{frame:06d}.txt"
    T = np.loadtxt(path)

    assert T.shape == (4, 4)

    return T


def rotation_angle_deg(R):
    value = (np.trace(R) - 1.0) / 2.0
    value = np.clip(value, -1.0, 1.0)
    return np.degrees(np.arccos(value))


print("=" * 70)
print("A2D2 TWO-FRAME POSE DIAGNOSTIC")
print("=" * 70)

T0 = load_pose(FRAMES[0])
T1 = load_pose(FRAMES[1])

R0 = T0[:3, :3]
R1 = T1[:3, :3]

p0 = T0[:3, 3]
p1 = T1[:3, 3]

print("\nFRAME 0")
print(T0)

print("\nFRAME 10")
print(T1)


# ============================================================
# Relative transform
# ============================================================

T_rel = np.linalg.inv(T0) @ T1

R_rel = T_rel[:3, :3]
t_rel = T_rel[:3, 3]

print("\n" + "=" * 70)
print("RELATIVE TRANSFORM: inv(T0) @ T10")
print("=" * 70)

print("\nT_relative:")
print(T_rel)

print("\nRelative translation:")
print(t_rel)

print(
    "\nRelative translation distance:",
    np.linalg.norm(t_rel)
)

print(
    "Relative rotation:",
    rotation_angle_deg(R_rel),
    "degrees"
)


# ============================================================
# Translation in WORLD coordinates
# ============================================================

world_delta = p1 - p0

print("\n" + "=" * 70)
print("WORLD POSITION CHANGE")
print("=" * 70)

print("\nP0:")
print(p0)

print("\nP10:")
print(p1)

print("\nP10 - P0:")
print(world_delta)

print(
    "\nWorld displacement:",
    np.linalg.norm(world_delta)
)


# ============================================================
# Camera basis vectors
# ============================================================

print("\n" + "=" * 70)
print("CAMERA BASIS VECTORS")
print("=" * 70)

print("\nFrame 0:")
print("X:", R0[:, 0])
print("Y:", R0[:, 1])
print("Z:", R0[:, 2])

print("\nFrame 10:")
print("X:", R1[:, 0])
print("Y:", R1[:, 1])
print("Z:", R1[:, 2])


# ============================================================
# Compare each possible forward axis
# ============================================================

print("\n" + "=" * 70)
print("MOTION RELATIVE TO FRAME 0")
print("=" * 70)

for name, axis in [
    ("X", R0[:, 0]),
    ("Y", R0[:, 1]),
    ("Z", R0[:, 2]),
]:

    forward_component = np.dot(world_delta, axis)

    print(
        f"\nWorld displacement projected onto frame-0 {name}: "
        f"{forward_component:.6f} m"
    )


# ============================================================
# Relative rotation matrix sanity
# ============================================================

print("\n" + "=" * 70)
print("RELATIVE ROTATION MATRIX")
print("=" * 70)

print(R_rel)

print("\ndeterminant:")
print(np.linalg.det(R_rel))

print("\northogonality error:")
print(np.linalg.norm(R_rel.T @ R_rel - np.eye(3)))


# ============================================================
# Alternative relative transform
# ============================================================

T_rel_alt = T1 @ np.linalg.inv(T0)

print("\n" + "=" * 70)
print("ALTERNATIVE: T10 @ inv(T0)")
print("=" * 70)

print(T_rel_alt)

print("\nRotation angle:")
print(
    rotation_angle_deg(
        T_rel_alt[:3, :3]
    )
)

print("\nTranslation:")
print(
    T_rel_alt[:3, 3]
)


print("\n" + "=" * 70)
print("DONE")
print("=" * 70)