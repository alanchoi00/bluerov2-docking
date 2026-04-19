import numpy as np


def apply_dock_pose(xyz, rpy, led_offsets):
    """Transform LED offsets from dock-local frame to world frame.

    Args:
        xyz: sequence of 3 floats [x, y, z] metres — dock translation in world
        rpy: sequence of 3 floats [roll, pitch, yaw] radians — dock rotation
        led_offsets: array-like of shape (N, 3) — LED positions in dock frame

    Returns:
        numpy array of shape (N, 3) — LED positions in world frame
    """
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    R = np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )
    offsets = np.asarray(led_offsets, dtype=float)
    return (R @ offsets.T).T + np.asarray(xyz, dtype=float)
