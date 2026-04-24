"""Quaternion and SO(3) helpers thin wrappers over scipy.spatial.transform.

Quaternion convention: (x, y, z, w) scalar-last, matching geometry_msgs/Quaternion
and scipy's default. All functions accept and return numpy arrays so callers
don't need to know about scipy's Rotation objects.
"""

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    return (Rotation.from_quat(q1) * Rotation.from_quat(q2)).as_quat(canonical=True)


def quat_inverse(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]])


def rotvec_to_quat(r: np.ndarray) -> np.ndarray:
    return Rotation.from_rotvec(r).as_quat(canonical=True)


def quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    return Rotation.from_quat(q).as_rotvec()


def geodesic_distance(q1: np.ndarray, q2: np.ndarray) -> float:
    return float((Rotation.from_quat(q1).inv() * Rotation.from_quat(q2)).magnitude())


def slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    rot = Rotation.from_quat(np.vstack([q1, q2]))
    return Slerp([0.0, 1.0], rot)([t]).as_quat(canonical=True).squeeze()
