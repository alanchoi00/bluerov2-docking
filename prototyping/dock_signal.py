"""Compute the dock pose in the vehicle body frame -- the 'measurement' the
CoarsePbvsController consumes -- from BlueRovSim's world pose `eta`.

This replaces the perception + TF stack in the lightweight harness: instead of
rendering markers and running the filter, we compute the relative pose directly
from the known geometry (optionally perturbed by a local noise sampler).

Body frame: +x forward, +y left, +z up (REP-103). eta = [x, y, z, phi, theta, psi]
with ZYX Euler (yaw psi, pitch theta, roll phi), matching BlueRovSim's J(eta).
"""

import numpy as np
from scipy.spatial.transform import Rotation


def wrap_to_pi(angle: float) -> float:
    """Wrap an angle (rad) to [-pi, pi]."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _rotation_world_from_body(eta: np.ndarray) -> Rotation:
    phi, theta, psi = eta[3], eta[4], eta[5]
    # R = Rz(psi) Ry(theta) Rx(phi) -- body -> world (matches BlueRovSim's J1)
    return Rotation.from_euler("ZYX", [psi, theta, phi])


def dock_pose_in_body(
    eta: np.ndarray,
    dock_pos_world: np.ndarray,
    dock_heading: float,
) -> tuple[np.ndarray, float]:
    """Dock position in the body frame + heading error to face the dock.

    Args:
        eta: vehicle world pose, shape (6,) [x, y, z, phi, theta, psi].
        dock_pos_world: dock position in the world frame, shape (3,).
        dock_heading: world yaw the vehicle should hold to face the dock (rad).

    Returns:
        (rel_pos_body, yaw_err): dock position expressed in the body frame
        (forward, left, up), and the wrapped heading error (rad).
    """
    eta = np.asarray(eta, dtype=float)
    p_rel_world = np.asarray(dock_pos_world, dtype=float) - eta[:3]
    rel_pos_body = _rotation_world_from_body(eta).inv().apply(p_rel_world)
    yaw_err = wrap_to_pi(dock_heading - eta[5])
    return rel_pos_body, yaw_err
