"""Multi-marker pose fusion via Mahalanobis-weighted mean (Xu 2021, Eqs 13-15).

Given N per-marker candidate dock poses with per-measurement covariances,
produce a single fused pose. Position and orientation are each fused
separately via inverse-covariance weighted mean (position in R^3, orientation
in the tangent space of a reference quaternion). Measurement covariance for
each marker carries BOTH position and rotation uncertainty as independent
3x3 blocks they have different units and different physical origins
(PnP error scales differently for position vs rotation), so combining them
into a coupled block would conflate those.
"""

from dataclasses import dataclass

import numpy as np

from perception.aruco.lib.geometry import (
    quat_inverse,
    quat_multiply,
    quat_to_rotvec,
    rotvec_to_quat,
)


@dataclass(frozen=True)
class MarkerMeasurement:
    """A per-marker dock-origin estimate with its measurement covariances."""

    marker_id: int
    position: np.ndarray  # shape (3,) in camera frame
    orientation: np.ndarray  # quaternion (x, y, z, w) in camera frame
    position_covariance: np.ndarray  # shape (3, 3), metres^2
    orientation_covariance: np.ndarray  # shape (3, 3), radians^2 (tangent-space)


@dataclass(frozen=True)
class FusedPose:
    position: np.ndarray  # (3,)
    orientation: np.ndarray  # quaternion (x, y, z, w)
    position_covariance: np.ndarray  # (3, 3), metres^2
    orientation_covariance: np.ndarray  # (3, 3), radians^2 (tangent-space)


def fuse_markers(measurements: list[MarkerMeasurement]) -> FusedPose:
    if not measurements:
        raise ValueError("fuse_markers requires at least one measurement")

    if len(measurements) == 1:
        m = measurements[0]
        return FusedPose(
            position=m.position.copy(),
            orientation=m.orientation.copy(),
            position_covariance=m.position_covariance.copy(),
            orientation_covariance=m.orientation_covariance.copy(),
        )

    # Position: inverse-covariance weighted mean (Xu 2021 closed form)
    pos_info_sum = np.zeros((3, 3))
    pos_info_weighted_sum = np.zeros(3)
    for m in measurements:
        info = np.linalg.inv(m.position_covariance)
        pos_info_sum += info
        pos_info_weighted_sum += info @ m.position
    fused_pos_cov = np.linalg.inv(pos_info_sum)
    fused_pos = fused_pos_cov @ pos_info_weighted_sum

    # Orientation: inverse-rotation-covariance weighted mean in tangent space
    # of the first marker's quaternion. For small tangent deltas (which is
    # what spatial_consensus ensures), this is a valid first-order fusion.
    q_ref = measurements[0].orientation
    rot_info_sum = np.zeros((3, 3))
    rot_info_weighted_sum = np.zeros(3)
    for m in measurements:
        rot_info = np.linalg.inv(m.orientation_covariance)
        delta = quat_to_rotvec(quat_multiply(quat_inverse(q_ref), m.orientation))
        rot_info_sum += rot_info
        rot_info_weighted_sum += rot_info @ delta
    fused_rot_cov = np.linalg.inv(rot_info_sum)
    fused_tangent = fused_rot_cov @ rot_info_weighted_sum
    fused_orientation = quat_multiply(q_ref, rotvec_to_quat(fused_tangent))

    return FusedPose(
        position=fused_pos,
        orientation=fused_orientation,
        position_covariance=fused_pos_cov,
        orientation_covariance=fused_rot_cov,
    )
