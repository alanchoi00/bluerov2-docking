"""Pure coarse-approach guidance: dock pose to a body-frame error at a standoff
point on the dock entry axis (boresight). No ROS dependencies."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


def standoff_point_in_target(
    dock_pos,
    dock_quat_xyzw,
    aim_offset_in_dock,
    standoff_distance_m: float,
) -> np.ndarray:
    """Standoff point expressed in the target frame.

    aim = dock_pos + R_dock . aim_offset_in_dock
    standoff = aim + R_dock . (0, -standoff_distance, 0)   # in front along dock -Y
    """
    r_dock = Rotation.from_quat(list(dock_quat_xyzw))
    aim = np.asarray(dock_pos, dtype=float) + r_dock.apply(
        np.asarray(aim_offset_in_dock, dtype=float)
    )
    return aim + r_dock.apply(np.array([0.0, -float(standoff_distance_m), 0.0]))


@dataclass(frozen=True)
class GuidanceResult:
    """Body-frame error to the standoff point plus diagnostics for telemetry."""

    rel_pos_body: np.ndarray  # [forward, left, up], metres
    yaw_err: float  # rad; rotate ROV by this to face the boresight
    range_to_standoff_m: float
    axis_offset_m: float  # perpendicular distance from ROV to the boresight line
    vertical_error_m: float


def _heading_error(rel_pos_body, boresight_body, range_to_standoff, blend_range_m):
    """Yaw error: pursue the standoff point when far (point the nose at it), blend
    to boresight alignment as it is approached. blend_range_m <= 0 disables the
    blend (always boresight). w=1 (boresight) at range 0, w=0 (pursuit) at/beyond
    blend_range_m."""
    bore = np.array([boresight_body[0], boresight_body[1]])
    bore_norm = np.linalg.norm(bore)
    if bore_norm > 1e-9:
        bore = bore / bore_norm
    point = np.array([rel_pos_body[0], rel_pos_body[1]])
    point_norm = np.linalg.norm(point)
    if blend_range_m > 0.0 and point_norm > 1e-6:
        w = float(np.clip(1.0 - range_to_standoff / blend_range_m, 0.0, 1.0))
        blended = w * bore + (1.0 - w) * (point / point_norm)
        return math.atan2(blended[1], blended[0])
    return math.atan2(bore[1], bore[0])


def compute_guidance(
    dock_pos,
    dock_quat_xyzw,
    rov_pos,
    rov_quat_xyzw,
    aim_offset_in_dock,
    standoff_distance_m: float,
    heading_blend_range_m: float = 0.0,
) -> GuidanceResult:
    r_dock = Rotation.from_quat(list(dock_quat_xyzw))
    r_rov = Rotation.from_quat(list(rov_quat_xyzw))

    aim = np.asarray(dock_pos, dtype=float) + r_dock.apply(
        np.asarray(aim_offset_in_dock, dtype=float)
    )
    standoff = aim + r_dock.apply(np.array([0.0, -float(standoff_distance_m), 0.0]))

    rel_world = standoff - np.asarray(rov_pos, dtype=float)
    rel_pos_body = r_rov.inv().apply(rel_world)  # [forward, left, up]
    range_to_standoff = float(np.linalg.norm(rel_pos_body))

    boresight_world = r_dock.apply(np.array([0.0, 1.0, 0.0]))  # dock +Y
    boresight_body = r_rov.inv().apply(boresight_world)
    yaw_err = _heading_error(
        rel_pos_body, boresight_body, range_to_standoff, heading_blend_range_m
    )

    d = np.asarray(rov_pos, dtype=float) - aim
    along = float(np.dot(d, boresight_world))
    perp = d - along * boresight_world
    axis_offset = float(np.linalg.norm(perp))

    return GuidanceResult(
        rel_pos_body=rel_pos_body,
        yaw_err=float(yaw_err),
        range_to_standoff_m=range_to_standoff,
        axis_offset_m=axis_offset,
        vertical_error_m=float(rel_pos_body[2]),
    )


def standoff_pose_in_target(
    dock_pos,
    dock_quat_xyzw,
    aim_offset_in_dock,
    standoff_distance_m: float,
):
    """Standoff point + desired ROV heading in the target frame, for viz.

    Returns (position ndarray, quaternion xyzw tuple). Desired heading aligns
    body +X with the dock boresight (dock +Y), i.e. facing into the dock."""
    r_dock = Rotation.from_quat(list(dock_quat_xyzw))
    pos = standoff_point_in_target(
        dock_pos, dock_quat_xyzw, aim_offset_in_dock, standoff_distance_m
    )
    desired = r_dock * Rotation.from_euler("z", math.pi / 2)
    return pos, tuple(float(q) for q in desired.as_quat())
