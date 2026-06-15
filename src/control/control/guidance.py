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
    yaw_err: float  # rad; rotate ROV by this to face the dock
    range_to_standoff_m: float
    axis_offset_m: float  # perpendicular distance from ROV to the boresight line
    vertical_error_m: float
    range_to_dock_m: float  # distance from ROV to the dock (aim point)


def compute_guidance(
    dock_pos,
    dock_quat_xyzw,
    rov_pos,
    rov_quat_xyzw,
    aim_offset_in_dock,
    standoff_distance_m: float,
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

    # heading: face the dock (aim point). Well-conditioned everywhere (the dock is
    # always ~standoff_distance ahead), keeps the dock in the camera FOV, and
    # equals boresight alignment once on-axis at the standoff.
    aim_in_body = r_rov.inv().apply(aim - np.asarray(rov_pos, dtype=float))
    yaw_err = math.atan2(aim_in_body[1], aim_in_body[0])
    range_to_dock = float(np.linalg.norm(aim_in_body))

    boresight_world = r_dock.apply(np.array([0.0, 1.0, 0.0]))  # dock +Y
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
        range_to_dock_m=range_to_dock,
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
