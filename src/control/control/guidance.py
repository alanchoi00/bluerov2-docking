"""Pure coarse-approach guidance: dock pose -> body-frame error to a standoff
point on the dock entry axis (boresight). No ROS dependencies."""

from __future__ import annotations

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
