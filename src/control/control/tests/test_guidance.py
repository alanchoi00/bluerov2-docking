"""Black-box tests for the pure coarse-approach guidance math."""

import math

import numpy as np

from control.guidance import standoff_point_in_target

AIM_OFFSET = (0.0, 0.310, 0.042)  # 401/402 midpoint in dock frame


def test_standoff_point_identity_dock():
    # Dock at origin, identity orientation. Aim point = aim offset; standoff
    # sits standoff_distance in front along dock -Y.
    pt = standoff_point_in_target(
        dock_pos=(0.0, 0.0, 0.0),
        dock_quat_xyzw=(0.0, 0.0, 0.0, 1.0),
        aim_offset_in_dock=AIM_OFFSET,
        standoff_distance_m=1.0,
    )
    assert np.allclose(pt, [0.0, -0.690, 0.042], atol=1e-6)


def test_standoff_point_yawed_dock():
    # Dock yawed +90deg about Z: dock +Y now points to world -X, dock -Y -> world +X.
    q = (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))
    pt = standoff_point_in_target(
        dock_pos=(0.0, 0.0, 0.0),
        dock_quat_xyzw=q,
        aim_offset_in_dock=AIM_OFFSET,
        standoff_distance_m=1.0,
    )
    # aim = R*(0,0.310,0.042) = (-0.310, 0, 0.042); standoff = aim + R*(0,-1,0) = aim + (1,0,0)
    assert np.allclose(pt, [0.690, 0.0, 0.042], atol=1e-6)
