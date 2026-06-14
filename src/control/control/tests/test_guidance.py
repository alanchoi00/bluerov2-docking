"""Black-box tests for the pure coarse-approach guidance math."""

import math

import numpy as np

from control.guidance import compute_guidance, standoff_point_in_target

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


def test_rov_at_origin_facing_x():
    # Dock+ROV at origin, both identity. Standoff is at world (0,-0.690,0.042).
    # ROV faces world +X, so the standoff is to its right (-Y=right) and slightly up.
    g = compute_guidance(
        dock_pos=(0.0, 0.0, 0.0),
        dock_quat_xyzw=(0.0, 0.0, 0.0, 1.0),
        rov_pos=(0.0, 0.0, 0.0),
        rov_quat_xyzw=(0.0, 0.0, 0.0, 1.0),
        aim_offset_in_dock=AIM_OFFSET,
        standoff_distance_m=1.0,
    )
    # rel_pos_body = R_rov^-1 (standoff - rov) = standoff (identity) = [fwd, left, up]
    assert np.allclose(g.rel_pos_body, [0.0, -0.690, 0.042], atol=1e-6)
    # boresight = dock +Y = world +Y; in body that is +left -> yaw_err = +pi/2
    assert math.isclose(g.yaw_err, math.pi / 2, abs_tol=1e-6)


def test_on_axis_at_standoff_is_zeroed():
    # ROV sitting AT the standoff point, facing dock +Y (turned +90deg about Z).
    standoff = (0.0, -0.690, 0.042)
    rov_q = (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))  # +X -> +Y
    g = compute_guidance(
        dock_pos=(0.0, 0.0, 0.0),
        dock_quat_xyzw=(0.0, 0.0, 0.0, 1.0),
        rov_pos=standoff,
        rov_quat_xyzw=rov_q,
        aim_offset_in_dock=AIM_OFFSET,
        standoff_distance_m=1.0,
    )
    assert np.allclose(g.rel_pos_body, [0.0, 0.0, 0.0], atol=1e-6)
    assert math.isclose(g.yaw_err, 0.0, abs_tol=1e-6)
    assert math.isclose(g.range_to_standoff_m, 0.0, abs_tol=1e-6)
    assert math.isclose(g.axis_offset_m, 0.0, abs_tol=1e-6)


def test_axis_offset_detects_off_boresight():
    # ROV offset +0.5 in world X from the boresight line (which runs along world Y
    # through x=0, z=0.042 for an identity dock). Perp distance should be 0.5.
    g = compute_guidance(
        dock_pos=(0.0, 0.0, 0.0),
        dock_quat_xyzw=(0.0, 0.0, 0.0, 1.0),
        rov_pos=(0.5, -0.690, 0.042),
        rov_quat_xyzw=(0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)),
        aim_offset_in_dock=AIM_OFFSET,
        standoff_distance_m=1.0,
    )
    assert math.isclose(g.axis_offset_m, 0.5, abs_tol=1e-6)
