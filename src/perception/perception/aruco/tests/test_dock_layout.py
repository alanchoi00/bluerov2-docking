import math

import numpy as np
import pytest

from perception.aruco.lib.dock_layout import (
    MARKER_POSE_IN_DOCK,
    MARKER_SIZE,
    EXPECTED_MARKER_IDS,
)


def test_all_expected_markers_present():
    assert set(MARKER_POSE_IN_DOCK.keys()) == set(EXPECTED_MARKER_IDS)
    assert set(MARKER_SIZE.keys()) == set(EXPECTED_MARKER_IDS)


def test_front_wing_baseline_is_850mm():
    pos_201 = np.array(MARKER_POSE_IN_DOCK[201][0])
    pos_202 = np.array(MARKER_POSE_IN_DOCK[202][0])
    separation = np.linalg.norm(pos_201 - pos_202)
    assert separation == pytest.approx(0.850, abs=1e-6)


def test_backplate_cluster_at_backplate_y():
    for mid in (301, 302, 303, 304, 305, 401, 402):
        assert MARKER_POSE_IN_DOCK[mid][0][1] == pytest.approx(0.310, abs=1e-6)


def test_front_wings_at_front_y():
    for mid in (201, 202):
        assert MARKER_POSE_IN_DOCK[mid][0][1] == pytest.approx(-0.315, abs=1e-6)


def test_all_markers_have_unit_quaternion():
    for mid, (_, quat) in MARKER_POSE_IN_DOCK.items():
        norm = sum(c * c for c in quat) ** 0.5
        assert norm == pytest.approx(1.0, abs=1e-6), f"Marker {mid} has non-unit quaternion"


def test_all_markers_face_forward():
    # All markers face the approaching ROV (normal = dock -Y direction).
    # Per OpenCV ArUco convention: marker +Z = out of face, +Y = up (= dock +Z),
    # +X = right (= dock +X). That's a +90deg rotation about dock +X axis ->
    # quaternion (sin(pi/4), 0, 0, cos(pi/4)) in (x, y, z, w).
    expected = (math.sin(math.pi / 4), 0.0, 0.0, math.cos(math.pi / 4))
    for mid, (_, quat) in MARKER_POSE_IN_DOCK.items():
        for i, (got, want) in enumerate(zip(quat, expected)):
            assert got == pytest.approx(want, abs=1e-9), (
                f"Marker {mid} component {i}: got {got}, want {want}"
            )


def test_marker_sizes_are_positive():
    for mid, size in MARKER_SIZE.items():
        assert size > 0, f"Marker {mid} has non-positive size"


def test_backplate_cluster_vertical_gap_matches_design():
    # 401/402 at z=+0.042, 301 at z=-0.03675. Vertical center-to-center
    # separation should be 78.75mm (per spec Sec.2.5).
    z_401 = MARKER_POSE_IN_DOCK[401][0][2]
    z_301 = MARKER_POSE_IN_DOCK[301][0][2]
    assert (z_401 - z_301) == pytest.approx(0.07875, abs=1e-6)
