"""Tests for the dock-pose-in-body-frame computation (the 'measurement' the
controller consumes). Pure geometry; no BlueRovSim needed."""

import numpy as np
import pytest

from dock_signal import dock_pose_in_body, wrap_to_pi


def test_dock_straight_ahead():
    eta = np.zeros(6)  # vehicle at world origin, zero attitude
    rel, yaw_err = dock_pose_in_body(eta, np.array([2.0, 0.0, 0.0]), dock_heading=0.0)
    np.testing.assert_allclose(rel, [2.0, 0.0, 0.0], atol=1e-9)
    assert yaw_err == pytest.approx(0.0)


def test_dock_to_the_left_is_positive_body_y():
    eta = np.zeros(6)
    rel, _ = dock_pose_in_body(eta, np.array([0.0, 1.0, 0.0]), 0.0)
    np.testing.assert_allclose(rel, [0.0, 1.0, 0.0], atol=1e-9)


def test_dock_above_is_positive_body_z():
    eta = np.zeros(6)
    rel, _ = dock_pose_in_body(eta, np.array([0.0, 0.0, 1.0]), 0.0)
    np.testing.assert_allclose(rel, [0.0, 0.0, 1.0], atol=1e-9)


def test_translation_is_subtracted():
    eta = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # vehicle 1 m forward
    rel, _ = dock_pose_in_body(eta, np.array([2.0, 0.0, 0.0]), 0.0)
    np.testing.assert_allclose(rel, [1.0, 0.0, 0.0], atol=1e-9)


def test_vehicle_yawed_90_dock_world_ahead_is_on_the_right():
    # vehicle facing world +y; a dock at world +x is to the vehicle's RIGHT -> body -y
    eta = np.array([0.0, 0.0, 0.0, 0.0, 0.0, np.pi / 2])
    rel, _ = dock_pose_in_body(eta, np.array([2.0, 0.0, 0.0]), 0.0)
    np.testing.assert_allclose(rel, [0.0, -2.0, 0.0], atol=1e-9)


def test_yaw_err_is_dock_heading_minus_vehicle_yaw():
    eta = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.2])
    _, yaw_err = dock_pose_in_body(eta, np.array([1.0, 0.0, 0.0]), dock_heading=0.5)
    assert yaw_err == pytest.approx(0.3)


def test_wrap_to_pi():
    assert wrap_to_pi(np.deg2rad(340.0)) == pytest.approx(np.deg2rad(-20.0))
    assert wrap_to_pi(np.deg2rad(-190.0)) == pytest.approx(np.deg2rad(170.0))
    assert wrap_to_pi(0.0) == pytest.approx(0.0)
