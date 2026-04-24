import numpy as np
import pytest

from perception.aruco.lib.geometry import (
    quat_multiply,
    quat_inverse,
    quat_to_rotvec,
    rotvec_to_quat,
    geodesic_distance,
    slerp,
)


# Quaternion convention: (x, y, z, w) matches geometry_msgs/Quaternion.


def test_quat_multiply_identity():
    q = np.array([0.1, 0.2, 0.3, 0.9273618])  # arbitrary unit quat
    q = q / np.linalg.norm(q)
    identity = np.array([0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(quat_multiply(q, identity), q, atol=1e-9)
    np.testing.assert_allclose(quat_multiply(identity, q), q, atol=1e-9)


def test_quat_inverse_roundtrip():
    q = np.array([0.1, 0.2, 0.3, 0.9273618])
    q = q / np.linalg.norm(q)
    q_inv = quat_inverse(q)
    product = quat_multiply(q, q_inv)
    np.testing.assert_allclose(product, [0.0, 0.0, 0.0, 1.0], atol=1e-9)


def test_rotvec_quat_roundtrip_small_angle():
    r = np.array([0.01, 0.02, -0.03])  # small rotation
    q = rotvec_to_quat(r)
    r_back = quat_to_rotvec(q)
    np.testing.assert_allclose(r_back, r, atol=1e-9)


def test_rotvec_quat_roundtrip_pi_over_four():
    # rotation of pi/4 around z
    r = np.array([0.0, 0.0, np.pi / 4])
    q = rotvec_to_quat(r)
    np.testing.assert_allclose(
        q, [0.0, 0.0, np.sin(np.pi / 8), np.cos(np.pi / 8)], atol=1e-9
    )
    r_back = quat_to_rotvec(q)
    np.testing.assert_allclose(r_back, r, atol=1e-9)


def test_geodesic_distance_identity_is_zero():
    q = np.array([0.0, 0.0, 0.0, 1.0])
    assert geodesic_distance(q, q) == pytest.approx(0.0)


def test_geodesic_distance_symmetric():
    q1 = rotvec_to_quat(np.array([0.1, 0.0, 0.0]))
    q2 = rotvec_to_quat(np.array([0.0, 0.2, 0.0]))
    d12 = geodesic_distance(q1, q2)
    d21 = geodesic_distance(q2, q1)
    np.testing.assert_allclose(d12, d21, atol=1e-9)


def test_geodesic_distance_known_angle():
    # two quats, 0.5 rad apart about x
    q1 = rotvec_to_quat(np.array([0.0, 0.0, 0.0]))
    q2 = rotvec_to_quat(np.array([0.5, 0.0, 0.0]))
    assert geodesic_distance(q1, q2) == pytest.approx(0.5, abs=1e-9)


def test_slerp_endpoints():
    q1 = rotvec_to_quat(np.array([0.0, 0.0, 0.0]))
    q2 = rotvec_to_quat(np.array([0.0, 0.0, 1.0]))
    np.testing.assert_allclose(slerp(q1, q2, 0.0), q1, atol=1e-9)
    np.testing.assert_allclose(slerp(q1, q2, 1.0), q2, atol=1e-9)


def test_slerp_midpoint_halfway():
    q1 = rotvec_to_quat(np.array([0.0, 0.0, 0.0]))
    q2 = rotvec_to_quat(np.array([0.0, 0.0, 1.0]))
    mid = slerp(q1, q2, 0.5)
    expected = rotvec_to_quat(np.array([0.0, 0.0, 0.5]))
    np.testing.assert_allclose(mid, expected, atol=1e-9)
