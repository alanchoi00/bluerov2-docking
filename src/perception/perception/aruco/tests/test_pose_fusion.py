import numpy as np
import pytest

from perception.aruco.lib.geometry import rotvec_to_quat
from perception.aruco.lib.pose_fusion import (
    MarkerMeasurement,
    fuse_markers,
)


def _measurement(
    mid: int,
    position: list[float],
    rotvec: list[float] | None = None,
    sigma_pos: float = 0.01,
    sigma_rot: float = 0.01,
) -> MarkerMeasurement:
    return MarkerMeasurement(
        marker_id=mid,
        position=np.array(position),
        orientation=rotvec_to_quat(np.array(rotvec or [0.0, 0.0, 0.0])),
        position_covariance=sigma_pos**2 * np.eye(3),
        orientation_covariance=sigma_rot**2 * np.eye(3),
    )


def test_single_input_passes_through():
    m = _measurement(301, [1.0, 2.0, 3.0])
    result = fuse_markers([m])
    np.testing.assert_allclose(result.position, [1.0, 2.0, 3.0], atol=1e-9)
    np.testing.assert_allclose(
        result.position_covariance, 0.01**2 * np.eye(3), atol=1e-12
    )


def test_two_equal_weight_inputs_gives_midpoint():
    m1 = _measurement(301, [1.0, 0.0, 0.0], sigma_pos=0.01)
    m2 = _measurement(401, [0.0, 1.0, 0.0], sigma_pos=0.01)
    result = fuse_markers([m1, m2])
    np.testing.assert_allclose(result.position, [0.5, 0.5, 0.0], atol=1e-9)


def test_fusion_reduces_covariance():
    # With N identical measurements, fused covariance should be cov / N
    m1 = _measurement(301, [1.0, 0.0, 0.0], sigma_pos=0.02)
    m2 = _measurement(401, [1.0, 0.0, 0.0], sigma_pos=0.02)
    result = fuse_markers([m1, m2])
    expected_variance = (0.02**2) / 2.0
    np.testing.assert_allclose(
        result.position_covariance, expected_variance * np.eye(3), atol=1e-12
    )


def test_unequal_weights_biases_toward_low_cov_input():
    # m1 has tiny variance -> should dominate
    m1 = _measurement(301, [1.0, 0.0, 0.0], sigma_pos=0.001)
    m2 = _measurement(401, [0.0, 0.0, 0.0], sigma_pos=0.1)
    result = fuse_markers([m1, m2])
    assert result.position[0] > 0.9, f"expected biased toward m1, got {result.position}"


def test_empty_input_raises():
    with pytest.raises(ValueError):
        fuse_markers([])


def test_orientation_averaged_in_tangent_space():
    # Two markers with slightly different rotations about z fused should be
    # between them
    m1 = MarkerMeasurement(
        marker_id=301,
        position=np.zeros(3),
        orientation=rotvec_to_quat(np.array([0.0, 0.0, 0.1])),
        position_covariance=0.01**2 * np.eye(3),
        orientation_covariance=0.01**2 * np.eye(3),
    )
    m2 = MarkerMeasurement(
        marker_id=401,
        position=np.zeros(3),
        orientation=rotvec_to_quat(np.array([0.0, 0.0, 0.2])),
        position_covariance=0.01**2 * np.eye(3),
        orientation_covariance=0.01**2 * np.eye(3),
    )
    result = fuse_markers([m1, m2])
    # midpoint rotation ~ 0.15 rad about z
    expected = rotvec_to_quat(np.array([0.0, 0.0, 0.15]))
    # quaternions might differ by sign; compare via geodesic distance
    from perception.aruco.lib.geometry import geodesic_distance

    assert geodesic_distance(result.orientation, expected) < 1e-6
