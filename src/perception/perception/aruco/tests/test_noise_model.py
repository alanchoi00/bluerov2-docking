import numpy as np
import pytest

from perception.aruco.lib.noise_model import (
    marker_position_covariance,
    marker_rotation_covariance,
)


# Reasonable alpha for the active Candidate B kernel (sigma = alpha * r^2 / s).
# alpha=0.001 gives ~1cm std at 1m for a 100mm marker, matching typical ArUco PnP.
# Tests below assert INVARIANTS (monotonicity, positive-definiteness), not
# specific numeric values, so they pass for any sensible kernel+alpha.
TEST_ALPHA = 0.001


def test_returns_3x3_positive_definite():
    cov = marker_position_covariance(range_m=1.0, marker_size_m=0.1, alpha=TEST_ALPHA)
    assert cov.shape == (3, 3)
    # Positive definite: all eigenvalues > 0
    eigvals = np.linalg.eigvalsh(cov)
    assert np.all(eigvals > 0), f"Covariance not PD: eigvals {eigvals}"


def test_covariance_grows_with_range():
    # At fixed marker size, a 2m marker should have larger cov than a 0.5m one
    cov_near = marker_position_covariance(
        range_m=0.5, marker_size_m=0.1, alpha=TEST_ALPHA
    )
    cov_far = marker_position_covariance(
        range_m=2.0, marker_size_m=0.1, alpha=TEST_ALPHA
    )
    assert np.linalg.det(cov_far) > np.linalg.det(cov_near)


def test_covariance_shrinks_with_marker_size():
    # At fixed range, a bigger marker should have smaller cov
    cov_small = marker_position_covariance(
        range_m=1.0, marker_size_m=0.047, alpha=TEST_ALPHA
    )
    cov_big = marker_position_covariance(
        range_m=1.0, marker_size_m=0.200, alpha=TEST_ALPHA
    )
    assert np.linalg.det(cov_big) < np.linalg.det(cov_small)


def test_symmetric():
    cov = marker_position_covariance(range_m=1.5, marker_size_m=0.1, alpha=TEST_ALPHA)
    np.testing.assert_allclose(cov, cov.T, atol=1e-12)


def test_alpha_is_usable_at_different_values():
    # Kernel should produce valid (PD) covariance across a reasonable alpha range.
    for alpha in (1e-4, 1e-3, 1e-2, 1e-1):
        cov = marker_position_covariance(range_m=1.0, marker_size_m=0.1, alpha=alpha)
        eigvals = np.linalg.eigvalsh(cov)
        assert np.all(eigvals > 0), f"Not PD at alpha={alpha}: {eigvals}"


def test_rotation_returns_3x3_positive_definite():
    cov = marker_rotation_covariance(range_m=1.0, marker_size_m=0.1, alpha=TEST_ALPHA)
    assert cov.shape == (3, 3)
    eigvals = np.linalg.eigvalsh(cov)
    assert np.all(eigvals > 0)


def test_rotation_covariance_grows_with_range():
    cov_near = marker_rotation_covariance(
        range_m=0.5, marker_size_m=0.1, alpha=TEST_ALPHA
    )
    cov_far = marker_rotation_covariance(
        range_m=2.0, marker_size_m=0.1, alpha=TEST_ALPHA
    )
    assert np.linalg.det(cov_far) > np.linalg.det(cov_near)


def test_rotation_covariance_shrinks_with_marker_size():
    cov_small = marker_rotation_covariance(
        range_m=1.0, marker_size_m=0.047, alpha=TEST_ALPHA
    )
    cov_big = marker_rotation_covariance(
        range_m=1.0, marker_size_m=0.200, alpha=TEST_ALPHA
    )
    assert np.linalg.det(cov_big) < np.linalg.det(cov_small)


def test_rotation_scales_one_power_less_than_position():
    # sigma_pos ~ r^2 / s, sigma_rot ~ r / s -> at fixed s, doubling r should
    # quadruple sigma_pos (16x variance) but only double sigma_rot (4x variance).
    r1, r2 = 1.0, 2.0
    s = 0.1
    pos1 = marker_position_covariance(range_m=r1, marker_size_m=s, alpha=TEST_ALPHA)[
        0, 0
    ]
    pos2 = marker_position_covariance(range_m=r2, marker_size_m=s, alpha=TEST_ALPHA)[
        0, 0
    ]
    rot1 = marker_rotation_covariance(range_m=r1, marker_size_m=s, alpha=TEST_ALPHA)[
        0, 0
    ]
    rot2 = marker_rotation_covariance(range_m=r2, marker_size_m=s, alpha=TEST_ALPHA)[
        0, 0
    ]
    assert pos2 / pos1 == pytest.approx(16.0, rel=1e-9)
    assert rot2 / rot1 == pytest.approx(4.0, rel=1e-9)
