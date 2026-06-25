"""Tests for the filter initialization-eligibility gate.

Root cause these guard against (see investigation 2026-06-25): under software
rendering the very first fused dock pose at startup is often a SINGLE-marker
estimate, which suffers the planar PnP flip/tilt ambiguity. The KF used to
initialize on it unconditionally, anchoring a tilted pose with 100x-inflated
covariance that took seconds to correct. The gate refuses to initialize until
the fused measurement is backed by enough markers to constrain orientation.
"""

from perception.aruco.lib.init_policy import is_initialization_eligible


def test_single_marker_is_not_eligible():
    # One marker cannot resolve the planar flip ambiguity -> never init on it.
    assert is_initialization_eligible(num_markers=1, min_markers=2) is False


def test_zero_markers_is_not_eligible():
    assert is_initialization_eligible(num_markers=0, min_markers=2) is False


def test_exactly_min_markers_is_eligible():
    assert is_initialization_eligible(num_markers=2, min_markers=2) is True


def test_above_min_markers_is_eligible():
    assert is_initialization_eligible(num_markers=5, min_markers=3) is True


def test_below_min_markers_is_not_eligible():
    assert is_initialization_eligible(num_markers=2, min_markers=3) is False
