"""Tests for the filter initialization-eligibility gate."""

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
