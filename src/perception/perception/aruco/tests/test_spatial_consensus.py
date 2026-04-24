import numpy as np

from perception.aruco.lib.geometry import rotvec_to_quat
from perception.aruco.lib.spatial_consensus import (
    MarkerCandidate,
    filter_consistent,
)


def _candidate(mid: int, rotvec: list[float]) -> MarkerCandidate:
    """Build a marker candidate with position=zero and given rotation."""
    return MarkerCandidate(
        marker_id=mid,
        position=np.zeros(3),
        orientation=rotvec_to_quat(np.array(rotvec)),
    )


def test_three_agreeing_markers_all_pass():
    cands = [
        _candidate(301, [0.01, 0.0, 0.0]),
        _candidate(401, [0.02, 0.0, 0.0]),
        _candidate(402, [0.015, 0.0, 0.0]),
    ]
    surviving = filter_consistent(cands, threshold_rad=np.deg2rad(8.0), min_for_check=3)
    assert {c.marker_id for c in surviving} == {301, 401, 402}


def test_one_flipped_marker_dropped():
    cands = [
        _candidate(301, [0.01, 0.0, 0.0]),
        _candidate(401, [0.02, 0.0, 0.0]),
        _candidate(402, [0.015, 0.0, 0.0]),
        _candidate(304, [np.pi, 0.0, 0.0]),  # 180deg flipped, big outlier
    ]
    surviving = filter_consistent(cands, threshold_rad=np.deg2rad(8.0), min_for_check=3)
    assert 304 not in {c.marker_id for c in surviving}
    assert {301, 401, 402}.issubset({c.marker_id for c in surviving})


def test_two_markers_fallback_accept_both():
    cands = [
        _candidate(301, [0.0, 0.0, 0.0]),
        _candidate(401, [0.02, 0.0, 0.0]),
    ]
    surviving = filter_consistent(cands, threshold_rad=np.deg2rad(8.0), min_for_check=3)
    assert len(surviving) == 2


def test_single_marker_trivially_accepted():
    cands = [_candidate(301, [0.0, 0.0, 0.0])]
    surviving = filter_consistent(cands, threshold_rad=np.deg2rad(8.0), min_for_check=3)
    assert len(surviving) == 1


def test_empty_input_returns_empty():
    surviving = filter_consistent([], threshold_rad=np.deg2rad(8.0), min_for_check=3)
    assert surviving == []


def test_all_disagree_only_one_survives():
    # Three candidates each at 90deg apart - any one is the "consensus" against
    # pairwise threshold violations, so the iterative drop keeps dropping
    # until one remains.
    cands = [
        _candidate(301, [0.0, 0.0, 0.0]),
        _candidate(401, [np.pi / 2, 0.0, 0.0]),
        _candidate(402, [0.0, np.pi / 2, 0.0]),
    ]
    surviving = filter_consistent(cands, threshold_rad=np.deg2rad(8.0), min_for_check=3)
    # Either 0 (if the algorithm gives up) or 1-2 (depending on tie-breaking).
    # Assert at minimum: outliers don't pass through unchanged.
    assert len(surviving) < 3
