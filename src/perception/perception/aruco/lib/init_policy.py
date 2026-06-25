"""Decision logic for when the dock-pose Kalman filter may initialize."""


def is_initialization_eligible(num_markers: int, min_markers: int) -> bool:
    """Return True iff a fused measurement is well-constrained enough to init on.

    Args:
        num_markers: how many markers were fused into THIS measurement (the
            count that survived consensus in aruco_fusion, carried on the
            DockPoseMeasurement message).
        min_markers: the configured minimum (a node parameter). The geometric
            floor that resolves the planar flip ambiguity is 2 non-collinear
            markers; 3 gives margin against a near-collinear pair and matches
            the existing consensus floor (min_markers_for_consensus=3).

    Returns:
        True if the measurement may initialize the filter, False to keep waiting.
    """
    return num_markers >= min_markers
