"""Decision logic for when the dock-pose Kalman filter may initialize.

The filter must NOT anchor itself on an under-determined first measurement.
A single ArUco marker yields a full 6-DOF PnP pose, but its orientation is
subject to the classic planar flip/tilt ambiguity: the corner-reprojection
error of the true pose and a mirrored "flipped" pose are nearly identical, so
the solver can return a pose tilted by tens of degrees with small (over-
confident) reported covariance. Two or more non-collinear markers share a
baseline that breaks the ambiguity, so orientation becomes well-constrained.

This module owns ONLY the eligibility predicate (pure logic, unit-tested). The
node (dock_pose_filter) owns the side effects: while ineligible it stays
WARMING_UP and publishes no dock pose; on the first eligible measurement it
initializes and switches to normal predict/update.
"""


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
