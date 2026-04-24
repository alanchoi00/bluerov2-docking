"""Spatial outlier rejection via pairwise rotation consistency.

Implements Kim, Bong, Jeong 2024 (Appl. Sci. 14:10225, Sec.3.2):
compute pairwise geodesic rotation distances, count how many pairings each
marker disagrees on (above threshold), iteratively drop the marker with the
highest disagreement count.

Fallback: when the number of markers is below min_for_check, skip the check
entirely (there's no meaningful "majority" with <3 markers).
"""

from dataclasses import dataclass

import numpy as np

from perception.aruco.lib.geometry import geodesic_distance


@dataclass(frozen=True)
class MarkerCandidate:
    """A per-marker candidate implied dock-origin pose in camera frame."""

    marker_id: int
    position: np.ndarray  # shape (3,)
    orientation: np.ndarray  # quaternion (x, y, z, w)


def filter_consistent(
    candidates: list[MarkerCandidate],
    threshold_rad: float,
    min_for_check: int,
) -> list[MarkerCandidate]:
    """Iteratively drop markers whose rotation disagrees most with the rest."""
    if len(candidates) < min_for_check:
        return list(candidates)

    survivors = list(candidates)

    while len(survivors) >= min_for_check:
        n = len(survivors)
        disagreement_count = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                d = geodesic_distance(
                    survivors[i].orientation, survivors[j].orientation
                )
                if d > threshold_rad:
                    disagreement_count[i] += 1
                    disagreement_count[j] += 1

        max_count = max(disagreement_count)
        if max_count == 0:
            # everyone agrees with everyone done
            break

        # drop the marker with the highest disagreement count; ties go to
        # the first-indexed one (deterministic)
        worst_index = disagreement_count.index(max_count)
        survivors.pop(worst_index)

    return survivors
