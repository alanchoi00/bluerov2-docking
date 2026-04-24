"""Single source of truth for ArUco marker placement on the dock body.

Dock frame: X=right, Y=into page, Z=up. Origin at the geometric centre of
the backplate cluster. (Note: the DAE mesh's own model origin sits at
the base-slope, 190mm below our dock frame origin in Z; this constant
translation is irrelevant to fusion since it cancels in the implied-origin
math.)

All markers face the approaching ROV (normal along dock -Y direction).
The marker frame is the OpenCV ArUco convention:
  marker +X: right edge of the printed pattern
  marker +Y: top edge of the printed pattern (= dock +Z)
  marker +Z: out of the marker face (= dock -Y)

That orientation is a +90deg rotation about dock's +X axis, quaternion
(sin(pi/4), 0, 0, cos(pi/4)) = (0.7071, 0, 0, 0.7071) in (x, y, z, w).

Reference: spec Sec.2.5 and project_aruco_research.md.
"""

import math

# All markers face forward (normal = dock -Y). Marker +Y edge points up.
# Rotation axis dock +X, angle +pi/2 -> quaternion (sin(pi/4), 0, 0, cos(pi/4)).
_FACING_FORWARD = (math.sin(math.pi / 4), 0.0, 0.0, math.cos(math.pi / 4))

# (position_xyz_metres, orientation_quaternion_xyzw)
MARKER_POSE_IN_DOCK: dict[
    int, tuple[tuple[float, float, float], tuple[float, float, float, float]]
] = {
    # Front-wing plates, 200x200mm, -Y normal
    201: ((-0.425, -0.315, 0.000), _FACING_FORWARD),
    202: ((0.425, -0.315, 0.000), _FACING_FORWARD),
    # Backplate cluster (centre of back plate)
    301: ((0.000, 0.310, -0.03675), _FACING_FORWARD),  # cluster bottom, 100mm
    401: ((-0.02625, 0.310, 0.042), _FACING_FORWARD),  # cluster top-left, 47.5mm
    402: ((0.02625, 0.310, 0.042), _FACING_FORWARD),  # cluster top-right, 47.5mm
    # Backplate corners, 60x60mm
    302: ((-0.280, 0.310, 0.095), _FACING_FORWARD),  # top-left
    303: ((0.280, 0.310, 0.095), _FACING_FORWARD),  # top-right
    304: ((0.280, 0.310, -0.095), _FACING_FORWARD),  # bottom-right
    305: ((-0.280, 0.310, -0.095), _FACING_FORWARD),  # bottom-left
}

MARKER_SIZE: dict[int, float] = {
    201: 0.200,
    202: 0.200,
    301: 0.100,
    302: 0.060,
    303: 0.060,
    304: 0.060,
    305: 0.060,
    401: 0.0475,
    402: 0.0475,
}

EXPECTED_MARKER_IDS: tuple[int, ...] = tuple(sorted(MARKER_POSE_IN_DOCK.keys()))
