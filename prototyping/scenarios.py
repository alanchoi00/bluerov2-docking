"""Tuning scenarios: initial vehicle poses relative to a fixed dock.

Dock sits at the world origin with its boresight along world +x; the vehicle
approaches from -x (so a vehicle at x=-2 sees the dock 2 m ahead). Isolation
scenarios excite one axis at a time (small offsets -> near-decoupled) for
per-axis starting gains; the combined scenario validates coupling.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Scenario:
    name: str
    eta0: np.ndarray  # initial vehicle world pose (6,): [x, y, z, phi, theta, psi]
    dock_pos_world: np.ndarray = field(default_factory=lambda: np.zeros(3))
    dock_heading: float = 0.0


def _eta(x=0.0, y=0.0, z=0.0, psi=0.0) -> np.ndarray:
    return np.array([x, y, z, 0.0, 0.0, psi], dtype=float)


# Isolation scenarios sit near the handoff range (x=-0.5) so surge is ~idle and
# the named axis dominates; the range scenario starts 2 m out.
SCENARIOS = [
    Scenario("range", _eta(x=-2.0)),
    Scenario("lateral", _eta(x=-0.5, y=0.3)),
    Scenario("vertical", _eta(x=-0.5, z=0.3)),
    Scenario("yaw", _eta(x=-0.5, psi=0.3)),
    Scenario("combined", _eta(x=-2.0, y=0.4, z=0.3, psi=0.3)),
]
