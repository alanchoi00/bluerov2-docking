"""Pure health gating + phase decision for the coarse approach node.

FilterHealth status integers (mirror interfaces/msg/FilterHealth.msg):
0 WARMING_UP, 1 HEALTHY, 2 DEGRADED, 3 STALE. The node asserts these match
the generated message constants at construction."""

from __future__ import annotations

from dataclasses import dataclass

WARMING_UP = 0
HEALTHY = 1
DEGRADED = 2
STALE = 3


@dataclass(frozen=True)
class GateResult:
    blocked: bool
    gain_scale: float
    dock_healthy: bool


def gate_for_health(status: int, degraded_gain_scale: float) -> GateResult:
    """Map a FilterHealth status to (blocked, gain_scale, dock_healthy).

    STALE/WARMING_UP -> blocked (zero command). DEGRADED -> regulate at the
    de-rated gain, flagged unhealthy. HEALTHY -> regulate at full gain."""
    if status in (STALE, WARMING_UP):
        return GateResult(blocked=True, gain_scale=0.0, dock_healthy=False)
    if status == DEGRADED:
        return GateResult(
            blocked=False, gain_scale=float(degraded_gain_scale), dock_healthy=False
        )
    return GateResult(blocked=False, gain_scale=1.0, dock_healthy=True)
