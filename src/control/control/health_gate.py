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


# Phase constants (mirror interfaces/msg/CoarseApproachStatus.msg).
APPROACHING = 0
AT_STANDOFF = 1
BLOCKED = 2


@dataclass(frozen=True)
class Tolerances:
    position_m: float
    axis_offset_m: float
    heading_rad: float
    debounce_cycles: int


def within_tolerances(
    range_to_standoff_m: float,
    axis_offset_m: float,
    heading_err_rad: float,
    tol: Tolerances,
) -> tuple[bool, bool]:
    within_pos = (
        range_to_standoff_m < tol.position_m and axis_offset_m < tol.axis_offset_m
    )
    within_head = abs(heading_err_rad) < tol.heading_rad
    return within_pos, within_head


def decide_phase(
    blocked: bool,
    within_pos: bool,
    within_head: bool,
    healthy: bool,
    ready_counter: int,
    was_ready: bool,
    tol: Tolerances,
) -> tuple[int, bool, int]:
    """Return (phase, ready_for_handoff, new_ready_counter).

    Hysteresis via an up/down counter: +1 per in-tolerance, healthy cycle and -1
    otherwise, clamped to [0, debounce_cycles]. Latch AT_STANDOFF once the counter
    reaches debounce_cycles; stay latched until it decays to 0. This needs a
    sustained fall out of tolerance to drop ready, so it does not flicker at the
    tolerance boundary."""
    if blocked:
        return BLOCKED, False, 0
    good = within_pos and within_head and healthy
    counter = ready_counter + 1 if good else ready_counter - 1
    counter = max(0, min(counter, tol.debounce_cycles))
    ready = counter > 0 if was_ready else counter >= tol.debounce_cycles
    return (AT_STANDOFF if ready else APPROACHING), ready, counter
