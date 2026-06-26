"""Pure FSM transition conditions for the docking orchestrator. No ROS deps.

Health status integers mirror interfaces/msg/FilterHealth.msg:
0 WARMING_UP, 1 HEALTHY, 2 DEGRADED, 3 STALE.
"""

WARMING_UP = 0
HEALTHY = 1
DEGRADED = 2
STALE = 3


def is_loss(health_status: int) -> bool:
    """Is loss if health status is either WARMING_UP (0) or STALE (3)"""
    return health_status in (WARMING_UP, STALE)


def is_drift(range_to_dock_m: float, demote_range_m: float) -> bool:
    """Is drift if the ROV is backed out past the demote range."""
    return range_to_dock_m > demote_range_m


def sustained(counter: int, condition: bool, threshold: int) -> tuple[int, bool]:
    """
    Increment counter (clamped at threshold) when condition True, otherwise resets counter to 0.
    Also returns `tripped` (new counter >= threshold) that drives the demote.
    Note: since the FSM ticks at a fixed rate, N cycles is a deterministic.
    """
    new_counter = min(counter + 1, threshold) if condition else 0
    return (new_counter, new_counter >= threshold)
