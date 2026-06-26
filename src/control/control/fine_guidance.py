"""Pure fine-alignment guidance: align-then-advance law for terminal docking.

Reuses control.guidance.compute_guidance for the body-frame error geometry
(call it with standoff_distance_m ~= 0 so the target IS the dock entry, not a
standoff). This module adds the terminal-phase policy: do not drive forward
until laterally/vertically/angularly centred, so the vehicle enters the dock
throat straight instead of scraping it. No ROS dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass

from control.pbvs import CmdVel

ALIGNING = 0
SEATED = 1
BLOCKED = 2


@dataclass(frozen=True)
class AlignTol:
    lateral_m: float
    vertical_m: float
    yaw_rad: float


@dataclass(frozen=True)
class SeatedTol:
    range_m: float
    lateral_m: float
    vertical_m: float
    yaw_rad: float
    debounce_cycles: int


def aligned(rel_pos_body, yaw_err: float, tol: AlignTol) -> bool:
    _, lateral, vertical = rel_pos_body

    return bool(
        abs(lateral) < tol.lateral_m
        and abs(vertical) < tol.vertical_m
        and abs(yaw_err) < tol.yaw_rad
    )


def advance_command(cmd: CmdVel, is_aligned: bool) -> CmdVel:
    # Design alternative may explore:
    # a SOFT blend, surge *= clamp(1 - max_axis_err/tol, 0, 1), for a smoother
    # approach but verify it never advances while badly misaligned, and the
    # test asserts the hard-gate behaviour (surge == 0 when not aligned), so
    # keep that contract if I blend.
    return cmd if is_aligned else CmdVel(0, cmd.sway, cmd.heave, cmd.yaw_rate)


def within_seated(
    range_to_dock_m: float, rel_pos_body, yaw_err: float, tol: SeatedTol
) -> bool:
    _, lateral, vertical = rel_pos_body
    return bool(
        range_to_dock_m <= tol.range_m
        and (
            abs(lateral) < tol.lateral_m
            and abs(vertical) < tol.vertical_m
            and abs(yaw_err) < tol.yaw_rad
        )
    )


def decide_seated(
    within_seated: bool, healthy: bool, counter: int, was_seated: bool, debounce: int
) -> tuple[int, bool, int]:
    counter = max(
        0, min(counter + 1 if within_seated and healthy else counter - 1, debounce)
    )
    seated = counter > 0 if was_seated else counter >= debounce
    return (SEATED if seated else ALIGNING), seated, counter
