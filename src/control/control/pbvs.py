"""Coarse-approach PBVS control law for BlueROV2 docking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CoarsePbvsParams:
    """Tunable gains + fixed limits. Field names are reused verbatim as ROS
    parameter names by the #30 node, so the tuned YAML loads directly there."""

    kp_surge: float
    kp_sway: float
    kd_sway: float
    kp_heave: float
    kd_heave: float
    kp_yaw: float
    kd_yaw: float

    v_max_surge: float = 0.5  # m/s
    v_max_sway: float = 0.3  # m/s
    v_max_heave: float = 0.3  # m/s
    v_max_yaw: float = 0.5  # rad/s


@dataclass(frozen=True)
class CmdVel:
    """Body-frame velocity command (maps 1:1 to geometry_msgs/Twist in #30)."""

    surge: float
    sway: float
    heave: float
    yaw_rate: float


def clamp(value: float, limit: float) -> float:
    """Symmetric saturation to [-limit, +limit]."""
    return float(np.clip(value, -limit, limit))


def rate(curr: float, prev: float | None) -> float:
    """Finite difference rate (curr - prev). If prev is None, return 0."""
    if prev is None:
        return 0.0
    return curr - prev


def approach_speed_limit(
    range_to_dock_m: float, slope_per_s: float, v_floor: float, v_ceiling: float
) -> float:
    """Distance-gated surge speed cap: allow slope * range_to_dock, floored at
    v_floor (final creep) and ceilinged at v_ceiling. Slows the approach the
    closer the ROV gets to the dock, bounding the worst-case collision speed."""
    return float(np.clip(slope_per_s * range_to_dock_m, v_floor, v_ceiling))


class CoarsePbvsController:
    """Decoupled P/PD regulator: body-frame error to body velocity command."""

    def __init__(self, params: CoarsePbvsParams) -> None:
        self._p = params
        self.reset()

    def reset(self) -> None:
        """Clear derivative state before a fresh approach."""
        self._prev_lateral: float | None = None
        self._prev_vertical: float | None = None
        self._prev_yaw_err: float | None = None

    def step(self, rel_pos_body: np.ndarray, yaw_err: float, dt: float) -> CmdVel:
        """Compute one velocity command from the current relative dock pose."""

        range_ahead, lateral_left, vertical_up = rel_pos_body

        # plain P: linear braking ramp over the last v_max/kp metres, and reverse
        # (negative) if it overshoots the standoff so it backs out instead of
        # coasting into the dock.
        surge = clamp(self._p.kp_surge * range_ahead, self._p.v_max_surge)

        sway = clamp(
            self._p.kp_sway * lateral_left
            + self._p.kd_sway * rate(lateral_left, self._prev_lateral) / dt,
            self._p.v_max_sway,
        )

        heave = clamp(
            self._p.kp_heave * vertical_up
            + self._p.kd_heave * rate(vertical_up, self._prev_vertical) / dt,
            self._p.v_max_heave,
        )

        yaw_rate = clamp(
            self._p.kp_yaw * yaw_err
            + self._p.kd_yaw * rate(yaw_err, self._prev_yaw_err) / dt,
            self._p.v_max_yaw,
        )

        self._prev_lateral = lateral_left
        self._prev_vertical = vertical_up
        self._prev_yaw_err = yaw_err

        return CmdVel(surge, sway, heave, yaw_rate)
