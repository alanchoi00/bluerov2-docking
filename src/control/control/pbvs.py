"""Coarse-approach PBVS control law for BlueROV2 docking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CoarsePbvsParams:
    """Tunable gains + fixed limits."""

    kp_surge: float
    kd_surge: float
    kp_sway: float
    kd_sway: float
    kp_heave: float
    kd_heave: float
    ki_heave: float  # integral gain: closes the buoyancy steady-state offset
    kp_yaw: float
    kd_yaw: float

    v_max_surge: float  # m/s
    v_max_sway: float  # m/s
    v_max_heave: float  # m/s
    v_max_yaw: float  # rad/s
    i_max_heave: float  # anti-windup: max heave (m/s) the integral term alone may add


@dataclass(frozen=True)
class CmdVel:
    """Body-frame velocity command."""

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
        self._prev_forward: float | None = None
        self._prev_lateral: float | None = None
        self._prev_vertical: float | None = None
        self._prev_yaw_err: float | None = None
        # Accumulated vertical error for integral heave action. Cleared here so a
        # re-engaged approach (phase gate / health block calls reset()) does not
        # carry stale windup from a previous run.
        self._int_vertical: float = 0.0

    def step(self, rel_pos_body: np.ndarray, yaw_err: float, dt: float) -> CmdVel:
        """Compute one velocity command from the current relative dock pose."""

        range_ahead, lateral_left, vertical_up = rel_pos_body

        # PD on the forward axis: kd_surge damps the closing velocity so the
        # vehicle brakes as it nears the target instead of coasting through it
        # (there is no physical stop at the dock). kd_surge=0 -> pure P.
        surge = clamp(
            self._p.kp_surge * range_ahead
            + self._p.kd_surge * rate(range_ahead, self._prev_forward) / dt,
            self._p.v_max_surge,
        )

        sway = clamp(
            self._p.kp_sway * lateral_left
            + self._p.kd_sway * rate(lateral_left, self._prev_lateral) / dt,
            self._p.v_max_sway,
        )

        # PID on the vertical axis. The vehicle is positively buoyant: a constant
        # upward disturbance that pure P/PD cannot hold against, so it settles a
        # few cm high and the body bumps the top of the dock. The integral term
        # accumulates the residual error to supply the steady down-thrust that
        # holds the vehicle at the true dock height. Anti-windup lives in
        # _update_vertical_integral so a saturated heave does not wind it up.
        self._int_vertical = self._update_vertical_integral(vertical_up, dt)
        heave = clamp(
            self._p.kp_heave * vertical_up
            + self._p.kd_heave * rate(vertical_up, self._prev_vertical) / dt
            + self._p.ki_heave * self._int_vertical,
            self._p.v_max_heave,
        )

        yaw_rate = clamp(
            self._p.kp_yaw * yaw_err
            + self._p.kd_yaw * rate(yaw_err, self._prev_yaw_err) / dt,
            self._p.v_max_yaw,
        )

        self._prev_forward = range_ahead
        self._prev_lateral = lateral_left
        self._prev_vertical = vertical_up
        self._prev_yaw_err = yaw_err

        return CmdVel(surge, sway, heave, yaw_rate)

    def _update_vertical_integral(self, vertical_up: float, dt: float) -> float:
        """Accumulate the vertical error for integral heave action, with anti-windup.

        self._int_vertical holds the bare error integral; ki_heave is applied once,
        in step(). Folding ki in here would double-apply the gain and decay the
        accumulated history each tick (a leaky filter, not an integrator).
        """
        if self._p.ki_heave == 0.0:
            return 0.0

        # Clamp the integral so its output contribution ki*int stays within
        # +/- i_max_heave: anti-windup.
        self._int_vertical = clamp(
            self._int_vertical + vertical_up * dt,
            self._p.i_max_heave / self._p.ki_heave,
        )
        return self._int_vertical
