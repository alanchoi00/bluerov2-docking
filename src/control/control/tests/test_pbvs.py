"""Contract tests for the coarse-approach PBVS control law."""

import numpy as np
import pytest

from control.pbvs import (
    CmdVel,
    CoarsePbvsController,
    CoarsePbvsParams,
    approach_speed_limit,
)


def make_params(**overrides) -> CoarsePbvsParams:
    base = dict(
        kp_surge=0.4,
        kd_surge=0.0,
        kp_sway=0.5,
        kd_sway=0.1,
        kp_heave=0.5,
        kd_heave=0.1,
        kp_yaw=0.8,
        kd_yaw=0.1,
        # generous limits so the arithmetic tests do not hit saturation
        v_max_surge=1.0,
        v_max_sway=1.0,
        v_max_heave=1.0,
        v_max_yaw=1.0,
    )
    base.update(overrides)
    return CoarsePbvsParams(**base)


def controller(**overrides) -> CoarsePbvsController:
    return CoarsePbvsController(make_params(**overrides))


def test_zero_error_gives_zero_command():
    cmd = controller().step(np.array([0.0, 0.0, 0.0]), yaw_err=0.0, dt=0.1)
    assert isinstance(cmd, CmdVel)
    assert cmd.surge == pytest.approx(0.0)
    assert cmd.sway == pytest.approx(0.0)
    assert cmd.heave == pytest.approx(0.0)
    assert cmd.yaw_rate == pytest.approx(0.0)


@pytest.mark.parametrize(
    "range_ahead, v_max_surge, expected",
    [
        (2.0, 1.0, 0.8),  # plain P: 0.4 * 2.0
        (1.0, 1.0, 0.4),  # plain P: 0.4 * 1.0
        (-0.3, 1.0, -0.12),  # overshot the standoff -> reverse to back out
        (10.0, 0.5, 0.5),  # saturates at +v_max_surge
        (-10.0, 0.5, -0.5),  # saturates at -v_max_surge (reverse)
    ],
)
def test_surge_first_step(range_ahead, v_max_surge, expected):
    cmd = controller(v_max_surge=v_max_surge).step(
        np.array([range_ahead, 0.0, 0.0]), 0.0, 0.1
    )
    assert cmd.surge == pytest.approx(expected)


def test_surge_brakes_on_closing_velocity():
    # kd_surge damps the closing velocity. First step is pure P (no prev); as
    # range_ahead shrinks (closing on the target) the derivative term subtracts,
    # braking the surge below the pure-P value so the vehicle does not coast in.
    ctrl = controller(kp_surge=1.0, kd_surge=0.5)
    first = ctrl.step(np.array([1.0, 0.0, 0.0]), 0.0, 0.1)
    second = ctrl.step(np.array([0.9, 0.0, 0.0]), 0.0, 0.1)
    assert first.surge == pytest.approx(1.0)  # pure P on the first step
    # 1.0 * 0.9 + 0.5 * (0.9 - 1.0) / 0.1 = 0.9 - 0.5 = 0.4
    assert second.surge == pytest.approx(0.4)
    assert second.surge < 1.0 * 0.9


@pytest.mark.parametrize(
    "lateral, v_max_sway, expected",
    [
        (0.2, 1.0, 0.1),  # pure P on the first step: 0.5 * 0.2
        (-0.2, 1.0, -0.1),  # antisymmetric in sign
        (10.0, 0.3, 0.3),  # saturates at v_max_sway
    ],
)
def test_sway_first_step(lateral, v_max_sway, expected):
    cmd = controller(v_max_sway=v_max_sway).step(
        np.array([0.5, lateral, 0.0]), 0.0, 0.1
    )
    assert cmd.sway == pytest.approx(expected)


def test_sway_second_step_adds_derivative():
    c = controller()
    c.step(np.array([0.5, 0.2, 0.0]), 0.0, 0.1)  # seeds prev_lateral = 0.2
    cmd = c.step(np.array([0.5, 0.1, 0.0]), 0.0, 0.1)
    # P = 0.5 * 0.1 = 0.05 ; D = 0.1 * (0.1 - 0.2)/0.1 = -0.1 ; sum = -0.05
    assert cmd.sway == pytest.approx(0.5 * 0.1 + 0.1 * (0.1 - 0.2) / 0.1)


def test_heave_first_step_is_pure_p():
    cmd = controller().step(np.array([0.5, 0.0, 0.2]), 0.0, 0.1)
    assert cmd.heave == pytest.approx(0.5 * 0.2)


def test_heave_second_step_adds_derivative():
    c = controller()
    c.step(np.array([0.5, 0.0, 0.2]), 0.0, 0.1)
    cmd = c.step(np.array([0.5, 0.0, 0.1]), 0.0, 0.1)
    assert cmd.heave == pytest.approx(0.5 * 0.1 + 0.1 * (0.1 - 0.2) / 0.1)


def test_yaw_first_step_is_pure_p():
    cmd = controller().step(np.array([0.5, 0.0, 0.0]), yaw_err=0.2, dt=0.1)
    assert cmd.yaw_rate == pytest.approx(0.8 * 0.2)  # 0.16


def test_yaw_second_step_adds_derivative():
    c = controller()
    c.step(np.array([0.5, 0.0, 0.0]), 0.2, 0.1)
    cmd = c.step(np.array([0.5, 0.0, 0.0]), 0.1, 0.1)
    # P = 0.8*0.1 = 0.08 ; D = 0.1*(0.1-0.2)/0.1 = -0.1 ; sum = -0.02
    assert cmd.yaw_rate == pytest.approx(0.8 * 0.1 + 0.1 * (0.1 - 0.2) / 0.1)


def test_reset_clears_derivative_state():
    c = controller()
    c.step(np.array([0.5, 1.0, 0.0]), 0.0, 0.1)  # seeds a large prev error
    c.reset()
    # after reset the first step must be pure P again (no phantom derivative)
    cmd = c.step(np.array([0.5, 0.2, 0.0]), 0.0, 0.1)
    assert cmd.sway == pytest.approx(0.5 * 0.2)


def test_all_commands_within_limits():
    rng = np.random.default_rng(42)
    limits = dict(v_max_surge=0.5, v_max_sway=0.3, v_max_heave=0.3, v_max_yaw=0.5)
    c = controller(**limits)
    for _ in range(100):
        rel = rng.uniform(-5.0, 5.0, size=3)
        yaw = rng.uniform(-np.pi, np.pi)
        cmd = c.step(rel, yaw, 0.1)
        assert abs(cmd.surge) <= limits["v_max_surge"] + 1e-9
        assert abs(cmd.sway) <= limits["v_max_sway"] + 1e-9
        assert abs(cmd.heave) <= limits["v_max_heave"] + 1e-9
        assert abs(cmd.yaw_rate) <= limits["v_max_yaw"] + 1e-9


def test_approach_speed_limit_ramp_floor_ceiling():
    assert approach_speed_limit(2.0, 0.2, 0.05, 0.3) == pytest.approx(0.3)  # ceiling
    assert approach_speed_limit(1.0, 0.2, 0.05, 0.3) == pytest.approx(0.2)  # linear ramp
    assert approach_speed_limit(0.1, 0.2, 0.05, 0.3) == pytest.approx(0.05)  # floor
