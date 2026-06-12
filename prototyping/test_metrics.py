"""Tests for step-response metrics. Pure numpy; no BlueRovSim needed."""

import numpy as np
import pytest

from metrics import (
    converged,
    overshoot,
    saturation_fraction,
    settling_time,
    steady_state_error,
)


def test_settling_time_basic():
    t = np.linspace(0.0, 10.0, 101)  # dt = 0.1
    signal = np.where(t < 2.0, 1.0, 0.0)  # error 1 until t=2, then 0 (target 0)
    assert settling_time(t, signal, target=0.0, tol=0.05) == pytest.approx(2.0, abs=0.1)


def test_settling_time_none_if_never_settles():
    t = np.linspace(0.0, 10.0, 101)
    signal = np.ones_like(t)
    assert settling_time(t, signal, 0.0, 0.05) is None


def test_settling_time_requires_staying_within_band():
    t = np.linspace(0.0, 10.0, 101)
    signal = np.empty_like(t)
    signal[:20] = 1.0   # out
    signal[20:30] = 0.0  # briefly in
    signal[30:40] = 1.0  # leaves again
    signal[40:] = 0.0    # finally settles at t=4.0
    assert settling_time(t, signal, 0.0, 0.05) == pytest.approx(4.0, abs=0.1)


def test_overshoot_when_crossing_target():
    # target 0 from initial 1; dips to -0.2 -> overshoot 0.2
    signal = np.array([1.0, 0.5, 0.0, -0.2, -0.1, 0.0])
    assert overshoot(signal, target=0.0, initial=1.0) == pytest.approx(0.2)


def test_overshoot_zero_when_monotone():
    signal = np.array([1.0, 0.5, 0.2, 0.05, 0.0])
    assert overshoot(signal, 0.0, 1.0) == pytest.approx(0.0)


def test_steady_state_error_uses_tail():
    signal = np.concatenate([np.ones(90), np.full(10, 0.02)])
    assert steady_state_error(signal, target=0.0, n_tail=10) == pytest.approx(0.02)


def test_converged_true_within_limit():
    t = np.linspace(0.0, 10.0, 101)
    signal = np.where(t < 3.0, 1.0, 0.0)
    assert converged(t, signal, 0.0, 0.05, t_limit=10.0) is True


def test_converged_false_when_too_slow():
    t = np.linspace(0.0, 10.0, 101)
    signal = np.where(t < 8.0, 1.0, 0.0)
    assert converged(t, signal, 0.0, 0.05, t_limit=5.0) is False


def test_saturation_fraction():
    cmd = np.array([0.1, 0.5, 0.5, -0.5, 0.0])  # 3 of 5 at the 0.5 limit
    assert saturation_fraction(cmd, 0.5) == pytest.approx(3 / 5)
