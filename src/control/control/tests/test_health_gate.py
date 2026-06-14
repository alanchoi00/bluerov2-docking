"""Black-box tests for the pure health gate and phase decision."""

from control.health_gate import (
    GateResult,
    WARMING_UP,
    HEALTHY,
    DEGRADED,
    STALE,
    gate_for_health,
)


def test_healthy_passes_full_gain():
    g = gate_for_health(HEALTHY, degraded_gain_scale=0.5)
    assert g == GateResult(blocked=False, gain_scale=1.0, dock_healthy=True)


def test_degraded_derates_and_flags():
    g = gate_for_health(DEGRADED, degraded_gain_scale=0.5)
    assert g == GateResult(blocked=False, gain_scale=0.5, dock_healthy=False)


def test_stale_blocks():
    g = gate_for_health(STALE, degraded_gain_scale=0.5)
    assert g.blocked is True and g.gain_scale == 0.0


def test_warming_up_blocks():
    g = gate_for_health(WARMING_UP, degraded_gain_scale=0.5)
    assert g.blocked is True and g.gain_scale == 0.0
