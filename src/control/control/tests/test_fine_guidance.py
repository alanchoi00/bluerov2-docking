"""Unit tests for the align-then-advance fine guidance law."""

import numpy as np

from control.pbvs import CmdVel
from control import fine_guidance as fg


ALIGN = fg.AlignTol(lateral_m=0.03, vertical_m=0.03, yaw_rad=0.05)
SEAT = fg.SeatedTol(
    range_m=0.10, lateral_m=0.03, vertical_m=0.03, yaw_rad=0.05, debounce_cycles=5
)


def test_aligned_true_when_lateral_vertical_yaw_within_tol():
    rel = np.array([0.5, 0.01, -0.01])  # large forward, tiny lateral/vertical
    assert fg.aligned(rel, yaw_err=0.02, tol=ALIGN) is True


def test_aligned_false_when_lateral_out_of_tol():
    rel = np.array([0.5, 0.10, 0.0])
    assert fg.aligned(rel, yaw_err=0.0, tol=ALIGN) is False


def test_aligned_ignores_forward_distance():
    # Far forward but perfectly centered -> aligned (forward is what surge closes)
    rel = np.array([5.0, 0.0, 0.0])
    assert fg.aligned(rel, yaw_err=0.0, tol=ALIGN) is True


def test_advance_command_zeroes_surge_when_not_aligned():
    cmd = CmdVel(surge=0.3, sway=0.1, heave=-0.05, yaw_rate=0.2)
    out = fg.advance_command(cmd, is_aligned=False)
    assert out.surge == 0.0
    assert out.sway == 0.1 and out.heave == -0.05 and out.yaw_rate == 0.2


def test_advance_command_passes_surge_when_aligned():
    cmd = CmdVel(surge=0.3, sway=0.0, heave=0.0, yaw_rate=0.0)
    out = fg.advance_command(cmd, is_aligned=True)
    assert out.surge == 0.3


def test_within_seated_true_at_dock():
    rel = np.array([0.05, 0.01, 0.0])
    assert fg.within_seated(0.05, rel, yaw_err=0.0, tol=SEAT) is True


def test_within_seated_false_when_too_far():
    rel = np.array([0.5, 0.0, 0.0])
    assert fg.within_seated(0.5, rel, yaw_err=0.0, tol=SEAT) is False


def test_decide_seated_latches_after_debounce():
    counter, was = 0, False
    phase = seated = None
    for _ in range(SEAT.debounce_cycles):
        phase, seated, counter = fg.decide_seated(
            within_seated=True,
            healthy=True,
            counter=counter,
            was_seated=was,
            debounce=SEAT.debounce_cycles,
        )
        was = seated
    assert seated is True
    assert phase == fg.SEATED


def test_decide_seated_aligning_before_latch():
    phase, seated, counter = fg.decide_seated(
        within_seated=True, healthy=True, counter=0, was_seated=False, debounce=5
    )
    assert seated is False
    assert phase == fg.ALIGNING
