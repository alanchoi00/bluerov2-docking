"""Unit tests for the pure FSM transition condition helpers."""

from orchestrator.transitions import is_loss, is_drift, sustained


def test_is_loss_true_for_warming_up_and_stale():
    assert is_loss(0) is True   # WARMING_UP
    assert is_loss(3) is True   # STALE


def test_is_loss_false_for_healthy_and_degraded():
    assert is_loss(1) is False  # HEALTHY
    assert is_loss(2) is False  # DEGRADED (usable, de-rated)


def test_is_drift_compares_range_to_threshold():
    assert is_drift(1.6, 1.5) is True
    assert is_drift(1.4, 1.5) is False
    assert is_drift(1.5, 1.5) is False  # strict greater-than


def test_sustained_increments_then_trips_at_threshold():
    counter, tripped = 0, False
    for _ in range(3):
        counter, tripped = sustained(counter, True, threshold=3)
    assert counter == 3
    assert tripped is True


def test_sustained_resets_on_false():
    counter, _ = sustained(2, True, threshold=5)
    assert counter == 3
    counter, tripped = sustained(counter, False, threshold=5)
    assert counter == 0
    assert tripped is False


def test_sustained_clamps_at_threshold():
    counter = 5
    for _ in range(10):
        counter, tripped = sustained(counter, True, threshold=5)
    assert counter == 5
    assert tripped is True


def test_sustained_not_tripped_before_threshold():
    counter, tripped = sustained(0, True, threshold=3)
    assert counter == 1
    assert tripped is False
