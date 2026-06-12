"""Step-response metrics for a per-axis error signal over time.

Pure numpy. Used for the printed tuning table and the convergence regression
test. A signal is the per-axis error vs time; `target` is usually 0 (drive
error to zero), except surge where the target is the handoff range.
"""

import numpy as np


def settling_time(t, signal, target: float, tol: float):
    """First time after which |signal - target| stays within tol forever.

    Returns the settling time (s), or None if it never settles (ends outside
    the band). 'Stays' matters: a brief dip into the band that later leaves
    does not count -- we find the last excursion and settle just after it.
    """
    t = np.asarray(t, dtype=float)
    s = np.asarray(signal, dtype=float)
    within = np.abs(s - target) <= tol
    if not within[-1]:
        return None
    outside_idx = np.where(~within)[0]
    if outside_idx.size == 0:
        return float(t[0])  # within band the whole time
    return float(t[outside_idx[-1] + 1])


def overshoot(signal, target: float, initial: float) -> float:
    """Largest excursion past the target, on the far side from `initial`.

    0.0 if the response approaches monotonically without crossing.
    """
    s = np.asarray(signal, dtype=float)
    beyond = (target - s) if initial >= target else (s - target)
    return float(max(0.0, np.max(beyond)))


def steady_state_error(signal, target: float, n_tail: int = 10) -> float:
    """Mean absolute error over the last n_tail samples."""
    s = np.asarray(signal, dtype=float)
    return float(np.mean(np.abs(s[-n_tail:] - target)))


def converged(t, signal, target: float, tol: float, t_limit: float) -> bool:
    """True if the signal settles within the band by t_limit."""
    st = settling_time(t, signal, target, tol)
    return st is not None and st <= t_limit


def saturation_fraction(command, v_max: float) -> float:
    """Fraction of samples at or above the command saturation limit."""
    c = np.asarray(command, dtype=float)
    return float(np.mean(np.abs(c) >= v_max - 1e-12))
