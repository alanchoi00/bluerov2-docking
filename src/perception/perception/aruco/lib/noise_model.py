"""Per-marker position covariance kernel.

This is the adaptive-noise mechanism that gives the downstream Kalman filter
range-and-size-aware measurement uncertainty. Larger covariance → less trust.

Currently active: Candidate B (PnP-error quadratic form).
See the comment block in marker_position_covariance() for alternative
candidates and how to swap them in.

Reference:
- PnP error analysis: σ ∝ r² / (s · f_px), where r is range, s is marker
  side length, f_px is camera focal length in pixels. For fixed camera
  intrinsics, σ ∝ r² / s.
- Xu et al. 2021, JMSE 9:1432 — Candidate A (exponential) below.
"""

import numpy as np


def marker_position_covariance(
    range_m: float,
    marker_size_m: float,
    alpha: float,
) -> np.ndarray:
    r"""Per-marker position measurement covariance (3x3, metres²).

    Args:
        range_m: distance from camera to marker centre (m), always > 0
        marker_size_m: physical side length of the black square (m), always > 0
        alpha: empirically-calibrated scale constant (units: m⁻¹ for
            Candidate B — multiplies (r²/s) to produce σ in metres)

    Returns:
        3x3 positive-definite covariance matrix. Passed upstream to pose_fusion
        as the measurement noise for this marker's contribution.

    Three candidate kernels are documented below. Candidate B is currently
    active (uncommented). To swap: comment out Candidate B and uncomment
    the replacement. Re-run tests.

    All three are monotonically increasing in (range / marker_size), which
    is the physically-motivated regime: PnP error scales with range over
    marker pixel size. They differ in tail behavior and numerical stability.

    Candidate A: Xu 2021 extended, exponential form
    -------------------------------------------------------
    Paper-backed, but the exponential tail saturates catastrophically when
    r/(s·α) >> 1: k → 0, so 1/k explodes and must be floored. In practice
    this means the kernel transitions from "over-trusting" to "completely
    useless" over a narrow α range. We saw this in sim: α=2 (the plan's
    starting point) gave variance ≈ 1e6 everywhere in our operating range.

        import math
        k = math.exp(-(range_m / (marker_size_m * alpha)) ** 2 / 2.0)
        k = max(k, 1e-6)
        variance = (1.0 / k) - 1.0
        return variance * np.eye(3)

    Candidate B (ACTIVE): Theoretical PnP error bound, quadratic form
    -------------------------------------------------------
    Derived from first principles: ArUco PnP position error scales as
    r²/(s·f_px) where f_px is focal length in pixels. For fixed camera
    intrinsics we roll f_px into α, giving σ = α · r² / s.

    Variance grows unbounded at far range (no floor needed) but that's
    physically correct — at very far range you really do have unbounded
    uncertainty. Downstream Mahalanobis gating handles it fine.

    Alpha tuning:
    - α ≈ 0.001 gives ~1cm std at 1m for a 100mm marker (matches typical
      ArUco PnP on a VGA-resolution camera).

    Candidate C: Clipped-linear, empirically robust
    -------------------------------------------------------
    Ad-hoc but numerically safe. Ties sigma linearly to r/s with floors
    and ceilings. Hardest to defend theoretically, easiest to debug.

        ratio = range_m / marker_size_m
        sigma = alpha * np.clip(ratio, 5.0, 200.0)
        return (sigma ** 2) * np.eye(3)
    """

    # Candidate B: PnP-error quadratic kernel
    sigma = alpha * (range_m**2) / marker_size_m
    return (sigma**2) * np.eye(3)


def marker_rotation_covariance(
    range_m: float,
    marker_size_m: float,
    alpha: float,
) -> np.ndarray:
    """Per-marker rotation measurement covariance (3x3, radians²).

    PnP rotation error scales as σ_rot ≈ α · r / s (linear in range, inverse
    in marker size). This is one factor of r less than position error because
    rotation error is proportional to angular pixel resolution, not metric
    resolution. Derived from the same pixel-noise physics as the position
    kernel above, so the same α constant applies.

    Examples at α=0.001:
    - r=1m,  s=0.1m  (301 at fine-align):  σ_rot ≈ 0.01 rad (0.57°)
    - r=0.3m, s=0.1m (close engagement):    σ_rot ≈ 0.003 rad (0.17°)
    - r=3m,  s=0.2m  (201 at coarse):      σ_rot ≈ 0.015 rad (0.86°)
    - r=4m,  s=0.06m (304 at extreme):     σ_rot ≈ 0.067 rad (3.8°)
    """
    sigma = alpha * range_m / marker_size_m
    return (sigma**2) * np.eye(3)
