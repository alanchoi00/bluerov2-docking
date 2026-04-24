"""6-state error-state Kalman filter for dock pose in odom frame.

State: [x_dock, y_dock, z_dock, deltarx, deltary, deltarz]
  - position is tracked directly in metres
  - orientation is tracked as a small-angle perturbation (deltar) about a
    reference quaternion q_ref which is carried alongside the state.
    After each update, the perturbation is absorbed into q_ref and the
    deltar part of the state is zeroed.

Process model (Thesis B): F = I_6 (dock is stationary).
Measurement model: identity (measurement IS the pose, in the same frame).

Reference: Ligorio & Sabatini 2013, Sensors 13:1919 error-state EKF template.
"""

import numpy as np

from perception.aruco.lib.geometry import (
    quat_inverse,
    quat_multiply,
    quat_to_rotvec,
    rotvec_to_quat,
)


class DockPoseKalmanFilter:
    """Error-state Kalman filter for a static dock in odom frame."""

    def __init__(self) -> None:
        self._position: np.ndarray | None = None
        self._q_ref: np.ndarray | None = None  # reference quaternion
        self._error_state: np.ndarray = np.zeros(
            6
        )  # [deltax, deltay, deltaz, deltarx, deltary, deltarz]
        self._covariance: np.ndarray | None = None

    @property
    def is_initialized(self) -> bool:
        return self._position is not None

    @property
    def position(self) -> np.ndarray:
        assert self._position is not None
        return self._position + self._error_state[:3]

    @property
    def orientation(self) -> np.ndarray:
        assert self._q_ref is not None
        delta = self._error_state[3:]
        return quat_multiply(self._q_ref, rotvec_to_quat(delta))

    @property
    def covariance(self) -> np.ndarray:
        assert self._covariance is not None
        return self._covariance

    def initialize(
        self,
        position: np.ndarray,
        orientation: np.ndarray,
        covariance: np.ndarray,
    ) -> None:
        assert covariance.shape == (6, 6)
        self._position = position.copy()
        self._q_ref = orientation / np.linalg.norm(orientation)
        self._error_state = np.zeros(6)
        self._covariance = covariance.copy()

    def predict(self, dt: float, process_noise: np.ndarray) -> None:
        # F = I: error state doesn't change; covariance grows by Q.
        assert self._covariance is not None
        self._covariance = self._covariance + process_noise

    def update(
        self,
        measurement_position: np.ndarray,
        measurement_orientation: np.ndarray,
        measurement_position_covariance: np.ndarray,
        measurement_rotation_covariance: np.ndarray,
    ) -> None:
        """Unconditional update caller is responsible for gating.

        Measurement covariances for position and rotation are passed as
        separate 3x3 blocks with different physical units (m^2 and rad^2).
        They're combined here into a block-diagonal 6x6 R (ArUco PnP corner
        noise has no meaningful position/rotation cross-correlation).
        """
        assert self.is_initialized
        assert measurement_position_covariance.shape == (3, 3)
        assert measurement_rotation_covariance.shape == (3, 3)

        R = np.zeros((6, 6))
        R[:3, :3] = measurement_position_covariance
        R[3:, 3:] = measurement_rotation_covariance

        y_pos = measurement_position - self.position
        delta_q = quat_multiply(quat_inverse(self._q_ref), measurement_orientation)
        y_rot = quat_to_rotvec(delta_q) - self._error_state[3:]
        y = np.concatenate([y_pos, y_rot])

        # H = I_6 in error-state formulation
        S = self._covariance + R
        K = self._covariance @ np.linalg.inv(S)

        self._error_state = self._error_state + K @ y
        I6 = np.eye(6)
        self._covariance = (I6 - K) @ self._covariance

        # Absorb error-state into nominal: shift position, compose rotation.
        self._position = self._position + self._error_state[:3]
        self._q_ref = quat_multiply(self._q_ref, rotvec_to_quat(self._error_state[3:]))
        self._q_ref = self._q_ref / np.linalg.norm(self._q_ref)
        self._error_state = np.zeros(6)

    def try_update(
        self,
        measurement_position: np.ndarray,
        measurement_orientation: np.ndarray,
        measurement_position_covariance: np.ndarray,
        measurement_rotation_covariance: np.ndarray,
        gate_chi2: float,
    ) -> bool:
        """Apply Mahalanobis gating. Returns True iff the update was applied.

        Also stores split diagnostics on self for caller inspection:
            self.last_d_sq_total, last_d_sq_pos, last_d_sq_rot, last_innovation
        """
        assert self.is_initialized
        R6 = np.zeros((6, 6))
        R6[:3, :3] = measurement_position_covariance
        R6[3:, 3:] = measurement_rotation_covariance

        y_pos = measurement_position - self.position
        delta_q = quat_multiply(quat_inverse(self._q_ref), measurement_orientation)
        y_rot = quat_to_rotvec(delta_q)
        y = np.concatenate([y_pos, y_rot])

        S = self._covariance + R6
        S_inv = np.linalg.inv(S)
        d_sq = float(y @ S_inv @ y)
        d_sq_pos = float(y_pos @ np.linalg.inv(S[:3, :3]) @ y_pos)
        d_sq_rot = float(y_rot @ np.linalg.inv(S[3:, 3:]) @ y_rot)

        self.last_d_sq_total = d_sq
        self.last_d_sq_pos = d_sq_pos
        self.last_d_sq_rot = d_sq_rot
        self.last_innovation = y.copy()

        if d_sq > gate_chi2:
            return False
        self.update(
            measurement_position,
            measurement_orientation,
            measurement_position_covariance,
            measurement_rotation_covariance,
        )
        return True


def make_process_noise(dt: float, regime: str) -> np.ndarray:
    """Process noise covariance Q for the 6-state constant-pose filter.

    Q encodes how much we expect the dock's pose to drift between predict
    steps. Larger Q -> filter responds faster to new measurements (less
    smoothing). Smaller Q -> filter resists noise harder (more smoothing,
    more lag).

    Args:
        dt: timestep since last predict (seconds)
        regime: "static" | "sway" | "drift"

    Returns:
        6x6 positive-definite Q matrix (metres^2 and radians^2 on diagonal).
    """

    if regime == "static":
        q_pos = 1e-4 * dt
        q_rot = 1e-4 * dt
    elif regime == "sway":
        q_pos = 1e-2 * dt
        q_rot = 7.6e-3 * dt
    elif regime == "drift":
        q_pos = 9e-2 * dt
        q_rot = 3e-2 * dt
    else:
        raise ValueError(
            f"unknown regime: {regime!r} (expected 'static'|'sway'|'drift')"
        )

    q = np.zeros((6, 6))
    q[0, 0] = q_pos
    q[1, 1] = q_pos
    q[2, 2] = q_pos
    q[3, 3] = q_rot
    q[4, 4] = q_rot
    q[5, 5] = q_rot
    return q
