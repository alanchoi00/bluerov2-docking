import numpy as np

from perception.aruco.lib.geometry import rotvec_to_quat
from perception.aruco.lib.kalman import (
    DockPoseKalmanFilter,
    make_process_noise,
)


def _pose(position: list[float], rotvec: list[float] | None = None):
    return (
        np.array(position),
        rotvec_to_quat(np.array(rotvec or [0.0, 0.0, 0.0])),
    )


def test_initialize_from_first_measurement():
    kf = DockPoseKalmanFilter()
    pos, quat = _pose([1.0, 2.0, 3.0])
    cov = 0.01 * np.eye(6)
    kf.initialize(pos, quat, cov)
    assert kf.is_initialized
    np.testing.assert_allclose(kf.position, pos, atol=1e-9)
    np.testing.assert_allclose(kf.orientation, quat, atol=1e-9)


def test_predict_only_grows_covariance():
    kf = DockPoseKalmanFilter()
    pos, quat = _pose([0.0, 0.0, 0.0])
    kf.initialize(pos, quat, 0.001 * np.eye(6))
    initial_cov_det = np.linalg.det(kf.covariance)

    q = make_process_noise(dt=0.033, regime="static")
    kf.predict(dt=0.033, process_noise=q)

    assert np.linalg.det(kf.covariance) > initial_cov_det


def test_converges_on_noisy_stream_around_ground_truth():
    kf = DockPoseKalmanFilter()
    truth_pos = np.array([1.0, 2.0, 3.0])
    truth_quat = rotvec_to_quat(np.array([0.0, 0.0, 0.5]))

    rng = np.random.default_rng(seed=42)
    noise_std = 0.02

    kf.initialize(truth_pos + rng.normal(0, noise_std, 3), truth_quat, 0.01 * np.eye(6))

    q = make_process_noise(dt=0.033, regime="static")
    pos_cov = (noise_std**2) * np.eye(3)
    rot_cov = (noise_std**2) * np.eye(3)
    for _ in range(100):
        kf.predict(dt=0.033, process_noise=q)
        measurement_pos = truth_pos + rng.normal(0, noise_std, 3)
        kf.update(measurement_pos, truth_quat, pos_cov, rot_cov)

    # After 100 noisy measurements, position should be within 5mm of truth
    np.testing.assert_allclose(kf.position, truth_pos, atol=0.005)


def test_mahalanobis_gate_rejects_outlier():
    kf = DockPoseKalmanFilter()
    pos, quat = _pose([0.0, 0.0, 0.0])
    kf.initialize(pos, quat, 1e-4 * np.eye(6))  # very confident
    q = make_process_noise(dt=0.033, regime="static")

    # Feed a measurement 10m away should be rejected as outlier
    far_pos = np.array([10.0, 0.0, 0.0])
    tight_cov = 0.01**2 * np.eye(3)
    accepted = kf.try_update(
        far_pos,
        quat,
        tight_cov,
        tight_cov,
        gate_chi2=18.548,  # chi^2_{6, 0.995}
    )
    assert not accepted
    np.testing.assert_allclose(kf.position, [0.0, 0.0, 0.0], atol=1e-6)


def test_process_noise_regimes_ordered():
    dt = 0.033
    q_static = make_process_noise(dt=dt, regime="static")
    q_sway = make_process_noise(dt=dt, regime="sway")
    q_drift = make_process_noise(dt=dt, regime="drift")
    assert np.trace(q_static) < np.trace(q_sway)
    assert np.trace(q_sway) < np.trace(q_drift)


def test_process_noise_scales_with_dt():
    q_short = make_process_noise(dt=0.01, regime="static")
    q_long = make_process_noise(dt=1.0, regime="static")
    assert np.trace(q_long) > np.trace(q_short)


def test_process_noise_returns_pd_6x6():
    q = make_process_noise(dt=0.033, regime="static")
    assert q.shape == (6, 6)
    eigvals = np.linalg.eigvalsh(q)
    assert np.all(eigvals > 0)
