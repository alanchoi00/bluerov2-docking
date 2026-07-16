# Results summary

- Analysis code: `alanchoi00/underwater-aruco-validation@a737503`
- OpenCV: `4.10.0`
- Fitted law: `max_range ~ 34.9 x side_length`
- Host CPU (latency context): `11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz`
- Intrinsics: ZED shipped K, no refraction correction (see spec 3.2)
- Metric range carries ~+/-10% from focal-length uncertainty.
- Pose error is **vs the board reference**, not vs ground truth. None exists.
- Latency is detectMarkers on the analysis host, not on ROV compute.
- Angle is reported as two regimes (test1 head-on, test2 oblique ~57 deg), NOT a swept curve: there is no controlled angle sweep and angle is confounded with range/size (spec 3.1c).
- 3 bin(s) with fewer than 10 trials were suppressed from the detection-rate figures (their Wilson CI spans most of the axis and would dominate the plot with no information); the full per-bin table, including these, is still in `detection_trials_test1.csv` / `detection_trials_test2.csv`. Suppressed: 75 mm, 8-12 px, n=2; 75 mm, 100-140 px, n=2; 149 mm, 200-300 px, n=1.
- 1 bin(s) with fewer than 10 trials were suppressed from the pose-error-vs-range figure (pooled across marker sizes; n=1-3 bins were near-empty and would carry a std of NaN or an unrepresentative one); the full per-bin table, including these, is still in `trans_err_vs_range.csv`. Suppressed: 3.0-4.0 m, n=3.
- 1 bin(s) with fewer than 10 trials were suppressed from the rotation-error-vs-range figure (same pooling and suppression as the translation-error figure above); the full per-bin table, including these, is still in `rot_err_vs_range.csv`. Suppressed: 3.0-4.0 m, n=3.
- The pose-error-vs-range figure is pooled across marker sizes, not broken down per size: the leave-one-out board reference used by `pose_error_vs_reference` gets weaker when 201/202 are the marker under test (only the narrow centre cluster remains as reference) and stronger when a small marker is under test (201+202, 427 mm apart, remain) -- a per-size split would measure that reference confound, not marker quality, and read backwards (see `analysis/metrics.py`).

## test1

- frames: 677, detections: 1376
- turn frames: 101
- mis-ID rate: 0.0015
- ids seen: [0, 201, 202, 301, 302, 303, 304, 305, 401, 402]
- detector latency: 2.66 ms median, 5.01 ms p95 (analysis host, NOT ROV compute)
- incidence angle: median 13.2 deg (p10-p90 4.2-28.5); 20 detections above 40 deg

  - 201 (149 mm): max range 5.10 m
  - 202 (149 mm): max range 4.87 m
  - 301 (75 mm): max range 3.05 m
  - 302 (44 mm): max range 1.75 m
  - 303 (44 mm): max range 2.17 m
  - 304 (44 mm): max range 1.55 m
  - 305 (44 mm): max range 1.75 m
  - 401 (36 mm): max range 1.10 m
  - 402 (36 mm): max range 1.14 m

## test2

- frames: 737, detections: 161
- turn frames: 33
- mis-ID rate: 0.0000
- ids seen: [201, 202]
- detector latency: 2.03 ms median, 3.61 ms p95 (analysis host, NOT ROV compute)
- incidence angle: median 57.5 deg (p10-p90 25.1-62.6); 122 detections above 40 deg

  - 201 (149 mm): max range 4.98 m
  - 202 (149 mm): max range 4.73 m

## Stage 4 - IMU validation (camera-independent)

- Gravity check PASSES: board tilt from vertical over 323 frames = 6.64 deg median, std 6.89 deg (low std = PnP attitude consistent with the absolute IMU gravity reference). This is the camera-independent validation.

- Yaw check: 7/25 turn segments were evaluable (board-pose coverage >= 80%); the rest had the board leave frame partway through the turn, so vision under-reports the rotation there and they are excluded rather than compared. Gyro measures yaw about the IMU's up (+z) axis while PnP yaw is about the optical frame's down (+y) axis, so an opposite sign between the two is the expected convention, not a discrepancy.
  Even restricted to these full-coverage segments, vision's magnitude does not consistently track the gyro's (see per-segment numbers below) -- the yaw check does not corroborate the gravity check on this dataset. The gravity check above remains the sole camera-independent validation; the yaw numbers are reported for completeness, not as a passing check.
  - turn 882.9-884.2s (coverage 100%): vision -23.6 deg, gyro 19.3 deg
  - turn 911.6-912.9s (coverage 100%): vision -6.2 deg, gyro 15.3 deg
  - turn 915.1-916.1s (coverage 100%): vision 7.5 deg, gyro 16.0 deg
  - turn 926.4-927.6s (coverage 100%): vision -0.7 deg, gyro 11.8 deg
  - turn 953.4-954.4s (coverage 100%): vision -3.3 deg, gyro 14.3 deg
  - turn 962.7-963.7s (coverage 100%): vision -1.7 deg, gyro 14.6 deg
  - turn 965.5-967.0s (coverage 100%): vision -4.0 deg, gyro 20.6 deg
