# Results summary

- Analysis code: `alanchoi00/underwater-aruco-validation@unknown`
- OpenCV: `4.10.0`
- Fitted law: `max_range ~ 34.9 x side_length`
- Host CPU (latency context): `11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz`
- Intrinsics: ZED shipped K, no refraction correction (see spec 3.2)
- Metric range carries ~+/-10% from focal-length uncertainty.
- Pose error is **vs the board reference**, not vs ground truth. None exists.
- Latency is detectMarkers on the analysis host, not on ROV compute.
- Angle is reported as two regimes (test1 head-on, test2 oblique ~57 deg), NOT a swept curve: there is no controlled angle sweep and angle is confounded with range/size (spec 3.1c).

## test1

- frames: 677, detections: 1376
- turn frames: 101
- mis-ID rate: 0.0015
- ids seen: [0, 201, 202, 301, 302, 303, 304, 305, 401, 402]
- detector latency: 3.10 ms median, 5.77 ms p95 (analysis host, NOT ROV compute)
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
- detector latency: 2.34 ms median, 4.08 ms p95 (analysis host, NOT ROV compute)
- incidence angle: median 57.5 deg (p10-p90 25.1-62.6); 122 detections above 40 deg

  - 201 (149 mm): max range 4.98 m
  - 202 (149 mm): max range 4.73 m

## Stage 4 - IMU validation (camera-independent)

- Gravity: board tilt from vertical over 323 frames = 6.64 deg median, std 6.89 deg (low std = PnP attitude consistent with the IMU).
- Yaw turns: 18/25 turn segments had board poses at both ends:

  - turn 870.9-872.8s: vision 25.2 deg, gyro -27.4 deg (compare magnitude, not sign)
  - turn 882.9-884.2s: vision -23.6 deg, gyro 19.3 deg (compare magnitude, not sign)
  - turn 891.5-893.2s: vision 7.7 deg, gyro -28.5 deg (compare magnitude, not sign)
  - turn 905.2-907.0s: vision 23.5 deg, gyro -32.5 deg (compare magnitude, not sign)
  - turn 911.6-912.9s: vision -6.2 deg, gyro 15.3 deg (compare magnitude, not sign)
  - turn 915.1-916.1s: vision 7.5 deg, gyro 16.0 deg (compare magnitude, not sign)
  - turn 919.2-921.0s: vision 3.6 deg, gyro -31.0 deg (compare magnitude, not sign)
  - turn 926.4-927.6s: vision -0.7 deg, gyro 11.8 deg (compare magnitude, not sign)
  - turn 933.5-934.9s: vision 9.2 deg, gyro -28.0 deg (compare magnitude, not sign)
  - turn 941.9-943.0s: vision -0.1 deg, gyro 15.0 deg (compare magnitude, not sign)
  - turn 944.6-946.3s: vision 8.8 deg, gyro -34.8 deg (compare magnitude, not sign)
  - turn 953.4-954.4s: vision -3.3 deg, gyro 14.3 deg (compare magnitude, not sign)
  - turn 957.1-958.5s: vision 4.8 deg, gyro -25.6 deg (compare magnitude, not sign)
  - turn 962.7-963.7s: vision -1.7 deg, gyro 14.6 deg (compare magnitude, not sign)
  - turn 965.5-967.0s: vision -4.0 deg, gyro 20.6 deg (compare magnitude, not sign)
  - turn 969.2-970.9s: vision 19.1 deg, gyro -30.5 deg (compare magnitude, not sign)
  - turn 981.0-982.8s: vision 37.8 deg, gyro -25.6 deg (compare magnitude, not sign)
  - turn 993.3-995.0s: vision 15.1 deg, gyro -29.8 deg (compare magnitude, not sign)
