# estimate_pose.py

import cv2
import numpy as np

CALIB_FILE = "camera_calib.npz"
CAM_INDEX = 0  # or path to video file as string, e.g. "test.mp4"

# Marker side length in metres (must match your printed marker size)
MARKER_LENGTH = 0.99  # 99 cm, adjust to your print


def load_calibration(path):
    data = np.load(path)
    camera_matrix = data["camera_matrix"]
    dist_coeffs = data["dist_coeffs"]
    return camera_matrix, dist_coeffs

def main():
    # Load calibration
    try:
        camera_matrix, dist_coeffs = load_calibration(CALIB_FILE)
    except Exception as e:
        print(f"Error loading calibration file {CALIB_FILE}: {e}")
        return

    # Prepare ArUco dictionary and detector
    aruco = cv2.aruco
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)

    # Open camera/video
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("Error: Could not open camera or video.")
        return

    print("Press ESC to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No frame received, exiting.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect markers
        corners, ids, rejected = detector.detectMarkers(gray)

        if ids is not None and len(ids) > 0:
            # Draw marker borders
            aruco.drawDetectedMarkers(frame, corners, ids)

            # Estimate pose for each marker
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                corners, MARKER_LENGTH, camera_matrix, dist_coeffs
            )

            for i, marker_id in enumerate(ids.flatten()):
                rvec = rvecs[i]
                tvec = tvecs[i]

                # Draw coordinate axes on the marker
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, MARKER_LENGTH * 0.5)

                # Convert rotation vector to rotation matrix, then to yaw/pitch/roll if you like
                R, _ = cv2.Rodrigues(rvec)
                # Simple: just show translation (x, y, z)
                tx, ty, tz = tvec[0]
                text = f"ID {marker_id}: x={tx:.2f}m y={ty:.2f}m z={tz:.2f}m"
                cv2.putText(frame, text, (10, 30 + 30 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("ArUco Pose Estimation", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
