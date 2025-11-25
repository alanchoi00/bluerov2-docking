# calibrate_camera.py

import cv2
import numpy as np

# --- CONFIG ---

# Number of INNER corners in the chessboard pattern (columns, rows)
CHESSBOARD_SIZE = (8, 6)  # adjust to your printed board

# Size of one square in metres (e.g. 0.024 for 24 mm)
SQUARE_SIZE = 0.030

# How many good views to collect before calibrating
NUM_IMAGES_NEEDED = 15

OUTPUT_FILE = "camera_calib.npz"
CAM_INDEX = 0  # change if using different camera

# --- SETUP OBJECT POINTS ---

# Prepare object points like (0,0,0), (1,0,0), ..., (8,5,0) scaled by SQUARE_SIZE
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []  # 3d points in world space
imgpoints = []  # 2d points in image plane

def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    collected = 0
    print(f"Press SPACE to capture chessboard views ({NUM_IMAGES_NEEDED} needed), ESC to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        found, corners = cv2.findChessboardCorners(
            gray, CHESSBOARD_SIZE,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if found:
            # Refine corners
            corners_refined = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )
            cv2.drawChessboardCorners(frame, CHESSBOARD_SIZE, corners_refined, found)

        cv2.putText(frame, f"Collected: {collected}/{NUM_IMAGES_NEEDED}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Calibration", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break
        elif key == 32 and found:  # SPACE
            print("Captured chessboard view.")
            objpoints.append(objp.copy())
            imgpoints.append(corners_refined)
            collected += 1

            if collected >= NUM_IMAGES_NEEDED:
                print("Collected enough images. Calibrating...")
                break

    cap.release()
    cv2.destroyAllWindows()

    if collected < 3:
        print("Not enough images to calibrate.")
        return

    # Calibrate
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    print("Calibration RMS error:", ret)
    print("Camera matrix:\n", camera_matrix)
    print("Distortion coefficients:\n", dist_coeffs.ravel())

    # Save to file
    np.savez(OUTPUT_FILE,
             camera_matrix=camera_matrix,
             dist_coeffs=dist_coeffs)

    print(f"Saved calibration to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
