import os
import warnings
import cv2
import time
import numpy as np

# Suppress noisy background warnings
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", category=UserWarning)

from detector import DriverFaceDetector
from state_monitor import DriverStateMonitor

from metrics import (
    extract_raw_features,
    GazeCalibrator,
    EAR_INDEX,
    VERTICAL_Y_INDEX,
    FORWARD_X_INDEX,
)

# Initialize global tools
detector = DriverFaceDetector()
calibrator = GazeCalibrator()
state_monitor = DriverStateMonitor()

# Calibration configuration
REQUIRED_CALIBRATION_FRAMES = 90  # ~3 seconds at 30 FPS
calibration_buffer = []

def reset_calibration():
    global calibration_buffer

    calibration_buffer.clear()
    calibrator.reset()
    state_monitor.reset()

    print("[INFO] Recalibration requested.")


def process_driver_frame(frame):
    """
    Processes each incoming frame and handles live state transitions.
    """
    global calibration_buffer

    # 1. Detect raw facial landmarks using MediaPipe
    detection_results = detector.find_face_landmarks(frame)
    frame = detector.draw_mesh(frame, detection_results)

    # If no face is in the frame, we skip calculations
    if not detection_results.multi_face_landmarks:
        face_result = state_monitor.update_face_missing()

        cv2.putText(
            frame,
            f"STATE: {face_result.state}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            face_result.color,
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            f"Face missing: {face_result.duration:.1f}s",
            (10, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        return frame

    # 2. Extract raw landmark feature coordinates
    raw_features = extract_raw_features(detection_results.multi_face_landmarks)
    
    if raw_features is None:
        cv2.putText(
            frame,
            "FEATURE EXTRACTION FAILED",
            (10, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return frame

    # 3. Handle Calibration vs. Live tracking states
    if not calibrator.is_calibrated:
        # State A: Gathering Calibration Data
        calibration_buffer.append(raw_features)
        remaining = REQUIRED_CALIBRATION_FRAMES - len(calibration_buffer)

        # Draw a calibration progress screen
        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1) # Dim the screen
        cv2.putText(frame, "CALIBRATING... LOOK AT THE WINDSHIELD", (30, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Keep steady: {remaining} frames remaining", (30, 200), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

        if len(calibration_buffer) >= REQUIRED_CALIBRATION_FRAMES:
            # Trigger the mathematical averaging once we have enough frames
            calibrator.calibrate(calibration_buffer)
            calibration_buffer.clear() # Free memory
            
    else:
        # State B: Live Tracking Mode (We use Normalized coordinates!)
        normalized_features = calibrator.get_relative_features(raw_features)
        
        # Grab our normalized features
        avg_ear = normalized_features[EAR_INDEX]

        # Forward-vector X changes as the nose moves left or right.
        horizontal_deviation = normalized_features[FORWARD_X_INDEX]

        # Vertical-vector Y changes as the head tilts up or down.
        vertical_deviation = normalized_features[VERTICAL_Y_INDEX]

        # Quick baseline classification rule to test that normalized values work:
        state_result = state_monitor.update(
            avg_ear=avg_ear,
            horizontal_deviation=horizontal_deviation,
            vertical_deviation=vertical_deviation,
        )

        state = state_result.state
        color = state_result.color

        # Render status onto the live video feed
        cv2.putText(
            frame,
            f"STATE: {state}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
            cv2.LINE_AA,
        )
        
        cv2.putText(
            frame,
            (
                f"Observed: {state_monitor.current_observation} "
                f"({state_result.duration:.1f}s)"
            ),
            (10, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        
        # Display debug values to see the normalization in real time
        cv2.putText(
            frame,
            (
                f"EAR: {avg_ear:.2f} | "
                f"Horizontal: {horizontal_deviation:.3f} | "
                f"Vertical: {vertical_deviation:.3f}"
            ),
            (10, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return frame


def start_camera_stream(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print(f"[ERROR] Could not open video stream on camera index {camera_index}")
        return

    print(f"[INFO] Camera stream started on index {camera_index}.")
    print("[INFO] Press 'q' inside the video window to safely exit.")
    print("[INFO] Press 'c' to trigger a recalibration at any point.")

    prev_frame_time = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Mirror view for realistic driver perspective
            frame = cv2.flip(frame, 1)

            # Core processing
            frame = process_driver_frame(frame)

            # FPS calculation
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow('DMM Eye-Tracking Stream', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                reset_calibration()

    except Exception as e:
        print(f"[CRITICAL ERROR] Pipeline crashed: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Camera hardware released safely.")


if __name__ == "__main__":
    start_camera_stream(camera_index=0)