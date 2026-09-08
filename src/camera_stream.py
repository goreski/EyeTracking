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

from detector import (
    DriverFaceDetector,
    crop_eye_region,
    LEFT_EYE_INDICES,
    RIGHT_EYE_INDICES,
)
from models.eye_quality import EyeQualityModel
from eye_pair_fusion import fuse_eye_pair
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
eye_quality_model = EyeQualityModel.from_environment()

# Calibration configuration
REQUIRED_CALIBRATION_FRAMES = 90  # ~3 seconds at 30 FPS
calibration_buffer = []

# Eye thumbnail overlay configuration (top-right corner of the stream)
EYE_THUMB_WIDTH = 180
EYE_THUMB_HEIGHT = 135
EYE_THUMB_MARGIN = 10
SHOW_EYE_LANDMARKS = False  # Toggle to draw eye-contour landmarks in the thumbnails


def overlay_eye_thumbnail(frame, eye_crop, eye_points, quality_result, fused_estimate, x, y, eye_name):
    """Resizes an eye crop and pastes it into the frame at (x, y)."""
    if eye_crop is None or eye_crop.size == 0:
        return

    h, w = frame.shape[:2]
    if y + EYE_THUMB_HEIGHT > h or x + EYE_THUMB_WIDTH > w:
        return

    crop_h, crop_w = eye_crop.shape[:2]
    thumb = cv2.resize(eye_crop, (EYE_THUMB_WIDTH, EYE_THUMB_HEIGHT))

    if SHOW_EYE_LANDMARKS and eye_points is not None:
        scale_x = EYE_THUMB_WIDTH / crop_w
        scale_y = EYE_THUMB_HEIGHT / crop_h
        for px, py in eye_points:
            cv2.circle(thumb, (int(px * scale_x), int(py * scale_y)), 2, (0, 255, 0), -1)

    if quality_result is None:
        lines = [f"{eye_name}: model off"]
        color = (0, 165, 255)
    else:
        open_visible, closed, occluded, sunglasses = quality_result.probabilities
        predicted_probability = float(np.max(quality_result.probabilities))
        lines = [
            f"{eye_name}: {quality_result.predicted_class} ({predicted_probability:.0%})",
            f"open {open_visible:.0%}  closed {closed:.0%}",
            f"occ {occluded:.0%}  sun {sunglasses:.0%}",
        ]
        color = (0, 255, 0) if quality_result.usable_probability >= 0.85 else (0, 165, 255)

        if fused_estimate is not None and fused_estimate.inferred_from_other_eye:
            lines.append(f"inferred open {fused_estimate.open_probability:.0%} (from other eye)")

    # Show the model's complete decision, not only the usable probability,
    # using a translucent label band so the eye image stays visible underneath.
    label_band_height = min(18 + len(lines) * 20, EYE_THUMB_HEIGHT)
    label_overlay = thumb.copy()
    cv2.rectangle(label_overlay, (0, 0), (EYE_THUMB_WIDTH, label_band_height), (0, 0, 0), -1)
    thumb = cv2.addWeighted(label_overlay, 0.45, thumb, 0.55, 0)
    frame[y:y + EYE_THUMB_HEIGHT, x:x + EYE_THUMB_WIDTH] = thumb
    cv2.rectangle(frame, (x, y), (x + EYE_THUMB_WIDTH, y + EYE_THUMB_HEIGHT), (0, 255, 0), 1)

    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (x + 4, y + 18 + index * 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )


def draw_eye_thumbnails(
    frame,
    left_eye_crop,
    left_eye_points,
    left_eye_quality,
    right_eye_crop,
    right_eye_points,
    right_eye_quality,
):
    # Anchored to the top-right corner of the frame.
    frame_width = frame.shape[1]
    right_thumb_x = frame_width - EYE_THUMB_MARGIN - EYE_THUMB_WIDTH
    left_thumb_x = right_thumb_x - EYE_THUMB_MARGIN - EYE_THUMB_WIDTH

    # Borrow the other eye's open/closed distribution when one eye is occluded.
    left_eye_fused, right_eye_fused = fuse_eye_pair(left_eye_quality, right_eye_quality)

    # Swapped so the displayed order matches the mirrored (flipped) frame.
    overlay_eye_thumbnail(
        frame,
        right_eye_crop,
        right_eye_points,
        right_eye_quality,
        right_eye_fused,
        left_thumb_x,
        EYE_THUMB_MARGIN,
        "R",
    )
    overlay_eye_thumbnail(
        frame,
        left_eye_crop,
        left_eye_points,
        left_eye_quality,
        left_eye_fused,
        right_thumb_x,
        EYE_THUMB_MARGIN,
        "L",
    )


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

    # If no face is in the frame, we skip calculations
    if not detection_results.multi_face_landmarks:
        frame = detector.draw_mesh(frame, detection_results)
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

    # Crop eyes before the mesh overlay is drawn, so crops stay clean.
    face_landmarks = detection_results.multi_face_landmarks[0].landmark
    left_eye_crop, left_eye_points = crop_eye_region(frame, face_landmarks, LEFT_EYE_INDICES)
    right_eye_crop, right_eye_points = crop_eye_region(frame, face_landmarks, RIGHT_EYE_INDICES)
    left_eye_quality = eye_quality_model.predict(left_eye_crop)
    right_eye_quality = eye_quality_model.predict(right_eye_crop)

    frame = detector.draw_mesh(frame, detection_results)

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
        draw_eye_thumbnails(
            frame,
            left_eye_crop,
            left_eye_points,
            left_eye_quality,
            right_eye_crop,
            right_eye_points,
            right_eye_quality,
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

        # Render status onto the live video feed, below the eye thumbnails
        cv2.putText(
            frame,
            f"STATE: {state}",
            (10, 130),
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
            (10, 200),
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
            (10, 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    draw_eye_thumbnails(
        frame,
        left_eye_crop,
        left_eye_points,
        left_eye_quality,
        right_eye_crop,
        right_eye_points,
        right_eye_quality,
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
    if eye_quality_model.is_available:
        print(f"[INFO] Eye-quality ONNX model loaded: {eye_quality_model.model_path}")
    else:
        print("[INFO] Eye-quality model is off. Train a model or set EYE_QUALITY_MODEL_PATH.")
        if eye_quality_model.load_error:
            print(f"[WARNING] {eye_quality_model.load_error}")

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
