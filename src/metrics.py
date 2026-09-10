import numpy as np
from detector import calculate_ear, get_head_pose_vector

EAR_INDEX = 0

VERTICAL_X_INDEX = 1
VERTICAL_Y_INDEX = 2
VERTICAL_Z_INDEX = 3

FORWARD_X_INDEX = 4
# Pitch (looking up/down) should be read from FORWARD_Y_INDEX, not
# VERTICAL_Y_INDEX. The forehead-chin vector's Y-component is cos(pitch)
# shaped: nearly flat near zero and identical in sign for up vs. down, so it
# can't distinguish looking up from looking down. The nose-tip vector's
# Y-component is sin(pitch) shaped: linear near zero and correctly signed
# (positive = down, negative = up).
FORWARD_Y_INDEX = 5
FORWARD_Z_INDEX = 6

class GazeCalibrator:
    def __init__(self):
        self.baseline_features: np.ndarray | None = None
        self.is_calibrated = False

    def reset(self):
        """
        Removes the current calibration baseline.
        """
        self.baseline_features = None
        self.is_calibrated = False
        print("[CALIBRATION RESET] Waiting for a new baseline.")

    def calibrate(self, calibration_frames):
        """
        Averages samples recorded while the driver looks forward.
        """
        if not calibration_frames:
            print("[CALIBRATION ERROR] No valid frames were provided.")
            return False

        frames = np.asarray(calibration_frames, dtype=np.float64)

        if frames.ndim != 2:
            print("[CALIBRATION ERROR] Invalid calibration data shape.")
            return False

        self.baseline_features = np.mean(frames, axis=0)
        self.is_calibrated = True

        print("[CALIBRATION SUCCESSFUL] New baseline established.")
        return True

    def get_relative_features(self, current_features):
        """
        Returns movement relative to the calibrated baseline.
        EAR remains an absolute measurement.
        """
        if not self.is_calibrated or self.baseline_features is None:
            return current_features

        current_features = np.asarray(current_features, dtype=np.float64)

        if current_features.shape != self.baseline_features.shape:
            raise ValueError(
                "Current feature shape does not match the calibration baseline."
            )

        deltas = np.zeros_like(current_features)

        # Preserve the absolute eye-aspect ratio.
        deltas[EAR_INDEX] = current_features[EAR_INDEX]

        # Normalize spatial measurements against the baseline.
        deltas[1:] = current_features[1:] - self.baseline_features[1:]

        return deltas

def extract_raw_features(multi_face_landmarks):
    """
    Extracts uncalibrated facial features from a MediaPipe result.
    """
    if not multi_face_landmarks:
        return None

    face_landmarks = multi_face_landmarks[0].landmark
    landmarks = [[lm.x, lm.y, lm.z] for lm in face_landmarks]

    left_eye_idx = [159, 160, 145, 144, 33, 133]
    right_eye_idx = [386, 387, 374, 373, 362, 263]

    left_ear = calculate_ear(landmarks, left_eye_idx)
    right_ear = calculate_ear(landmarks, right_eye_idx)
    avg_ear = (left_ear + right_ear) / 2.0

    head_vectors = get_head_pose_vector(landmarks)

    return np.insert(head_vectors, 0, avg_ear)