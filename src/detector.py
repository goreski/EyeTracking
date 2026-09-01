import cv2
import mediapipe as mp
import numpy as np

from mediapipe.python.solutions import face_mesh as mp_face_mesh
from mediapipe.python.solutions import drawing_utils as mp_drawing
from typing import Any

# Full eye-contour landmark indices (denser than the 6-point EAR set).
LEFT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466,
                     388, 387, 386, 385, 384, 398]
RIGHT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173,
                      157, 158, 159, 160, 161, 246]


def crop_eye_region(frame, landmarks, eye_indices, padding=8):
    """
    Crops the eye bounding box (plus padding) out of the frame.

    `landmarks` are MediaPipe's normalized (0-1) landmark objects, not the
    [x, y, z] lists produced by extract_raw_features.

    Returns (crop, local_points) where local_points are the eye-contour
    pixel coordinates translated into the crop's own coordinate space, or
    (None, None) if the box is degenerate.
    """
    h, w = frame.shape[:2]
    points = np.array(
        [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_indices],
        dtype=np.int32,
    )

    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)

    x_min = max(int(x_min) - padding, 0)
    y_min = max(int(y_min) - padding, 0)
    x_max = min(int(x_max) + padding, w)
    y_max = min(int(y_max) + padding, h)

    # Degenerate box (e.g. landmarks collapsed) yields an empty crop.
    if x_max <= x_min or y_max <= y_min:
        return None, None

    # Copy so later drawing on `frame` (e.g. the mesh overlay) can't leak into the crop.
    crop = frame[y_min:y_max, x_min:x_max].copy()
    local_points = points - np.array([x_min, y_min])

    return crop, local_points


def euclidean_distance(point_a, point_b):
    """
    Calculates 2D Euclidean distance between two facial landmarks.

    The z-coordinate is intentionally excluded because MediaPipe's z values
    are relative estimates and may have a different scale from x and y.
    """
    point_a = np.asarray(point_a[:2], dtype=np.float64)
    point_b = np.asarray(point_b[:2], dtype=np.float64)

    return float(np.linalg.norm(point_a - point_b))


def calculate_ear(landmarks, eye_indices):
    """
    Calculates the Eye Aspect Ratio.

    Expected eye_indices layout:
        [
            upper_point_1,
            upper_point_2,
            lower_point_1,
            lower_point_2,
            left_corner,
            right_corner
        ]
    """
    if len(eye_indices) != 6:
        raise ValueError("eye_indices must contain exactly six landmark indices.")

    upper_1 = landmarks[eye_indices[0]]
    upper_2 = landmarks[eye_indices[1]]
    lower_1 = landmarks[eye_indices[2]]
    lower_2 = landmarks[eye_indices[3]]
    left_corner = landmarks[eye_indices[4]]
    right_corner = landmarks[eye_indices[5]]

    vertical_distance_1 = euclidean_distance(upper_1, lower_1)
    vertical_distance_2 = euclidean_distance(upper_2, lower_2)
    horizontal_distance = euclidean_distance(left_corner, right_corner)

    # Avoid division by zero when landmarks overlap or tracking is unstable.
    if horizontal_distance < 1e-6:
        return 0.0

    return (
        vertical_distance_1 + vertical_distance_2
    ) / (2.0 * horizontal_distance)


def normalize_vector(vector):
    """
    Converts a vector to unit length.
    """
    vector = np.asarray(vector, dtype=np.float64)
    magnitude = np.linalg.norm(vector)

    if magnitude < 1e-6:
        return np.zeros(3, dtype=np.float64)

    return vector / magnitude


def get_head_pose_vector(landmarks):
    """
    Produces two approximate normalized face-orientation vectors:

    1. Vertical vector:
       Direction from the chin toward the forehead.

    2. Forward vector:
       Direction from the center of the face toward the nose tip.

    Returns:
        np.ndarray containing:
        [
            vertical_x,
            vertical_y,
            vertical_z,
            forward_x,
            forward_y,
            forward_z
        ]

    These vectors are relative geometric measurements, not true yaw,
    pitch and roll angles.
    """
    points = np.asarray(landmarks, dtype=np.float64)

    # MediaPipe Face Mesh landmark indices
    forehead_index = 10
    chin_index = 152
    nose_tip_index = 1
    left_cheek_index = 234
    right_cheek_index = 454

    forehead = points[forehead_index]
    chin = points[chin_index]
    nose_tip = points[nose_tip_index]
    left_cheek = points[left_cheek_index]
    right_cheek = points[right_cheek_index]

    vertical_vector = normalize_vector(forehead - chin)

    face_center = (left_cheek + right_cheek) / 2.0
    forward_vector = normalize_vector(nose_tip - face_center)

    return np.concatenate((vertical_vector, forward_vector))

class DriverFaceDetector:
    def __init__(self):
        self.mp_face_mesh = mp_face_mesh
        self.mp_drawing = mp_drawing
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,  # Crucial for iris tracking!
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def find_face_landmarks(self, frame) -> Any:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        return results

    def draw_mesh(self, frame, results) -> np.ndarray:
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # FIXED LINE HERE: Changed FACEMESH_TESSELLATION to FACEMESH_CONTOURS
                self.mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=list(self.mp_face_mesh.FACEMESH_CONTOURS),
                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0)),
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(thickness=1, color=(0, 255, 0))
                )
        return frame