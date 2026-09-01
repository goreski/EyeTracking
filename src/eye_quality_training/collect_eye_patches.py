"""Collect manually labelled left/right eye patches for the eye-quality model."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import cv2

from labels import CLASS_NAMES

# Allow direct execution with: python src\eye_quality_training\collect_eye_patches.py
SRC_DIRECTORY = Path(__file__).resolve().parents[1]
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from detector import (
    DriverFaceDetector,
    LEFT_EYE_INDICES,
    RIGHT_EYE_INDICES,
    crop_eye_region,
)


PAIR_KEYS = {ord(str(index + 1)): class_name for index, class_name in enumerate(CLASS_NAMES)}
LEFT_KEYS = {
    ord("a"): "open_visible",
    ord("s"): "closed",
    ord("d"): "occluded",
    ord("f"): "sunglasses",
}
RIGHT_KEYS = {
    ord("j"): "open_visible",
    ord("k"): "closed",
    ord("l"): "occluded",
    ord("p"): "sunglasses",
}


def save_patch(output_directory: Path, class_name: str, eye_name: str, patch) -> Path | None:
    """Saves one patch and returns its path, or None if the crop is unavailable."""
    if patch is None or patch.size == 0:
        return None

    class_directory = output_directory / class_name
    class_directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = class_directory / f"{timestamp}_{eye_name}.jpg"

    if not cv2.imwrite(str(output_path), patch):
        raise RuntimeError(f"Could not save patch: {output_path}")

    return output_path


def annotate_frame(frame):
    """Draws short, always-visible data-collection controls."""
    instructions = (
        "1-4: save both | A/S/D/F: left | J/K/L/P: right | Q: quit",
        "1/A/J open  2/S/K closed  3/D/L occluded  4/F/P sunglasses",
    )
    for index, line in enumerate(instructions):
        y = 28 + index * 25
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)


def collect(camera_index: int, output_directory: Path) -> None:
    """Runs the manual eye-patch collection window."""
    detector = DriverFaceDetector()
    camera = cv2.VideoCapture(camera_index)

    if not camera.isOpened():
        detector.face_mesh.close()
        raise RuntimeError(f"Could not open camera index {camera_index}.")

    output_directory.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Saving labelled patches in: {output_directory.resolve()}")
    print("[INFO] Use the visible keyboard controls; press Q to finish.")

    latest_left_patch = None
    latest_right_patch = None

    try:
        while True:
            success, frame = camera.read()
            if not success:
                print("[WARNING] Could not read a camera frame.")
                break

            frame = cv2.flip(frame, 1)
            results = detector.find_face_landmarks(frame)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                latest_left_patch, _ = crop_eye_region(frame, landmarks, LEFT_EYE_INDICES)
                latest_right_patch, _ = crop_eye_region(frame, landmarks, RIGHT_EYE_INDICES)
                detector.draw_mesh(frame, results)
                cv2.putText(frame, "Face detected", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                latest_left_patch = None
                latest_right_patch = None
                cv2.putText(frame, "Face not detected", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)

            annotate_frame(frame)
            cv2.imshow("Eye Quality Patch Collection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            if key in PAIR_KEYS:
                class_name = PAIR_KEYS[key]
                saved = [
                    save_patch(output_directory, class_name, "left", latest_left_patch),
                    save_patch(output_directory, class_name, "right", latest_right_patch),
                ]
                count = sum(path is not None for path in saved)
                print(f"[SAVED] {count} eye patches as {class_name}.")
            elif key in LEFT_KEYS:
                class_name = LEFT_KEYS[key]
                saved = save_patch(output_directory, class_name, "left", latest_left_patch)
                print(f"[SAVED] Left eye as {class_name}." if saved else "[WARNING] No left-eye patch to save.")
            elif key in RIGHT_KEYS:
                class_name = RIGHT_KEYS[key]
                saved = save_patch(output_directory, class_name, "right", latest_right_patch)
                print(f"[SAVED] Right eye as {class_name}." if saved else "[WARNING] No right-eye patch to save.")
    finally:
        camera.release()
        detector.face_mesh.close()
        cv2.destroyAllWindows()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index (default: 0).")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data") / "eye_quality",
        help="Directory in which class folders are created.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    collect(camera_index=arguments.camera, output_directory=arguments.output)
