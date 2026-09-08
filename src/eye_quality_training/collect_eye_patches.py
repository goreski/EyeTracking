"""Collect manually labelled left/right eye patches for the eye-quality model."""

from __future__ import annotations

import argparse
import csv
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

MANIFEST_COLUMNS = ("frame_id", "eye", "class", "path")


def append_manifest_row(output_directory: Path, frame_id: int, eye_name: str, class_name: str, path: Path) -> None:
    """Records which camera frame a saved patch came from.

    Patches saved with the same frame_id came from the exact same instant
    (guaranteed when saved while frozen -- see the 'Z' key in `collect()`),
    which lets estimate_pair_correlation.py later match them into genuine
    left/right ground-truth pairs, including asymmetric ones.
    """
    manifest_path = output_directory / "pairs_manifest.csv"
    is_new_file = not manifest_path.exists()

    with manifest_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        if is_new_file:
            writer.writeheader()
        writer.writerow({"frame_id": frame_id, "eye": eye_name, "class": class_name, "path": str(path)})


def save_patch(output_directory: Path, class_name: str, eye_name: str, patch, frame_id: int) -> Path | None:
    """Saves one patch and returns its path, or None if the crop is unavailable."""
    if patch is None or patch.size == 0:
        return None

    class_directory = output_directory / class_name
    class_directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = class_directory / f"{frame_id:08d}_{timestamp}_{eye_name}.jpg"

    if not cv2.imwrite(str(output_path), patch):
        raise RuntimeError(f"Could not save patch: {output_path}")

    append_manifest_row(output_directory, frame_id, eye_name, class_name, output_path)
    return output_path


def annotate_frame(frame):
    """Draws short, always-visible data-collection controls."""
    instructions = (
        "1-4: save both | A/S/D/F: left | J/K/L/P: right | Z: freeze pair | Q: quit",
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
    print(
        "[INFO] Press Z to freeze the current frame, then label each eye "
        "independently (e.g. one closed, one open) to record a true paired "
        "example for estimating cross-eye correlation. Press Z again to resume."
    )

    latest_left_patch = None
    latest_right_patch = None
    base_frame = None
    frozen = False
    frame_id = 0

    try:
        while True:
            if not frozen:
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

                base_frame = frame
                frame_id += 1

            display_frame = base_frame.copy()
            annotate_frame(display_frame)
            if frozen:
                cv2.putText(
                    display_frame,
                    "FROZEN -- label each eye, then press Z to resume",
                    (10, 135),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            cv2.imshow("Eye Quality Patch Collection", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            if key == ord("z"):
                frozen = not frozen
                print("[INFO] Frame frozen for paired labeling." if frozen else "[INFO] Resumed live feed.")
            elif key in PAIR_KEYS:
                class_name = PAIR_KEYS[key]
                saved = [
                    save_patch(output_directory, class_name, "left", latest_left_patch, frame_id),
                    save_patch(output_directory, class_name, "right", latest_right_patch, frame_id),
                ]
                count = sum(path is not None for path in saved)
                print(f"[SAVED] {count} eye patches as {class_name}.")
            elif key in LEFT_KEYS:
                class_name = LEFT_KEYS[key]
                saved = save_patch(output_directory, class_name, "left", latest_left_patch, frame_id)
                print(f"[SAVED] Left eye as {class_name}." if saved else "[WARNING] No left-eye patch to save.")
            elif key in RIGHT_KEYS:
                class_name = RIGHT_KEYS[key]
                saved = save_patch(output_directory, class_name, "right", latest_right_patch, frame_id)
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
