"""Probabilistic per-eye quality inference for gaze estimation.

The same model instance is intentionally run once for each eye crop. This
shares learned visual features while retaining independent probabilities for
the driver's left and right eye.

The runtime expects an ONNX classifier whose output is four *logits* in this
fixed order: open_visible, closed, occluded, sunglasses. Model weights are not
kept in the repository; provide their path through EYE_QUALITY_MODEL_PATH.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import os

import cv2
import numpy as np
import onnxruntime as ort


CLASS_NAMES = ("open_visible", "closed", "occluded", "sunglasses")
USABLE_CLASS_INDEX = 0
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "artifacts" / "eye_quality" / "eye_quality.onnx"
DEFAULT_CALIBRATION_PATH = PROJECT_ROOT / "artifacts" / "eye_quality" / "eye_quality_calibration.json"


@dataclass(frozen=True)
class EyeQualityResult:
    """Calibrated probabilities for one extracted eye patch."""

    probabilities: np.ndarray

    @property
    def usable_probability(self) -> float:
        """Probability that this eye can provide iris/gaze information."""
        return float(self.probabilities[USABLE_CLASS_INDEX])

    @property
    def predicted_class(self) -> str:
        return CLASS_NAMES[int(np.argmax(self.probabilities))]


class EyeQualityModel:
    """ONNX Runtime backend for a calibrated four-class eye classifier.

    ``cv2.dnn`` cannot reliably execute graphs produced by PyTorch's current
    (dynamo-based) ONNX exporter -- it silently runs them and returns garbage
    logits instead of raising an error. ONNX Runtime is the reference
    implementation and is what training/export is actually verified against.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        input_size: tuple[int, int] = (64, 32),
        temperature: float = 1.0,
    ):
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero.")

        self.input_size = input_size
        self.temperature = float(temperature)
        self.model_path = Path(model_path) if model_path else None
        self.session: ort.InferenceSession | None = None
        self.input_name: str | None = None
        self.load_error: str | None = None

        if self.model_path is not None:
            self._load_model()

    @classmethod
    def from_environment(cls) -> "EyeQualityModel":
        """Uses environment overrides or the default training-output paths."""
        model_path = Path(os.getenv("EYE_QUALITY_MODEL_PATH", str(DEFAULT_MODEL_PATH)))
        calibration_path = Path(
            os.getenv("EYE_QUALITY_CALIBRATION_PATH", str(DEFAULT_CALIBRATION_PATH))
        )
        temperature = cls._load_temperature(calibration_path)

        temperature_override = os.getenv("EYE_QUALITY_TEMPERATURE")
        if temperature_override is not None:
            temperature = float(temperature_override)

        return cls(model_path=model_path, temperature=temperature)

    @staticmethod
    def _load_temperature(calibration_path: Path) -> float:
        """Reads a saved calibration temperature, falling back safely to 1.0."""
        if not calibration_path.is_file():
            return 1.0

        try:
            calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
            temperature = float(calibration["temperature"])
            return temperature if temperature > 0 else 1.0
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return 1.0

    @property
    def is_available(self) -> bool:
        return self.session is not None

    def _load_model(self) -> None:
        if self.model_path is None:
            return

        if not self.model_path.is_file():
            self.load_error = f"Model file not found: {self.model_path}"
            return

        try:
            self.session = ort.InferenceSession(
                str(self.model_path), providers=["CPUExecutionProvider"]
            )
            self.input_name = self.session.get_inputs()[0].name
            self.load_error = None
        except Exception as error:  # ONNX Runtime raises its own exception types.
            self.session = None
            self.load_error = f"Could not load ONNX model: {error}"

    def predict(self, eye_crop: np.ndarray | None) -> EyeQualityResult | None:
        """Returns calibrated probabilities for one BGR eye crop.

        ``None`` means no usable model or crop is available. Callers should
        treat that as unknown rather than fabricate a gaze-confidence value.
        """
        if self.session is None or eye_crop is None or eye_crop.size == 0:
            return None

        blob = cv2.dnn.blobFromImage(
            eye_crop,
            scalefactor=1.0 / 255.0,
            size=self.input_size,
            mean=(0.0, 0.0, 0.0),
            swapRB=True,
            crop=False,
        )
        logits = np.asarray(
            self.session.run(None, {self.input_name: blob})[0], dtype=np.float64
        ).reshape(-1)

        if logits.size != len(CLASS_NAMES):
            raise ValueError(
                "Eye-quality model must output four logits in the order "
                f"{CLASS_NAMES}; received {logits.size} values."
            )

        return EyeQualityResult(probabilities=self._softmax(logits / self.temperature))

    @staticmethod
    def _softmax(values: np.ndarray) -> np.ndarray:
        shifted = values - np.max(values)
        exp_values = np.exp(shifted)
        return exp_values / np.sum(exp_values)
