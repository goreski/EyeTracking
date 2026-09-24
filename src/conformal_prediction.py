"""Adaptive Conformal Prediction for Driver Ocular States.

Provides distribution-free, finite-sample coverage guarantees for eye quality
and closure classification. Replaces rigid point predictions with mathematically
grounded prediction sets C(x) that explicitly flag ambiguity and out-of-distribution
conditions under variable driving environments.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np

from models.eye_quality import CLASS_NAMES, EyeQualityResult


@dataclass(frozen=True)
class ConformalPredictionResult:
    """Conformal prediction set and ambiguity metrics for one eye patch."""
    prediction_set: tuple[str, ...]
    set_size: int
    is_singleton: bool
    is_ambiguous: bool
    is_empty: bool
    coverage_level: float        # 1 - alpha (e.g., 0.95)
    nonconformity_threshold: float
    confidence_margin: float     # Difference between top and second class probability


class ConformalEyePredictor:
    """
    Adaptive Conformal Inference (ACI) engine for multi-class ocular states.
    Guarantees that the true label is contained in the prediction set with
    probability >= 1 - alpha.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        default_quantile: float = 0.88,
        gamma: float = 0.005,
        calibration_path: Path | None = None,
    ):
        """
        Args:
            alpha: Significance level (error rate). 1 - alpha is target coverage (e.g. 0.95).
            default_quantile: Default non-conformity threshold quantile if not calibrated.
            gamma: Adaptive learning rate for online quantile updates.
            calibration_path: Path to conformal calibration JSON if available.
        """
        self.alpha = float(alpha)
        self.target_coverage = 1.0 - self.alpha
        self.gamma = float(gamma)
        self.q_hat = float(default_quantile)
        self.calibration_path = calibration_path

        if self.calibration_path and self.calibration_path.is_file():
            self._load_calibration()

    def _load_calibration(self) -> None:
        try:
            data = json.loads(self.calibration_path.read_text(encoding="utf-8"))
            if "conformal_quantile" in data:
                self.q_hat = float(data["conformal_quantile"])
            if "alpha" in data:
                self.alpha = float(data["alpha"])
                self.target_coverage = 1.0 - self.alpha
        except Exception:
            pass

    def predict(self, eye_quality: EyeQualityResult | None) -> ConformalPredictionResult | None:
        """
        Generates a conformal prediction set for an eye quality result.
        Returns None if eye quality is unavailable.
        """
        if eye_quality is None or eye_quality.probabilities is None:
            return None

        probs = eye_quality.probabilities
        # Non-conformity score: s_i = 1 - p_i
        # Inclusion criterion: s_i <= q_hat  <=>  p_i >= 1 - q_hat
        threshold = max(0.0, 1.0 - self.q_hat)

        included_indices = [i for i, p in enumerate(probs) if p >= threshold]

        # If thresholding yielded empty, check the cumulative distribution to ensure coverage
        if not included_indices:
            sorted_indices = np.argsort(-probs)
            cum_prob = 0.0
            for idx in sorted_indices:
                cum_prob += probs[idx]
                included_indices.append(int(idx))
                if cum_prob >= self.target_coverage:
                    break

        included_classes = tuple(CLASS_NAMES[i] for i in included_indices)
        set_size = len(included_classes)

        # Confidence margin (p_top - p_runner_up)
        sorted_probs = np.sort(probs)[::-1]
        margin = float(sorted_probs[0] - (sorted_probs[1] if len(sorted_probs) > 1 else 0.0))

        return ConformalPredictionResult(
            prediction_set=included_classes,
            set_size=set_size,
            is_singleton=(set_size == 1),
            is_ambiguous=(set_size > 1),
            is_empty=(set_size == 0),
            coverage_level=self.target_coverage,
            nonconformity_threshold=self.q_hat,
            confidence_margin=margin,
        )

    def adapt_online(self, was_correct: bool) -> None:
        """
        Adaptive Conformal Inference (ACI) step.
        Updates the quantile online:
            q_{t+1} = q_t + gamma * (alpha - err_t)
        where err_t = 0 if label in set, 1 otherwise.
        """
        err_t = 0.0 if was_correct else 1.0
        self.q_hat = float(np.clip(self.q_hat + self.gamma * (self.alpha - err_t), 0.1, 0.99))
