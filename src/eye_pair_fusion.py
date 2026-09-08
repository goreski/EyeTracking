"""Bayesian fusion of left/right eye-quality predictions.

Blinks are almost always binocular, so when one eye's own classifier read is
unreliable (occluded), the other eye's open/closed distribution is an
informative prior for the hidden eye's true state.

The correlation strength (P(this eye open | other eye open)) is fitted from
data whenever it's available: collect_eye_patches.py can freeze a frame (the
'Z' key) so both eyes are labelled from the exact same instant, and
estimate_pair_correlation.py turns those paired frames into
artifacts/eye_quality/eye_pair_correlation.json. Until that file exists,
DEFAULT_SAME_STATE_PROBABILITY below is used as a documented assumption,
not a measured one.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import os

from models.eye_quality import CLASS_NAMES, EyeQualityResult

OPEN_INDEX = CLASS_NAMES.index("open_visible")
CLOSED_INDEX = CLASS_NAMES.index("closed")
OCCLUDED_INDEX = CLASS_NAMES.index("occluded")

# P(this eye open | other eye open), used symmetrically for P(closed | closed)
# too. Falls back to this assumption until a fitted value is available.
DEFAULT_SAME_STATE_PROBABILITY = 0.93

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORRELATION_PATH = PROJECT_ROOT / "artifacts" / "eye_quality" / "eye_pair_correlation.json"


def _load_same_state_probability() -> float:
    """Reads a fitted correlation from disk, falling back safely to the default."""
    correlation_path = Path(os.getenv("EYE_PAIR_CORRELATION_PATH", str(DEFAULT_CORRELATION_PATH)))

    if not correlation_path.is_file():
        return DEFAULT_SAME_STATE_PROBABILITY

    try:
        correlation = json.loads(correlation_path.read_text(encoding="utf-8"))
        probability = float(correlation["same_state_probability"])
        return probability if 0.0 < probability < 1.0 else DEFAULT_SAME_STATE_PROBABILITY
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return DEFAULT_SAME_STATE_PROBABILITY


SAME_STATE_PROBABILITY = _load_same_state_probability()

# Occluded-class probability above which an eye's own open/closed read is
# treated as unreliable and worth replacing with an inference from the
# other eye instead.
OCCLUSION_TRUST_THRESHOLD = 0.5


@dataclass(frozen=True)
class FusedEyeEstimate:
    """A probability that this eye is open, plus where that number came from."""

    open_probability: float
    inferred_from_other_eye: bool


def _open_closed_distribution(result: EyeQualityResult | None) -> tuple[float, float] | None:
    """Returns (P(open), P(closed)) renormalized over just those two classes."""
    if result is None:
        return None

    open_probability = result.probabilities[OPEN_INDEX]
    closed_probability = result.probabilities[CLOSED_INDEX]
    total = open_probability + closed_probability

    if total < 1e-6:
        return None

    return open_probability / total, closed_probability / total


def _fuse_one_eye(
    this_eye: EyeQualityResult | None, other_eye: EyeQualityResult | None
) -> FusedEyeEstimate | None:
    if this_eye is None:
        return None

    this_eye_occluded = this_eye.probabilities[OCCLUDED_INDEX] >= OCCLUSION_TRUST_THRESHOLD
    other_distribution = _open_closed_distribution(other_eye)

    if not this_eye_occluded or other_distribution is None:
        own_distribution = _open_closed_distribution(this_eye)
        if own_distribution is None:
            return None
        return FusedEyeEstimate(open_probability=own_distribution[0], inferred_from_other_eye=False)

    other_open, other_closed = other_distribution

    # Marginalize over the other eye's open/closed state using the assumed
    # binocular correlation: P(this=open) = sum_s P(this=open | other=s) P(other=s)
    inferred_open_probability = (
        other_open * SAME_STATE_PROBABILITY + other_closed * (1.0 - SAME_STATE_PROBABILITY)
    )
    return FusedEyeEstimate(open_probability=inferred_open_probability, inferred_from_other_eye=True)


def fuse_eye_pair(
    left: EyeQualityResult | None, right: EyeQualityResult | None
) -> tuple[FusedEyeEstimate | None, FusedEyeEstimate | None]:
    """Infers each eye's open-probability, borrowing from the other eye when
    this eye is occluded and the other eye is not.
    """
    return _fuse_one_eye(left, right), _fuse_one_eye(right, left)
