"""Fit the left/right eye open-state correlation from paired collection data.

`collect_eye_patches.py` records a `pairs_manifest.csv` alongside the labelled
patches. A frame_id shared between a "left" row and a "right" row means both
patches were saved from the exact same instant (guaranteed by freezing the
frame with the 'Z' key before labeling each eye) -- i.e. genuine ground
truth for how often the two eyes share the same open/closed state,
including asymmetric cases (e.g. one eye deliberately closed, the other
open).

This script estimates P(same open/closed state) from those pairs and writes
it to artifacts/eye_quality/eye_pair_correlation.json, which
eye_pair_fusion.py loads in place of its hardcoded default.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

MIN_RECOMMENDED_PAIRS = 20
OPEN_CLASS = "open_visible"
CLOSED_CLASS = "closed"
BINARY_CLASSES = {OPEN_CLASS, CLOSED_CLASS}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data") / "eye_quality" / "pairs_manifest.csv")
    parser.add_argument("--output", type=Path, default=Path("artifacts") / "eye_quality" / "eye_pair_correlation.json")
    return parser.parse_args()


def load_frame_rows(manifest_path: Path) -> dict[str, dict[str, str]]:
    """Returns {frame_id: {"left": class_name, "right": class_name}} for
    frames where both eyes were labelled, restricted to the open/closed
    classes the correlation applies to.
    """
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"No manifest found at {manifest_path}. Collect some paired frames first "
            "(press Z in collect_eye_patches.py to freeze a frame, then label each eye)."
        )

    frames: dict[str, dict[str, str]] = defaultdict(dict)
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["class"] not in BINARY_CLASSES:
                continue
            frames[row["frame_id"]][row["eye"]] = row["class"]

    return {
        frame_id: eyes
        for frame_id, eyes in frames.items()
        if "left" in eyes and "right" in eyes
    }


def estimate_same_state_probability(paired_frames: dict[str, dict[str, str]]) -> tuple[float, int]:
    if not paired_frames:
        return 0.0, 0

    same_state_count = sum(
        1 for eyes in paired_frames.values() if eyes["left"] == eyes["right"]
    )
    return same_state_count / len(paired_frames), len(paired_frames)


def main() -> None:
    arguments = parse_arguments()
    paired_frames = load_frame_rows(arguments.manifest)
    same_state_probability, pair_count = estimate_same_state_probability(paired_frames)

    if pair_count == 0:
        raise ValueError(
            "No frames had both a left and a right open/closed label. "
            "Freeze a frame with Z and label both eyes to build paired data."
        )

    print(f"[INFO] Paired open/closed frames found: {pair_count}")
    print(f"[RESULT] P(same open/closed state) = {same_state_probability:.3f}")

    if pair_count < MIN_RECOMMENDED_PAIRS:
        print(
            f"[WARNING] Only {pair_count} paired frames -- this estimate is noisy. "
            f"Collect at least {MIN_RECOMMENDED_PAIRS} before trusting it in place of the default."
        )

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(
            {
                "same_state_probability": same_state_probability,
                "pair_count": pair_count,
                "source": "empirical",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[SUCCESS] Correlation written to: {arguments.output.resolve()}")


if __name__ == "__main__":
    main()
