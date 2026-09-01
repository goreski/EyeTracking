"""Create a reproducible train/validation/test split from collected eye patches."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import random
import shutil
from typing import Literal

from labels import CLASS_NAMES


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
ManifestColumn = Literal["source", "split", "class", "destination"]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data") / "eye_quality")
    parser.add_argument("--output", type=Path, default=Path("data") / "eye_quality_split")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_ratios(train_ratio: float, val_ratio: float) -> None:
    if not 0 < train_ratio < 1 or not 0 < val_ratio < 1 or train_ratio + val_ratio >= 1:
        raise ValueError("train-ratio and val-ratio must be between 0 and 1 and sum to less than 1.")


def image_paths(class_directory: Path) -> list[Path]:
    return sorted(path for path in class_directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)


def split_paths(paths: list[Path], train_ratio: float, val_ratio: float) -> dict[str, list[Path]]:
    train_end = round(len(paths) * train_ratio)
    val_end = train_end + round(len(paths) * val_ratio)
    return {
        "train": paths[:train_end],
        "val": paths[train_end:val_end],
        "test": paths[val_end:],
    }


def prepare_dataset(source: Path, output: Path, train_ratio: float, val_ratio: float, seed: int) -> None:
    validate_ratios(train_ratio, val_ratio)
    if not source.is_dir():
        raise FileNotFoundError(f"Source dataset directory does not exist: {source}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output}. Choose a new output path to avoid overwriting a split."
        )

    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    manifest_rows: list[dict[ManifestColumn, str]] = []

    for class_name in CLASS_NAMES:
        source_class_directory = source / class_name
        if not source_class_directory.is_dir():
            raise FileNotFoundError(f"Missing class directory: {source_class_directory}")

        paths = image_paths(source_class_directory)
        if not paths:
            raise ValueError(f"No images found in: {source_class_directory}")

        rng.shuffle(paths)
        splits = split_paths(paths, train_ratio, val_ratio)
        print(
            f"[INFO] {class_name}: "
            + ", ".join(f"{split}={len(items)}" for split, items in splits.items())
        )

        for split_name, split_items in splits.items():
            destination_directory = output / split_name / class_name
            destination_directory.mkdir(parents=True, exist_ok=True)
            for index, source_path in enumerate(split_items):
                destination_name = f"{index:06d}_{source_path.name}"
                destination_path = destination_directory / destination_name
                shutil.copy2(source_path, destination_path)
                manifest_rows.append(
                    {
                        "source": str(source_path),
                        "split": split_name,
                        "class": class_name,
                        "destination": str(destination_path),
                    }
                )

    manifest_path = output / "split_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("source", "split", "class", "destination"))
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"[SUCCESS] Dataset split written to: {output.resolve()}")
    print("[WARNING] This is an image-level split. For final evaluation, split entire drivers/sessions before collection.")


if __name__ == "__main__":
    arguments = parse_arguments()
    prepare_dataset(
        source=arguments.source,
        output=arguments.output,
        train_ratio=arguments.train_ratio,
        val_ratio=arguments.val_ratio,
        seed=arguments.seed,
    )
