"""Fine-tune MobileNetV3-Small and export the eye-quality model to ONNX."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
except ImportError as error:  # Clear guidance when this standalone environment is not installed yet.
    raise SystemExit(
        "Training dependencies are missing. Activate a training environment and run "
        "`pip install -r src\\eye_quality_training\\requirements.txt`."
    ) from error

from labels import CLASS_NAMES


INPUT_HEIGHT = 32
INPUT_WIDTH = 64


class EyeQualityDataset(datasets.ImageFolder):
    """ImageFolder with a fixed output-label order required by the runtime."""

    def find_classes(self, directory: str):
        expected_directories = [Path(directory) / class_name for class_name in CLASS_NAMES]
        missing = [path.name for path in expected_directories if not path.is_dir()]
        if missing:
            raise FileNotFoundError(f"Missing class folders in {directory}: {', '.join(missing)}")
        return list(CLASS_NAMES), {class_name: index for index, class_name in enumerate(CLASS_NAMES)}


class EyeQualityNet(nn.Module):
    """MobileNetV3-Small with deployment-time RGB normalization built in."""

    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        self.backbone.classifier[3] = nn.Linear(self.backbone.classifier[3].in_features, len(CLASS_NAMES))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.backbone((image - self.mean) / self.std)

    def set_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.backbone.features.parameters():
            parameter.requires_grad = trainable


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data") / "eye_quality_split")
    parser.add_argument("--output", type=Path, default=Path("artifacts") / "eye_quality")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--freeze-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--fine-tune-learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0, help="Use 0 on Windows unless multiprocessing is configured.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_transforms():
    train_transform = transforms.Compose(
        [
            transforms.Resize((INPUT_HEIGHT, INPUT_WIDTH)),
            transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.90, 1.10)),
            transforms.ColorJitter(brightness=0.20, contrast=0.20),
            transforms.ToTensor(),
        ]
    )
    evaluation_transform = transforms.Compose(
        [transforms.Resize((INPUT_HEIGHT, INPUT_WIDTH)), transforms.ToTensor()]
    )
    return train_transform, evaluation_transform


def make_loaders(data_directory: Path, batch_size: int, workers: int):
    train_transform, evaluation_transform = build_transforms()
    train_dataset = EyeQualityDataset(data_directory / "train", transform=train_transform)
    validation_dataset = EyeQualityDataset(data_directory / "val", transform=evaluation_transform)
    test_dataset = EyeQualityDataset(data_directory / "test", transform=evaluation_transform)

    loaders = {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers),
        "val": DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=workers),
        "test": DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers),
    }
    return train_dataset, loaders


def class_weights(dataset: EyeQualityDataset, device: torch.device) -> torch.Tensor:
    counts = torch.bincount(torch.tensor(dataset.targets), minlength=len(CLASS_NAMES)).float()
    if torch.any(counts == 0):
        empty_classes = [CLASS_NAMES[index] for index, count in enumerate(counts) if count == 0]
        raise ValueError(f"Training split has no examples for: {', '.join(empty_classes)}")
    weights = counts.sum() / (len(CLASS_NAMES) * counts)
    print("[INFO] Training class counts:", dict(zip(CLASS_NAMES, counts.int().tolist())))
    return weights.to(device)


def evaluate(model: EyeQualityNet, loader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    logits_batches = []
    label_batches = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_loss += criterion(logits, labels).item() * labels.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_examples += labels.size(0)
            logits_batches.append(logits.cpu())
            label_batches.append(labels.cpu())

    if total_examples == 0:
        raise ValueError("Evaluation split contains no images.")
    return (
        total_loss / total_examples,
        total_correct / total_examples,
        torch.cat(logits_batches),
        torch.cat(label_batches),
    )


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Fits post-hoc temperature scaling on held-out validation logits."""
    log_temperature = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=50)
    criterion = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = criterion(logits / log_temperature.exp(), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(min=0.05, max=20.0).item())


def train(arguments: argparse.Namespace) -> None:
    if arguments.epochs < 1 or arguments.freeze_epochs < 0 or arguments.freeze_epochs > arguments.epochs:
        raise ValueError("epochs must be positive and freeze-epochs must be between 0 and epochs.")

    set_seed(arguments.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training on {device}.")

    train_dataset, loaders = make_loaders(arguments.data, arguments.batch_size, arguments.workers)
    model = EyeQualityNet().to(device)
    weights = class_weights(train_dataset, device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    model.set_backbone_trainable(False)
    optimizer = torch.optim.AdamW(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=arguments.learning_rate,
        weight_decay=1e-4,
    )

    arguments.output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = arguments.output / "eye_quality_best.pt"
    best_validation_loss = float("inf")

    for epoch in range(1, arguments.epochs + 1):
        if epoch == arguments.freeze_epochs + 1:
            model.set_backbone_trainable(True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=arguments.fine_tune_learning_rate, weight_decay=1e-4)
            print("[INFO] Unfroze the MobileNet feature extractor.")

        model.train()
        for images, labels in loaders["train"]:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

        validation_loss, validation_accuracy, _, _ = evaluate(model, loaders["val"], criterion, device)
        print(
            f"[EPOCH {epoch:02d}/{arguments.epochs}] "
            f"val_loss={validation_loss:.4f} val_accuracy={validation_accuracy:.3f}"
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(model.state_dict(), checkpoint_path)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    test_loss, test_accuracy, _, _ = evaluate(model, loaders["test"], criterion, device)
    _, validation_accuracy, validation_logits, validation_labels = evaluate(model, loaders["val"], criterion, device)
    temperature = fit_temperature(validation_logits, validation_labels)

    print(f"[RESULT] test_loss={test_loss:.4f} test_accuracy={test_accuracy:.3f}")
    print(f"[RESULT] validation_accuracy={validation_accuracy:.3f} temperature={temperature:.3f}")

    model.eval().cpu()
    onnx_path = arguments.output / "eye_quality.onnx"
    example_input = torch.zeros(1, 3, INPUT_HEIGHT, INPUT_WIDTH, dtype=torch.float32)
    onnx_program = torch.onnx.export(
        model,
        (example_input,),
        input_names=["eye_patch"],
        output_names=["logits"],
        dynamo=True,
    )
    onnx_program.save(str(onnx_path))

    calibration_path = arguments.output / "eye_quality_calibration.json"
    calibration_path.write_text(
        json.dumps(
            {
                "temperature": temperature,
                "class_names": CLASS_NAMES,
                "input_shape": ["N", 3, INPUT_HEIGHT, INPUT_WIDTH],
                "input_description": "RGB pixels scaled to 0-1; ImageNet normalization is embedded in ONNX.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[SUCCESS] ONNX model: {onnx_path.resolve()}")
    print(f"[SUCCESS] Calibration: {calibration_path.resolve()}")


if __name__ == "__main__":
    train(parse_arguments())
