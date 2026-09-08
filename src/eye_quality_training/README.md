# Eye-quality training data collection

This standalone utility saves the existing MediaPipe eye crops into labelled
folders. It does not run the live driver-state pipeline and does not require a
trained eye-quality model.

Run it from the project root:

```powershell
python src\eye_quality_training\collect_eye_patches.py --camera 0
```

By default it creates this dataset structure:

```text
data/eye_quality/
  open_visible/
  closed/
  occluded/
  sunglasses/
```

The collector uses manual labels. Press a key while the correct eye state is
visible:

| Keys | Saved patch(es) | Class |
| --- | --- | --- |
| `1` | both eyes | `open_visible` |
| `2` | both eyes | `closed` |
| `3` | both eyes | `occluded` |
| `4` | both eyes | `sunglasses` |
| `A` / `J` | left / right eye | `open_visible` |
| `S` / `K` | left / right eye | `closed` |
| `D` / `L` | left / right eye | `occluded` |
| `F` / `P` | left / right eye | `sunglasses` |
| `Q` | — | quit |

Use the individual-eye keys whenever the two eyes have different visibility.
Collect diverse people, camera mounts, lighting, glasses, and partial
occlusions. Do not save many near-identical adjacent frames: vary the pose and
conditions between samples.

### Paired ground truth for cross-eye correlation

Press `Z` to freeze the current frame, then label each eye independently
(e.g. one closed, one open) before pressing `Z` again to resume. Because both
patches come from the exact same frozen instant, this records genuine
ground truth for how often the two eyes share the same open/closed
state -- including deliberately asymmetric cases a live, moving feed can't
reliably capture. Every save (frozen or not) is also logged to
`pairs_manifest.csv` in the output directory.

Once you have a reasonable number of frozen pairs (20+), fit the
correlation used by `src/eye_pair_fusion.py` to infer an occluded eye's
state from the other eye:

```powershell
python src\eye_quality_training\estimate_pair_correlation.py
```

This writes `artifacts/eye_quality/eye_pair_correlation.json`. Until that
file exists, the fusion module falls back to a documented default
assumption (0.93) instead of a measured value.

## Split, train, and export

Install the standalone training dependencies into a separate environment:

```powershell
pip install -r src\eye_quality_training\requirements.txt
```

Create a reproducible split without changing the raw collected data:

```powershell
python src\eye_quality_training\prepare_dataset.py
```

Fine-tune MobileNetV3-Small, calibrate its probabilities on the validation
set, and export a compatible ONNX model:

```powershell
python src\eye_quality_training\train_eye_quality.py --epochs 15
```

Outputs are written to `artifacts/eye_quality/`:

```text
eye_quality_best.pt
eye_quality.onnx
eye_quality_calibration.json
```

The exported ONNX model includes ImageNet normalization. Its input is RGB eye
patches with shape `N x 3 x 32 x 64` and values scaled to 0-1. Its output is
four logits in the model-runtime class order.

The live stream automatically loads these default files after training:

```powershell
python src\camera_stream.py
```

It reads the temperature from `eye_quality_calibration.json`. Do not estimate
that value from training accuracy.
