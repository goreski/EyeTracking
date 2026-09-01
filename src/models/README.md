# Eye-quality model

`eye_quality.py` runs one shared classifier independently on the left and
right eye patches. It therefore produces a separate probability for each eye,
without requiring two separately trained models.

## Output contract

Export a trained classifier to ONNX. It must accept one RGB image tensor of
shape `N x 3 x 32 x 64` and return four uncalibrated logits, in this exact
order:

1. `open_visible`
2. `closed`
3. `occluded`
4. `sunglasses`

The live stream defines `P(usable for gaze)` as the `open_visible` probability.
A normal training run writes `eye_quality.onnx` and its calibration JSON to
`artifacts/eye_quality/`; the stream finds both files automatically.

Fit the temperature on a held-out calibration set, not the training set. To
use a different model location, set `EYE_QUALITY_MODEL_PATH`; optional
overrides are `EYE_QUALITY_CALIBRATION_PATH` and `EYE_QUALITY_TEMPERATURE`.
Until a valid model is available, the stream deliberately shows `model off`
instead of inventing a probability.
