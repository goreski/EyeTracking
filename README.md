# Eye-Tracking for Driver Attention (DMM Project)

This project focuses on researching driver attention, distraction, and drowsiness using computer vision techniques. By leveraging live camera streams and facial landmark detection, the system analyzes driver states in real-time to promote road safety.

---

## 🏗️ System Architecture

The project uses a three-tiered pipeline to process live video and classify driver states:

```text
[Live Camera Stream] 
        │
        ▼
[1. Face Mesh / Landmark Detection] ──> Extracts 3D coordinates of eyes/face
        │
        ▼
[2. Feature Extraction Engine]     ──> Calculates EAR (Blinks) & Gaze Vectors (Direction)
        │
        ▼
[3. State Decision Logic]         ──> Classifies: "Attentive", "Distracted", "Drowsy"
```

---

## Environments and quick switching

The project uses two separate virtual environments:

```text
.venv/                Live webcam application: OpenCV + MediaPipe + ONNX Runtime
.venv-eye-training/   Eye-quality model training: PyTorch + ONNX tools
```

Run these commands from the project root in PowerShell.

### Live camera application

Create this environment once with `uv`:

```powershell
uv venv .venv --python 3.12
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

Activate it to run the live stream:

```powershell
.\.venv\Scripts\Activate.ps1
python src\camera_stream.py
```

The eye-quality classifier is loaded with ONNX Runtime, not `cv2.dnn`.
`cv2.dnn` cannot reliably execute graphs from PyTorch's current (dynamo-based)
ONNX exporter -- it runs without error but returns garbage logits.

To leave the environment when finished:

```powershell
deactivate
```

### Eye-quality training

Create this environment once with `uv`:

```powershell
uv venv .venv-eye-training --python 3.12
uv pip install --python .venv-eye-training\Scripts\python.exe -r src\eye_quality_training\requirements.txt
```

Activate it for dataset preparation or training:

```powershell
.\.venv-eye-training\Scripts\Activate.ps1
python src\eye_quality_training\prepare_dataset.py
python src\eye_quality_training\train_eye_quality.py --epochs 15
```

Then leave it before returning to the live application:

```powershell
deactivate
```

After training, `src\camera_stream.py` automatically loads
`artifacts\eye_quality\eye_quality.onnx` and its matching calibration JSON.
No PowerShell environment variables are required for that default location.

You can also run a command without activating either environment:

```powershell
& .\.venv-eye-training\Scripts\python.exe src\eye_quality_training\train_eye_quality.py --epochs 15
```
