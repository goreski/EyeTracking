# Conference paper plan — ICETC 2026

Notes from the planning chat on 2026-10-10.

## 1. Starting point: what the project already has

- **Eye-state classifier:** a 4-class MobileNetV3-Small (open_visible / closed / occluded / sunglasses), exported to ONNX and calibrated with temperature scaling.
- **Binocular Bayesian fusion** (`src/eye_pair_fusion.py`): fills in an occluded eye's state from the other eye, using a measured correlation between the two eyes.
- **Gaze zones** (`src/spherical_gaze.py`): a von Mises–Fisher model of gaze over cockpit areas (road, mirrors, phone, etc.).
- **Conformal prediction sets** (`src/conformal_prediction.py`), including the adaptive online version (ACI).
- **Evaluation setup:** subject-level splits (`src/evaluation/splits.py`), provenance tracking, and a staged `evaluate.py`.

The fusion, gaze-zone and conformal parts are the contributions of the main ongoing research, so this paper must not present any of them as new.

### Current preliminary result (`reports/eye_20260925_130830.json`)

- Accuracy 89.7% (95% CI 81.0–94.7%), n = 78 test images
- ECE 0.067, NLL 0.216, Brier 0.120, temperature T = 0.853
- Conformal (alpha = 0.05): coverage 1.0, mean set size 1.63
- **Caveats:** image-level split (not subject-independent); the validation split was reused as the conformal calibration split. These numbers are not publishable as they are.

## 2. Chosen paper idea

Originally framed for driver monitoring, then reframed for ICETC (an education technology conference) toward **learner attention analytics**.

**Working title:** *How Reliable Are Webcam Eye-State Signals for Learner Attention Analytics? Subject Leakage and Miscalibration in Eye-State Classification*

(Original driver-monitoring title: *How Optimistic Are Eye-State Benchmarks? Subject Leakage and Miscalibration in Open/Closed Eye Classification for Driver Monitoring*)

---

## Handoff notes (cloud session, 2026-10-10)

- The pasted plan ends at section 2. Any later sections (experiments, timeline, deadline) still need to be added.
- Not yet on GitHub, so local only: `src/evaluation/` (including `splits.py`), `evaluate.py`, provenance tracking, and `reports/eye_20260925_130830.json`. Push these from the original machine so the subject-level re-evaluation can continue elsewhere.
- Also not uploaded: the session export `session-export-1791622832386.zip`.
