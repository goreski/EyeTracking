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