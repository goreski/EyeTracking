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

**Claim:** Many published open/closed eye results on public datasets report about 97–99% accuracy, but they use image-level random splits. Images of the same person end up in both training and test sets, which inflates the numbers. Calibration also breaks down when the model moves to a different dataset or camera. Overconfident predictions are an ethical problem when such models are used to monitor students (engagement analytics, online proctoring).

**Experiments:**
1. Train MobileNetV3 on MRL Eye, CEW and RT-BENE under three protocols: image-level split, subject-level split, and leave-one-dataset-out.
2. Report the accuracy drop between protocols, with confidence intervals.
3. Measure calibration (ECE, NLL, Brier, reliability diagrams) before and after temperature scaling, within one dataset and across datasets.
4. Optional: in-domain test on 10–15 people recorded at a laptop webcam in a study setting (check whether ethics consent is needed).
5. A small "abstain when uncertain" result to support the ethics argument.
6. Release the split manifests and provenance-stamped reports as a reproducible protocol.

**Why it doesn't overlap with the main research:** it doesn't touch fusion, gaze zones or conformal prediction, and it justifies the evaluation protocol the main paper will use.

**Fallback idea (if fusion is not central to the main work):** reading an occluded eye from the visible one. Use RT-BENE's blink labels for both eyes in the same frame to measure how often the eyes agree at scale, then simulate occluding one eye and measure what Bayesian fusion recovers.

**To do before writing:** search the literature to confirm nobody has already done this exact leakage/calibration study for eye-state classification.

## 3. Venue: ICETC 2026

- 18th International Conference on Education Technology and Computers, Porto, Portugal, December 14–17, 2026. IEEE proceedings (Xplore, Ei, Scopus).
- Website: https://www.icetc.org/index.html
- Submission: https://easychair.org/my/conference?conf=icetc2026
- Templates: https://www.icetc.org/submission.html
- **Deadline:** the website shows October 10, 2026; an email received says **November 10, 2026**. Make sure that email came from the only official address, **icetc.edu@outlook.com**.
- **Format (IEEE):** at least 5 two-column pages; more than 6 pages costs USD 70 per extra page. Review is double-blind.
- Abstract-only submissions (presentation without publication) are also accepted.
- **Recommended track:** Track 4, Learning Analytics and Data-Driven Education. Alternative: Track 7, Digital Ethics ("AI surveillance in schools").
- **Note:** a full ICETC paper counts as publication, so the same study can't go to ETRA or IEEE IV later. Selected papers may be invited to submit an extended version to *IEEE Transactions on Learning Technologies*.

## 4. Four-week plan (5–6 page IEEE paper)

| Week | Work |
|---|---|
| 1 (Oct 10–17) | Download MRL Eye, CEW and RT-BENE (RT-BENE requires an access request). Build subject-level manifests with `splits.py`. Optional: record learner webcam data and check ethics requirements. |
| 2 (Oct 18–24) | Train under the three protocols. Add `evaluate.py` commands so every number goes into a provenance-stamped report. |
| 3 (Oct 25–31) | Calibration analysis before and after temperature scaling, within one dataset and across datasets. Abstention result. Generate figures and tables from the reports. |
| 4 (Nov 1–10) | Write the paper in the IEEE template. Leave 2–3 days for revision and anonymisation. |

**Outline:** Introduction (webcam attention analytics in education and the problem of overconfident surveillance) → Related work (eye-state datasets, engagement detection, calibration) → Method (classifier, the three protocols, calibration) → Results (accuracy drop between protocols, calibration across datasets) → Discussion (implications for learning-analytics and proctoring tools) → Conclusion.

## 5. Next steps

- [ ] Confirm the November 10 deadline came from icetc.edu@outlook.com.
- [ ] Download the datasets.
- [ ] Write manifest builders for MRL, CEW and RT-BENE that plug into `src/evaluation/splits.py`.
- [ ] Add training and evaluation commands for the three protocols to `evaluate.py`.
- [ ] Literature check for prior leakage/calibration studies on eye-state classification.
- [ ] Paper skeleton: title, abstract draft, section structure.

## 6. Handoff notes (cloud session, 2026-10-10)

- The full planning chat is in `docs/paper_planning_chat_2026-10-10.md`.
- These still exist only on the original computer, not on GitHub: `src/evaluation/` (including `splits.py`), `evaluate.py`, the provenance tracking code and `reports/eye_20260925_130830.json`. Push them from that machine before starting the week 1 code.
