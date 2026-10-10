# Planning chat: conference paper from the eye-tracking work (2026-10-10)

Exported from the local Claude session; tool calls and system text removed.


## You

i need to write an conference paper from eyetracking work i done so far, and not to overlap with onging research can you give me an idea


## Claude

I'll look at what you've built first so the idea comes from your actual work.


## Claude

My suggestion is a short evaluation paper that uses the work you've already done but stays away from the new methods in your pipeline. I'm assuming "ongoing research" means your own main paper or thesis. If you meant other groups' work, see the note at the end.

## What you have

- **Eye-state classifier:** a 4-class MobileNetV3 (open / closed / occluded / sunglasses), exported to ONNX and calibrated with temperature scaling.
- **Binocular Bayesian fusion:** fills in an occluded eye's state from the other eye, using a measured correlation between the two eyes.
- **Gaze zones:** a von Mises–Fisher model of gaze over cockpit areas (road, mirrors, phone, etc.).
- **Conformal prediction sets, including the adaptive online version (ACI).**
- **Evaluation setup:** subject-level splits (`src/evaluation/splits.py`), provenance tracking, and a staged `evaluate.py`. `splits.py` already plans for MRL, RT-BENE and your own Basler camera data.

The fusion, gaze-zone and conformal parts are your main contributions, so this paper shouldn't present any of them as new.

## Recommended paper: how much eye-state benchmarks overstate accuracy

**Working title:** *How Optimistic Are Eye-State Benchmarks? Subject Leakage and Miscalibration in Open/Closed Eye Classification for Driver Monitoring*

**Claim:** Many published open/closed eye results on public datasets report about 97–99% accuracy, but they use image-level random splits. Images of the same person end up in both training and test sets, which inflates the numbers. Calibration also breaks down when the model moves to a different dataset or camera. Your own report already flags this problem: `reports/eye_20260925_130830.json` warns that its split is image-level.

**Experiments, mostly using code you've already written:**
1. Train your MobileNetV3 on MRL Eye, CEW and RT-BENE under three protocols: image-level split, subject-level split, and leave-one-dataset-out.
2. Report the accuracy drop between protocols, with confidence intervals (your metrics code already computes them).
3. Measure calibration (ECE, NLL, Brier) before and after temperature scaling, within one dataset and across datasets. The likely finding is that temperature fitted on one dataset doesn't carry over to another.
4. Optionally, test on your own Basler camera recordings as a real cockpit-like domain.
5. Release the split manifests and provenance-stamped reports as a reproducible benchmark protocol.

**Why it fits:**
- **It doesn't overlap with your main paper.** It doesn't touch fusion, gaze zones or conformal prediction, and it actually justifies the evaluation protocol your main paper will use. You could cite it there.
- **It's cheap.** It runs on public data and your existing setup, and a laptop GPU is enough.
- **Venues:** ETRA short paper, IEEE IV / ITSC, or AutomotiveUI workshops.

**Fallback if fusion isn't central to your main work:** a paper on reading an occluded eye from the visible one. RT-BENE has blink labels for both eyes in the same frame, so you could measure how often the two eyes agree at scale. Then simulate occluding one eye and show how much your Bayesian fusion recovers.

## Things to know before committing

- **Your current numbers aren't publishable as they are:** 78 test images, an image-level split, and the calibration split reused for the conformal quantile. Any paper needs at least one public dataset with subject-level splits.
- **I haven't checked the literature for this specific eye-state leakage study.** Subject leakage is a known problem in ML generally, so before writing you should confirm nobody has done this exact study for eye-state classification. I can run that search next.

**One question so I don't aim at the wrong thing:** what is your main ongoing work — the full probabilistic pipeline (fusion + gaze zones + conformal + temporal), or something else? And is there a target venue or deadline? Once I know, I can draft an outline or set up the dataset and protocol experiments in `evaluate.py`.


## You

i like recommende paper, i am planning to attend https://www.icetc.org/index.html


## Claude

Two problems already: the deadline is today and the topic is education. Let me check the tracks and submission rules before advising.


## Claude

I checked the ICETC 2026 site. There are two problems you should know about before we go further.

**1. The submission deadline is today, October 10, 2026.** It was already extended once, on Aug 31. A full IEEE paper needs at least 5 two-column pages, and the experiments for the recommended paper haven't been run yet: no MRL, CEW or RT-BENE training, and no subject-level splits. A full paper by tonight isn't realistic.

**2. ICETC is an education technology conference.** All 10 tracks are about learning: online learning, AI in education, learning analytics, assistive tech and so on. A driver-monitoring paper would probably be rejected as out of scope.

## How to make it work

**Change the setting from drivers to learners, and keep the core claim.** Webcam eye-state and attention estimation is widely used for student engagement analytics and online proctoring. The paper's message carries over directly: reported accuracy is inflated by subject leakage, and the models are overconfident, which is an ethical problem when you're watching students. Two tracks fit:
- **Track 4, Learning Analytics:** "student behavior data", "real-time adaptive learning analytics".
- **Track 7, Digital Ethics:** "ethical considerations in AI surveillance in schools". This fits the calibration angle well: a model that says "I can't tell" is better than one that confidently flags a student as inattentive.

Possible title: *How Reliable Are Webcam Eye-State Signals for Learner Attention Analytics? Subject Leakage and Miscalibration in Eye-State Classification*

**For the deadline, I'd submit an abstract today rather than a full paper.** ICETC accepts abstracts for presentation without publication. That gets you to Porto, and it doesn't publish the study. You could still publish the full version later at ETRA, IEEE IV or a journal. That matters: a full ICETC paper would count as prior publication of the same work. Your current results (89.7% accuracy on an image-level split, ECE 0.067) can go in as preliminary findings, with the subject-level and cross-dataset experiments described as work in progress.

If you'd rather have a published full paper at ICETC, email the organisers today (icetc.edu@outlook.com) and ask whether late full-paper submissions are allowed. I can't send it for you, but I can draft it.

## Next step

I can write the abstract now in about 250–300 words, framed for Track 4 or Track 7, using your real numbers from `reports/eye_20260925_130830.json`. Tell me:
- Track 4 or Track 7?
- Abstract only, or should I also draft the extension email?
- Who are the co-authors and what's the affiliation? Reviews are double-blind, so I'll leave names off the submission itself either way.


## You

i got mail saying Submission Deadline: November 10, 2026


## Claude

With a deadline of November 10 you have about four weeks, which is enough for the full paper. Before you plan around it, check that the email came from **icetc.edu@outlook.com**. The site says that's the conference's only official address and warns about fake ones. The website also still shows October 10 as the deadline, so if you're unsure, reply to that address and ask them to confirm.

One thing to decide up front: a full ICETC paper gets published, so you can't later submit the same study to ETRA or IEEE IV. ICETC does invite selected papers to submit an extended version to *IEEE Transactions on Learning Technologies*. If you take the learner-attention framing, that's a natural place for the extended version.

## Four-week plan (5–6 page IEEE paper)

| Week | Work |
|---|---|
| **1 (Oct 10–17)** | Download MRL Eye, CEW and RT-BENE. Build subject-level manifests with your `splits.py`. Optionally record 10–15 people at a laptop webcam in a study setting; that's your in-domain learner data. Ask now whether your institution needs ethics consent for this. |
| **2 (Oct 18–24)** | Train MobileNetV3 under three protocols: image-level split, subject-level split, and leave-one-dataset-out. Add `evaluate.py` commands so every number goes into a provenance-stamped report. |
| **3 (Oct 25–31)** | Calibration analysis: ECE, NLL and reliability diagrams, before and after temperature scaling, within one dataset and across datasets. Add a small "abstain when uncertain" result to support the ethics argument. Generate figures and tables from the reports. |
| **4 (Nov 1–10)** | Write the paper in the IEEE template. Leave 2–3 days for revision and anonymisation, since review is double-blind. |

**Paper outline:** Introduction (webcam attention analytics in education and the problem of overconfident surveillance) → Related work (eye-state datasets, engagement detection, calibration) → Method (classifier, the three protocols, calibration) → Results (how much accuracy drops between protocols, calibration across datasets) → Discussion (implications for learning-analytics and proctoring tools) → Conclusion.

**Track:** I'd choose **Track 4, Learning Analytics**. The ethics angle can go in the discussion section.

## What I can start now

The quickest win is week 1's setup. I can write manifest builders for MRL, CEW and RT-BENE that plug into `src/evaluation/splits.py`, plus training and evaluation commands in `evaluate.py` for the three protocols. The download itself is yours to do: MRL is a direct download, but RT-BENE requires you to request access.

Should I start on that code, or would you rather start with the paper skeleton (title, abstract draft, section structure)?


## You

can you save this chat to some file i can transfer to another computer


## Claude

I'll save it in two forms: a readable Markdown file with the full discussion and plan, and the app's raw transcript export.
