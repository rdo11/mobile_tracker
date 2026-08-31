# MOBILE TRACKER — RELEASE & PROMOTION PACKAGE
Last updated: 2026-08-29 (v18 deployed as king)

> Purpose: everything needed to publish this project publicly (GitHub, Hugging Face,
> LinkedIn) in a way that builds real credibility under YOUR name. Follow the order —
> the sequence is designed so everything you post is true, verifiable and privacy-clean.

---

## 1. PROJECT STORY (the elevator pitch)

Privacy-first dashcam that reads traffic: detects and tracks every vehicle live,
identifies make+model+generation offline on the Mac, reads license plates (DK/DE,
electric/historic detection), blurs plates for GDPR, and auto-collects a growing
European vehicle dataset — built from scratch on a MacBook + iPhone.

Key differentiator vs existing dashcams: dashcams record, this one UNDERSTANDS —
and it runs fully offline after training (no API at inference).

---

## 2. THE NUMBERS (credibility = specific metrics)

### Model (current king, v19 — deployed 2026-08-30)
- Architecture: ConvNeXt-Tiny (28M params), trained from scratch
- Classes: 646 European car classes (make + model, incl. generations)
- Evaluation: **permanent leak-free holdout** (3,570 real dashcam crops excluded
  from ALL training builds) + honest intersection tests (crops no model ever trained on)
- King honest numbers (v19, intersection-tested): top-1 ~57% / top-5 ~78-80%
  (with TTA at inference: +~1pt)
- Inference: runs on Apple Silicon (MPS), fully offline, no GPU needed

### The pipeline
- YOLOv8 + ByteTrack: real-time vehicle detection/tracking (~25ms/frame on M5)
- ConvNeXt-Tiny: offline make/model/generation recognition, 646 classes
- GTSRB: speed-limit sign recognition (EU 3-lamp light classifier)
- EasyOCR: license plate reading + electric/historic plate detection
- GDPR-style plate blur on stream + recordings (ON by default)
- Test-time augmentation at inference (+flip/+scale averaging)

### Engineering journey (what we learned — real, verified)
- v15: fine-tuning the king on more classes → drifted, REJECTED
- v16: 30x oversampling → model memorized duplicates (97% val / 44% real) — REJECTED
- Fixed: 5x oversample + from-scratch + leak-free splits (the formula)
- v18→v20: more data verified; big-arch/336px tests showed tiny+224px is the sweet spot
- v21+: permanent frozen holdout so every comparison is trustworthy
- Honest verdict from v20 (intersection test): king 57.2% vs v20 56.5% — plateau at
  current data; the next lever is NEW car variety (regions/vehicle types), not more
  of the same footage

> NOTE: early public numbers (74.5%, 85.3%, 58.5%) were measured on contaminated
> holdouts (each model had seen part of the other's eval set). The permanent
> holdout methodology fixed this. Quote only intersection/frozen-holdout numbers.

---

## 3. PUBLICATION ORDER (IMPORTANT — do in this order)

### STEP 1 (NOW): GitHub code repo — safe, no data
- Publish: train_classifier.py, compare_ckpts.py, extract_crops.py, build_merged.py,
  classifier.py, main.py (dashboard), config.yaml (secrets stripped)
- Include: README with pipeline diagram, INSTALL.md, LICENSE (MIT)
- Do NOT include: .env, models/*.pt (or only as release assets), raw data
- Repo name suggestion: `euro-dashcam-vision` or `mobile-tracker`

### STEP 2 (AFTER FIRST LIVE DRIVE): Real footage, blurred
- Drive with LIVE_DRIVE_CHECKLIST.md (blur_plates: true in config)
- Verify plates are unreadable in the recording
- Clip the best 30-60s into a demo GIF/video (blurred)
- This becomes the hero asset of the README + LinkedIn post

### STEP 3: Hugging Face — model card + dataset
- Model card: v18 metrics, method, confusion analysis, how to run
- Dataset (ONLY after real blurred footage exists): HF dataset of your own crops
- IMPORTANT: do NOT upload YouTube-derived crops as a public dataset (rights)
- If you want a dataset NOW: publish the code-generated labels format instead

### STEP 4: LinkedIn posts (drafts below)

---

## 4. LINKEDIN POST DRAFTS

### Post 1 — The launch (after STEP 1-2)
Title: I built a dashcam that reads traffic — offline, privacy-first, on a MacBook.

I turned a MacBook + phone into an AI dashcam that:
• Detects & tracks every car in real time (YOLOv8 + ByteTrack, ~25ms/frame on Apple Silicon)
• Identifies make + model + generation OFFLINE (ConvNeXt-Tiny, 630 European classes)
• Reads DK/DE license plates incl. electric/historic detection
• Blurs every plate (GDPR-friendly) on stream and recordings
• Auto-collects a European vehicle dataset while driving — no API at inference

The result: ~57% top-1 / ~79% top-5 on a permanent leak-free real-dashcam holdout
(3,570 crops no model ever trained on), fully offline on a MacBook.
Built from scratch: scraping → labeling → training → deployment. All on a Mac.

Stack: Python, PyTorch (MPS), YOLOv8, ByteTrack, OpenCV, FastAPI, EasyOCR.
Why it matters: dashcams record, but don't understand. I taught mine to read traffic —
and I did it fully privacy-first.

#AI #ComputerVision #Dashcam #EdgeAI #MachineLearning #PyTorch #Privacy #Mobility #ComputerVisionEngineer

### Post 2 — The engineering lessons (technical credibility)
3 lessons from training a 630-class vehicle classifier on a MacBook:

1. More data beats bigger models. Going from 4.7k → 13k unique crops moved us from
   44% → 58.5% top-1. Increasing model size? ~0%. (We tested tiny/small/base/large.)
2. Watch your oversampling. 30x duplication → the model memorized and collapsed
   on real crops (97% val, 44% holdout). 5x + leak-free split fixed it.
3. Your validation set is only as good as its split. We found a track-level leak
   that inflated every number. Fixed, re-measured, everything changed.

#MachineLearning #MLOps #DataScience #ComputerVision #LLM #Training #Metrics

### Post 3 — The dataset pitch (after STEP 3)
Building the largest privacy-first European vehicle dataset — and sharing it.

Existing car datasets are US-only and 15 years old. European cars (Golf, Octavia,
Clio generations...) are underrepresented. So I built my own: real dashcam footage,
auto-blurred plates, 630 classes, DeepSeek-verified labels.

Now on Hugging Face: [link] — fully GDPR-clean (no readable plates).

#Dataset #OpenSource #HuggingFace #AI #VehicleRecognition #EuropeanTech

---

## 5. HASHTAGS (reusable)
#AI #ComputerVision #Dashcam #EdgeAI #MachineLearning #PyTorch #Privacy
#Mobility #Automotive #DataScience #OpenSource #HuggingFace #MLOps
#ComputerVisionEngineer #VehicleRecognition #EuropeanTech

---

## 6. CHECKLIST BEFORE EACH POST
- [ ] Plates blurred in ALL visuals (config blur_plates: true, verified)
- [ ] No .env / API keys in any repo or post
- [ ] YouTube-derived crops NOT in public dataset
- [ ] Metrics quoted are the leak-free holdout numbers
- [ ] Model weights: decide MIT vs GPL-agnostic license before publishing
- [ ] Repo README has a real demo GIF (from your own footage)
- [ ] LinkedIn profile updated: add "Built a 630-class vehicle recognition system" experience entry

---

## 7. FILES & WHERE THEY LIVE
- Code: ~/Projects/mobile_tracker/
- This doc: ~/Projects/mobile_tracker/RELEASE_PROMOTION.md
- Live-drive checklist: LIVE_DRIVE_CHECKLIST.md
- Metrics history: PROJECT_NOTES.md / MEMORY.md
- Deployed model: models/r50_dashcam.pt (v18), old king backup: models/r50_dashcam_v17_backup.pt
