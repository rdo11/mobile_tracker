# Mobile Tracker — Privacy-First Dashcam Vehicle Recognition

A privacy-first dashcam system that **understands traffic**: detects and tracks every
vehicle live, identifies make + model + generation offline on Apple Silicon, reads
license plates (DK/DE, electric/historic detection), blurs plates for GDPR, and
auto-collects a European vehicle dataset while driving.

Built from scratch on a MacBook — no cloud API at inference.

---

## What it does

| Capability | How |
|---|---|
| Vehicle detection + tracking | YOLOv8 + ByteTrack, ~25ms/frame on Apple Silicon (MPS) |
| Make/model/generation (offline) | ConvNeXt-Tiny classifier, 600+ European classes |
| License plates | EasyOCR reading + DK/DE electric/historic detection |
| Privacy | GDPR-style plate blur on stream + recordings (on by default) |
| Signs / road context | GTSRB speed-limit + EU traffic-light module |
| Dataset growth | Auto-collects labeled crops from every drive |
| Live dashboard | FastAPI + WebSockets UI (`localhost:8500`) |

## Why it exists

Dashcams record, but don't understand. This project teaches one to read traffic —
with full privacy (plates auto-blurred) and fully offline inference.

## Model & results

- **Architecture:** ConvNeXt-Tiny (28M params), trained from scratch
- **Classes:** 646 European car classes (make + model, incl. generations)
- **Evaluation:** permanent leak-free holdout (3,570 real dashcam crops,
  excluded from ALL training builds) + honest intersection tests (crops no
  model ever trained on)
- **Deployed model (v19):** top-1 ~57% / top-5 ~79% on the honest intersection
  test (597 crops held out from every training build). Test-time augmentation
  adds ~+1pt.
- **Engineering journey (verified):** 30x oversampling causes memorization
  (97% internal val → 44% real). The winning formula: from-scratch training,
  5x oversample, leak-free splits, a permanent frozen evaluation holdout, and
  data diversity over architecture (tiny > base/large; 224px > 336px).

## Repository layout

```
main.py               # live dashboard + capture loop (FastAPI/uvicorn :8500)
tracker.py            # YOLOv8 + ByteTrack tracking
classifier.py         # offline ConvNeXt classifier + TTA
anpr_privacy.py       # plate reading + GDPR blur
road_context.py       # GTSRB signs + traffic lights
train_classifier.py   # training script (MPS or CUDA pod)
compare_ckpts.py      # holdout evaluation
extract_crops.py      # crop mining from dashcam videos
label_crops.py        # DeepSeek/Gemini labeling
build_merged.py       # dataset builder (canonicalization, leak-free split)
config.yaml           # runtime config (blur on by default)
frontend/index.html   # driving UI
```

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Live system (phone/camera → Mac → dashboard)
python main.py            # then open http://localhost:8500

# Train a classifier from your own footage
python extract_crops.py <video>.mp4
python label_crops.py --provider deepseek
python build_merged.py --oversample 5 --allow-new
python train_classifier.py --data storage/dataset/train_v6 --arch convnext_tiny
```

## Privacy & data

- **`blur_plates: true` by default** — plates are blurred on the stream and any
  recording before they touch disk.
- License plates are used for **in-memory re-ID only** (never streamed/stored).
- The public dataset (Hugging Face) contains only **self-recorded footage with
  plates blurred**, or crops derived from publicly available dashcam footage
  under a research license — see the dataset card for provenance.

## License

MIT — see [LICENSE](LICENSE). Models/dataset have their own licenses in their
respective cards.
