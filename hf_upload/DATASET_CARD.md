---
license: cc-by-nc-4.0
language:
  - en
tags:
  - cars
  - vehicle-make-model
  - dashcam
  - european-cars
  - image-classification
pretty_name: European Dashcam Vehicle Crops (provenance-documented)
---

# European Dashcam Vehicle Crops

Vehicle crops extracted from publicly available dashcam driving footage of
European cities, labeled with make + model (+ generation).

## Provenance & licensing (please read)
- Crops are **derived from publicly available YouTube dashcam footage**
  (city drives: Berlin, Paris, Rome, Vienna, Prague, Barcelona, Munich, London,
  Istanbul, Stockholm, Lyon, Budapest, Milan, Bern, Hamburg, and more).
- Provided for **non-commercial research use** (CC-BY-NC-4.0). The original
  videos remain the property of their respective uploaders.
- If you use this dataset, please credit the source channels (see the
  `sources/` file) and link back to this card.
- **No readable license plates** are included: crops are small vehicle boxes
  (100-500 px) and the full system blurs plates by default.

## Content
- **~21,300 crops** across **~820 classes** (make + model, generations merged at
  training time into model-level classes)
- Labels produced by DeepSeek vision + second opinions from Gemini/Grok
  (recovered hard cases)
- Structure: one folder per class, JPG crops of detected vehicles

## How it was built
1. yt-dlp → 1080p dashcam videos of European cities
2. YOLOv8 detection + ByteTrack tracking → one sharp crop per car per
   distance bucket (far/mid/near)
3. DeepSeek batch labeling (50/request) → dashcam_raw/<Label>/
4. Gemini/Grok second-opinion recovery of low-confidence rejects
5. build_merged.py: accent-fold, dedupe, 5x oversample, leak-free split

## Evaluation note
The companion model (rdo1/euro-dashcam-vehicle-classifier) is evaluated on a
permanent frozen holdout excluded from ALL training builds + intersection
tests — see the model card for honest numbers.
