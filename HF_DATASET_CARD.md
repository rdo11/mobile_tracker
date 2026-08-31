---
license: mit
language:
  - en
  - cs
  - sk
tags:
  - cars
  - vehicle-make-model
  - dashcam
  - european-cars
  - image-classification
  - wikimedia
pretty_name: European Vehicle Make/Model (Wikimedia)
---

# European Vehicle Make/Model Dataset (Wikimedia Commons)

~10,800 web images (Wikimedia) + ~3,800 verified real dashcam crops
(YouTube EU drives + recorded drives, two-provider label verification) across
~480 make/model classes. Web set cleaned with YOLOv8 (non-car noise removed);
real crops are accent-folded, deduplicated and cross-checked between DeepSeek
and Gemini vision models before training.

Built for a dashcam vehicle-identification project (see
`PROJECT_NOTES.md`): detecting make + model of cars seen on Danish/German
roads, trained as a fallback to vision-API classification.

## Content

- **247 classes** — the most common cars on DK/DE roads (VW Golf, Skoda
  Octavia, Tesla Model 3, Toyota Yaris, ...)
- **~9,520 images** (572MB), one folder per class
- All Wikimedia *generation* categories merged into one class per model:
  `Volkswagen_Golf_VII` + `Volkswagen_Golf_VIII` → `Volkswagen Golf`
- YOLOv8n car-detection filter: images without a clearly detected car were
  dropped (~10% of the raw scrape)

## Cleaning pipeline

1. `wikimedia_scraper.py` — recursive category scrape from Commons
2. `class_key()` — generation merging (Roman numerals, chassis codes, Mk,
   single-letter generations; Tesla "Model S/X/Y" preserved as models)
3. YOLOv8n filter (`filter_crops.py --no-crop`): drop no-car images,
   keep originals

## Validation

ResNet18 fine-tuned on this set (train/val split 85/15, stratified):
**top-1 48.2%, top-5 ~75-85%** on the validation split. Expect lower
accuracy on rare/similar models; the raw Commons data contains street-scene
and multi-car photos that limit precision.

## License / provenance

All images are from Wikimedia Commons under their individual licenses
(many CC-BY-SA). This dataset is provided as a convenience aggregation —
verify per-image licenses before redistribution. Filenames are URL-encoded
Commons file names.

## Usage

```python
import torch
from torchvision import models

ckpt = torch.load("checkpoint.pt", map_location="cpu")
classes = ckpt["classes"]
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(ckpt["state_dict"])
model.eval()
```