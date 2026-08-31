---
license: mit
language:
  - en
tags:
  - cars
  - vehicle-make-model
  - dashcam
  - image-classification
  - computer-vision
  - pytorch
  - convnext
pipeline_tag: image-classification
pretty_name: European Dashcam Vehicle Classifier (v19)
---

# European Dashcam Vehicle Classifier (v19)

Privacy-first dashcam vehicle recognition: detects and identifies make + model +
generation of European cars, fully offline on Apple Silicon.

## Model
- **Architecture:** ConvNeXt-Tiny (28M params), trained from scratch
- **Classes:** 646 European vehicle classes (make + model, incl. generations)
- **Input:** 224×224 RGB image of a vehicle crop
- **Output:** class logits → softmax probabilities

## Honest evaluation
Measured on an **intersection holdout** (597 real dashcam crops that no model ever
trained on) — this methodology was deliberately built after discovering that naive
holdout comparisons were contaminated:

| Model | Top-1 | Top-5 |
|---|---|---|
| **v19 (this model)** | **57.2%** | **79.7%** |
| v20 | 56.5% | 79.4% |
| v21 | 54.7% | 77.9% |
| v22 | 53.7% | 79.9% |

Test-time augmentation (horizontal flip + scale averaging) adds ~+1pt.

## Usage (PyTorch)
```python
import torch
from torchvision import models

ckpt = torch.load("model.pt", map_location="cpu")
classes = ckpt["classes"]

model = models.convnext_tiny(weights=None)
model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, len(classes))
model.load_state_dict(ckpt["state_dict"])
model.eval()
# preprocess: resize to 224x224, normalize with ImageNet mean/std, predict
```

## Training data
~21,300 real dashcam crops (extracted from public dashcam footage of European
cities) + ~10,800 web images, 5x oversampled, from-scratch training. See the
dataset card for provenance and licensing.

## Privacy
The full system blurs license plates on stream + recordings by default (GDPR).
This model operates on vehicle crops only.
