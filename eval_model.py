"""
eval_model.py — evaluate a checkpoint: top-1 and top-5 accuracy on the val
split, plus per-class breakdown for the worst classes.

Usage:
    python eval_model.py --data dataset_filtered --ckpt models/vehicle_make_model_filt.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

from train_classifier import ImageFolderDS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    classes = ckpt["classes"]
    print(f"classes: {len(classes)}")

    from torchvision import models

    sd = ckpt["state_dict"]
    arch = "resnet50" if sd["fc.weight"].shape[1] == 2048 else "resnet18"
    print(f"detected arch: {arch}")
    factory = getattr(models, arch)
    m = factory(weights=None)
    m.fc = nn.Linear(m.fc.in_features, len(classes))
    m.load_state_dict(sd)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = m.to(device)
    m.eval()

    ds = ImageFolderDS(args.data, classes, "val", False)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=args.batch, shuffle=False, num_workers=4)

    top1 = top5 = total = 0
    per_class = {c: [0, 0] for c in classes}
    with torch.no_grad():
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            out = m(x)
            p = out.topk(5, 1).indices
            hit1 = p[:, 0] == y
            hit5 = (p == y.unsqueeze(1)).any(1)
            top1 += hit1.sum().item()
            top5 += hit5.sum().item()
            total += y.size(0)
            for i in range(y.size(0)):
                per_class[classes[y[i].item()]][0] += int(hit1[i])
                per_class[classes[y[i].item()]][1] += 1

    print(f"top-1: {top1 / total:.3f}  top-5: {top5 / total:.3f}  (n={total})")
    worst = sorted(per_class.items(), key=lambda kv: kv[1][0] / max(1, kv[1][1]))[:10]
    print("worst classes:")
    for c, (ok, n) in worst:
        if n >= 2:
            print(f"  {c}: {ok}/{n} = {ok / n:.0%}")


if __name__ == "__main__":
    main()