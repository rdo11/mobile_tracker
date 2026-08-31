#!/usr/bin/env python3
"""compare_ckpts.py — evaluate checkpoints on the REAL-crop holdout set.

Unlike eval_model.py (which re-splits its own val), this walks a flat
<root>/<class>/*.jpg directory and evaluates EVERY image against each
checkpoint's own label space. Classes absent from a checkpoint are skipped
for that checkpoint and reported, so v5 vs v6 comparisons stay honest.

Usage: .venv/bin/python compare_ckpts.py --data storage/dataset/dashcam_val \
           --ckpt models/r50_dashcam.pt models/r50_dashcam_v6.pt
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_model(ckpt_path: str) -> tuple[nn.Module, list[str]]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    classes = list(ckpt["classes"])
    sd = ckpt["state_dict"]
    from torchvision import models
    if "fc.weight" in sd:                                  # resnet family
        arch = "resnet50" if sd["fc.weight"].shape[1] == 2048 else "resnet18"
        m = getattr(models, arch)(weights=None)
        m.fc = nn.Linear(m.fc.in_features, len(classes))
    elif "classifier.2.weight" in sd:                      # convnext family
        # distinguish Tiny vs Small by depth of stage-5 (Tiny=9 blocks, Small=27)
        n5 = max(int(k.split(".")[2]) for k in sd
                 if k.startswith("features.5.") and k.endswith(".weight"))
        arch = "convnext_small" if n5 > 17 else "convnext_tiny"
        m = getattr(models, arch)(weights=None)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, len(classes))
    else:
        raise KeyError(f"unrecognized checkpoint layout: {list(sd)[:3]}")
    m.load_state_dict(sd)
    m.eval()
    return m, classes


def load_items(root: Path) -> list[tuple[Path, str]]:
    items = []
    for cls in sorted(os.listdir(root)):
        d = root / cls
        if d.is_dir():
            items += [(p, cls.replace("_", " ")) for p in sorted(d.glob("*.jpg"))]
    return items


def prep(p: Path) -> torch.Tensor:
    img = cv2.imread(str(p))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224)).astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    return torch.from_numpy(img).permute(2, 0, 1)


def evaluate(model: nn.Module, classes: list[str],
             items: list[tuple[Path, str]], batch: int = 64) -> dict[str, float]:
    idx = {c: i for i, c in enumerate(classes)}
    usable = [(p, idx[c]) for p, c in items if c in idx]
    top1 = top5 = 0
    per_class: dict[str, list[int]] = {}
    device = next(model.parameters()).device
    with torch.no_grad():
        for k in range(0, len(usable), batch):
            chunk = usable[k:k + batch]
            x = torch.stack([prep(p) for p, _ in chunk]).to(device)
            probs = model(x).softmax(1)
            top5_idx = probs.topk(5, 1).indices.cpu()
            for j, (_, y) in enumerate(chunk):
                hit1 = int(top5_idx[j, 0].item() == y)
                hit5 = int(y in top5_idx[j].tolist())
                top1 += hit1
                top5 += hit5
                name = classes[y]
                pc = per_class.setdefault(name, [0, 0])
                pc[0] += hit1
                pc[1] += 1
    n = max(1, len(usable))
    return {"top1": top1 / n, "top5": top5 / n, "n": n,
            "per_class": per_class, "skipped": len(items) - len(usable)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--ckpt", nargs="+", required=True)
    args = ap.parse_args()
    items = load_items(args.data)
    print(f"holdout: {len(items)} crops from {args.data}\n")
    results = {}
    for path in args.ckpt:
        m, classes = load_model(str(path))
        r = evaluate(m, classes, items)
        results[path] = r
        print(f"{path}")
        print(f"  {len(classes)} classes | skipped {r['skipped']} crops "
              f"(class absent)")
        print(f"  top-1 {r['top1']:.3f}   top-5 {r['top5']:.3f}   "
              f"(n={r['n']})\n")
    if len(results) == 2:
        (a, ra), (b, rb) = results.items()
        winner = a if ra["top1"] >= rb["top1"] else b
        print(f"WINNER on real crops: {winner} "
              f"({max(ra['top1'], rb['top1']):.3f} vs "
              f"{min(ra['top1'], rb['top1']):.3f})")


if __name__ == "__main__":
    main()
