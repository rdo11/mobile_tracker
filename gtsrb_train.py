#!/usr/bin/env python3
"""gtsrb_train.py — train the traffic-sign reader on German Traffic Sign
Recognition Benchmark (43 classes, 39k cropped signs).

Output checkpoint format mirrors our other models:
    {"state_dict": ..., "classes": [human-readable names]}

Speed-limit integration: road_context can call this model on candidate ring
interiors instead of EasyOCR digits. Class names carry the km/h value so the
caller maps 'Speed limit (50)' -> 50 directly.

Usage: python gtsrb_train.py --data GTSRB/Final_Training/Images \
           --out models/gtsrb_signs.pt --epochs 12
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

GTSRB_NAMES = [
    "Speed limit (20)", "Speed limit (30)", "Speed limit (50)",
    "Speed limit (60)", "Speed limit (70)", "Speed limit (80)",
    "End of speed limit (80)", "Speed limit (100)", "Speed limit (120)",
    "No overtaking", "No overtaking (trucks)", "Right-of-way",
    "Priority road", "Yield", "Stop", "No vehicles",
    "No trucks", "No entry", "Caution", "Curve left", "Curve right",
    "Double curve", "Bumpy road", "Slippery", "Narrow right", "Roadwork",
    "Traffic signals", "Pedestrians", "Children", "Bicycles", "Ice/snow",
    "Animals", "End of all limits", "Turn right ahead", "Turn left ahead",
    "Ahead only", "Ahead or right", "Ahead or left", "Keep right",
    "Keep left", "Roundabout", "End no overtaking", "End no overtaking (trucks)",
]

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class GtsrbDS(Dataset):
    def __init__(self, items: list[tuple[Path, int]], train: bool, size: int = 48):
        self.items, self.train, self.size = items, train, size

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        if self.train:
            if random.random() < 0.5:
                img = cv2.flip(img, 1)
            k = random.choice([0, 1])
            if k:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        img = cv2.resize(img, (self.size, self.size)).astype(np.float32) / 255.0
        img = (img - MEAN) / STD
        return torch.from_numpy(img).permute(2, 0, 1), y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("models/gtsrb_signs.pt"))
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    print("device:", device)

    per_class: dict[int, list[Path]] = {}
    root = args.data
    for d in sorted(root.iterdir()):
        if d.is_dir() and d.name.isdigit():
            per_class[int(d.name)] = [p for p in d.rglob("*.ppm")]
    assert len(per_class) == 43, f"expected 43 classes, got {len(per_class)}"

    tr_items, va_items = [], []
    rng = random.Random(42)
    for cid, files in sorted(per_class.items()):
        files = sorted(files)
        rng.shuffle(files)
        n_val = max(1, int(len(files) * 0.1))
        va_items += [(p, cid) for p in files[:n_val]]
        tr_items += [(p, cid) for p in files[n_val:]]
    print(f"train {len(tr_items)}  val {len(va_items)}")

    dl_tr = DataLoader(GtsrbDS(tr_items, True), batch_size=args.batch,
                       shuffle=True, num_workers=4, persistent_workers=True)
    dl_va = DataLoader(GtsrbDS(va_items, False), batch_size=args.batch,
                       num_workers=4)

    from torchvision import models
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    m.fc = nn.Linear(m.fc.in_features, 43)
    m = m.to(device)

    crit = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best = 0.0
    classes = GTSRB_NAMES
    for ep in range(args.epochs):
        m.train()
        tot = cor = ls = 0.0
        for x, y in dl_tr:
            x, y = x.to(device, non_blocking=True), y.to(device)
            opt.zero_grad()
            out = m(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            ls += loss.item() * x.size(0)
            cor += (out.argmax(1) == y).sum().item()
            tot += x.size(0)
        sched.step()
        m.eval()
        vc = vt = 0
        with torch.no_grad():
            for x, y in dl_va:
                x, y = x.to(device), y.to(device)
                vc += (m(x).argmax(1) == y).sum().item()
                vt += x.size(0)
        acc = vc / max(1, vt)
        print(f"epoch {ep+1}/{args.epochs}: train {cor/max(1,tot):.3f} "
              f"val {acc:.3f} loss {ls/max(1,tot):.4f}", flush=True)
        if acc > best:
            best = acc
            m = m.cpu()
            args.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": m.state_dict(), "classes": classes}, args.out)
            m = m.to(device)
    print(f"done. best val {best:.3f} -> {args.out}")


if __name__ == "__main__":
    main()
