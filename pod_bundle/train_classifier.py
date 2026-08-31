"""
train_classifier.py — fine-tune a small vehicle make/model classifier on MPS.

Trains a torchvision ResNet18/ResNet50 (or ConvNeXt-Tiny) on folders of
labeled vehicle images (one folder per class, e.g. from the Wikimedia
scraper and the Grok-labeled dashcam crops) and saves a checkpoint
compatible with classifier.py ({"state_dict": ..., "classes": [...]}).

Usage:
    python train_classifier.py --data storage/dataset --epochs 8
                              [--arch resnet18|resnet50|convnext_tiny]
                              [--out models/vehicle_make_model.pt]
                              [--min-per-class 30]
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

_ROMAN = {"I", "II", "III", "IV", "V", "VI", "VII", "VIII"}
_KEEP_SUFFIX = {"SE", "EV"}  # e.g. Mini Cooper SE, Porsche Macan EV
# Trucks/bikes get coarse YOLO labels live — never trained as classes.
COARSE_ONLY_MAKES = {"MAN", "Scania", "DAF", "Iveco", "Renault Trucks",
                     "Mercedes-Benz Trucks", "Volvo Trucks",
                     "KTM", "Ducati", "Triumph", "Aprilia", "BMW Motorrad"}


def class_key(folder_name: str) -> str:
    """Map a generation-specific folder to its model-level class.

    'Volkswagen_Golf_VII'  -> 'Volkswagen Golf'  (Roman numerals)
    'BMW_3_Series_(F30)'   -> 'BMW 3 Series'     (chassis codes)
    'Volkswagen_Passat_B8' -> 'Volkswagen Passat'
    'Opel_Corsa_E'         -> 'Opel Corsa'
    'Smart_Fortwo_451'     -> 'Smart Fortwo'
    """
    parts = folder_name.replace("_", " ").split()
    if not parts:
        return folder_name
    # trailing '(E90)' / '(W204)' / '(2014)' / '(8P)'
    if parts[-1].startswith("(") and parts[-1].endswith(")") \
            and parts[-1][1:-1].replace(".", "").isalnum():
        parts = parts[:-1]
    # trailing 'Mk1'..'Mk4'
    if parts and len(parts[-1]) > 2 and parts[-1][:2] == "Mk" and parts[-1][2:].isdigit():
        parts = parts[:-1]
    # trailing Roman numeral
    if parts and parts[-1] in _ROMAN:
        parts = parts[:-1]
    # trailing single letter generation (Corsa D, Astra K) — keep SE/EV;
    # only strip when a real model token remains (never "BMW X1", "Audi A4",
    # and never Tesla "Model S/X/Y" where the letter IS the model name)
    if parts and len(parts[-1]) == 1 and parts[-1].isalpha() \
            and parts[-1] not in _KEEP_SUFFIX and len(parts) >= 3 \
            and not (parts[0] == "Tesla" and len(parts) == 3 and parts[1] == "Model"):
        parts = parts[:-1]
    # trailing letter+digits generation (Passat B8, Transporter T6.1, Cooper R56)
    import re

    if parts and re.fullmatch(r"[A-Z]\d+(\.\d+)?", parts[-1]) and len(parts) >= 3:
        parts = parts[:-1]
    # trailing plain digits >= 3 chars (Fortwo 451, Forfour 453)
    if parts and len(parts[-1]) >= 3 and parts[-1].isdigit() and len(parts) >= 3:
        parts = parts[:-1]
    return " ".join(parts)


def collect_classes(data_dir: Path, min_per_class: int) -> tuple[list[str], dict[str, Path], int]:
    """Folders grouped by model-level class; returns (classes, folder map).

    Every generation folder (Golf_I .. Golf_VIII) contributes to one
    'Volkswagen Golf' class — more data per class, no similar-class confusion.
    The third return value counts classes dropped by `min_per_class` (callers
    should surface it: silently losing classes cost us a rerun once).
    """
    groups: dict[str, list[Path]] = {}
    for folder in sorted(data_dir.iterdir()):
        if not folder.is_dir():
            continue
        make = folder.name.split("_", 1)[0]
        if make in COARSE_ONLY_MAKES:
            continue  # trucks/bikes: coarse labels only, never trained
        groups.setdefault(class_key(folder.name), []).append(folder)
    classes: list[str] = []
    folder_map: dict[str, Path] = {}
    dropped = 0
    for key, folders in sorted(groups.items()):
        n = sum(len(list(f.glob("*.jpg"))) + len(list(f.glob("*.jpeg")))
                + len(list(f.glob("*.png"))) for f in folders)
        if n >= min_per_class:
            classes.append(key)
            folder_map[key] = folders[0]
        else:
            dropped += 1
    return classes, folder_map, dropped


class ImageFolderDS(Dataset):
    def __init__(self, data_dir: Path, classes: list[str], split: str,
                 train: bool, size: int = 224):
        self.items: list[tuple[str, int]] = []
        rng = random.Random(42)
        for idx, cls in enumerate(classes):
            folders = [f for f in data_dir.iterdir()
                       if f.is_dir() and class_key(f.name) == cls]
            files: list[Path] = []
            for f in folders:
                files += [p for p in f.glob("*") if p.suffix.lower() in
                          (".jpg", ".jpeg", ".png")]
            files = sorted(files)
            rng.shuffle(files)
            n_val = max(2, int(len(files) * 0.15))
            picked = files[: -n_val] if train else files[-n_val:]
            self.items += [(str(p), idx) for p in picked]
        self.train = train
        self.size = size

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        path, label = self.items[i]
        img = cv2.imread(path)
        if img is None:
            return self[(i + 1) % len(self)]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        if self.train:
            scale = random.uniform(0.85, 1.15)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
            if random.random() < 0.5:
                img = cv2.flip(img, 1)
        img = cv2.resize(img, (self.size, self.size))
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        return torch.from_numpy(img).permute(2, 0, 1), label


def build_model(arch: str, n_classes: int) -> nn.Module:
    from torchvision import models
    if arch == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    elif arch == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    elif arch == "convnext_tiny":
        m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
    elif arch == "convnext_small":
        m = models.convnext_small(weights=models.ConvNeXt_Small_Weights.DEFAULT)
    elif arch == "convnext_base":
        m = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT)
    else:
        raise ValueError(arch)
    # replace the head for our class count (resnet: fc; convnext: classifier[-1])
    if hasattr(m, "fc"):
        m.fc = nn.Linear(m.fc.in_features, n_classes)
    elif hasattr(m, "classifier") and isinstance(m.classifier, nn.Sequential):
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, n_classes)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="storage/dataset", type=Path)
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-per-class", type=int, default=1,
                    help="min images per class to keep (default 1 keeps ALL "
                         "classes; a higher value SILENTLY drops classes)")
    ap.add_argument("--init", default=None,
                    help="checkpoint to start from (e.g. models/r50_filt_60.pt)")
    ap.add_argument("--freeze", action="store_true",
                    help="freeze backbone, train head only (domain adaptation)")
    ap.add_argument("--size", type=int, default=224,
                    help="training resolution (224 standard; 336 = higher-res experiment)")
    ap.add_argument("--out", default="models/vehicle_make_model.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() \
        else ("mps" if (getattr(torch.backends, "mps", None)
                        and torch.backends.mps.is_available()) else "cpu")
    print(f"device: {device}")
    if device == "mps":
        print("WARNING: MPS (Apple GPU) training has silently corrupted gradients")
        print("         on this machine. Prefer CUDA (RunPod). If you proceed,")
        print("         validate the result on real crops, not just val_acc.")

    classes, _folder_map, dropped = collect_classes(args.data, args.min_per_class)
    if dropped:
        print(f"NOTE: {dropped} classes dropped by --min-per-class "
              f"{args.min_per_class} (use 1 to keep all classes)")
    if len(classes) < 2:
        print(f"Not enough classes with >= {args.min_per_class} images in "
              f"{args.data}. Found: {classes}")
        sys.exit(1)
    print(f"classes ({len(classes)}): {classes}")

    ds_tr = ImageFolderDS(args.data, classes, "train", True, size=args.size)
    ds_va = ImageFolderDS(args.data, classes, "val", False, size=args.size)
    print(f"train: {len(ds_tr)}  val: {len(ds_va)}")

    # CUDA pods get parallel decode workers; on macOS/MPS forked DataLoader
    # workers randomly segfault (cv2 + fork) and kill the whole run.
    workers = 4 if device == "cuda" else 0
    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, num_workers=workers)
    dl_va = DataLoader(ds_va, batch_size=args.batch, shuffle=False, num_workers=workers)

    model = build_model(args.arch, len(classes)).to(device)
    if args.init:
        ck = torch.load(args.init, map_location="cpu")
        sd = ck["state_dict"] if "state_dict" in ck else ck
        # Drop classifier-head keys when the class count changed (e.g. 247 ->
        # 252): strict=False tolerates missing keys but NOT shape mismatches,
        # so a stale fc would crash the load. Fresh head, pretrained backbone.
        model_shapes = {k: v.shape for k, v in model.state_dict().items()}
        head_keys = [k for k in sd
                     if k.startswith(("fc.", "classifier.2.", "head."))
                     and k in model_shapes and sd[k].shape != model_shapes[k]]
        if head_keys:
            sd = {k: v for k, v in sd.items() if k not in head_keys}
            print(f"init: dropped {len(head_keys)} head keys (class count "
                  f"changed) — fresh head, backbone preserved")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"init from {args.init}: missing {len(missing)} "
              f"({missing[:4]}...) unexpected {len(unexpected)}")
    if args.freeze and args.init:
        for name, p in model.named_parameters():
            if not name.startswith("fc."):
                p.requires_grad = False
        print("backbone frozen, training head only")
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best = 0.0
    for ep in range(args.epochs):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in dl_tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)
        sched.step()
        acc_tr = correct / total

        model.eval()
        v_total, v_correct = 0, 0
        with torch.no_grad():
            for x, y in dl_va:
                x, y = x.to(device), y.to(device)
                out = model(x)
                v_correct += (out.argmax(1) == y).sum().item()
                v_total += x.size(0)
        acc_va = v_correct / max(1, v_total)
        print(f"epoch {ep + 1}/{args.epochs}: train_acc {acc_tr:.3f} "
              f"val_acc {acc_va:.3f} loss {loss_sum / max(1, total):.4f}")
        if acc_va > best:
            best = acc_va
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.cpu().state_dict(),
                        "classes": classes}, args.out)
            model.to(device)

    print(f"done. best val acc {best:.3f} -> {args.out}")


if __name__ == "__main__":
    main()