"""
filter_crops.py — clean the dataset with YOLO detection.

For every image:
  - run YOLOv8n detection
  - if no car/bus/truck/motorcycle detected -> DROP the image (noise removal)
  - if a car is detected -> CROP to the largest car box (with context margin)
    and write the crop as a new image into a mirrored dataset directory

Usage:
    python filter_crops.py --data dataset --out dataset_clean
                          [--conf 0.25] [--margin 0.25]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

# YOLO COCO class ids we keep as vehicles
KEEP = {2: "car", 5: "bus", 7: "truck", 3: "motorcycle", 1: "bicycle"}
CAR = 2  # only car crops go to the classifier


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--margin", type=float, default=0.25,
                    help="extra context around the box, fraction of box size")
    ap.add_argument("--no-crop", action="store_true",
                    help="keep original image instead of cropping to car box")
    args = ap.parse_args()

    model = YOLO("yolov8n.pt")
    args.out.mkdir(parents=True, exist_ok=True)

    total = kept = dropped = 0
    for cls_dir in sorted(args.data.iterdir()):
        if not cls_dir.is_dir():
            continue
        out_dir = args.out / cls_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        for p in sorted(cls_dir.glob("*")):
            if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            total += 1
            img = cv2.imread(str(p))
            if img is None:
                dropped += 1
                continue
            res = model(img, conf=args.conf, verbose=False)[0]
            box = None
            for det in res.boxes:
                if int(det.cls) == CAR:
                    box = det.xyxy[0].tolist()
                    break  # largest is not guaranteed; keep first car
            if box is None:
                dropped += 1
                continue
            x1, y1, x2, y2 = map(int, box)
            h, w = img.shape[:2]
            mx, my = int((x2 - x1) * args.margin), int((y2 - y1) * args.margin)
            x1, y1 = max(0, x1 - mx), max(0, y1 - my)
            x2, y2 = min(w, x2 + mx), min(h, y2 + my)
            if x2 - x1 < 16 or y2 - y1 < 16:
                dropped += 1
                continue
            cv2.imwrite(str(out_dir / p.name),
                        img if args.no_crop else img[y1:y2, x1:x2])
            kept += 1
            if total % 500 == 0:
                print(f"progress: {total} images, kept {kept}, dropped {dropped}",
                      flush=True)
    print(f"done: {total} total, {kept} kept, {dropped} dropped "
          f"({dropped / max(1, total):.1%} noise)", flush=True)


if __name__ == "__main__":
    main()