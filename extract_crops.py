#!/usr/bin/env python3
"""Offline crop extraction: one sharp crop per unique car per distance bucket,
NO API calls. Saves to storage/dataset/dashcam_youtube/<bucket>/.

Distance buckets (by bbox long side):
  far :  90-150 px  (distant cars — teaching the model to guess from afar)
  mid : 150-300 px  (medium distance)
  near: 300px+      (close, high detail)

Usage: .venv/bin/python extract_crops.py <video.mp4> [video.mp4 ...]
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
OUT_ROOT = ROOT / "storage" / "dataset" / "dashcam_youtube"

BUCKETS = {"far": (70, 150), "mid": (150, 300), "near": (300, 10**9)}
MIN_AREA = 2500
CONF = 0.35
IMGSZ = 320
MIN_CROP_TAIL = 6  # frames after last sighting before finalizing a track
SAMPLE_HZ = 2.0    # target detection rate (~2 samples/sec keeps tracks alive)
# Multi-crop harvesting: keep the sharpest crop per (bucket, time-window) so a
# car visible for minutes yields several angles/distances instead of ONE crop.
WINDOW_SECS = 30.0
MAX_PER_TRACK = 15


def bucket_for(long_side: int) -> str | None:
    for name, (lo, hi) in BUCKETS.items():
        if lo <= long_side < hi:
            return name
    return None


def frame_sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def process(video: Path) -> tuple[int, int]:
    yolo = YOLO(ROOT / "yolov8n.pt")
    names = yolo.names
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / SAMPLE_HZ)))
    tag = video.stem.replace(" ", "_")[:40]
    print(f"[{tag}] {n} frames @ {fps:.1f} fps, step {step}, "
          f"{n / fps / 60:.1f} min")

    out_dir = OUT_ROOT / tag
    for bucket in BUCKETS:
        (out_dir / bucket).mkdir(parents=True, exist_ok=True)

    track_best: dict[int, dict] = {}   # tid -> {"wins": {(bucket, win): (sharp, w, h, crop)}}
    last_seen: dict[int, int] = {}
    finalized: set[int] = set()
    frame_no = 0
    saved = 0

    def finalize(tid: int) -> None:
        nonlocal saved
        if tid in finalized:
            return
        finalized.add(tid)
        cur = track_best.get(tid)
        if not cur:
            return
        # sharpest first, capped per track
        wins = sorted(cur["wins"].items(), key=lambda kv: -kv[1][0])[:MAX_PER_TRACK]
        for (bucket, _win), (_sharp, w, h, crop) in wins:
            path = out_dir / bucket / f"t{tid}_{w}x{h}.jpg"
            cv2.imwrite(str(path), crop)
            saved += 1

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1
        if frame_no % step != 0:
            continue
        for det in yolo.track(frame, persist=True, conf=CONF, imgsz=IMGSZ,
                              classes=[2], verbose=False)[0].boxes:
            if det.id is None:
                continue
            if names[int(det.cls[0])] != "car":
                continue
            tid = int(det.id[0])
            x1, y1, x2, y2 = [int(v) for v in det.xyxy[0]]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue
            long_side = max(x2 - x1, y2 - y1)
            bucket = bucket_for(long_side)
            if bucket is None:
                continue  # < 90px: too far, not worth training data
            crop = frame[y1:y2, x1:x2]
            area = crop.shape[0] * crop.shape[1]
            if area < MIN_AREA:
                continue
            last_seen[tid] = frame_no
            sharp = frame_sharpness(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
            win = int((frame_no / fps) / WINDOW_SECS)
            key = (bucket, win)
            cur = track_best.get(tid)
            if cur is None:
                track_best[tid] = {"wins": {key: (sharp, x2 - x1, y2 - y1, crop)}}
            else:
                prev = cur["wins"].get(key)
                if prev is None or sharp > prev[0]:
                    cur["wins"][key] = (sharp, x2 - x1, y2 - y1, crop)
        for tid in list(last_seen):
            if tid not in finalized and frame_no - last_seen[tid] > MIN_CROP_TAIL:
                finalize(tid)
    cap.release()
    for tid in list(last_seen):
        finalize(tid)

    print(f"[{tag}] done: {len(finalized)} cars, {saved} crops -> {out_dir}")
    return len(finalized), saved


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    args = ap.parse_args()
    total_cars = total_crops = 0
    for v in args.videos:
        path = Path(v)
        if not path.exists():
            print(f"skip: {v} not found")
            continue
        cars, crops = process(path)
        total_cars += cars
        total_crops += crops
    print(f"\nTOTAL: {total_cars} cars, {total_crops} crops in {OUT_ROOT}")


if __name__ == "__main__":
    main()
