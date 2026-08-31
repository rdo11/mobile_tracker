#!/usr/bin/env python3
"""Offline dashcam labeling: extract one best crop per unique car from a
recording, ask Grok once per car, save labeled crops to the dataset.

Usage: .venv/bin/python label_drive.py [recording.mp4] [--min-area 3000]
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from grok_classifier import GrokClassifier

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "storage" / "dataset" / "dashcam_raw"
HARD_LABEL_DIR = ROOT / "storage" / "dataset" / "dashcam_hard"
MIN_AREA = 3000
MAX_CALLS = 60
CONF_SAVE = 0.6
MIN_CROP_TAIL = 6  # frames after detection before re-picking crop


def frame_sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("recording", nargs="?", default=None)
    ap.add_argument("--min-area", type=int, default=MIN_AREA)
    ap.add_argument("--max-calls", type=int, default=MAX_CALLS)
    args = ap.parse_args()

    if args.recording is None:
        vids = sorted((ROOT / "storage" / "recordings").glob("session_*.mp4"),
                      key=lambda p: p.stat().st_mtime)
        rec = vids[-1]
    else:
        rec = Path(args.recording)
    print(f"Recording: {rec}")

    yolo = YOLO("yolov8n.pt")
    names = yolo.names
    cap = cv2.VideoCapture(str(rec))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Frames: {n} ({n / 30 / 60:.1f} min)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    HARD_LABEL_DIR.mkdir(parents=True, exist_ok=True)

    grok = GrokClassifier({"model": "grok-4.6", "fallback_conf": 1.0})
    track_best: dict[int, dict] = {}  # track_id -> best crop seen
    last_seen: dict[int, int] = {}
    finalized: set[int] = set()
    calls, saved = 0, 0
    queue: list[tuple[int, np.ndarray]] = []

    def finalize(tid: int) -> None:
        nonlocal calls, saved
        if tid in finalized:
            return
        finalized.add(tid)
        best = track_best.get(tid)
        if not best or best["area"] < args.min_area:
            return
        calls += 1
        print(f"[{calls}] track {tid}: {best['w']}x{best['h']} sharp {best['sharp']:.1f} "
              f"-> ", end="", flush=True)
        t0 = time.time()
        res = grok._query(best["crop"])
        dt = time.time() - t0
        if not res or res.get("label", "Unknown Unknown").startswith("Unknown"):
            print(f"Unknown ({dt:.1f}s) - saved to hard/")
            cv2.imwrite(str(HARD_LABEL_DIR / f"t{tid}_{best['w']}x{best['h']}.jpg"), best["crop"])
            return
        label = res["label"].replace("/", "_")
        conf = res.get("confidence", 0.0)
        print(f"{label} conf {conf:.2f} ({dt:.1f}s)")
        if conf < CONF_SAVE:
            return
        folder = OUT_DIR / label
        folder.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(str(folder / f"{stamp}_t{tid}.jpg"), best["crop"])
        saved += 1

    frame_no = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1
        step = max(1, n // 1200)  # dense sampling keeps track IDs stable
        if frame_no % step != 0:
            continue
        for det in yolo.track(frame, persist=True, verbose=False)[0].boxes:
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
            crop = frame[y1:y2, x1:x2]
            area = crop.shape[0] * crop.shape[1]
            if max(x2 - x1, y2 - y1) < 90:
                continue  # too far away — fuzzy dot, not worth a token
            last_seen[tid] = frame_no
            sharp = frame_sharpness(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
            cur = track_best.get(tid)
            if cur is None or area > cur["area"] or sharp > cur["sharp"] * 1.2:
                track_best[tid] = {
                    "crop": crop, "area": area, "sharp": sharp,
                    "w": crop.shape[1], "h": crop.shape[0], "frame": frame_no,
                }
        # finalize tracks that left the frame
        for tid in list(last_seen):
            if tid not in finalized and frame_no - last_seen[tid] > MIN_CROP_TAIL:
                finalize(tid)
        if calls >= args.max_calls:
            print(f"Reached --max-calls {args.max_calls}, stopping.")
            break
    cap.release()
    for tid in list(last_seen):
        finalize(tid)

    print(f"\nDone. Grok calls: {calls}, saved labeled crops: {saved}")
    print(f"Output: {OUT_DIR}")
    print(f"Hard/unknown (for the eval set): {HARD_LABEL_DIR}")

    if saved:
        print("\nNext: rsync dataset + dashcam_raw to RunPod and fine-tune, or")
        print("      copy dashcam_raw into the local dataset for a merge-eval.")


if __name__ == "__main__":
    main()