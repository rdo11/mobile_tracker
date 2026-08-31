"""End-to-end offline test: YOLO detect + track + classify on the real drive recording."""
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from classifier import VehicleClassifier  # noqa: E402
from tracker import VehicleTracker  # noqa: E402

VIDEO = "storage/recordings/session_20260817_134751.mp4"
det_cfg = {"model": "yolov8n.pt", "device": "auto", "conf_threshold": 0.35,
           "iou_threshold": 0.45, "imgsz": 640, "vehicle_classes": [2, 3, 5, 7],
           "tracker": "bytetrack.yaml", "tracker_persist": True}
cls_cfg = {"model_path": "models/stanford_cars_convnext", "device": "auto",
           "color_analysis": True}

print("loading tracker (downloads yolov8n.pt on first run)...")
tr = VehicleTracker(det_cfg)
assert tr.load(), tr.error
vc = VehicleClassifier(cls_cfg)
assert vc.load(), vc.error

cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    print("cannot open", VIDEO)
    sys.exit(1)

total, detected = 0, 0
seen: dict[int, str] = {}
t0 = time.time()
for i in range(7661):
    ok, frame = cap.read()
    if not ok:
        break
    total += 1
    dets = tr.update(frame)
    if not dets:
        continue
    detected += 1
    for d in dets:
        x1, y1, x2, y2 = [int(v) for v in d.bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        if x2 - x1 < 40 or y2 - y1 < 40:
            continue
        if d.track_id in seen:
            continue
        roi = frame[y1:y2, x1:x2]
        a = vc.classify(roi)
        seen[d.track_id] = (
            f"ID #{d.track_id} {d.cls_name} {d.conf:.0%} -> "
            f"{a.make_model} {a.model_confidence:.0%} ({a.year_range}) "
            f"{a.color} {a.color_confidence:.0%}"
        )
        print(f"  frame {i:3d}: {seen[d.track_id]}")

cap.release()
dt = time.time() - t0
print(f"\nframes: {total}, with detections: {detected}, unique tracks: {len(seen)}")
print(f"avg pipeline time: {dt / max(1, total) * 1000:.0f} ms/frame ({total / max(dt, 1e-9):.1f} fps)")
