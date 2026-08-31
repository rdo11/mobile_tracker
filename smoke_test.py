"""Offline smoke test: synthetic scene -> privacy engine -> classifier -> recorder."""
import json
import os
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from anpr_privacy import PrivacyEngine, PlateDatabase  # noqa: E402
from classifier import ColorAnalyzer, VehicleClassifier  # noqa: E402
from recorder import SessionLog, VideoRecorder  # noqa: E402

failures = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        failures.append(name)


# --- synthetic scene: 720p, a dark "car" with a high-contrast plate-like strip ---
frame = np.full((720, 1280, 3), (60, 70, 80), np.uint8)  # grayish bg
cv2.rectangle(frame, (400, 300), (880, 640), (90, 90, 100), -1)          # car body
cv2.rectangle(frame, (560, 500), (780, 540), (200, 200, 200), -1)        # plate area
cv2.putText(frame, "ABC1234", (585, 533), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2)

anpr_cfg = {"enabled": True, "plate_model": "", "ocr_engine": "none",
            "ocr_min_confidence": 0.4, "plate_database": ""}
priv_cfg = {"mode": "pixel", "pixel_size": 12, "gaussian_kernel": 41,
            "pad_ratio": 0.2, "blur_full_frame": False, "show_blur_overlay": True}

print("== PrivacyEngine (contour find + pixelation) ==")
eng = PrivacyEngine(anpr_cfg, priv_cfg)
regions = eng.find_plates(frame, (400, 300, 880, 640))
check("contour plate found", len(regions) >= 1, str([r.bbox for r in regions]))
if regions:
    r = regions[0]
    x1, y1, x2, y2 = r.bbox
    before = frame[y1:y2, x1:x2].copy()
    n = eng.anonymize(frame, regions)
    after = frame[y1:y2, x1:x2]
    check("plate region pixelated (mean |Δpixel| > 0)", float(np.abs(after.astype(int) - before.astype(int)).mean()) > 1.0)
    check("all regions anonymized", n == len(regions), f"{n} of {len(regions)}")
    eng.draw_blur_overlay(frame, regions)
    cv2.imwrite("/tmp/smoke_anonymized.jpg", frame)
    print("  wrote /tmp/smoke_anonymized.jpg")

print("== ColorAnalyzer ==")
ca = ColorAnalyzer()
for name, (b, g, r) in {"Red": (40, 40, 200), "White": (240, 240, 240),
                        "Black": (15, 15, 15), "Blue": (200, 90, 20)}.items():
    car = np.full((100, 200, 3), (b, g, r), np.uint8)
    col, conf = ca.analyze(car)
    check(f"color {name}", col == name, f"got {col} ({conf})")

print("== VehicleClassifier (no checkpoint -> Unknown, color still works) ==")
vc = VehicleClassifier({"model_path": "", "color_analysis": True})
vc.load()
red_car = np.full((100, 200, 3), (40, 40, 200), np.uint8)
attrs = vc.classify(red_car)
check("make_model Unknown without checkpoint", attrs.make_model == "Unknown")
check("color detected without checkpoint", attrs.color == "Red", attrs.to_dict().__str__())
print("  split_label:", VehicleClassifier._split_label("Audi A4 2016-2020"),
      VehicleClassifier._split_label("Skoda Octavia 2013"))

print("== Recorder: anonymized video + sqlite ==")
with tempfile.TemporaryDirectory() as tmp:
    rec_cfg = {"enabled": True, "recordings_dir": tmp, "codec": "mp4v",
               "output_ext": "mp4"}
    rec = VideoRecorder(rec_cfg, fps=10.0)
    rec.start()
    for i in range(12):
        rec.write(frame)
    rec.stop()
    vids = os.listdir(tmp)
    check("video file written", len(vids) == 1 and vids[0].endswith(".mp4"), str(vids))
    log = SessionLog(os.path.join(tmp, "session_vehicles.sqlite"))
    log.upsert({"track_id": 104, "cls_name": "car", "cls_conf": 0.92,
                "make_model": "Audi A4", "year_range": "2018", "color": "Silver",
                "color_conf": 0.8, "plate_text": "ABC1234", "plate_status": "ANONYMIZED",
                "plate_db_match": False})
    log.upsert({"track_id": 104, "cls_name": "car", "cls_conf": 0.95,
                "make_model": "Audi A4", "year_range": "2018", "color": "Silver",
                "color_conf": 0.81, "plate_text": "ABC1234", "plate_status": "ANONYMIZED",
                "plate_db_match": False})
    log.upsert({"track_id": 7, "cls_name": "truck", "cls_conf": 0.8,
                "make_model": "Unknown", "year_range": "Unknown", "color": "White",
                "color_conf": 0.9, "plate_text": "", "plate_status": "none",
                "plate_db_match": False})
    rows = log.recent()
    check("sqlite upsert per track", len(rows) == 2, f"{len(rows)} rows")
    t104 = [r for r in rows if r["track_id"] == "104"][0]
    check("frames_seen incremented", t104["frames_seen"] == 2, str(t104["frames_seen"]))
    check("plate text stored internally", t104["plate_text"] == "ABC1234")
    log.close()

print("== PlateDatabase ==")
with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
    json.dump({"ABC 1234": {"owner": "fleet"}, "XYZ-999": {}}, fh)
    db_path = fh.name
pdb = PlateDatabase(db_path)
check("lookup normalized", pdb.lookup("abc1234") == {"owner": "fleet"})
check("lookup miss", pdb.lookup("ZZZ000") is None)
os.unlink(db_path)

print()
if failures:
    print(f"SMOKE TEST FAILED ({len(failures)}): {failures}")
    sys.exit(1)
print("SMOKE TEST OK")
