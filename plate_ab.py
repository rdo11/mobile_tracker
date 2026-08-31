#!/usr/bin/env python3
"""plate_ab.py — proper A/B: raw vs enhanced OCR on REAL plates.

Scans the 13-min EU drive, samples every Nth frame, finds plates with the
plate YOLO model, then OCRs each plate crop BOTH ways:
  A) raw        (no gate, no enhance)
  B) enhanced   (size gate 28px + LANCZOS x2 + CLAHE)
Counts successful reads + reads a human would trust (>= conf gate).
"""
import sys, time
import cv2
sys.path.insert(0, ".")
from anpr_privacy import PrivacyEngine

eng = PrivacyEngine({"enabled": True, "plate_model": "storage/models/plate.pt",
                     "ocr_engine": "easyocr", "ocr_lang": ["en"],
                     "ocr_min_confidence": 0.4}, {})
eng.load()
assert eng.ocr_available, "OCR not available"

cap = cv2.VideoCapture("storage/recordings/session_20260819_131447.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 2000)
stats = {"A_raw": {"reads": 0, "chars": 0}, "B_enh": {"reads": 0, "chars": 0}}
n_plates = n_gated = 0
processed = 0
t0 = time.time()
while processed < 2000:
    ok, frame = cap.read()
    if not ok:
        break
    processed += 1
    if processed % 20 != 0:          # ~1.5 frames/sec sampled
        continue
    res = eng.plate_model(frame, verbose=False, conf=0.35)[0]
    if res.boxes is None or len(res.boxes) == 0:
        continue
    xyxy = res.boxes.xyxy.cpu().numpy().astype(int)
    for i in range(len(xyxy)):
        x1, y1, x2, y2 = xyxy[i]
        pad = 6
        x1p, y1p = max(0, x1 - pad), max(0, y1 - pad)
        x2p, y2p = min(frame.shape[1], x2 + pad), min(frame.shape[0], y2 + pad)
        crop = frame[y1p:y2p, x1p:x2p]
        if crop.shape[0] < 12 or crop.size == 0:
            continue
        n_plates += 1
        ALL = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        # A) raw
        ra = eng.ocr_reader.readtext(crop, detail=1, allowlist=ALL, paragraph=False)
        bestA = max([(c, len(t)) for _, t, c in ra if len(t) >= 4], default=(0, 0))
        if bestA[0] >= 0.4:
            stats["A_raw"]["reads"] += 1
            stats["A_raw"]["chars"] += bestA[1]
        # B) enhanced (only meaningful if >= 28px, else skip = not counted)
        if crop.shape[0] < 28:
            n_gated += 1
            continue
        e = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
        lab = cv2.cvtColor(e, cv2.COLOR_BGR2LAB); l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(2.0, (8, 8)).apply(l)
        e = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
        rb = eng.ocr_reader.readtext(e, detail=1, allowlist=ALL, paragraph=False)
        bestB = max([(c, len(t)) for _, t, c in rb if len(t) >= 4], default=(0, 0))
        if bestB[0] >= 0.4:
            stats["B_enh"]["reads"] += 1
            stats["B_enh"]["chars"] += bestB[1]

print(f"frames {processed} | plates {n_plates} | skipped (<28px) {n_gated} | {time.time()-t0:.0f}s")
for k, s in stats.items():
    print(f"  {k}: {s['reads']} reads ({s['reads']/max(1,n_plates):.0%}) avg {s['chars']/max(1,s['reads']):.1f} chars")
