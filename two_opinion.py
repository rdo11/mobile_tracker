#!/usr/bin/env python3
"""two_opinion.py — reconcile over-aggressive quarantine with a second provider.

Each crop quarantined by Gemini is re-checked by DeepSeek. A crop is KEPT in
quarantine only if BOTH providers disagree with the folder label. If DeepSeek
AGREES with the folder label, Gemini was wrong -> restore the crop to
dashcam_raw (and drop it from quarantine). Low-confidence DeepSeek = keep as-is.
"""
import base64
import shutil
import sys
import unicodedata
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "storage" / "dataset" / "dashcam_raw"
QUAR = ROOT / "storage" / "dataset" / "quarantine"
BATCH = 10


def fold(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s)
                   if not unicodedata.combining(c))


def same(a, b):
    ka = "".join(ch for ch in fold(a).lower() if ch.isalnum())
    kb = "".join(ch for ch in fold(b).lower() if ch.isalnum())
    if not ka or not kb:
        return False
    return ka == kb or ka in kb or kb in ka


def main():
    from deepseek_classifier import DeepSeekClassifier
    engine = DeepSeekClassifier({"enabled": True,
                                 "model": "deepseek-v4-flash-vision-exp"})
    if not engine.available:
        sys.exit("no DEEPSEEK_API_KEY in .env")
    query = engine._query_batch

    items = []  # (src_path, quarantine_dir_path, folder_label)
    for d in sorted(QUAR.iterdir()):
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.jpg")):
            items.append((p, d, d.name.replace("_", " ")))
    print(f"{len(items)} quarantined crops to re-check with DeepSeek")

    restored = kept = lowconf = 0
    for r in range(0, len(items), BATCH):
        chunk = items[r:r + BATCH]
        b64s, meta = [], []
        for p, d, lbl in chunk:
            img = cv2.imread(str(p))
            if img is None:
                continue
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
            if ok:
                b64s.append(base64.b64encode(buf.tobytes()).decode())
                meta.append((p, d, lbl))
        if not meta:
            continue
        results = query(b64s)
        if not results:
            print(f"[batch {r//BATCH}] no parseable response — pausing 30s")
            import time
            time.sleep(30)
            continue
        for (p, d, lbl), res in zip(meta, results):
            if not res:
                continue
            conf = float(res.get("confidence", 0.0))
            pred = str(res.get("label", "")).strip()
            if conf < 0.4:
                lowconf += 1
                continue
            if same(lbl, pred):
                # DeepSeek agrees with original label -> Gemini was wrong, restore
                dst = RAW / d.name / p.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(p), dst)
                restored += 1
                print(f"  RESTORE {d.name}/{p.name} (DS said '{pred}' {conf:.2f})")
            else:
                kept += 1
        print(f"[batch {r//BATCH}] restored {restored} | kept {kept} | lowconf {lowconf}")

    print(f"\nDONE: restored={restored} kept-quarantined={kept} lowconf-skipped={lowconf}")


if __name__ == "__main__":
    main()
