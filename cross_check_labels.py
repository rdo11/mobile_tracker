#!/usr/bin/env python3
"""cross_check_labels.py — second-opinion labeling for dashcam_raw.

Sends each crop (batched) to one provider and compares its answer against the
folder label. DISAGREEMENTS are moved to storage/dataset/quarantine/<folder>/
so build_merged.py stops training on suspect data. Agreements stay untouched.

Usage:
  .venv/bin/python cross_check_labels.py --provider gemini --max-requests 5
"""
from __future__ import annotations

import argparse
import base64
import shutil
import sys
import unicodedata
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "storage" / "dataset" / "dashcam_raw"
QUAR = ROOT / "storage" / "dataset" / "quarantine"
BATCH = 50


def fold(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s)
                   if not unicodedata.combining(c))


def same(a: str, b: str) -> bool:
    ka, kb = fold(a).lower(), fold(b).lower()
    ka = "".join(ch for ch in ka if ch.isalnum())
    kb = "".join(ch for ch in kb if ch.isalnum())
    if not ka or not kb:
        return False
    return ka == kb or ka in kb or kb in ka   # 'vwgolf' vs 'wolksvarengolf8'-style slack


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=("gemini", "deepseek"), default="gemini")
    ap.add_argument("--model", default=None,
                    help="override model name (free quota is PER MODEL — cycle "
                         "gemini-3.5-flash-lite / gemini-3.6-flash / ... daily)")
    ap.add_argument("--max-requests", type=int, default=5)
    ap.add_argument("--only-folders", nargs="*", default=None,
                    help="limit to these dashcam_raw folder names")
    args = ap.parse_args()

    if args.provider == "deepseek":
        from deepseek_classifier import DeepSeekClassifier
        engine = DeepSeekClassifier({"enabled": True,
                                     "model": "deepseek-v4-flash-vision-exp"})
        query = engine._query_batch
        key = "DEEPSEEK_API_KEY"
    else:
        from gemini_classifier import GeminiClassifier
        model = args.model or "gemini-3.5-flash-lite"
        engine = GeminiClassifier({"enabled": True, "model": model,
                                   "batch_size": BATCH, "flush_interval": 1})
        query = engine._query
        key = f"GEMINI_API_KEY ({model})"
    if not engine.available:
        sys.exit(f"ERROR: no {key} in .env")

    items = []          # (path, folder_label)
    for d in sorted(RAW.iterdir()):
        if not d.is_dir():
            continue
        if args.only_folders and d.name not in args.only_folders:
            continue
        items += [(p, d.name.replace("_", " ")) for p in sorted(d.glob("*.jpg"))]
    # newest-labeled first: today's additions are the prime noise suspects,
    # so limited verification budgets hit them before older, trusted labels.
    items.sort(key=lambda t: t[0].stat().st_mtime, reverse=True)
    print(f"{len(items)} labeled crops to verify ({key})")

    agree = disagree = lowconf = 0
    QUAR.mkdir(parents=True, exist_ok=True)
    sent = 0
    for r in range(args.max_requests):
        chunk = items[sent:sent + BATCH]
        if not chunk:
            break
        b64s, meta = [], []
        for p, lbl in chunk:
            img = cv2.imread(str(p))
            if img is None:
                continue
            h, w = img.shape[:2]
            if max(h, w) < 90:
                continue
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
            if ok:
                b64s.append(base64.b64encode(buf.tobytes()).decode())
                meta.append((p, lbl))
        results = query(b64s)
        if not results:
            # transient (rate limit / truncated response): pause, skip this
            # chunk, keep going — a full run matters more than one batch
            import time
            print(f"[req {r+1}] no parseable response — pausing 45s, skipping chunk")
            time.sleep(45)
            sent += len(meta)
            continue
        moved = 0
        for (p, lbl), res in zip(meta, results):
            sent += 1
            if not res:
                continue
            conf = float(res.get("confidence", 0.0))
            pred = str(res.get("label", "")).strip()
            if conf < 0.4:
                lowconf += 1
                continue
            if same(lbl, pred):
                agree += 1
            else:
                disagree += 1
                dst = QUAR / lbl.replace(" ", "_") / p.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(p), dst)
                moved += 1
                if moved <= 6:
                    print(f"  QUARANTINE {p.relative_to(RAW)} -> said '{pred}' ({conf:.2f})")
        print(f"[req {r+1}] verified {len(meta)} | agree {agree} total "
              f"| quarantined this round: {moved}")

    print(f"\nDONE: agree={agree} disagree(=moved)={disagree} low-conf(skipped)={lowconf}")
    print(f"Quarantine: {QUAR}")


if __name__ == "__main__":
    main()
