#!/usr/bin/env python3
"""Batch-label extracted crops via Gemini vision, one request per 50 crops.

Saves each labeled crop into storage/dataset/dashcam_raw/<Label>/ so it can be
merged into the training set. Crops already present in dashcam_raw are skipped
(unique by filename).

Usage: .venv/bin/python label_crops.py [--max-requests 5] [--max-images 250]
"""
import argparse
import shutil
import unicodedata
from pathlib import Path

import cv2

from gemini_classifier import GeminiClassifier

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "storage" / "dataset" / "dashcam_youtube"
OUT = ROOT / "storage" / "dataset" / "dashcam_raw"
CONF_SAVE = 0.6
BATCH = 50


def fold_label(label: str) -> str:
    """'Škoda Octavia Combi' -> 'Skoda_Octavia_Combi'.

    Accent-fold + spaces->underscores so every crop lands in ONE canonical
    folder per label (the old behavior created 'Citroen_C3' AND 'Citroën C3'
    as separate classes — orphaning crops at merge time).
    """
    folded = "".join(c for c in unicodedata.normalize("NFKD", label)
                     if not unicodedata.combining(c))
    return folded.strip().replace(" ", "_").replace("/", "_")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-requests", type=int, default=5)
    ap.add_argument("--max-images", type=int, default=250)
    ap.add_argument("--provider", choices=("gemini", "deepseek"), default="gemini",
                    help="gemini = free tier (~5 req/day); deepseek = paid, "
                         "pennies per 1000 crops, no daily cap — use it to "
                         "bulk-label or cross-check gemini labels")
    ap.add_argument("--batch", type=int, default=BATCH)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    crops = sorted(SRC.rglob("*.jpg"))
    # Skip crops that already sit in dashcam_raw (unique by filename) so the
    # request budget goes to NEW images instead of re-labeling old ones.
    labeled_names = {p.name for p in OUT.rglob("*.jpg")}
    # Rejects: crops the API already said "Unknown/low-conf" on — don't burn
    # credits re-sending them every round (cheap memory, big token saver).
    rejects = set()
    rej_file = ROOT / "storage" / "dataset" / "label_rejects.txt"
    if rej_file.exists():
        rejects = set(l.strip() for l in rej_file.read_text().splitlines() if l.strip())
    n_all = len(crops)
    crops = [p for p in crops
             if p.name not in labeled_names and str(p.relative_to(SRC)) not in rejects]
    # Shuffle: without this, every round re-scans the SAME sorted head of the
    # queue and the tail never gets attempted (starved 1,700 crops overnight).
    import random
    random.Random(int(__import__("time").time())).shuffle(crops)
    print(f"Found {n_all} crops in {SRC} ({len(crops)} not yet labeled, "
          f"{len(rejects)} known-rejects skipped)")

    def _mark_reject(rel: str) -> None:
        with rej_file.open("a") as fh:
            fh.write(rel + "\n")

    if args.provider == "deepseek":
        from deepseek_classifier import DeepSeekClassifier
        engine = DeepSeekClassifier({"enabled": True,
                                     "model": "deepseek-v4-flash-vision-exp",
                                     "batch_size": args.batch, "flush_interval": 1})
        key_name = "DEEPSEEK_API_KEY"
    else:
        engine = GeminiClassifier({"enabled": True, "model": "gemini-3.5-flash-lite",
                                   "batch_size": args.batch, "flush_interval": 1})
        key_name = "GEMINI_API_KEY"
    if not engine.available:
        print(f"ERROR: no {key_name}. Add it to .env")
        return

    total_imgs = 0
    for req_idx in range(args.max_requests):
        if total_imgs >= args.max_images:
            break
        batch = crops[total_imgs:total_imgs + args.batch]
        if not batch:
            break
        b64s = []
        meta = []
        for p in batch:
            img = cv2.imread(str(p))
            if img is None:
                continue
            h, w = img.shape[:2]
            if max(h, w) < 90:
                continue
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
            if not ok:
                continue
            b64s.append(buf.tobytes())
            meta.append(p)
        if not meta:
            break
        print(f"[req {req_idx + 1}] sending {len(meta)} images ...", flush=True)
        import base64
        payload = [base64.b64encode(b).decode() for b in b64s]
        if args.provider == "deepseek":
            results = engine._query_batch(payload)
        else:
            results = engine._query(payload)
        if not results:
            print(f"[req {req_idx + 1}] no parseable results — stopping")
            break
        labeled = 0
        for path, res in zip(meta, results):
            if not res:
                continue
            conf = res.get("confidence", 0.0)
            label = res.get("label", "Unknown")
            if label.startswith("Unknown") or conf < CONF_SAVE:
                # permanently mark as reject so future rounds skip it
                _mark_reject(str(path.relative_to(SRC)))
                continue
            safe = fold_label(label)
            folder = OUT / safe
            # Global dedupe: the same source filename must exist in AT MOST ONE
            # label folder. Earlier runs saved contradictory copies into two
            # folders when different API calls disagreed — poison for training.
            clash = [p for p in OUT.rglob(path.name) if p.parent != folder]
            if clash:
                print(f"  skip {path.name}: already labeled "
                      f"{clash[0].parent.name} (refusing contradiction '{safe}')")
                continue
            folder.mkdir(parents=True, exist_ok=True)
            dst = folder / path.name
            if not dst.exists():
                shutil.copy2(path, dst)
                labeled += 1
        print(f"[req {req_idx + 1}] labeled {labeled} crops")
        total_imgs += len(meta)

    print(f"\nDone. {total_imgs} images processed -> {OUT}")


if __name__ == "__main__":
    main()
