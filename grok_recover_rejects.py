#!/usr/bin/env python3
"""grok_recover_rejects.py — third-opinion recovery of DeepSeek rejects via Grok.

DeepSeek: 0% on these; Gemini: 77-100% but quota-limited. Grok (paid, no cap)
re-checks every remaining reject in BATCHES (50/request) for speed.
"""
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import base64
import shutil
import unicodedata
from pathlib import Path

import cv2

ROOT = Path(str(Path(__file__).resolve().parent))
SRC = ROOT / 'storage' / 'dataset' / 'dashcam_youtube'
OUT = ROOT / 'storage' / 'dataset' / 'dashcam_raw'
REJ_FILE = ROOT / 'storage' / 'dataset' / 'label_rejects.txt'
CONF_SAVE = 0.6
BATCH = 50


def fold_label(label: str) -> str:
    folded = "".join(c for c in unicodedata.normalize("NFKD", label)
                     if not unicodedata.combining(c))
    return folded.strip().replace(" ", "_").replace("/", "_")


def main():
    from grok_classifier import GrokClassifier
    g = GrokClassifier({'enabled': True})
    if not g.available:
        print('ERROR: no XAI_API_KEY')
        return

    rejects = [l.strip() for l in REJ_FILE.read_text().splitlines() if l.strip()]
    labeled = {p.name for p in OUT.rglob('*.jpg')}
    todo = [rel for rel in rejects if rel.rsplit('/', 1)[-1] not in labeled]
    print(f'{len(todo)} rejects to re-check with Grok ({g.model}) in batches of {BATCH}')

    recovered = 0
    for r in range(0, len(todo), BATCH):
        chunk = todo[r:r + BATCH]
        b64s, meta = [], []
        for rel in chunk:
            p = SRC / rel
            img = cv2.imread(str(p))
            if img is None:
                continue
            ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 88])
            if ok:
                b64s.append(base64.b64encode(buf.tobytes()).decode())
                meta.append((rel, p))
        if not meta:
            continue
        results = g._query_batch(b64s)
        if not results:
            print(f'[req {r//BATCH+1}] no parseable results — pausing 30s')
            import time
            time.sleep(30)
            continue
        for (rel, p), res in zip(meta, results):
            if not res:
                continue
            conf = float(res.get('confidence', 0.0))
            label = str(res.get('label', 'Unknown')).strip()
            if label.startswith('Unknown') or conf < CONF_SAVE:
                continue
            safe = fold_label(label)
            folder = OUT / safe
            folder.mkdir(parents=True, exist_ok=True)
            dst = folder / p.name
            if not dst.exists():
                shutil.copy2(p, dst)
            recovered += 1
            if recovered <= 6:
                print(f'  RECOVERED: {p.name:<28} -> {safe} ({conf:.2f})')
        print(f'[req {r//BATCH+1}] processed {len(meta)} | recovered {recovered} so far')

    recovered_names = {p.name for p in OUT.rglob('*.jpg')}
    kept = [rel for rel in rejects if rel.rsplit('/', 1)[-1] not in recovered_names]
    REJ_FILE.write_text('\n'.join(kept) + ('\n' if kept else ''))
    print(f'\nDONE: recovered {recovered} | still rejected {len(kept)}')


if __name__ == '__main__':
    main()
