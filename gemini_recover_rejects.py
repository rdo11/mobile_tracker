#!/usr/bin/env python3
"""gemini_recover_rejects.py — second opinion on DeepSeek's 'Unknown' rejects.

DeepSeek was over-conservative (77% recovery rate in testing). Gemini re-checks
every rejected crop; confident labels are saved to dashcam_raw and the crop is
removed from the reject list so future rounds can see it's labeled.
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
    from gemini_classifier import GeminiClassifier
    g = GeminiClassifier({'enabled': True, 'model': 'gemini-3.5-flash-lite',
                          'batch_size': BATCH, 'flush_interval': 1})
    if not g.available:
        print('ERROR: no GEMINI_API_KEY')
        return

    rejects = [l.strip() for l in REJ_FILE.read_text().splitlines() if l.strip()]
    print(f'{len(rejects)} rejects to re-check with Gemini')

    # remove rejects that are already labeled now (from prior rounds)
    labeled = {p.name for p in OUT.rglob('*.jpg')}
    todo = [rel for rel in rejects if rel.rsplit('/', 1)[-1] not in labeled]
    print(f'{len(todo)} still unlabeled')

    recovered = 0
    still_reject = 0
    new_rejects = []
    for r in range(0, len(todo), BATCH):
        chunk = todo[r:r + BATCH]
        b64s, meta = [], []
        for rel in chunk:
            p = SRC / rel
            img = cv2.imread(str(p))
            if img is None:
                continue
            h, w = img.shape[:2]
            if max(h, w) < 60:
                still_reject += 1
                continue
            ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 88])
            if ok:
                b64s.append(base64.b64encode(buf.tobytes()).decode())
                meta.append((rel, p))
        if not meta:
            continue
        results = g._query(b64s)
        if not results:
            print(f'[req {r//BATCH+1}] no results — stopping')
            break
        for (rel, p), res in zip(meta, results):
            if not res:
                still_reject += 1
                continue
            conf = float(res.get('confidence', 0.0))
            label = str(res.get('label', 'Unknown')).strip()
            if label.startswith('Unknown') or conf < CONF_SAVE:
                still_reject += 1
                continue
            safe = fold_label(label)
            folder = OUT / safe
            folder.mkdir(parents=True, exist_ok=True)
            dst = folder / p.name
            if not dst.exists():
                shutil.copy2(p, dst)
            recovered += 1
            if recovered <= 8:
                print(f'  RECOVERED: {p.name:<30} -> {safe} ({conf:.2f})')
        print(f'[req {r//BATCH+1}] processed {len(meta)} | recovered {recovered} so far')

    # rewrite the reject file WITHOUT the recovered ones
    recovered_names = {p.name for p in OUT.rglob('*.jpg')}
    kept = [rel for rel in rejects
            if rel.rsplit('/', 1)[-1] not in recovered_names]
    REJ_FILE.write_text('\n'.join(kept) + ('\n' if kept else ''))
    print(f'\nDONE: recovered {recovered} | still rejected {len(kept)} '
          f'(reject file updated)')


if __name__ == '__main__':
    main()
