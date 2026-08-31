#!/usr/bin/env python3
"""verify_all_labels.py — full Grok/Gemini verification of every labeled crop.

For each crop in dashcam_raw/<class>/<name>.jpg, ask the chosen provider
(batched) and RECORD agree/disagree to disk (verify_results/), so a run can be
interrupted/resumed and the results are never lost.

Output:
  verify_results/agree.txt      crop path + confirmed label
  verify_results/disagree.txt   crop path + current label + provider label + conf
  verify_results/lowconf.txt    crop path (provider couldn't decide)
  verify_results/report.txt     summary

Usage: .venv/bin/python verify_all_labels.py [--provider grok|gemini] [--max N]
"""
import sys
sys.path.insert(0, '/Users/radovanhloska/Projects/mobile_tracker')
import base64
import unicodedata
from collections import defaultdict
from pathlib import Path

import cv2

ROOT = Path('/Users/radovanhloska/Projects/mobile_tracker')
RAW = ROOT / 'storage' / 'dataset' / 'dashcam_raw'
OUT_DIR = ROOT / 'verify_results'
BATCH = 50
CONF_AGREE = 0.6


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
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--provider', choices=('grok', 'gemini'), default='grok')
    ap.add_argument('--max', type=int, default=0, help='limit crops (0=all)')
    args = ap.parse_args()

    if args.provider == 'grok':
        from grok_classifier import GrokClassifier
        engine = GrokClassifier({'enabled': True})
        query = engine._query_batch
    else:
        from gemini_classifier import GeminiClassifier
        engine = GeminiClassifier({'enabled': True,
                                   'model': 'gemini-3.5-flash-lite',
                                   'batch_size': BATCH, 'flush_interval': 1})
        query = engine._query
    if not engine.available:
        sys.exit(f'no {args.provider} key')

    OUT_DIR.mkdir(exist_ok=True)
    agree_f = open(OUT_DIR / 'agree.txt', 'a')
    disagree_f = open(OUT_DIR / 'disagree.txt', 'a')
    lowconf_f = open(OUT_DIR / 'lowconf.txt', 'a')

    # resume: skip already-recorded crops
    done = set()
    for f in ('agree.txt', 'disagree.txt', 'lowconf.txt'):
        p = OUT_DIR / f
        if p.exists():
            for line in p.read_text().splitlines():
                if line.strip():
                    done.add(line.split('|')[0].strip())

    items = []
    for d in sorted(RAW.iterdir()):
        if not d.is_dir() or 'Unknown' in d.name:
            continue
        for p in sorted(d.glob('*.jpg')):
            if str(p) not in done:
                items.append((p, d.name))
    if args.max:
        import random
        random.Random(1).shuffle(items)
        items = items[:args.max]
    print(f'{len(items)} crops to verify ({args.provider}), resuming around {len(done)} done')

    agree = disagree = lowconf = 0
    per_class = defaultdict(lambda: [0, 0])
    req_n = 0
    for r in range(0, len(items), BATCH):
        chunk = items[r:r + BATCH]
        b64s, meta = [], []
        for p, lbl in chunk:
            img = cv2.imread(str(p))
            if img is None:
                continue
            ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 88])
            if ok:
                b64s.append(base64.b64encode(buf.tobytes()).decode())
                meta.append((p, lbl))
        if not meta:
            continue
        req_n += 1
        if args.provider == 'gemini':
            results = query(b64s)
        else:
            results = query(b64s)
        if not results:
            print(f'[req {req_n}] no results — pause 30s (may be out of credits)')
            import time
            time.sleep(30)
            continue
        for (p, lbl), res in zip(meta, results):
            if not res:
                continue
            conf = float(res.get('confidence', 0.0))
            pred = str(res.get('label', 'Unknown')).strip()
            if conf < CONF_AGREE:
                lowconf += 1
                lowconf_f.write(f'{p}|{lbl}\n')
                continue
            if same(lbl, pred):
                agree += 1
                per_class[lbl][0] += 1
                per_class[lbl][1] += 1
                agree_f.write(f'{p}|{lbl}\n')
            else:
                disagree += 1
                per_class[lbl][1] += 1
                disagree_f.write(f'{p}|{lbl}|{pred}|{conf:.2f}\n')
        agree_f.flush(); disagree_f.flush(); lowconf_f.flush()
        if req_n % 10 == 0:
            print(f'[req {req_n}] agree {agree} | disagree {disagree} | lowconf {lowconf}')

    agree_f.close(); disagree_f.close(); lowconf_f.close()
    with open(OUT_DIR / 'report.txt', 'w') as fh:
        fh.write(f'provider: {args.provider}\n')
        fh.write(f'checked: {agree + disagree + lowconf}\n')
        fh.write(f'AGREE: {agree}\nDISAGREE: {disagree}\nLOWCONF: {lowconf}\n')
        fh.write(f'rate: {agree / max(1, agree + disagree) * 100:.1f}%\n')
        worst = sorted(per_class.items(), key=lambda kv: kv[1][1] - kv[1][0], reverse=True)
        for name, (a, t) in worst[:20]:
            fh.write(f'{name}\t{a}/{t}\n')
    print(f'\nDONE — results saved to {OUT_DIR}/')


if __name__ == '__main__':
    main()
