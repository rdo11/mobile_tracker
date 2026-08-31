#!/usr/bin/env python3
"""build_clean_dataset.py — build a cleaned dataset from verification results.

Strategy (conservative, no new API calls):
- agree.txt crops: keep with the current label (double-confirmed)
- disagree.txt crops: RELABEL with the provider's answer when confidence is high
  (>= 0.85); below that, drop the crop (too uncertain)
- lowconf.txt crops: drop

Output: storage/dataset/dashcam_clean/<Label>/<crop> — ready to train on.
"""
import shutil
import unicodedata
from pathlib import Path

ROOT = Path('/Users/radovanhloska/Projects/mobile_tracker')
RAW = ROOT / 'storage' / 'dataset' / 'dashcam_raw'
RES = ROOT / 'verify_results'
CLEAN = ROOT / 'storage' / 'dataset' / 'dashcam_clean'
RELABEL_CONF = 0.85


def fold_label(label: str) -> str:
    folded = "".join(c for c in unicodedata.normalize("NFKD", label)
                     if not unicodedata.combining(c))
    return folded.strip().replace(" ", "_").replace("/", "_")


def main():
    if CLEAN.exists():
        shutil.rmtree(CLEAN)
    CLEAN.mkdir(parents=True, exist_ok=True)

    kept = relabeled = dropped = 0

    # agrees: keep as-is
    if (RES / 'agree.txt').exists():
        for line in (RES / 'agree.txt').read_text().splitlines():
            if not line.strip():
                continue
            path, lbl = line.split('|', 1)
            p = Path(path)
            if not p.exists():
                continue
            d = CLEAN / lbl
            d.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, d / p.name)
            kept += 1

    # disagrees: relabel if confident, else drop
    if (RES / 'disagree.txt').exists():
        for line in (RES / 'disagree.txt').read_text().splitlines():
            if not line.strip():
                continue
            parts = line.split('|')
            path, cur_lbl, pred = parts[0], parts[1], parts[2]
            conf = float(parts[3]) if len(parts) > 3 else 0.0
            p = Path(path)
            if not p.exists():
                continue
            if conf >= RELABEL_CONF and not pred.startswith('Unknown'):
                safe = fold_label(pred)
                d = CLEAN / safe
                d.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, d / p.name)
                relabeled += 1
            else:
                dropped += 1

    n_crops = sum(1 for _ in CLEAN.rglob('*.jpg'))
    n_cls = len([d for d in CLEAN.iterdir() if d.is_dir()])
    with open(RES / 'clean_report.txt', 'w') as fh:
        fh.write(f'kept (double-confirmed): {kept}\n')
        fh.write(f'relabeled (conf>={RELABEL_CONF}): {relabeled}\n')
        fh.write(f'dropped (uncertain): {dropped}\n')
        fh.write(f'CLEAN total: {n_crops} crops / {n_cls} classes\n')
        fh.write(f'CLEAN dir: {CLEAN}\n')
    print(f'kept {kept} | relabeled {relabeled} | dropped {dropped}')
    print(f'CLEAN: {n_crops} crops / {n_cls} classes -> {CLEAN}')


if __name__ == '__main__':
    main()
