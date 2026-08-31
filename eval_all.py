#!/usr/bin/env python3
"""eval_all.py — honest four-way comparison on crops held out from ALL models.

Method (the v20 lesson): naive holdout compares are contaminated because each
model trained on part of the other's holdout. The only fair test = crops that
NEITHER model ever trained on = the INTERSECTION of the frozen eval_holdout
with the older holdout tars, restricted further to tracks not in ANY training.

Here we use: frozen eval_holdout (3,570) ∩ v19-holdout (2,606) → ~597 crops.
Both king (v19) and v20/v21/v22 trained with these excluded (king: v19-holdout;
v20+: frozen-holdout exclusion in build_merged.py).
"""
import sys
from pathlib import Path
import shutil

ROOT = Path(str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))
from compare_ckpts import load_model, load_items, prep  # noqa: E402
import torch  # noqa: E402

FROZEN = ROOT / 'storage' / 'dataset' / 'eval_holdout'
V19_TAR = ROOT / 'pod_bundle_v19' / 'dashcam_val_v19.tar'
WORK = Path('/tmp/eval_intersect')

def build_intersection():
    if WORK.exists():
        shutil.rmtree(WORK)
    # extract v19 holdout
    import tarfile
    with tarfile.open(V19_TAR) as tf:
        tf.extractall('/tmp/eval_v19_tmp')
    v19root = Path('/tmp/eval_v19_tmp') / 'storage' / 'dataset' / 'dashcam_val'
    v19 = {p.name: p.parent.name for p in v19root.rglob('*.jpg')}
    frozen = {p.name: p.parent.name for p in FROZEN.rglob('*.jpg')}
    inter = set(frozen) & set(v19)
    for name, cls in frozen.items():
        if name in inter:
            d = WORK / cls
            d.mkdir(parents=True, exist_ok=True)
            src = FROZEN / cls / name
            if src.exists():
                shutil.copy2(src, d / name)
    return len(inter)

def evaluate(m, classes, items, batch=64):
    idx = {c: i for i, c in enumerate(classes)}
    usable = [(p, idx[c]) for p, c in items if c in idx]
    t1 = t5 = 0
    with torch.no_grad():
        for k in range(0, len(usable), batch):
            chunk = usable[k:k+batch]
            x = torch.stack([prep(p) for p, _ in chunk])
            probs = m(x).softmax(1)
            topk = probs.topk(5, 1).indices
            for j, (_, y) in enumerate(chunk):
                t1 += int(topk[j, 0].item() == y)
                t5 += int(y in topk[j].tolist())
    return t1/len(usable), t5/len(usable), len(usable)

def main():
    n = build_intersection()
    print(f'INTERSECTION eval set: {n} crops (held out from all models)\n')
    items = load_items(WORK)
    models = [
        ('king v19', 'models/r50_dashcam.pt'),
        ('v20', 'models/v20/v20_tiny224.pt'),
        ('v21', 'models/v21/v21_tiny224.pt'),
        ('v22', 'models/v22/v22_tiny224.pt'),
    ]
    rows = []
    for label, path in models:
        p = ROOT / path
        if not p.exists():
            print(f'{label}: checkpoint missing ({p}) — skipping')
            continue
        m, classes = load_model(str(p))
        t1, t5, n2 = evaluate(m, classes, items)
        rows.append((label, t1, t5, n2))
        print(f'{label:<10} top-1 {t1:.3f}  top-5 {t5:.3f}  (n={n2})')
    if len(rows) >= 2:
        best = max(rows, key=lambda r: r[1])
        print(f'\nWINNER on honest intersection: {best[0]} ({best[1]:.3f} top-1)')
    print(f'\n(frozen-holdout full-set compare runs separately in v22_results)')

if __name__ == '__main__':
    main()
