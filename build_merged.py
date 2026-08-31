#!/usr/bin/env python3
"""build_merged.py — reproducible train-set builder (replaces ad-hoc shell injection).

What it does
------------
1. Canonicalizes every label: accent-fold, case-fix (Seat -> SEAT), strip
   estate/variant suffixes onto the base model class, alias fixes.
2. Builds storage/dataset/train_v6/ = hardlinks of the filtered web set +
   dashcam crops oversampled xN into their canonical class dirs.
3. Holds out ~1/MAX_FRACTION of real crops per class as a NEVER-TRAINED
   validation split: storage/dataset/dashcam_val/. This is the honest metric —
   val_acc inside train_v6 is contaminated by the oversampling copies.
4. Optionally adds real-crop-only classes (--allow-new) that have no web data.
   The v5 lesson: web-only classes fail on real footage, but a few REAL crops
   x30 CAN seed a working class.

Usage:
    .venv/bin/python build_merged.py                 # build train_v6 + dashcam_val
    .venv/bin/python build_merged.py --oversample 40 --allow-new --dry-run
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC_WEB = ROOT / "storage" / "dataset" / "filtered"
SRC_WIKI = ROOT / "storage" / "dataset" / "wikimedia"
SRC_RAW = ROOT / "storage" / "dataset" / "dashcam_raw"
OUT_TRAIN = ROOT / "storage" / "dataset" / "train_v6"
OUT_VAL = ROOT / "storage" / "dataset" / "dashcam_val"

# Estate/wagon/body-variant suffixes folded onto the base model class
VARIANT_RE = re.compile(
    r"_(Combi|Variant|SW|SW\d*|ST|MCV|Sports[ _]?Tourer|Active[ _]?Tourer"
    r"|Estate|Tourer|Touring|Avant|Sportback|Shooting[ _]?Brake"
    r"|Station[ _]?Wagon|Kombi|Caravan|Cabriolet|Cabrio|Convertible)$")
ALIASES = {
    "Kia_cee'd": "Kia_Ceed",
    "Kia_Ceed_Sportswagon": "Kia_Ceed",
    "Hyundai_Kona_Electric": "Hyundai_Kona",
    "Opel_Crossland_X": "Opel_Crossland",
    "Skoda_Enyaq_iV": "Skoda_Enyaq",
    # batch-3 DeepSeek label dupes (same car, different spellings)
    "MG_MG4": "MG4_EV",
    "Mazda_Mazda6": "Mazda6",
    "Polestar_Polestar_2": "Polestar_2",
}


def fold(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s)
                   if not unicodedata.combining(c))


def _key(s: str) -> str:
    """Match key: accents, case, spaces and punctuation all collapse
    ('Volkswagen up!' == 'Volkswagen_Up' == 'volkswagen-up')."""
    return re.sub(r"[^a-z0-9]", "", fold(s).lower())


def canon(label: str, web_keys_lower: dict[str, str]) -> str:
    """'Škoda Octavia Combi' -> 'Skoda_Octavia'; case-matched to the web set."""
    s = fold(label).strip().replace(" ", "_").replace("/", "_")
    s = ALIASES.get(s, s)
    # strip a trailing bracketed generation tag: 'Porsche_Macan_(XAB)' -> 'Porsche_Macan'
    s = re.sub(r"\(([A-Za-z0-9.]+)\)$", "", s).rstrip("_")
    # strip variant suffixes while the base is a known class
    keys = {_key(k) for k in web_keys_lower.values()}
    changed = True
    while changed:
        changed = False
        # Mercedes-style pairing: 'CLA' -> 'CLA_Class' when that's the name used
        if _key(s + "_Class") in keys:
            s += "_Class"
            changed = True
            continue
        m = VARIANT_RE.search(s)
        if m and (_key(s[:m.start()]) in keys
                  or _key(s[:m.start()] + "_Class") in keys):
            s = s[:m.start()]
            changed = True
            continue
        m2 = re.search(r"_([1-9])$", s)      # Golf_7 / Corsa_D-style digits
        if m2 and _key(s[:m2.start()]) in keys:
            s = s[:m2.start()]
            changed = True
            continue
        m3 = re.search(r"_([A-Z])$", s)      # Astra_K / Octavia_II letters
        if m3 and _key(s[:m3.start()]) in keys:
            s = s[:m3.start()]
            changed = True
    # case-canonicalize against existing web keys (Seat_Ibiza -> SEAT_Ibiza)
    return web_keys_lower.get(_key(s), s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oversample", type=int, default=30,
                    help="copies per real crop injected into the train set")
    ap.add_argument("--holdout-every", type=int, default=5,
                    help="every Nth crop (sorted) becomes dashcam_val, rest train")
    ap.add_argument("--min-web", type=int, default=0,
                    help="skip classes whose web count is below this")
    ap.add_argument("--allow-new", action="store_true",
                    help="add classes that only exist as real crops")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not SRC_WEB.is_dir():
        sys.exit(f"missing {SRC_WEB}")
    if not SRC_RAW.is_dir():
        sys.exit(f"missing {SRC_RAW}")

    # ---- collect ALL source dirs (filtered + wikimedia union), canonicalized
    # Web dirs themselves carry variants ('Opel_Crossland_X', 'Audi_A4_Avant')
    # that must merge onto their base class instead of fragmenting it.
    raw_dirs: list[tuple[str, Path]] = []
    for src in (SRC_WEB, SRC_WIKI):
        if src.is_dir():
            for d in sorted(os.listdir(src)):
                p = src / d
                if p.is_dir():
                    raw_dirs.append((d, p))
    raw_keys = {_key(d): d for d, _ in raw_dirs}
    # group canonical name -> list of source dirs (first wins on dupes)
    web_groups: dict[str, list[Path]] = {}
    for d, p in raw_dirs:
        c = canon(d, raw_keys)
        web_groups.setdefault(c, []).append(p)
    web_classes = sorted(web_groups)

    def web_img_count(cls: str) -> int:
        return sum(1 for g in web_groups.get(cls, []) for f in g.rglob("*.jpg"))

    def web_iter(cls: str):
        for g in web_groups.get(cls, []):
            yield from g.rglob("*")

    # ---- gather real crops per canonical class; split holdout vs train
    # GROUP-AWARE split: all crops of one tracked car share the 't<id>_' prefix
    # and MUST land on the same side, otherwise near-duplicate frames of the
    # same car leak between train and val and inflate the holdout metric.
    def track_of(p: Path) -> str:
        name = p.name.split("_")[0]          # 't123' from 't123_300x150.jpg'
        return name if name.startswith("t") and name[1:].isdigit() else p.stem

    val_items: dict[str, list[Path]] = defaultdict(list)
    train_items: dict[str, list[Path]] = defaultdict(list)
    n_orphan_new = 0
    # FROZEN EVAL HOLDOUT: crops in storage/dataset/eval_holdout/ are the permanent
    # evaluation set. They must NEVER enter any training build, or future comparisons
    # against it are invalid (the v20 lesson). Collect their filenames up front.
    EVAL_HOLDOUT = ROOT / "storage" / "dataset" / "eval_holdout"
    frozen_names: set[str] = set()
    if EVAL_HOLDOUT.is_dir():
        frozen_names = {p.name for p in EVAL_HOLDOUT.rglob("*.jpg")}
        print(f"eval_holdout frozen: {len(frozen_names)} crops excluded from train")
    # GLOBAL dedupe: the same physical crop can sit in several raw dirs that
    # canonicalize to the same class (e.g. BMW_3_Series + BMW_3_Series_Touring
    # -> BMW_3_Series). Processing dirs separately splits the same track across
    # train/val. Dedupe by filename so each crop is assigned exactly once.
    seen_files: set[str] = set()
    for d in sorted(os.listdir(SRC_RAW)):
        cdir = SRC_RAW / d
        if not cdir.is_dir():
            continue
        cls = canon(d, raw_keys)
        in_web = cls in set(web_classes)
        files = sorted(p for p in cdir.iterdir() if p.suffix.lower() == ".jpg")
        tracks: dict[str, list[Path]] = defaultdict(list)
        for p in files:
            if p.name in seen_files:
                continue  # already assigned from an earlier (duplicate) dir
            seen_files.add(p.name)
            if p.name in frozen_names:
                continue  # permanent eval crop — NEVER train on it
            tracks[track_of(p)].append(p)
        track_names = sorted(tracks)
        for i, t in enumerate(track_names):
            target = val_items if i % args.holdout_every == args.holdout_every - 1 \
                else train_items
            target[cls].extend(tracks[t])
        if not in_web:
            n_orphan_new += 1

    # ---- report
    print(f"{'class':38s} {'web':>5s} {'real-tr':>7s} {'val':>4s}  note")
    rows = []
    all_classes = sorted(set(web_classes) | set(train_items) | set(val_items))
    # drop junk classes from noisy API labels: 'Kia Unknown', 'Volkswagen
    # Unknown' etc. teach the model to output garbage — never train on them
    all_classes = [c for c in all_classes
                   if not re.search(r"\bUnknown\b", c) or web_img_count(c) > 0]
    for cls in all_classes:
        web_n = web_img_count(cls)
        tr_n = len(train_items.get(cls, []))
        va_n = len(val_items.get(cls, []))
        note = ""
        if web_n and tr_n:
            note = "web+real"
        elif tr_n and not web_n:
            note = "REAL-ONLY" + ("" if args.allow_new else " (skipped)")
        elif web_n < 5:
            note = "tiny-web"
        rows.append((cls, web_n, tr_n, va_n, note))
        print(f"{cls:38s} {web_n:>5d} {tr_n:>7d} {va_n:>4d}  {note}")
    skipped_real_only = [r for r in rows if r[4].startswith("REAL-ONLY") and "skipped" in r[4]]
    print(f"\nclasses: {len(all_classes)} total | "
          f"real-only skipped: {len(skipped_real_only)} | "
          f"crops held out for val: {sum(len(v) for v in val_items.values())}")

    if args.dry_run:
        print("\n(dry run — nothing written)")
        return

    # ---- write train_v6 (hardlinks; fall back to copy across filesystems)
    if OUT_TRAIN.exists():
        shutil.rmtree(OUT_TRAIN)
    if OUT_VAL.exists():
        shutil.rmtree(OUT_VAL)
    OUT_VAL.mkdir(parents=True, exist_ok=True)

    def link(src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(src, dst)
        except FileExistsError:
            pass
        except OSError:
            if not dst.exists():
                shutil.copy2(src, dst)

    def inject(files: list[Path], dst_dir: Path) -> None:
        for p in files:
            for k in range(args.oversample):
                link(p, dst_dir / f"{p.stem}__inj{k}{p.suffix}")

    n_train_cls = 0
    for cls in all_classes:
        tr_files = train_items.get(cls, [])
        va_files = val_items.get(cls, [])
        has_web = cls in web_groups
        if not has_web and not (tr_files and args.allow_new):
            continue
        out_cls_tr = OUT_TRAIN / cls
        if has_web:
            for f in sorted(web_iter(cls)):
                if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    link(f, out_cls_tr / f.name)
        inject(tr_files, out_cls_tr)
        for p in va_files:
            link(p, OUT_VAL / cls / p.name)
        n_train_cls += 1

    n_val_cls = len([d for d in OUT_VAL.iterdir() if d.is_dir()])
    tot_web = sum(1 for _ in OUT_TRAIN.rglob("*.jpg"))
    print(f"\nwrote {n_train_cls} train classes ({tot_web} imgs) -> {OUT_TRAIN}")
    print(f"wrote {n_val_cls} val classes -> {OUT_VAL}  <-- TRUE accuracy metric")


if __name__ == "__main__":
    main()
