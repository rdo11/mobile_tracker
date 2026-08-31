"""
wikimedia_scraper.py — collect vehicle photos from Wikimedia Commons for
European makes/models common in Denmark/Germany.

Labels come from category names (free, no API cost). Output goes to
storage/dataset/wikimedia/<make_model>/<n>.jpg (long side resized to 512).

Usage:
    python wikimedia_scraper.py --all              # full curated list
    python wikimedia_scraper.py "Volkswagen" "Golf"  # single model
"""

from __future__ import annotations

import logging
import re
import ssl
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path

import cv2

try:
    import certifi

    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except Exception:  # noqa: BLE001
    _SSL_CTX = None

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("scraper")

API = "https://commons.wikimedia.org/w/api.php"
UA = "MobileTrackerDatasetBuilder/1.0 (contact: local@user)"

OUT = Path("storage/dataset/wikimedia")
MAX_PER_MODEL = 100
MAX_SIZE = 512
OUT_OVERRIDE: Path | None = None  # set by --out; lets us write to the 4TB HDD

MODELS: dict[str, list[str]] = {
    # ---- Model-level only (no generations / model years / chassis codes).
    # Generations merge at training time anyway; parent categories give more
    # variety per class and far fewer crawls. ~100 images per class is plenty
    # for a small classifier. Truck/bike brands are NOT scraped — YOLO names
    # them coarse (Truck / Motorbike / ...) live, no make/model needed.
    "Volkswagen": ["Golf", "Polo", "Passat", "T-Roc", "Tiguan", "Touran",
                   "Up", "ID.3", "ID.4", "ID.5", "ID.7", "ID. Buzz",
                   "Caddy", "Transporter", "Crafter"],
    "BMW": ["1 Series", "2 Series", "3 Series", "4 Series", "5 Series",
            "6 Series", "7 Series", "8 Series",
            "X1", "X2", "X3", "X4", "X5", "X6", "X7", "i3", "i4", "iX"],
    "Mercedes-Benz": ["A-Class", "B-Class", "C-Class", "E-Class", "S-Class",
                      "GLA", "GLB", "GLC", "GLE", "GLS", "G-Class", "CLA", "CLS",
                      "EQA", "EQB", "EQC", "EQE", "EQS",
                      "Sprinter", "Vito", "V-Class", "Citan"],
    "Audi": ["A1", "A3", "A4", "A5", "A6", "A7", "A8", "Q2", "Q3", "Q5", "Q7",
             "Q8", "e-tron", "Q4 e-tron", "Q6 e-tron", "e-tron GT"],
    "Porsche": ["911", "Macan", "Macan EV", "Cayenne", "Taycan",
                "Panamera", "718"],
    "Opel": ["Corsa", "Astra", "Mokka", "Insignia", "Grandland", "Combo",
             "Vivaro", "Crossland"],
    "Škoda": ["Octavia", "Fabia", "Superb", "Kodiaq", "Karoq", "Scala", "Kamiq",
              "Enyaq", "Elroq", "Citigo"],
    "Seat": ["Leon", "Ibiza", "Arona", "Ateca", "Tarraco", "Alhambra"],
    "Cupra": ["Formentor", "Born", "Tavascan", "Ateca", "Leon"],
    "Peugeot": ["208", "308", "2008", "3008", "5008", "508", "Partner", "Boxer", "406"],
    "Citroën": ["C3", "C4", "C5 Aircross", "Berlingo", "Jumper"],
    "Renault": ["Clio", "Captur", "Megane", "Scenic", "Talisman", "Twingo",
                "Zoe", "Kangoo", "Master", "Trafic", "Arkana"],
    "Dacia": ["Sandero", "Duster", "Logan", "Jogger", "Spring"],
    "Fiat": ["500", "Panda", "Tipo", "500X", "Ducato"],
    "Alfa Romeo": ["Giulia", "Stelvio", "Tonale"],
    "Volvo": ["XC40", "XC60", "XC90", "S60", "S90", "V60", "V90",
              "EX30", "EX90", "S40", "V40"],
    "Saab": ["9-3", "9-5", "900", "9000"],
    "Polestar": ["2", "3", "4"],
    "Mini": ["Cooper", "Countryman", "Cooper SE"],
    "Smart": ["Fortwo", "Forfour", "#1"],
    "Ford": ["Focus", "Fiesta", "Kuga", "Puma", "Mondeo", "Galaxy", "S-Max",
             "Transit Custom", "Transit", "Ranger"],
    "Land Rover": ["Range Rover", "Range Rover Evoque", "Range Rover Velar",
                   "Discovery", "Discovery Sport", "Defender"],
    "Jaguar": ["F-Pace", "XF", "E-Pace", "I-Pace", "F-Type"],
    # ---- EVs (model-level, distinct models) --------------------------------
    "Hyundai": ["Ioniq 5", "Ioniq 6", "Ioniq 9", "Kona", "Kona Electric"],
    "Kia": ["Ceed", "Rio", "EV3", "EV6", "EV9", "Niro EV"],
    "Tesla": ["Model 3", "Model Y", "Model S", "Model X", "Cybertruck"],
    # ---- Asian / American brands common in DK/DE ---------------------------
    "Toyota": ["Yaris", "Corolla", "Aygo", "Auris", "C-HR", "RAV4", "Prius", "Camry"],
    "Nissan": ["Qashqai", "Leaf", "Juke", "Micra", "X-Trail", "Note"],
    "Mazda": ["2", "3", "6", "CX-3", "CX-5", "MX-5"],
    "Honda": ["Civic", "Jazz", "CR-V", "HR-V"],
    "Suzuki": ["Swift", "Vitara", "Ignis", "S-Cross", "Jimny", "SX4"],
    "Subaru": ["Outback", "Forester", "XV", "Impreza"],
    "Mitsubishi": ["ASX", "Outlander", "Space Star"],
    "Lexus": ["NX", "RX", "UX", "ES"],
    "Jeep": ["Renegade", "Compass", "Cherokee", "Wrangler"],
    "MG": ["4", "ZS", "HS"],
}


def api(params: dict) -> dict:
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=90, context=_SSL_CTX) as resp:
                return json_load(resp.read())
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < 3:
                time.sleep(15.0 * (attempt + 1))
                continue
            raise
        except (urllib.error.URLError, TimeoutError):
            if attempt < 3:
                time.sleep(10.0 * (attempt + 1))
                continue
            raise


def _fold(s: str) -> str:
    """ASCII-fold a string: 'Mégane' -> 'Megane', 'Škoda' -> 'Skoda'."""
    return "".join(c for c in unicodedata.normalize("NFKD", s)
                   if not unicodedata.combining(c))


def _norm(title: str) -> set[str]:
    """Significant words of a category title (punctuation stripped, accents
    folded so 'Megane' matches 'Mégane')."""
    return set(re.findall(r"[a-z0-9]+", _fold(title).lower()))


# Commons uses a different name than our MODELS entry in a few cases.
ALIASES: dict[tuple[str, str], str] = {
    ("Land Rover", "Range Rover"): "Range Rover",
    ("Land Rover", "Range Rover Evoque"): "Range Rover Evoque",
    ("Mazda", "2"): "Mazda2",
    ("Mazda", "3"): "Mazda3",
    ("Mazda", "6"): "Mazda6",
    ("MG", "4"): "MG4 EV",
    ("MG", "ZS"): "MG ZS EV",
    ("Porsche", "Macan EV"): "Porsche Macan (XAB)",
    ("Renault", "Megane"): "Renault Mégane",
    ("Renault", "Scenic"): "Renault Scénic",
    # Škoda/Citroën: ASCII categories are EMPTY shells on Commons — the real
    # generation structure lives under the accented names.
    ("Škoda", "Octavia"): "Škoda Octavia",
    ("Škoda", "Fabia"): "Škoda Fabia",
    ("Škoda", "Superb"): "Škoda Superb",
    ("Škoda", "Kodiaq"): "Škoda Kodiaq",
    ("Škoda", "Karoq"): "Škoda Karoq",
    ("Škoda", "Scala"): "Škoda Scala",
    ("Škoda", "Kamiq"): "Škoda Kamiq",
    ("Škoda", "Enyaq"): "Škoda Enyaq",
    ("Škoda", "Elroq"): "Škoda Elroq",
    ("Škoda", "Citigo"): "Škoda Citigo",
    ("Citroën", "C3"): "Citroën C3",
    ("Citroën", "C4"): "Citroën C4",
    ("Citroën", "C5 Aircross"): "Citroën C5 Aircross",
    ("Citroën", "Berlingo"): "Citroën Berlingo",
    ("Citroën", "Jumper"): "Citroën Jumper",
}

# Titles that pass the word check but are NOT car categories
# (e.g. 'MG3 machine guns', 'User mg-4', 'MG Bhe 4/8').
NOISE_TOKENS = {"user", "people", "gun", "machine", "beretta",
                "locomotive", "disambiguation", "train", "aircraft"}


def resolve_category(make: str, model: str) -> str:
    """Find the real Commons category for '<make> <model>' via search
    (naming varies: 'Golf VII', 'Octavia II', 'A4 (B8)'...).

    Requires EVERY query word to appear in the candidate title, so factory
    categories like 'Volkswagen Poznań' are rejected for 'Caddy III'.
    """
    alias = ALIASES.get((make, model))
    if alias:
        logger.info("    alias -> %s", alias)
        return alias
    for query in (f"{make} {model}", model):
        logger.info("    searching for '%s'...", query)
        data = api({"action": "query", "format": "json", "list": "search",
                    "srsearch": f"intitle:{query}", "srnamespace": "14", "srlimit": "10"})
        time.sleep(1.0)
        want = _norm(f"{make} {model}")
        best = ""
        for r in data.get("query", {}).get("search", []):
            title = r["title"].removeprefix("Category:")
            toks = _norm(title)
            if toks & NOISE_TOKENS:
                continue
            if want <= toks:
                # prefer the parent category (fewest words) over a
                # generation subcategory like 'BMW 3 Series (G20)'
                if not best or len(toks) < len(_norm(best)):
                    best = title
        if best:
            return best
    return f"{make} {model}"


def json_load(data: bytes) -> dict:
    import json
    return json.loads(data)


def category_files(category: str, depth: int = 0, seen: set | None = None,
                   cap: int = 160) -> list[str]:
    """Image URLs in a Commons category (+ one level of subcategories),
    stopping once `cap` URLs are collected — big categories have thousands
    of files and full pagination would take forever."""
    if seen is None:
        seen = set()
    if category in seen or depth > 3:
        return []
    seen.add(category)
    urls: list[str] = []
    cont: dict | None = None
    while len(urls) < cap:
        params = {
            "action": "query", "format": "json",
            "generator": "categorymembers",
            "gcmtitle": f"Category:{category}",
            "gcmtype": "file", "gcmlimit": "500",
            "prop": "imageinfo", "iiprop": "url|size",
            "iiurlwidth": "640",
        }
        if cont:
            params.update(cont)
        data = api(params)
        time.sleep(1.5)  # be polite to the API
        pages = data.get("query", {}).get("pages", {})
        for pg in pages.values():
            ii = pg.get("imageinfo", [{}])[0]
            w, h = ii.get("width", 0), ii.get("height", 0)
            if w < 150 or h < 150:
                continue
            urls.append(ii.get("thumburl") or ii["url"])
        cont = data.get("continue")
        if not cont:
            break
    if len(urls) >= cap:
        return urls
    # one level of subcategories (e.g. generation-specific categories)
    params = {
        "action": "query", "format": "json",
        "generator": "categorymembers",
        "gcmtitle": f"Category:{category}",
        "gcmtype": "subcat", "gcmlimit": "500",
    }
    data = api(params)
    time.sleep(1.5)
    for pg in data.get("query", {}).get("pages", {}).values():
        urls += category_files(pg["title"].split(":", 1)[1], depth + 1, seen, cap - len(urls))
        if len(urls) >= cap:
            break
    return urls


def download(url: str, dest: Path) -> bool:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=90, context=_SSL_CTX) as resp:
            raw = resp.read()
    except (urllib.error.URLError, TimeoutError):
        return False
    if len(raw) < 2000:
        return False
    import numpy as np
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return False
    h, w = img.shape[:2]
    if max(h, w) > MAX_SIZE:
        scale = MAX_SIZE / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    cv2.imwrite(str(dest), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return True


def scrape(make: str, model: str) -> int:
    label = _fold(f"{make}_{model}").replace(" ", "_").replace("/", "_")
    try:
        cat = resolve_category(make, model)
        if cat != f"{make} {model}":
            label = _fold(cat).replace(" ", "_").replace("/", "_")
            logger.info("  [%s %s] resolved -> %s", make, model, cat)
    except Exception:  # noqa: BLE001
        cat = f"{make} {model}"
    folder = OUT / label
    folder.mkdir(parents=True, exist_ok=True)
    logger.info("  [%s %s] category: %s", make, model, cat)
    existing = {p.stem for p in folder.glob("*.jpg")}
    try:
        urls = category_files(cat)
    except Exception as exc:  # noqa: BLE001
        logger.warning("  [%s %s] category failed: %s", make, model, exc)
        return 0
    got = 0
    for i, url in enumerate(urls):
        if got >= MAX_PER_MODEL:
            break
        if i % 8 == 0:
            time.sleep(1.5)  # be polite
        name = url.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        if name in existing:
            got += 1
            continue
        if download(url, folder / f"{name}.jpg"):
            got += 1
        time.sleep(0.4)
    logger.info("  [%s %s] saved %d (skipped %d existing)",
                make, model, got, len(existing))
    return got


def main() -> None:
    global OUT  # noqa: PLW0603
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="scrape the full curated list")
    ap.add_argument("--out", default=None,
                    help="output dir (default storage/dataset/wikimedia; use the "
                         "4TB HDD for big runs, e.g. --out /Volumes/4TB/datasets/wikimedia)")
    ap.add_argument("makes_models", nargs="*", help="make model pairs")
    args = ap.parse_args()
    if args.out:
        OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    resume = "--resume" in args.makes_models
    jobs: list[tuple[str, str]] = []
    if args.all:
        jobs = [(m, mod) for m, mods in MODELS.items() for mod in mods]
    elif len(args.makes_models) >= 2:
        resume = "--resume" in args.makes_models
        models = [a for a in args.makes_models if a != "--resume"]
        if len(models) >= 2:
            jobs = [(models[0], models[1])]
    else:
        print(__doc__)
        return
    total = 0
    skipped = 0
    for i, (make, model) in enumerate(jobs):
        if resume:
            folder = OUT / _fold(f"{make}_{model}").replace(" ", "_").replace("/", "_")
            if folder.exists() and list(folder.glob("*.jpg")):
                skipped += 1
                continue
        if i > 0:
            time.sleep(3.0)
        total += scrape(make, model)
    logger.info("DONE: %d models (%d resumed/skipped), %d images saved to %s",
                len(jobs), skipped, total, OUT)


if __name__ == "__main__":
    main()