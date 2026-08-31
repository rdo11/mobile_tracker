"""
validate_dataset.py — API validation gate for scraped images.

Batch-checks scraped images with the Gemini vision model ("is this actually a
photo of a <class>?") and deletes/moves the clearly-wrong ones. Runs BEFORE a
new class is added to the training dataset.

Cost estimate: ~10k images / 50 per call = ~200 calls, well under $1 on
flash-lite. Set GEMINI_API_KEY in .env.

Usage:
    python validate_dataset.py --data storage/dataset/wikimedia \
                               [--dry-run] [--threshold 0.5]
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("pip install google-genai")
    sys.exit(1)


PROMPT = """You are a data-quality filter for a vehicle make/model dataset.
An image will be attached along with its claimed class label. Answer in JSON:
{"is_correct": true/false, "reason": "one short phrase"}
"is_correct" is true only if the image clearly shows THE claimed car model
(make + model). Return false for: wrong car model, interiors, dashboards,
engine bays, close-up badges, concept cars, drawings, or street scenes where
the claimed car is not the main subject."""


def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would be deleted, don't delete")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="fraction of 'wrong' needed to flag a class")
    ap.add_argument("--batch", type=int, default=50)
    args = ap.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY not set (put it in .env)")
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    total_wrong = 0
    for cls_dir in sorted(args.data.iterdir()):
        if not cls_dir.is_dir():
            continue
        files = [p for p in sorted(cls_dir.glob("*"))
                 if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        if not files:
            continue
        wrong = []
        for i in range(0, len(files), args.batch):
            chunk = files[i:i + args.batch]
            parts = [types.Part(text=PROMPT),
                     types.Part(text=f"claimed class: {cls_dir.name}")]
            for p in chunk:
                parts.append(types.Part(
                    inline_data=types.Blob(mime_type="image/jpeg",
                                           data=encode_image(p))))
            try:
                resp = client.models.generate_content(
                    model="gemini-3.5-flash-lite",
                    contents=types.Content(parts=parts))
                verdicts = json.loads(resp.text.strip().strip("`"))
                for p, v in zip(chunk, verdicts):
                    if not v.get("is_correct", False):
                        wrong.append(p)
            except Exception as exc:  # noqa: BLE001
                print(f"  api error on {cls_dir.name} chunk {i}: {exc}",
                      flush=True)
        frac = len(wrong) / max(1, len(files))
        flag = "FLAG" if frac > args.threshold else "ok  "
        print(f"{flag} {cls_dir.name:35s} {len(wrong):3d}/{len(files):3d} wrong "
              f"({frac:.0%})", flush=True)
        if flag == "FLAG" and not args.dry_run:
            for p in wrong:
                p.unlink(missing_ok=True)
            total_wrong += len(wrong)
    print(f"done. removed {total_wrong} images" if not args.dry_run
          else f"done (dry run). would remove {total_wrong}")


if __name__ == "__main__":
    main()