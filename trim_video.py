"""trim_video.py — cut dead (non-road) parts out of a session recording.

Uses frame-difference motion scoring: while driving, consecutive frames differ
a lot; parked/static/"no road" footage scores near zero. Keeps active segments
(with small margins), fills short gaps, drops very short segments, and writes a
new mp4 via cv2.VideoWriter.
"""

from __future__ import annotations

import argparse
import os
import time

import cv2
import numpy as np

SCAN_STEP = 2          # analyze every Nth frame (speed)
GAP_FILL_FRAMES = 60   # merge segments closer than this (2 s @ 30 fps)
MIN_SEGMENT_FRAMES = 90  # drop segments shorter than this (3 s)
MARGIN_FRAMES = 12     # keep N frames of slack around each segment


def motion_scores(video: str, step: int = SCAN_STEP) -> list[float]:
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    prev = None
    scores: list[float] = []
    idx = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if idx % step == 0:
            small = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (320, 180))
            if prev is not None:
                scores.append(float(cv2.absdiff(small, prev).mean()))
            prev = small
        idx += 1
    cap.release()
    return scores


def active_segments(scores: list[float], step: int,
                    thresh: float = 1.0,
                    min_segment_frames: int = MIN_SEGMENT_FRAMES) -> list[tuple[int, int]]:
    """Frames with mean abs diff below `thresh` are 'dead' (parked/static/black).
    Dead footage scores ~0.2-0.4 while driving scores 2+ on dashcam streams,
    so a fixed low threshold separates them cleanly. Lower = keep more."""
    # Rolling mean over ~0.8s of video kills isolated noise spikes that
    # would otherwise keep a static segment looking "active".
    win = max(4, int(round(0.8 * 30 / step)))
    smoothed = np.convolve(scores, np.ones(win) / win, mode="same")
    active = smoothed > thresh
    # expand back to frame indices
    frame_active = np.repeat(active, step)
    segs: list[tuple[int, int]] = []
    start = None
    for i, a in enumerate(frame_active):
        if a and start is None:
            start = i
        elif not a and start is not None:
            segs.append((start, i - 1))
            start = None
    if start is not None:
        segs.append((start, len(frame_active) - 1))
    if not segs:
        return []
    # fill gaps
    merged = [segs[0]]
    for s in segs[1:]:
        if s[0] - merged[-1][1] <= GAP_FILL_FRAMES:
            merged[-1] = (merged[-1][0], s[1])
        else:
            merged.append(s)
    return [(max(0, a - MARGIN_FRAMES), b + MARGIN_FRAMES) for a, b in merged
            if b - a >= min_segment_frames]


def cut(video: str, out: str, threshold: float = 1.0,
        min_segment_frames: int = MIN_SEGMENT_FRAMES) -> tuple[int, int, list[tuple[int, int]]]:
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    scores = motion_scores(video)
    segs = active_segments(scores, SCAN_STEP, threshold, min_segment_frames)
    if not segs:
        cap.release()
        return total, 0, []

    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"cannot open writer {out}")

    kept = 0
    for (a, b) in segs:
        for i in range(a, min(b + 1, total)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, f = cap.read()
            if ok:
                writer.write(f)
                kept += 1
    writer.release()
    cap.release()
    return total, kept, segs


def main() -> None:
    parser = argparse.ArgumentParser(description="Trim dead footage from a session recording")
    parser.add_argument("video", nargs="?", default="storage/recordings/session_20260817_134751.mp4")
    parser.add_argument("--out", default="")
    parser.add_argument("--threshold", type=float, default=1.0,
                        help="motion threshold; frames below it are cut (default 1.0)")
    parser.add_argument("--min-segment", type=int, default=MIN_SEGMENT_FRAMES,
                        help="minimum kept segment length in frames (default 90 = 3 s)")
    args = parser.parse_args()

    t0 = time.time()
    total, kept, segs = cut(args.video, args.out or args.video.replace(".mp4", "_trimmed.mp4"),
                            args.threshold, args.min_segment)
    out_path = args.out or args.video.replace(".mp4", "_trimmed.mp4")
    print(f"source frames : {total} ({total / 30:.1f} s @ 30 fps)")
    print(f"kept frames   : {kept} ({kept / 30:.1f} s) — {100.0 * kept / max(1, total):.0f}% kept")
    for i, (a, b) in enumerate(segs):
        print(f"  segment {i + 1}: frames {a}-{b} = {(b - a) / 30:.1f} s")
    print(f"output        : {out_path} ({os.path.getsize(out_path) / 1e6:.0f} MB) in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
