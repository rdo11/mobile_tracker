"""
road_context.py — optional traffic-light state + speed-limit context.

Design contract (mirrors AsyncPlateReader/AsyncClassifier):
  * ALL work happens on one background worker thread; the capture loop only
    reads cached state, so nothing here can ever stall the video/recording.
  * Traffic lights come from the shared YOLO tracker (COCO class 9 added to
    vehicle_classes when road.enabled) — the loop hands crops to us.
  * Light STATE is classified by lamp-color analysis (HSV on bright pixels):
    good enough for dashcam distances, zero extra model downloads.
  * Speed limits: Phase-1 heuristic — circle candidates + digit OCR on the
    ring interior, scanned at a slow cadence. Replaced by a trained sign
    detector later; behind road.enabled so it costs nothing until tested.
  * Display rule (user requirement): light state is shown ONLY while a light
    is actually tracked (TTL); MAX limit persists until derestriction/new
    sign/hard expiry so a bad read cannot stick forever.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time

import cv2
import numpy as np
import torch

logger = logging.getLogger("mobile_tracker.road")

# OpenCV hue ranges (H 0-179)
_HUE_RED = ((0, 10), (170, 179))
_HUE_YELLOW = ((14, 38),)
_HUE_GREEN = ((44, 95),)


def _band_color(seg: np.ndarray, hue_range: tuple) -> tuple[int, float]:
    """Count saturated-bright pixels in a segment matching a hue band.
    Returns (count, fraction-of-lit)."""
    hsv = cv2.cvtColor(cv2.resize(seg, (24, 24)), cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    lit = (V > 140) & (S > 90)
    n_lit = int(lit.sum())
    if n_lit < 4:
        return 0, 0.0
    m = np.zeros_like(lit)
    for lo, hi in hue_range:
        m |= (H >= lo) & (H <= hi)
    return int((m & lit).sum()), float((m & lit).sum()) / n_lit


def classify_light(crop: np.ndarray) -> tuple[str, float]:
    """'red'|'yellow'|'green'|'' + confidence 0..1 from lamp color analysis.

    EU traffic-light boxes are TALL (3-lamp housing): red top, amber middle,
    green bottom. Global hue-voting across the whole box mixes the unlit lamps
    and breaks. We split a tall box into 3 horizontal bands and classify the
    BRIGHTEST band — that's the one lamp that's actually on. A non-tall box
    (single round lamp) falls back to whole-box voting.
    """
    try:
        if crop is None or crop.size == 0:
            return "", 0.0
        h, w = crop.shape[:2]
        tall = h > w * 1.25
        if tall and h >= 24:
            bands = [crop[int(h * i / 3):int(h * (i + 1) / 3)] for i in range(3)]
            cands = []
            for name, hue in (("red", _HUE_RED), ("yellow", _HUE_YELLOW),
                              ("green", _HUE_GREEN)):
                for seg in bands:
                    cnt, frac = _band_color(seg, hue)
                    cands.append((name, cnt, frac))
            name, cnt, frac = max(cands, key=lambda c: c[1])
            if cnt < 6:
                return "", 0.0
            return name, round(min(1.0, frac), 2)
        # non-tall: whole-box voting (original path)
        c = crop[int(h * .2):int(h * .8), int(w * .2):int(w * .8)]
        hsv = cv2.cvtColor(cv2.resize(c, (32, 32)), cv2.COLOR_BGR2HSV)
        H, S, V = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
        bright = V > 140
        lit = bright & (S > 90)
        n_lit = int(lit.sum())
        if n_lit < 12:
            return "", 0.0
        votes = {}
        for name, ranges in (("red", _HUE_RED), ("yellow", _HUE_YELLOW),
                             ("green", _HUE_GREEN)):
            m = np.zeros_like(lit)
            for lo, hi in ranges:
                m |= (H >= lo) & (H <= hi)
            votes[name] = int((m & lit).sum())
        best = max(votes, key=votes.get)
        conf = votes[best] / max(1, n_lit)
        if votes[best] < 6:
            return "", 0.0
        return best, round(min(1.0, conf), 2)
    except Exception:  # noqa: BLE001
        return "", 0.0


def find_speed_signs(frame: np.ndarray) -> list[tuple[int, int, int]]:
    """Phase-1 candidate finder: circular rings, upper 3/4 of frame.
    Returns [(cx, cy, r)] in frame coords. Heuristic — replaced by a trained
    sign detector later."""
    h, w = frame.shape[:2]
    scale = 480 / max(1, w)
    small = cv2.resize(frame, (480, int(h * scale)))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2,
                               minDist=40, param1=110, param2=42,
                               minRadius=7, maxRadius=46)
    out = []
    if circles is not None:
        for cx, cy, r in circles[0][:6]:
            if cy > small.shape[0] * 0.75:
                continue                       # signs live above the hood line
            out.append((int(cx / scale), int(cy / scale), int(r / scale)))
    return out


_GTSRB_SPEED = {
    0: 20, 1: 30, 2: 50, 3: 60, 4: 70, 5: 80, 7: 100, 8: 120,
}


class SignClassifier:
    """GTSRB-trained sign reader (models/gtsrb_signs.pt). Loaded lazily;
    when present it REPLACES digit OCR for speed-sign candidates — a real
    classifier beats reading characters one by one."""

    def __init__(self, path: str, device: str = "cpu"):
        self.available = False
        self.path = path
        try:
            import torch
            from torchvision import models
            ckpt = torch.load(path, map_location="cpu")
            self.classes = list(ckpt["classes"])
            m = models.resnet18(weights=None)
            m.fc = torch.nn.Linear(m.fc.in_features, len(self.classes))
            m.load_state_dict(ckpt["state_dict"])
            self.torch = torch
            self.model = m.to(device).eval()
            self.device = device
            self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            self.available = True
            logger.info("GTSRB sign classifier loaded (%d classes)", len(self.classes))
        except Exception as exc:  # noqa: BLE001
            logger.warning("sign classifier unavailable: %s", exc)

    def predict(self, crop: np.ndarray) -> tuple[str | None, float]:
        if not self.available or crop.size == 0:
            return None, 0.0
        img = cv2.cvtColor(cv2.resize(crop, (48, 48)), cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
        t = ((t.unsqueeze(0).to(self.device) - self.mean) / self.std)
        with self.torch.no_grad():
            probs = self.model(t).softmax(1)[0]
        idx = int(probs.argmax())
        name, conf = self.classes[idx], float(probs[idx])
        limit = _GTSRB_SPEED.get(idx)
        if "Speed limit" in name and limit is not None and conf >= 0.55:
            return limit, round(conf, 2)
        return None, round(conf, 2)


def read_digits(engine_ocr, crop: np.ndarray) -> tuple[int | None, float]:
    """OCR the interior of a suspected speed sign. Returns (limit, conf)."""
    if engine_ocr is None or crop.size == 0:
        return None, 0.0
    try:
        res = engine_ocr.readtext(crop, detail=1, allowlist="0123456789",
                                  paragraph=False)
        best, conf = None, 0.0
        for _bb, text, c in res:
            t = "".join(ch for ch in text if ch.isdigit())
            if not t:
                continue
            val = int(t)
            if 5 <= val <= 160 and c > conf:   # sane EU limits only
                best, conf = val, float(c)
        return best, round(conf, 2)
    except Exception:  # noqa: BLE001
        return None, 0.0


class AsyncRoadContext:
    """Background worker for light states + speed-sign scanning."""

    def __init__(self, cfg: dict, ocr_reader=None):
        self.cfg = cfg
        self.min_side = int(cfg.get("min_light_side", 24))
        self.light_ttl = float(cfg.get("light_ttl_secs", 1))
        self.persist = float(cfg.get("maxspeed_persist_secs", 300))
        self.scan_every = float(cfg.get("sign_scan_interval", 2.0))
        self.conf_min = float(cfg.get("light_conf_min", 0.55))
        self.ocr = ocr_reader
        self.signs = None
        sign_path = str(cfg.get("sign_model", ""))
        if cfg.get("enabled") and sign_path and os.path.exists(sign_path):
            self.signs = SignClassifier(sign_path,
                                        device="mps" if torch.backends.mps.is_available() else "cpu")

        self._results: dict[str, dict] = {}       # key -> cached classification
        self._last_req: dict[str, float] = {}
        self._last_scan = 0.0
        self._lock = threading.Lock()
        self._q: queue.Queue = queue.Queue(maxsize=16)

        # persistent speed-limit state machine
        # Safety-first: a limit is DISPLAYED only once CONFIRMED — either one
        # high-confidence read or two agreeing reads. Anything less shows as
        # explicit 'unknown' (never a guess that could constrain the driver).
        self.speed_limit: int | None = None
        self._limit_conf = 0.0
        self._confirmed = False
        self._pending_val: int | None = None   # first unconfirmed sighting
        self._pending_ts = 0.0
        self._limit_ts = 0.0

        self.available = True
        self.error = ""
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()
        logger.info("AsyncRoadContext worker started (enabled=%s)",
                    bool(cfg.get("enabled")))

    # ------------------------------------------------------------- public API
    def request_light(self, key: str, crop: np.ndarray, bbox=None) -> None:
        now = time.time()
        with self._lock:
            if now - self._last_req.get(key, 0.0) < 0.7:
                return
            self._last_req[key] = now
        try:
            self._q.put_nowait(("light", key, crop.copy(), bbox))
        except queue.Full:
            pass

    def request_sign_scan(self, frame: np.ndarray) -> None:
        """Rate-limited full-frame scan for speed signs."""
        now = time.time()
        with self._lock:
            if now - self._last_scan < self.scan_every:
                return
            self._last_scan = now
        small = cv2.resize(frame, (960, int(frame.shape[0] * 960 /
                                        frame.shape[1])))
        try:
            self._q.put_nowait(("scan", "", small, None))
        except queue.Full:
            pass

    def display_state(self) -> dict:
        """What the dashboard should show right now. Lights appear ONLY while
        fresh; MAX shows a number only when CONFIRMED, else explicit '?' —
        never an uncertain guess."""
        now = time.time()
        light, lconf, lbbox = "", 0.0, None
        with self._lock:
            fresh = {k: v for k, v in self._results.items()
                     if k.startswith("light") and now - v["ts"] <= self.light_ttl}
            if fresh:
                # freshest confident sighting wins
                k, v = max(fresh.items(), key=lambda kv: kv[1]["ts"])
                if v["conf"] >= self.conf_min:
                    light, lconf = v["label"], v["conf"]
                    lbbox = v.get("bbox")
            limit = self.speed_limit if self._confirmed else None
            if limit is not None and now - self._limit_ts > self.persist:
                limit = None                      # hard expiry: no stale lies
        out: dict = {"road_on": True}
        if light:
            out["light"] = light.upper()
            out["light_conf"] = lconf
            if lbbox:
                out["light_bbox"] = [int(v) for v in lbbox]
        out["maxspeed"] = limit                   # None -> frontend renders '?'
        return out

    # ---------------------------------------------------------------- worker
    def _run(self) -> None:
        while True:
            kind, key, img, bbox = self._q.get()
            try:
                if kind == "light":
                    label, conf = classify_light(img)
                    with self._lock:
                        self._results[f"light:{key}"] = {
                            "label": label, "conf": conf, "ts": time.time(),
                            "bbox": bbox}
                        # bound cache
                        if len(self._results) > 300:
                            cutoff = time.time() - 120
                            for k in [k for k, v in self._results.items()
                                      if v["ts"] < cutoff]:
                                self._results.pop(k, None)
                elif kind == "scan":
                    self._scan_frame(img)
            except Exception:  # noqa: BLE001
                logger.warning("road worker failed", exc_info=True)

    def _scan_frame(self, small: np.ndarray) -> None:
        for cx, cy, r in find_speed_signs(small):
            pad = int(r * 0.45)
            x1, y1 = max(0, cx - r - pad), max(0, cy - r - pad)
            x2, y2 = cx + r + pad, cy + r + pad
            crop = small[y1:y2, x1:x2]
            if crop.shape[0] < 24:
                continue
            # GTSRB classifier first (trained sign recognition); digit OCR
            # stays as fallback when the checkpoint isn't installed yet.
            if self.signs is not None and self.signs.available:
                val, conf = self.signs.predict(crop)
                if val is None:
                    continue
                self._confirm_limit(val, conf)
                return
            val, conf = read_digits(self.ocr, crop)
            if val is None or conf < 0.5:
                continue
            self._confirm_limit(val, conf)
            return

    def _confirm_limit(self, val: int, conf: float) -> None:
        now = time.time()
        # confirmation gate: single very-confident read OR two agreeing
        # reads within 20s. Disagreeing read resets the pending state.
        if conf >= 0.78:
            self.speed_limit, self._confirmed = val, True
            self._limit_ts = now
            self._pending_val = None
            logger.info("Speed limit %d (single high-conf %.2f)", val, conf)
        elif self._pending_val == val and now - self._pending_ts < 20:
            self.speed_limit, self._confirmed = val, True
            self._limit_ts = now
            self._pending_val = None
            logger.info("Speed limit %d confirmed (two agreeing reads)", val)
        else:
            if self._pending_val not in (None, val):
                logger.info("Speed candidate changed %s -> %s, resetting",
                            self._pending_val, val)
            self._pending_val, self._pending_ts = val, now
            return                                 # strongest candidate per scan
