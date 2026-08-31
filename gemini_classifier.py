"""
gemini_classifier.py — batched Gemini Vision fallback for uncertain cars.

Unlike GrokClassifier (one API call per crop), this queues uncertain crops
and sends them in ONE request once `batch_size` (default 50) is collected
or `flush_interval` seconds have passed — ~50x cheaper.

Same interface as GrokClassifier (enabled, fallback_conf, maybe_retry, get,
request, on_result) so main.py treats it identically.

API key is read from the GEMINI_API_KEY env var or ./.env
(NEVER logged or sent anywhere but generativelanguage.googleapis.com).
"""

from __future__ import annotations

import base64
import json
import logging
import os
import queue
import ssl
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from grok_classifier import _bbox_size, _crop_with_context, _load_env

logger = logging.getLogger("mobile_tracker.gemini")

try:
    import certifi

    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except Exception:  # noqa: BLE001
    _SSL_CTX = None

API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_MODEL = "gemini-3.5-flash-lite"

PROMPT = (
    "You are a vehicle recognition system. Each image in this request is a "
    "separate vehicle photo. Reply with ONLY a JSON array, no other text, "
    "with exactly one object per image, in the same order: "
    '[{"make": "brand e.g. Volkswagen", "model": "e.g. Golf 8", '
    '"year_range": "e.g. 2020-present or Unknown", "type": "car|truck|bus|van|motorcycle", '
    '"color": "one word, e.g. Blue", "confidence": 0.0-1.0}, ...]. '
    "If an image is too small, blurry, or not a vehicle, set confidence to 0.1."
)


class GeminiClassifier:
    """Async batched Gemini Vision classifier with per-track caching."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.enabled = bool(cfg.get("enabled", False))
        env = _load_env()
        self.api_key = env.get("GEMINI_API_KEY", "") or os.environ.get("GEMINI_API_KEY", "")
        self.model = cfg.get("model", DEFAULT_MODEL)
        self.min_crop_area = int(cfg.get("min_crop_area", 4096))
        self.fallback_conf = float(cfg.get("fallback_conf", 0.45))
        self.batch_size = int(cfg.get("batch_size", 50))
        self.flush_interval = float(cfg.get("flush_interval", 600))
        self.available = bool(self.api_key)
        if not self.available:
            logger.warning("Gemini classifier disabled (no GEMINI_API_KEY in env or .env)")
        self._q: queue.Queue = queue.Queue(maxsize=int(cfg.get("max_pending", 32)))
        self._results: dict[int, dict] = {}
        self._last_bbox: dict[int, tuple] = {}  # track_id -> (w, h) of last request
        self.on_result = None
        self.requested = 0
        self.responded = 0
        self.last_label = ""
        self.batches_sent = 0
        self._lock = threading.Lock()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()
        if self.enabled:
            logger.info("Gemini classifier enabled (model=%s, batch=%d, api_key=%s)",
                        self.model, self.batch_size,
                        "set" if self.available else "MISSING")

    # ------------------------------------------------------------ requests
    def request(self, track_id: int, frame: np.ndarray, bbox: tuple) -> bool:
        """Queue a crop of the vehicle for the next Gemini batch."""
        if not self.enabled or not self.available:
            return False
        crop = _crop_with_context(frame, bbox)
        if crop is None or crop.size < self.min_crop_area:
            return False
        h, w = crop.shape[:2]
        self._last_bbox[track_id] = (w, h)
        try:
            self._q.put_nowait((track_id, crop))
            self.requested += 1
            return True
        except queue.Full:
            return False

    def maybe_retry(self, track_id: int, frame: np.ndarray, bbox: tuple) -> None:
        """Re-queue a crop for a track that got no result yet (dropped queue)
        or that has grown meaningfully bigger — better signal."""
        if not self.enabled or not self.available:
            return
        with self._lock:
            cached = self._results.get(track_id)
        prev = self._last_bbox.get(track_id)
        if cached is not None and prev is not None:
            h, w = _bbox_size(frame, bbox)
            if h * w < (prev[0] * prev[1]) * 2.0:
                return
        self.request(track_id, frame, bbox)

    def get(self, track_id: int) -> dict | None:
        with self._lock:
            return self._results.get(track_id)

    # ------------------------------------------------------------- worker
    def _run(self) -> None:
        pending: list[tuple[int, np.ndarray]] = []
        last_flush = time.time()
        while True:
            try:
                item = self._q.get(timeout=5.0)
                pending.append(item)
            except queue.Empty:
                pass
            if len(pending) >= self.batch_size or (
                    pending and time.time() - last_flush >= self.flush_interval):
                self._flush(pending)
                pending = []
                last_flush = time.time()

    def _flush(self, pending: list[tuple[int, np.ndarray]]) -> None:
        items: list[tuple[int, str]] = []  # (track_id, base64 jpeg)
        for track_id, crop in pending:
            h, w = crop.shape[:2]
            if max(h, w) < 300:
                continue  # too small for vision
            ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                continue
            items.append((track_id, base64.b64encode(buf.tobytes()).decode()))
        if not items:
            return
        results = self._query([b64 for _, b64 in items])
        self.batches_sent += 1
        if not results:
            logger.warning("Gemini batch %d: no parseable results (%d imgs)",
                           self.batches_sent, len(items))
            return
        for (track_id, _b64), res in zip(items, results):
            if not res:
                continue
            self.responded += 1
            self.last_label = res.get("label", "")
            logger.info("Gemini #%d: %s (conf %.2f, batch %d)", track_id,
                        res.get("label"), res.get("confidence", 0), self.batches_sent)
            cb = self.on_result
            if cb:
                try:
                    cb(track_id, res)
                except Exception:  # noqa: BLE001
                    logger.warning("Gemini on_result callback failed", exc_info=True)
            with self._lock:
                self._results[track_id] = res
                if len(self._results) > 400:
                    self._results.clear()

    def _query(self, b64s: list[str]) -> list[dict | None] | None:
        parts: list[dict] = [{"text": PROMPT}]
        for b64 in b64s:
            parts.append({"inline_data": {"mime_type": "image/jpeg", "data": b64}})
        payload = {"contents": [{"parts": parts}]}
        url = f"{API_BASE}/{self.model}:generateContent"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key},
        )
        try:
            with urllib.request.urlopen(req, timeout=120, context=_SSL_CTX) as resp:
                data = json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            logger.warning("Gemini API HTTP %s: %s", exc.code, exc.read().decode()[:300])
            return None
        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            return None
        return self._parse_array(text, len(b64s))

    @staticmethod
    def _parse_array(text: str, n: int) -> list[dict | None] | None:
        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
        start, end = text.find("["), text.rfind("]")
        if start < 0 or end <= start:
            return None
        try:
            arr = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
        if not isinstance(arr, list):
            return None
        out: list[dict | None] = []
        for obj in arr[:n]:
            if not isinstance(obj, dict):
                out.append(None)
                continue
            try:
                confidence = float(obj.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            make = str(obj.get("make", "")).strip()
            model = str(obj.get("model", "")).strip()
            label = " ".join(p for p in (make, model) if p) or "Unknown"
            out.append({
                "make": make,
                "model": model,
                "label": label,
                "year_range": str(obj.get("year_range", "Unknown")).strip(),
                "type": str(obj.get("type", "")).strip(),
                "color": str(obj.get("color", "")).strip(),
                "confidence": confidence,
            })
        while len(out) < n:
            out.append(None)
        return out