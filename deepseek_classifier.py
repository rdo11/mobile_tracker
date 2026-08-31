"""
deepseek_classifier.py — DeepSeek Vision vehicle make/model recognition.

Sends one crop per tracked vehicle to the DeepSeek API (async worker thread so the
capture loop is never blocked), caches the result per track_id, and collects
labeled crops into a local dataset for future fine-tuning.

DeepSeek Vision uses the standard OpenAI chat-completions format with base64
image_url blocks — same shape as Grok, just a different endpoint/model.

API key is read from the DEEPSEEK_API_KEY env var or ./.env
(NEVER logged or sent anywhere but api.deepseek.com).
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

logger = logging.getLogger("mobile_tracker.deepseek")

try:
    import certifi

    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except Exception:  # noqa: BLE001
    _SSL_CTX = None

ENDPOINT = "https://api.deepseek.com/chat/completions"
MODEL_FALLBACKS = ["deepseek-v4-flash-vision-exp"]

PROMPT = (
    "You are a vehicle recognition system. Identify the vehicle in this photo "
    "and reply with ONLY a JSON object, no other text: "
    '{"make": "brand e.g. Volkswagen", "model": "e.g. Golf 8", '
    '"year_range": "e.g. 2020-present or 2013-2020 or Unknown", '
    '"type": "car|truck|semi|bus|van|motorcycle", '
    '"color": "one word, e.g. Blue", "confidence": 0.0-1.0}. '
    "If the image is too small, blurry or not a vehicle, set confidence to 0.1."
)


class DeepSeekClassifier:
    """Async DeepSeek Vision classifier with per-track caching."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.enabled = bool(cfg.get("enabled", False))
        env = _load_env()
        self.api_key = env.get("DEEPSEEK_API_KEY", "") or os.environ.get("DEEPSEEK_API_KEY", "")
        self.model = cfg.get("model", MODEL_FALLBACKS[0])
        self.min_crop_area = int(cfg.get("min_crop_area", 4096))
        self.fallback_conf = float(cfg.get("fallback_conf", 0.45))
        self.save_dataset = bool(cfg.get("save_dataset", True))
        self.dataset_dir = Path(cfg.get("dataset_dir", "storage/dataset"))
        self.min_confidence = float(cfg.get("min_confidence", 0.7))
        self.min_save_long_side = int(cfg.get("min_save_long_side", 200))
        self.max_saves_per_track = int(cfg.get("max_saves_per_track", 4))
        self.available = bool(self.api_key)
        if not self.available:
            logger.warning("DeepSeek classifier disabled (no DEEPSEEK_API_KEY in env or .env)")
        self._q: queue.Queue = queue.Queue(maxsize=int(cfg.get("max_pending", 8)))
        self._results: dict[int, dict] = {}
        self._saved: dict[int, int] = {}  # track_id -> times saved already
        self._last_bbox: dict[int, tuple] = {}  # track_id -> (w, h) of last request
        self.on_result = None  # optional callback(track_id, result) for late updates
        self.requested = 0
        self.responded = 0
        self.last_label = ""
        self._lock = threading.Lock()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()
        if self.enabled:
            logger.info("DeepSeek classifier enabled (model=%s, api_key=%s)", self.model,
                        "set" if self.available else "MISSING")

    # ------------------------------------------------------------ requests
    def request(self, track_id: int, frame: np.ndarray, bbox: tuple) -> bool:
        """Queue a crop of the vehicle for DeepSeek. Uses a padded region around
        the detection box so distant cars get more context. Returns True if
        queued."""
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
        """Re-queue a crop for a track that got no result yet (dropped queue),
        or that has grown meaningfully bigger — better data for fine-tuning.
        Called on every classify window."""
        if not self.enabled or not self.available:
            return
        with self._lock:
            cached = self._results.get(track_id)
        prev = self._last_bbox.get(track_id)
        if cached is not None and prev is not None:
            h, w = _bbox_size(frame, bbox)
            if h * w < (prev[0] * prev[1]) * 2.0:
                return
        if cached is not None and self._saved.get(track_id, 0) >= self.max_saves_per_track:
            return
        self.request(track_id, frame, bbox)

    def get(self, track_id: int) -> dict | None:
        with self._lock:
            return self._results.get(track_id)

    # ------------------------------------------------------------- worker
    def _run(self) -> None:
        while True:
            track_id, crop = self._q.get()
            try:
                result = self._query(crop)
            except Exception as exc:  # noqa: BLE001
                logger.warning("DeepSeek request failed: %s", exc)
                continue
            if not result:
                continue
            self.responded += 1
            self.last_label = result.get("label", "")
            logger.info("DeepSeek #%d: %s (conf %.2f, %s)", track_id,
                        result.get("label"), result.get("confidence", 0), result.get("model_used", ""))
            cb = self.on_result
            if cb:
                try:
                    cb(track_id, result)
                except Exception:  # noqa: BLE001
                    logger.warning("DeepSeek on_result callback failed", exc_info=True)
            with self._lock:
                self._results[track_id] = result
                if len(self._results) > 400:
                    self._results.clear()
                    self._saved.clear()
            if self.save_dataset and result.get("confidence", 0) >= self.min_confidence:
                h, w = crop.shape[:2]
                if max(h, w) < self.min_save_long_side:
                    continue
                with self._lock:
                    if self._saved.get(track_id, 0) >= self.max_saves_per_track:
                        continue
                    self._saved[track_id] = self._saved.get(track_id, 0) + 1
                self._save_sample(track_id, crop, result)

    def _query(self, crop: np.ndarray) -> dict | None:
        h, w = crop.shape[:2]
        long_side = max(h, w)
        if long_side < 512:
            scale = 512 / long_side
            crop = cv2.resize(crop, (int(w * scale), int(h * scale)),
                              interpolation=cv2.INTER_LANCZOS4)
        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return None
        b64 = base64.b64encode(buf.tobytes()).decode()
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": "Identify the vehicle."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ]},
            ],
            "thinking": {"type": "disabled"},
            "max_tokens": 120,
            "temperature": 0.0,
        }
        req = urllib.request.Request(
            ENDPOINT,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30, context=_SSL_CTX) as resp:
                data = json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            body = exc.read().decode()[:300]
            if "model" in body.lower() and ("not" in body.lower() or "invalid" in body.lower()):
                self._try_next_model()
            logger.warning("DeepSeek API HTTP %s: %s", exc.code, body)
            return None
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            return None
        result = self._parse(content)
        if result:
            result["model_used"] = self.model
        return result

    def _query_batch(self, b64s: list[str]) -> list[dict | None] | None:
        """One request, N images. Mirrors GeminiClassifier._query's contract
        exactly (same prompt, same parse) so label_crops.py can drive either
        provider interchangeably for dataset labeling / cross-checking."""
        from gemini_classifier import PROMPT, GeminiClassifier

        content: list[dict] = [{"type": "text", "text": PROMPT}]
        for b64 in b64s:
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            # Labeling needs no chain-of-thought: thinking is ON by default
            # at effort=high and would burn max_tokens in reasoning_content,
            # returning an EMPTY content field.
            "thinking": {"type": "disabled"},
            # generous ceiling: truncated JSON arrays were the silent killer
            # of large labeling batches (finish_reason=length -> unparseable)
            "max_tokens": 250 * len(b64s) + 500,
            "temperature": 0.0,
        }
        req = urllib.request.Request(
            ENDPOINT,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=180, context=_SSL_CTX) as resp:
                data = json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            body = exc.read().decode()[:300]
            if exc.code == 400 and ("model" in body.lower()
                                    and ("not" in body.lower() or "invalid" in body.lower())):
                self._try_next_model()
                logger.warning("model %s rejected, falling back (%s)", self.model, body[:120])
            logger.warning("DeepSeek batch HTTP %s: %s", exc.code, body)
            return None
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            return None
        return GeminiClassifier._parse_array(text, len(b64s))

    def _try_next_model(self) -> None:
        try:
            idx = MODEL_FALLBACKS.index(self.model)
            if idx + 1 < len(MODEL_FALLBACKS):
                self.model = MODEL_FALLBACKS[idx + 1]
                logger.info("DeepSeek model fallback -> %s", self.model)
        except ValueError:
            pass

    @staticmethod
    def _parse(content: str) -> dict | None:
        text = content.strip()
        try:
            if text.startswith("```"):
                text = text.strip("`")
                if text.startswith("json"):
                    text = text[4:]
            obj = json.loads(text)
        except json.JSONDecodeError:
            start, end = text.find("{"), text.rfind("}")
            if start < 0 or end <= start:
                return None
            try:
                obj = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None
        if not isinstance(obj, dict):
            return None
        try:
            confidence = float(obj.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        make = str(obj.get("make", "")).strip()
        model = str(obj.get("model", "")).strip()
        label = " ".join(p for p in (make, model) if p) or "Unknown"
        return {
            "make": make,
            "model": model,
            "label": label,
            "year_range": str(obj.get("year_range", "Unknown")).strip(),
            "type": str(obj.get("type", "")).strip(),
            "color": str(obj.get("color", "")).strip(),
            "confidence": confidence,
        }

    # ----------------------------------------------------------- dataset
    def _save_sample(self, track_id: int, crop: np.ndarray, result: dict) -> None:
        try:
            label = result["label"].replace("/", "_").replace(" ", "_")
            folder = self.dataset_dir / label
            folder.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            path = folder / f"{stamp}_t{track_id}.jpg"
            cv2.imwrite(str(path), crop)
            with self._lock:
                self._saved[track_id] = self._saved.get(track_id, 0) + 1
        except Exception as exc:  # noqa: BLE001
            logger.debug("Dataset save failed: %s", exc)
