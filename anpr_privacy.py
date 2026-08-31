"""
anpr_privacy.py — license-plate extraction + PRIVACY ENFORCEMENT ENGINE.

GDPR rule that this module enforces:
  * Plate characters are read ONLY in memory (OCR on the raw frame) for an
    internal lookup / session log.
  * The OUTPUT frame buffer is anonymized with pixelation or Gaussian blur
    over every detected plate region BEFORE it is written to disk or sent
    to the WebSocket stream. The saved video therefore never contains
    readable plates.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import re
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger("mobile_tracker.privacy")

PLATE_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


# ------------------------------------------------------------- plate country
# EU plate format heuristics: given an OCR'd plate string (no separators),
# guess the issuing country from its character pattern. Length + structure
# alone can't be 100% certain (countries share patterns), so we return a
# confidence and the closest match. Ordered: more distinctive patterns first.
PLATE_PATTERNS: list[tuple[str, list[tuple[str, float]]]] = [
    # (regex on cleaned plate, [(country, confidence), ...] — best wins)
    (r"^[A-Z]{1,3}[A-Z]{1,2}\d{1,4}[EH]$", [("DE", 0.9)]),       # B AB 1E/H — German e/historic
    (r"^[A-Z]{1,3}[A-Z]{1,2}\d{1,4}$", [("DE", 0.8)]),           # KA AB 123 — German city+district
    (r"^\d{1,2}[A-Z]{2}\d{1,4}$", [("CZ", 0.8)]),               # 1AB 1234 — Czech
    (r"^\d{3}[A-Z]{2}$", [("SK", 0.6)]),                        # 123 AA — Slovak 2023+
    (r"^[A-Z]{2}\d{3}[A-Z]{2}$", [("SK", 0.9), ("IT", 0.7)]),   # BA 123AA vs AB 123CD
    (r"^[A-Z]{1,2}\d{3,5}[A-Z]{1,2}$", [("AT", 0.7)]),          # W 12345 AB — Austrian
    (r"^[A-Z]{1,3}\d{5}$", [("PL", 0.8)]),                      # XYZ 12345 — Polish
    (r"^[A-Z]{3}\d{3}$", [("HU", 0.8), ("DE", 0.6)]),           # ABC 123 — Hungarian
    (r"^\d{4}[A-Z]{3}$", [("ES", 0.85)]),                       # 1234 ABC — Spanish
    (r"^[A-Z]{2}\d{2}[A-Z]{3}$", [("UK", 0.8)]),                # AB12 CDE — UK
    (r"^\d{1,2}[A-Z]{1,3}\d{1,4}$", [("BE", 0.6)]),             # 1 ABC 234 — Belgian
    (r"^\d{4}[A-Z]{2}$", [("FR", 0.5)]),                        # 1234 AB — French older
    (r"^\d{2}[A-Z]{2}\d{2}$", [("FR", 0.6)]),                   # 12 AB 34 — French
    (r"^[A-Z]{2}\d{5}$", [("SE", 0.6), ("PL", 0.5)]),           # AB 12345 — Swedish
    (r"^[A-Z]{1,3}\d{1,4}$", [("DE", 0.85), ("NL", 0.6)]),      # B1234 / KA1234 / AB1234
    (r"^[A-Z]{2}\d{3}$", [("NL", 0.5)]),                        # AB 123 — Dutch short
    (r"^[A-Z]{1,2}\d{2,5}$", [("CH", 0.5), ("DE", 0.4)]),       # ZH 12345 — Swiss canton
]


def detect_plate_country(text: str) -> tuple[str, float]:
    """Guess the issuing country of an OCR'd plate (no separators needed).

    Returns (country_code, confidence). If no pattern matches, returns
    ("Unknown", 0.0).
    """
    cleaned = re.sub(r"[^A-Z0-9]", "", text.upper())
    if len(cleaned) < 4 or len(cleaned) > 8:
        return "Unknown", 0.0
    best, best_conf = "Unknown", 0.0
    for pattern, candidates in PLATE_PATTERNS:
        if re.match(pattern, cleaned):
            for country, conf in candidates:
                if conf > best_conf:
                    best, best_conf = country, conf
    return best, best_conf


@dataclass
class PlateRegion:
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2 (image coords)
    source: str = "contour"          # "yolo" | "contour"
    confidence: float = 0.0
    text: str = ""                   # in-memory OCR result, never streamed
    ocr_confidence: float = 0.0
    country: str = ""                # guessed issuing country (DE/CZ/SK/...)
    country_confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "bbox": list(self.bbox),
            "source": self.source,
            "confidence": round(self.confidence, 3),
            "ocr_confidence": round(self.ocr_confidence, 3),
            "country": self.country,
            "country_confidence": round(self.country_confidence, 3),
        }


class PlateDatabase:
    """Tiny JSON plate database for in-memory matching, e.g.
    {"ABC1234": {"owner": "internal-fleet", "notes": ""}}"""

    def __init__(self, path: str = ""):
        self.path = path
        self._data: dict = {}
        self._normalized: dict[str, str] = {}
        self.load()

    def load(self) -> None:
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                self._data = json.load(fh)
            self._normalized = {self._norm(k): k for k in self._data}
            logger.info("Plate database loaded: %d entries", len(self._normalized))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Plate database unreadable (%s): %s", self.path, exc)

    @staticmethod
    def _norm(plate: str) -> str:
        return re.sub(r"[^A-Z0-9]", "", plate.upper())

    def lookup(self, plate_text: str) -> dict | None:
        key = self._norm(plate_text)
        if not key:
            return None
        original = self._normalized.get(key)
        return self._data[original] if original else None


class PrivacyEngine:
    """Finds plates (YOLO model or contour heuristic), reads them in memory
    via OCR, then blurs them on the output buffer."""

    def __init__(self, anpr_cfg: dict, privacy_cfg: dict):
        self.cfg = anpr_cfg
        self.priv = privacy_cfg
        self.plate_model = None
        self.ocr_engine: str = anpr_cfg.get("ocr_engine", "easyocr").lower()
        self.ocr_reader = None
        self.ocr_available = False
        self.plate_db = PlateDatabase(anpr_cfg.get("plate_database", ""))
        self._plate_model_loaded = False

    # ------------------------------------------------------------ plate finder
    def load(self) -> None:
        model_path = self.cfg.get("plate_model", "")
        if model_path:
            try:
                from ultralytics import YOLO

                self.plate_model = YOLO(model_path)
                self._plate_model_loaded = True
                logger.info("Plate YOLO model loaded: %s", model_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Plate model load failed, using contours: %s", exc)

        if self.cfg.get("ocr_engine", "none") != "none":
            self._load_ocr()

    def _load_ocr(self) -> None:
        try:
            if self.ocr_engine == "easyocr":
                import easyocr  # noqa: PLC0415 — lazy: heavy dependency

                self.ocr_reader = easyocr.Reader(
                    self.cfg.get("ocr_lang", ["en"]), gpu=False, verbose=False
                )
            elif self.ocr_engine == "paddleocr":
                from paddleocr import PaddleOCR  # noqa: PLC0415

                self.ocr_reader = PaddleOCR(
                    use_angle_cls=True, lang="en", show_log=False
                )
            self.ocr_available = True
            logger.info("OCR engine ready: %s", self.ocr_engine)
        except Exception as exc:  # noqa: BLE001
            self.ocr_available = False
            logger.error("OCR engine '%s' unavailable: %s — blur-only mode", self.ocr_engine, exc)

    # ------------------------------------------------------- region detection
    def find_plates(self, frame: np.ndarray, vehicle_bbox: tuple) -> list[PlateRegion]:
        """Locate plate regions inside (or near) a vehicle box."""
        if self._plate_model_loaded and self.plate_model is not None:
            regions = self._yolo_plates_crop(frame, vehicle_bbox)
            if regions:
                return regions
        return self._contour_plates(frame, vehicle_bbox)

    def _yolo_plates_crop(self, frame: np.ndarray, vehicle_bbox: tuple) -> list[PlateRegion]:
        """Run the plate model on the (padded) vehicle ROI, not the full frame —
        a plate is far too small in a 1920px dashcam frame to survive imgsz 640
        downscaling, but is plenty large inside the car box."""
        x1, y1, x2, y2 = [int(v) for v in vehicle_bbox]
        pad = int((x2 - x1) * 0.06)
        cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
        cx2, cy2 = min(frame.shape[1], x2 + pad), min(frame.shape[0], y2 + pad)
        if cy2 <= cy1 or cx2 <= cx1:
            return []
        try:
            res = self.plate_model(frame[cy1:cy2, cx1:cx2], verbose=False,
                                   conf=self.cfg.get("ocr_min_confidence", 0.3))
            box = res[0].boxes
            if box is None or len(box) == 0:
                return []
            xyxy = box.xyxy.cpu().numpy()
            confs = box.conf.cpu().numpy()
            out = []
            for (px1, py1, px2, py2), c in zip(xyxy, confs):
                out.append(PlateRegion(
                    bbox=(int(px1 + cx1), int(py1 + cy1), int(px2 + cx1), int(py2 + cy1)),
                    source="yolo",
                    confidence=float(c),
                ))
            return out
        except Exception as exc:  # noqa: BLE001
            logger.warning("YOLO plate inference failed: %s", exc)
            return []

    def _yolo_plates(self, frame: np.ndarray) -> list[PlateRegion]:
        if not self._plate_model_loaded or self.plate_model is None:
            return []
        try:
            res = self.plate_model(frame, verbose=False, conf=self.cfg.get("ocr_min_confidence", 0.3))
            box = res[0].boxes
            if box is None or len(box) == 0:
                return []
            xyxy = box.xyxy.cpu().numpy()
            confs = box.conf.cpu().numpy()
            out = []
            for (x1, y1, x2, y2), c in zip(xyxy, confs):
                out.append(PlateRegion(
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    source="yolo",
                    confidence=float(c),
                ))
            return out
        except Exception as exc:  # noqa: BLE001
            logger.warning("YOLO plate inference failed: %s", exc)
            return []

    def _contour_plates(self, frame: np.ndarray, vehicle_bbox: tuple) -> list[PlateRegion]:
        """Aspect-ratio contour heuristic over the lower half of the vehicle ROI."""
        x1, y1, x2, y2 = [int(v) for v in vehicle_bbox]
        h = y2 - y1
        roi_y1 = max(0, y1 + int(h * 0.35))   # plates sit in the lower part
        roi_y2 = min(frame.shape[0], y2)
        roi_x1, roi_x2 = max(0, x1), min(frame.shape[1], x2)
        if roi_y2 <= roi_y1 or roi_x2 <= roi_x1:
            return []

        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        roi_h, roi_w = roi.shape[:2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        edges = cv2.Canny(gray, 120, 200)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[PlateRegion] = []
        roi_area = roi_w * roi_h
        for cnt in contours:
            rect = cv2.minAreaRect(cnt)
            w, hh = rect[1]
            if w < 1 or hh < 1:
                continue
            aspect = max(w, hh) / min(w, hh)
            area = w * hh
            if not (2.0 <= aspect <= 6.5):          # plate aspect ratio
                continue
            if not (0.005 * roi_area <= area <= 0.35 * roi_area):
                continue
            pts = cv2.boxPoints(rect)
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            bx1 = roi_x1 + int(min(xs))
            by1 = roi_y1 + int(min(ys))
            bx2 = roi_x1 + int(max(xs))
            by2 = roi_y1 + int(max(ys))
            candidates.append(PlateRegion(
                bbox=(max(0, bx1), max(0, by1), min(frame.shape[1], bx2), min(frame.shape[0], by2)),
                source="contour",
                confidence=float(min(1.0, area / (0.05 * roi_area))),
            ))
        candidates.sort(key=lambda p: (p.bbox[2] - p.bbox[0]) * (p.bbox[3] - p.bbox[1]), reverse=True)
        return self._dedupe(candidates)

    @staticmethod
    def _dedupe(regions: list[PlateRegion]) -> list[PlateRegion]:
        """Drop near-duplicate candidates (nested/overlapping contour hits)."""
        kept: list[PlateRegion] = []
        for r in regions:
            dup = False
            for k in kept:
                ax1, ay1, ax2, ay2 = r.bbox
                bx1, by1, bx2, by2 = k.bbox
                ix1, iy1 = max(ax1, bx1), max(ay1, by1)
                ix2, iy2 = min(ax2, bx2), min(ay2, by2)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                small_area = (ax2 - ax1) * (ay2 - ay1)
                if small_area > 0 and inter / small_area > 0.5:
                    dup = True
                    break
            if not dup:
                kept.append(r)
        return kept[:2]

    # ------------------------------------------------------- in-memory OCR
    def extract_plate_text(self, frame: np.ndarray, region: PlateRegion) -> str:
        """Read the plate ONLY in memory. Returns the cleaned plate string or ''."""
        if not self.ocr_available or self.ocr_reader is None:
            return ""
        x1, y1, x2, y2 = region.bbox
        pad = 6
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(frame.shape[1], x2 + pad), min(frame.shape[0], y2 + pad)
        if x2 <= x1 or y2 <= y1:
            return ""
        crop = frame[y1:y2, x1:x2]
        # Size gate + optional enhancement. Below ~28px tall, characters are
        # physically unresolvable — reading them produces confident garbage
        # that would poison plate re-ID memory. Small-but-readable plates get
        # LANCZOS x2 + CLAHE contrast before OCR (naive nearest-neighbor was
        # tested once and HURT — do not reintroduce).
        min_h = int(self.cfg.get("ocr_min_height", 28))
        enhance = str(self.cfg.get("ocr_enhance", "lanczos")).lower()
        if crop.shape[0] < min_h:
            return ""
        if enhance == "lanczos" and max(crop.shape[:2]) < 160:
            crop = cv2.resize(crop, None, fx=2, fy=2,
                              interpolation=cv2.INTER_LANCZOS4)
            lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
            crop = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
        try:
            if self.ocr_engine == "paddleocr":
                out = self.ocr_reader.ocr(crop, cls=True)
                best = None
                for page in (out or []):
                    for line in (page or []):
                        if line and len(line) >= 2 and line[1][1] >= self.cfg.get("ocr_min_confidence", 0.4):
                            best = self._clean(line[1][0])
                if best:
                    region.ocr_confidence = 1.0
                    region.text = best
                    self._attach_country(region)
                return best or ""
            # easyocr
            results = self.ocr_reader.readtext(
                crop, detail=1, allowlist=PLATE_CHARS, paragraph=False
            )
            min_conf = self.cfg.get("ocr_min_confidence", 0.4)
            best, best_conf = "", 0.0
            for bbox, text, conf in results:
                cleaned = self._clean(text)
                if cleaned and conf >= min_conf and len(cleaned) >= 4 and conf > best_conf:
                    best, best_conf = cleaned, conf
            if best:
                region.ocr_confidence = best_conf
                region.text = best
                self._attach_country(region)
            return best
        except Exception as exc:  # noqa: BLE001
            logger.debug("OCR failed: %s", exc)
            return ""

    @staticmethod
    def _clean(text: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9]", "", text).upper()
        return cleaned

    @staticmethod
    def _attach_country(region: PlateRegion) -> None:
        """Guess the issuing country from OCR'd plate text and store it."""
        if not region.text:
            return
        country, conf = detect_plate_country(region.text)
        if country != "Unknown":
            region.country = country
            region.country_confidence = conf

    # ------------------------------------------------------- PRIVACY ENGINE
    def anonymize(self, frame: np.ndarray, regions: list[PlateRegion]) -> int:
        """Blur/pixelate every plate region ON THE OUTPUT BUFFER in place.
        Returns the number of anonymized regions."""
        if not regions:
            return 0
        mode = self.priv.get("mode", "pixel")
        pad_ratio = float(self.priv.get("pad_ratio", 0.2))
        count = 0
        for region in regions:
            x1, y1, x2, y2 = region.bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                continue
            pad_x = int((x2 - x1) * pad_ratio)
            pad_y = int(pad_x * 0.6)
            x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
            x2, y2 = min(frame.shape[1], x2 + pad_x), min(frame.shape[0], y2 + pad_y)
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            if mode == "pixel":
                ps = max(3, int(self.priv.get("pixel_size", 12)))
                sh, sw = roi.shape[:2]
                small = cv2.resize(roi, (max(1, sw // ps), max(1, sh // ps)), interpolation=cv2.INTER_LINEAR)
                roi = cv2.resize(small, (sw, sh), interpolation=cv2.INTER_NEAREST)
            else:
                k = int(self.priv.get("gaussian_kernel", 41))
                k = k if k % 2 == 1 else k + 1
                sigma = float(self.priv.get("gaussian_sigma", 0))
                roi = cv2.GaussianBlur(roi, (k, k), sigma)
            frame[y1:y2, x1:x2] = roi
            count += 1
        if bool(self.priv.get("blur_full_frame", False)):
            # In-place: callers hold the same buffer, rebinding the local name
            # here would silently skip the full-frame anonymization.
            frame[:] = self._blur_full(frame)
        return count

    def _blur_full(self, frame: np.ndarray) -> np.ndarray:
        mode = self.priv.get("mode", "pixel")
        if mode == "pixel":
            ps = max(8, int(self.priv.get("pixel_size", 12)))
            sh, sw = frame.shape[:2]
            small = cv2.resize(frame, (max(1, sw // ps), max(1, sh // ps)), interpolation=cv2.INTER_LINEAR)
            return cv2.resize(small, (sw, sh), interpolation=cv2.INTER_NEAREST)
        k = int(self.priv.get("gaussian_kernel", 41))
        k = k if k % 2 == 1 else k + 1
        return cv2.GaussianBlur(frame, (k, k), 0)

    @staticmethod
    def draw_blur_overlay(frame: np.ndarray, regions: list[PlateRegion], color=(0, 0, 255)) -> None:
        """Optional red 'PRIVACY: BLURRED' marker over anonymized plates."""
        for region in regions:
            x1, y1, x2, y2 = region.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            label = "PRIVACY: BLURRED"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), color, -1)
            cv2.putText(frame, label, (x1 + 3, max(th + 2, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def crop_bbox(frame: np.ndarray, bbox: tuple, pad_ratio: float = 0.15):
    """Padded crop around a detection box. Returns (crop, ox, oy) where (ox, oy)
    is the crop's top-left corner in the original frame."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    w, h = x2 - x1, y2 - y1
    padx, pady = int(w * pad_ratio), int(h * pad_ratio)
    ox, oy = max(0, x1 - padx), max(0, y1 - pady)
    ox2, oy2 = min(frame.shape[1], x2 + padx), min(frame.shape[0], y2 + pady)
    if ox2 <= ox or oy2 <= oy:
        return None, 0, 0
    return frame[oy:oy2, ox:ox2], ox, oy


class AsyncPlateReader:
    """Background license-plate reader.

    Plate detection (plate YOLO ~75ms) + OCR (easyocr ~300ms) are the single
    most expensive steps in the pipeline. Running them inline in the capture
    loop stalls the video when a car with a plate passes by — the recording
    drops frames and freezes. This worker moves that work off the critical
    path:

      * the capture loop only reads a cached per-track result (snapshot) and
        asks for a refresh when one is due (rate-limited by refresh_interval);
      * each track is re-read at most once per refresh_interval and only while
        it is big enough to contain a legible plate;
      * cached regions are stored in FRAME coords plus the detection box they
        were computed from, so the loop can re-anchor them onto the car's
        current position every frame for smooth blurring.
    """

    def __init__(self, engine: PrivacyEngine, cfg: dict, refresh_interval: float = 1.5):
        self.engine = engine
        self.refresh_interval = max(0.2, float(refresh_interval))
        self._results: dict[int, dict] = {}  # track_id -> {"regions", "bbox", "ts"}
        self._last_req: dict[int, float] = {}
        self._lock = threading.Lock()
        self._q: queue.Queue = queue.Queue(maxsize=32)
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def snapshot(self, track_id: int) -> dict | None:
        with self._lock:
            return self._results.get(track_id)

    def request(self, track_id: int, frame: np.ndarray, bbox: tuple) -> bool:
        """Queue a cheap cropped read for a track, at most once per interval."""
        now = time.time()
        with self._lock:
            if now - self._last_req.get(track_id, 0.0) < self.refresh_interval:
                return False
            self._last_req[track_id] = now
        crop, ox, oy = crop_bbox(frame, bbox)
        if crop is None or crop.size < 400:
            return False
        crop = crop.copy()  # a slice view would pin the whole 1080p frame in RAM
        try:
            self._q.put_nowait((track_id, crop, ox, oy, tuple(int(v) for v in bbox)))
            return True
        except queue.Full:
            return False

    def _run(self) -> None:
        while True:
            track_id, crop, ox, oy, bbox = self._q.get()
            try:
                h, w = crop.shape[:2]
                regions = self.engine.find_plates(crop, (0, 0, w, h))
                for r in regions:
                    if not r.text:
                        r.text = self.engine.extract_plate_text(crop, r)
                    r.bbox = (r.bbox[0] + ox, r.bbox[1] + oy,
                              r.bbox[2] + ox, r.bbox[3] + oy)
                with self._lock:
                    if regions:
                        self._results[track_id] = {
                            "regions": regions, "bbox": bbox, "ts": time.time()}
                    else:
                        self._results.pop(track_id, None)
                    # bound memory: forget stale tracks (track ids recycle)
                    if len(self._results) > 500:
                        cutoff = time.time() - 120.0
                        for tid in [t for t, r in self._results.items()
                                    if r["ts"] < cutoff]:
                            self._results.pop(tid, None)
            except Exception:  # noqa: BLE001
                logger.warning("async plate read failed", exc_info=True)
