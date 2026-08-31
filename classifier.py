"""
classifier.py — vehicle attribute classification.

Two tiers, both optional:
  1. HSV color analysis (always available, model-free) -> dominant color.
  2. Deep make/model/year classifier: loads a fine-tuned torchvision
     ResNet50/ConvNeXt checkpoint from config. Checkpoint format:
       {"state_dict": ..., "classes": ["Audi A4 2016-2020", "Skoda Octavia 2013-2019", ...]}
     or {"model": ..., "classes": ...} saved with torch.save.

Missing checkpoint = graceful "Unknown" — the pipeline never crashes.
"""

from __future__ import annotations

import logging
import os
import queue
import re
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

logger = logging.getLogger("mobile_tracker.classifier")

# HSV buckets: name -> (hue ranges OR None for achromatic, saturation range, value range)
COLOR_BUCKETS: list[tuple[str, tuple | None, tuple, tuple]] = [
    ("Black", None, (0.0, 255.0), (0.0, 60.0)),
    ("White", None, (0.0, 70.0), (190.0, 255.0)),
    ("Silver", None, (0.0, 60.0), (140.0, 190.0)),
    ("Gray", None, (0.0, 60.0), (60.0, 140.0)),
    ("Red", ((0, 12), (168, 180)), (60.0, 255.0), (60.0, 255.0)),
    ("Orange", ((12, 25),), (80.0, 255.0), (60.0, 255.0)),
    ("Yellow", ((25, 38),), (80.0, 255.0), (60.0, 255.0)),
    ("Green", ((38, 85),), (60.0, 255.0), (50.0, 255.0)),
    ("Blue", ((85, 135),), (60.0, 255.0), (50.0, 255.0)),
    ("Purple", ((135, 165),), (50.0, 255.0), (50.0, 255.0)),
    ("Brown", ((10, 25),), (40.0, 180.0), (50.0, 140.0)),
]

_YEAR_RE = re.compile(r"((?:19|20)\d{2})\s*[-–]\s*((?:19|20)\d{2})|((?:19|20)\d{2})")


@dataclass
class VehicleAttributes:
    make_model: str = "Unknown"
    year_range: str = "Unknown"
    color: str = "Unknown"
    color_confidence: float = 0.0
    model_confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "make_model": self.make_model,
            "year_range": self.year_range,
            "color": self.color,
            "color_confidence": round(self.color_confidence, 3),
            "model_confidence": round(self.model_confidence, 3),
        }


class ColorAnalyzer:
    """Model-free dominant vehicle color estimation via HSV sampling.

    Pixels are hard-assigned to exactly one bucket (no double counting from
    overlapping hue ranges), sampled from the car body band only (below the
    roofline, above the bumper/shadow zone) so windows, tires and road don't
    skew the vote. If no bucket reaches `min_share`, the color is Unknown.
    """

    def __init__(self, min_share: float = 0.35):
        self.min_share = min_share

    @staticmethod
    def _bucket(h: int, s: int, v: int) -> str:
        if v < 60:  # darkest pixels are indistinguishable -> Black
            return "Black"
        if s < 60:  # achromatic: white / silver / gray by lightness
            if v >= 190:
                return "White"
            if v >= 140:
                return "Silver"
            return "Gray"
        # chromatic
        if h < 12 or h >= 168:
            return "Red"
        if h < 25:
            if v <= 140 and s <= 180:  # darker + less saturated -> Brown
                return "Brown"
            return "Orange"
        if h < 38:
            return "Yellow"
        if h < 85:
            return "Green"
        if h < 135:
            return "Blue"
        if h < 165:
            return "Purple"
        return "Gray"

    def analyze(self, roi: np.ndarray) -> tuple[str, float]:
        try:
            h, w = roi.shape[:2]
            if h < 8 or w < 8:
                return "Unknown", 0.0
            y0, y1 = int(h * 0.30), int(h * 0.75)
            x0, x1 = int(w * 0.10), int(w * 0.90)
            band = roi[y0:y1, x0:x1]
            if band.size == 0:
                return "Unknown", 0.0
            sample = cv2.resize(band, (48, 24))
            hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(int)
            votes: dict[str, int] = {}
            for px in hsv:
                name = self._bucket(int(px[0]), int(px[1]), int(px[2]))
                votes[name] = votes.get(name, 0) + 1
            total = sum(votes.values()) or 1
            best = max(votes, key=votes.get)
            share = votes[best] / total
            if share < self.min_share:
                return "Unknown", 0.0
            return best, round(share, 3)
        except Exception:  # noqa: BLE001
            return "Unknown", 0.0

    # ------------------------------------------------------------------ v2
    def analyze_v2(self, roi: np.ndarray) -> tuple[str, float]:
        """LAB k-means dominant-cluster version: separates luminance from
        color, so bright-sun whites stop becoming 'Silver' and shaded panels
        stop becoming 'Black'. Picks the biggest cluster, maps it to a name.
        """
        try:
            h, w = roi.shape[:2]
            if h < 8 or w < 8:
                return "Unknown", 0.0
            band = roi[int(h * 0.25):int(h * 0.80), int(w * 0.08):int(w * 0.92)]
            if band.size == 0:
                return "Unknown", 0.0
            sample = cv2.resize(band, (64, 32))
            lab = cv2.cvtColor(sample, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
            crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
            _compact, labels, centers = cv2.kmeans(lab, 3, None, crit, 3,
                                                   cv2.KMEANS_PP_CENTERS)
            counts = np.bincount(labels.flatten(), minlength=3)
            best_c = centers[int(np.argmax(counts))]
            share = float(counts.max()) / max(1, len(labels))
            if share < self.min_share:
                return "Unknown", 0.0
            # convert cluster center back to BGR -> bucket name
            px = np.clip(best_c, 0, 255).astype(np.uint8).reshape(1, 1, 3)
            bgr = cv2.cvtColor(px, cv2.COLOR_LAB2BGR)[0, 0]
            hsv = cv2.cvtColor(np.array([[bgr]], np.uint8), cv2.COLOR_BGR2HSV)[0, 0]
            name = self._bucket(int(hsv[0]), int(hsv[1]), int(hsv[2]))
            return name, round(min(1.0, share), 3)
        except Exception:  # noqa: BLE001
            return "Unknown", 0.0


class VehicleClassifier:
    """Make/model/year classifier backed by an optional fine-tuned checkpoint."""

    def __init__(self, cls_cfg: dict):
        self.cfg = cls_cfg
        self.model = None
        self.classes: list[str] = []
        self.available = False
        self.error = ""
        self.color_analyzer = ColorAnalyzer(float(cls_cfg.get("color_min_share", 0.35)))
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    def load(self) -> bool:
        model_path = self.cfg.get("model_path", "")
        if not model_path:
            logger.info("Deep classifier disabled (no model_path in config)")
            return False
        if os.path.isdir(model_path):
            return self._load_transformers(model_path)
        return self._load_torchvision(model_path)

    def _load_transformers(self, folder: str) -> bool:
        """Load a HuggingFace ConvNextForImageClassification folder
        (config.json + preprocessor_config.json + model.safetensors)."""
        try:
            from transformers import ConvNextForImageClassification, ConvNextImageProcessor
        except ImportError:
            self.error = "transformers not installed"
            logger.error(self.error)
            return False
        try:
            import torch

            device = "mps" if (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()) else "cpu"
            self.model = ConvNextForImageClassification.from_pretrained(folder).to(device)
            self.processor = ConvNextImageProcessor.from_pretrained(folder)
            id2label = self.model.config.id2label
            self.classes = [id2label[k] for k in sorted(id2label, key=lambda x: int(x))]
            self.available = True
            logger.info("HF ConvNeXt classifier loaded: %d classes (device=%s)", len(self.classes), device)
            return True
        except Exception as exc:  # noqa: BLE001
            self.error = f"HF model load failed: {exc}"
            logger.exception("HF ConvNeXt load failed")
            return False

    def _load_torchvision(self, model_path: str) -> bool:
        try:
            import torch
            from torchvision import models, transforms  # noqa: F401, PLC0415
        except ImportError:
            self.error = "torch/torchvision not installed"
            logger.error(self.error)
            return False
        try:
            ckpt = torch.load(model_path, map_location="cpu")
            if isinstance(ckpt, dict) and "classes" in ckpt:
                self.classes = list(ckpt["classes"])
            if "state_dict" in ckpt:
                state = ckpt["state_dict"]
            elif "model" in ckpt and isinstance(ckpt["model"], dict):
                state = ckpt["model"]
            else:
                state = ckpt
            # Support resnet (fc) and convnext (classifier[2] -> head) heads.
            # ConvNeXt family: discriminate by backbone width from the
            # classifier head (Tiny/Small=768, Base=1024, Large=1536), and
            # Tiny vs Small by stage-5 depth (9 vs 27 blocks).
            convnext_arch = "convnext_tiny"
            try:
                width = state["classifier.2.weight"].shape[1]
                if width == 1536:
                    convnext_arch = "convnext_large"
                elif width == 1024:
                    convnext_arch = "convnext_base"
                else:
                    n5 = max(int(k.split(".")[2]) for k in state
                             if k.startswith("features.5.") and k.endswith(".weight"))
                    if n5 > 17:
                        convnext_arch = "convnext_small"
            except (KeyError, ValueError, StopIteration):
                pass
            for factory_name in ("resnet50", convnext_arch, "convnext_tiny", "resnet18"):
                try:
                    factory = getattr(models, factory_name)
                    model = factory(weights=None)
                    if hasattr(model, "fc"):
                        model.fc = torch.nn.Linear(model.fc.in_features, len(self.classes))
                    elif hasattr(model, "classifier") and isinstance(
                            getattr(model, "classifier"), torch.nn.Sequential):
                        # torchvision ConvNeXt: classifier is a Sequential
                        # whose LAST layer is the Linear head
                        model.classifier[-1] = torch.nn.Linear(
                            model.classifier[-1].in_features, len(self.classes))
                    missing, unexpected = model.load_state_dict(state, strict=False)
                    if len(unexpected) == 0 and len(missing) <= 2:
                        self.model = model
                        break
                except Exception:  # noqa: BLE001
                    continue
            if self.model is None:
                self.error = "checkpoint does not match resnet50/convnext_tiny/resnet18"
                logger.error(self.error)
                return False
            self.model.eval()
            # Inference-only: use MPS when available. (The historical MPS bug
            # affected TRAINING gradients only — inference is unaffected and
            # ~6x faster than CPU here.) Without this the deep model stays on
            # CPU and ConvNeXt costs ~200ms instead of ~25ms.
            if torch.backends.mps.is_available():
                self.model = self.model.to("mps")
                logger.info("deep classifier running on MPS")
            self._mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            self._std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            self.available = True
            logger.info("Deep classifier loaded: %d classes", len(self.classes))
            return True
        except Exception as exc:  # noqa: BLE001
            self.error = f"checkpoint load failed: {exc}"
            logger.exception("Deep classifier load failed")
            return False

    def classify(self, roi: np.ndarray) -> VehicleAttributes:
        attrs = VehicleAttributes()
        if self.cfg.get("color_analysis", True):
            color, conf = self.color_analyzer.analyze(roi)
            attrs.color, attrs.color_confidence = color, conf
        if self.model is not None:
            try:
                label, conf = self._predict_label(roi)
                attrs.make_model, attrs.year_range = self._split_label(label)
                attrs.model_confidence = conf
            except Exception as exc:  # noqa: BLE001
                logger.debug("Deep classification failed: %s", exc)
        return attrs

    def color_only(self, roi: np.ndarray) -> VehicleAttributes:
        """HSV color + shape only — no deep model. Used by the capture loop
        where the deep make/model inference runs on AsyncClassifier instead."""
        attrs = VehicleAttributes()
        if self.cfg.get("color_analysis", True):
            if str(self.cfg.get("color_algorithm", "hsv")) == "lab_kmeans":
                color, conf = self.color_analyzer.analyze_v2(roi)
            else:
                color, conf = self.color_analyzer.analyze(roi)
            attrs.color, attrs.color_confidence = color, conf
        return attrs

    def _predict_label(self, roi: np.ndarray) -> tuple[str, float]:
        import torch  # noqa: PLC0415

        device = next(self.model.parameters()).device
        processor = getattr(self, "processor", None)
        tta = bool(self.cfg.get("tta", True))
        with torch.no_grad():
            if processor is not None:  # HuggingFace ConvNeXt path
                inputs = processor(images=roi, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                logits = self.model(**inputs).logits[0]
            else:  # torchvision checkpoint path
                resized = cv2.resize(roi, (224, 224))
                tensor = torch.from_numpy(resized).permute(2, 0, 1).float().div(255.0)
                tensor = (tensor.unsqueeze(0) - self._mean) / self._std
                if not tta:
                    logits = self.model(tensor.to(device))[0]
                else:
                    # Test-time augmentation: original + horizontal flip + small
                    # scales, probabilities averaged. Free accuracy on MPS.
                    h224 = torch.flip(tensor, dims=[3])
                    xs = []
                    for t in (tensor, h224):
                        xs.append(self.model(t.to(device))[0])
                        t224 = torch.nn.functional.interpolate(t, size=(240, 240),
                                                               mode="bilinear")
                        xs.append(self.model(t224.to(device))[0])
                    logits = torch.stack(xs).mean(0)
            probs = torch.softmax(logits, dim=0)
            idx = int(torch.argmax(probs).item())
        return self.classes[idx] if idx < len(self.classes) else "Unknown", float(probs[idx])

    @staticmethod
    def _split_label(label: str) -> tuple[str, str]:
        """'Audi A4 2016-2020' or 'BMW_3_Series_Sedan_2012' -> (make model, year)"""
        if not label or label == "Unknown":
            return "Unknown", "Unknown"
        pretty = re.sub(r"_", " ", label)
        pretty = re.sub(r"\s+", " ", pretty).strip()
        m = _YEAR_RE.search(pretty)
        if m:
            y1, y2, single = m.groups()
            if y1 and y2:
                year = f"{y1}-{y2}"
            elif single:
                year = single
            else:
                year = "Unknown"
            model = (pretty[: m.start()] + pretty[m.end():]).strip()
            return (model or "Unknown"), year
        return label, "Unknown"


class AsyncClassifier:
    """Runs the deep make/model classifier off the capture thread.

    ResNet50 inference is ~40 ms on this Mac. Running it inline once per car
    every classify window stalls the video ~1×/second. This worker moves it
    off the critical path: the loop shows the last cached label per track and
    re-queues a fresh inference at most once per refresh_interval. Color
    analysis stays synchronous in the loop (it's ~1 ms and should appear
    immediately).
    """

    def __init__(self, classifier: VehicleClassifier, refresh_interval: float = 1.5):
        self.classifier = classifier
        # NOTE: do NOT snapshot classifier.available here — this worker is
        # constructed before load_engines() runs, so the flag would be frozen
        # False forever and every request() would be dropped (the silent
        # "local labels never appear live" bug).
        self.refresh_interval = max(0.2, float(refresh_interval))
        self._results: dict[int, tuple[str, float, float]] = {}  # tid -> (label, conf, ts)
        self._last_req: dict[int, float] = {}
        self._lock = threading.Lock()
        self._q: queue.Queue = queue.Queue(maxsize=32)
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()
        if self.available:
            logger.info("Async classifier worker started (refresh %.1fs)",
                        self.refresh_interval)

    @property
    def available(self) -> bool:
        """Live flag: True once the underlying model has loaded."""
        return self.classifier.available

    def get(self, track_id: int) -> tuple[str, float, float] | None:
        """(label, confidence, timestamp) for a track, or None."""
        with self._lock:
            return self._results.get(track_id)

    def request(self, track_id: int, roi: np.ndarray) -> bool:
        """Queue a crop for deep classification, at most once per interval."""
        if not self.available:
            return False
        now = time.time()
        with self._lock:
            if now - self._last_req.get(track_id, 0.0) < self.refresh_interval:
                return False
            self._last_req[track_id] = now
        try:
            self._q.put_nowait((track_id, roi.copy()))
            return True
        except queue.Full:
            return False

    def _run(self) -> None:
        while True:
            track_id, roi = self._q.get()
            try:
                label, conf = self.classifier._predict_label(roi)
            except Exception as exc:  # noqa: BLE001
                logger.debug("async classify failed: %s", exc)
                continue
            with self._lock:
                self._results[track_id] = (label, float(conf), time.time())
                # bound memory: track ids recycle
                if len(self._results) > 2000:
                    cutoff = time.time() - 120.0
                    for tid in [t for t, (_l, _c, ts) in self._results.items()
                                if ts < cutoff]:
                        self._results.pop(tid, None)
