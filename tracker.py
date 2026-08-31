"""
tracker.py — YOLOv8 vehicle detection + ByteTrack persistent tracking.

Uses ultralytics' native ByteTrack (bytetrack.yaml) / BoT-SORT (botsort.yaml)
integration. If ultralytics is unavailable or returns no IDs (first frames),
a lightweight IoU matcher keeps persistent IDs so the pipeline never stalls.

Every detection is exposed as a `Detection` dataclass consumed by the
classifier, ANPR/privacy engine, recorder and the WebSocket broadcaster.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger("mobile_tracker.tracker")

COCO_CLASSES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
    20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
    25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
    30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite", 34: "baseball bat",
    35: "baseball glove", 36: "skateboard", 37: "surfboard", 38: "tennis racket",
    39: "bottle", 40: "wine glass", 41: "cup", 42: "fork", 43: "knife", 44: "spoon",
    45: "bowl", 46: "banana", 47: "apple", 48: "sandwich", 49: "orange",
    50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza", 54: "donut",
    55: "cake", 56: "chair", 57: "couch", 58: "potted plant", 59: "bed",
    60: "dining table", 61: "toilet", 62: "tv", 63: "laptop", 64: "mouse",
    65: "remote", 66: "keyboard", 67: "cell phone", 68: "microwave", 69: "oven",
    70: "toaster", 71: "sink", 72: "refrigerator", 73: "book", 74: "clock",
    75: "vase", 76: "scissors", 77: "teddy bear", 78: "hair drier", 79: "toothbrush",
}


@dataclass
class Detection:
    track_id: int = -1
    cls: int = -1
    cls_name: str = "unknown"
    conf: float = 0.0
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)  # x1, y1, x2, y2
    class_confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "cls": self.cls,
            "cls_name": self.cls_name,
            "conf": round(self.conf, 3),
            "bbox": [round(v, 1) for v in self.bbox],
        }


class _IoUTracker:
    """Lightweight IoU matcher used when ByteTrack has no IDs yet (first
    frames) or ultralytics is unavailable. Gives every detection a stable id."""

    def __init__(self, iou_threshold: float = 0.3, max_misses: int = 30):
        self.iou_threshold = iou_threshold
        self.max_misses = max_misses
        self._tracks: dict[int, dict] = {}
        self._next_id = 1

    @staticmethod
    def _iou(a: tuple, b: tuple) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        return inter / (area_a + area_b - inter + 1e-9)

    def update(self, dets: list[Detection]) -> list[Detection]:
        unmatched = list(dets)
        assigned: list[Detection] = []
        for tid, track in sorted(self._tracks.items()):
            best, best_iou = None, 0.0
            for d in unmatched:
                iou = self._iou(track["bbox"], d.bbox)
                if iou > best_iou:
                    best, best_iou = d, iou
            if best is not None and best_iou >= self.iou_threshold:
                track["bbox"] = best.bbox
                track["misses"] = 0
                best.track_id = tid
                assigned.append(best)
                unmatched.remove(best)
            else:
                track["misses"] += 1
        for tid in [t for t, tr in self._tracks.items() if tr["misses"] > self.max_misses]:
            del self._tracks[tid]
        for d in unmatched:
            d.track_id = self._next_id
            self._tracks[self._next_id] = {"bbox": d.bbox, "misses": 0}
            self._next_id += 1
            assigned.append(d)
        return assigned


class VehicleTracker:
    """YOLOv8 detection + ByteTrack/BoT-SORT tracking with graceful fallback."""

    def __init__(self, detection_cfg: dict):
        self.cfg = detection_cfg
        self.device = self._resolve_device(detection_cfg.get("device", "auto"))
        self.vehicle_classes = list(detection_cfg.get("vehicle_classes", [2, 3, 5, 7]))
        self.conf = float(detection_cfg.get("conf_threshold", 0.35))
        self.iou = float(detection_cfg.get("iou_threshold", 0.45))
        self.imgsz = int(detection_cfg.get("imgsz", 640))
        self.tracker_cfg = detection_cfg.get("tracker", "bytetrack.yaml")
        self.persist = bool(detection_cfg.get("tracker_persist", True))
        self.model = None
        self.available = False
        self.error = ""
        self._iou_fallback = _IoUTracker()

    @staticmethod
    def _resolve_device(requested: str) -> str:
        if requested != "auto":
            return requested
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    def load(self) -> bool:
        """Load the YOLO model. Returns False (with self.error set) if the
        heavy dependencies are missing so the caller can keep running."""
        try:
            from ultralytics import YOLO
        except ImportError:
            self.error = "ultralytics is not installed — run 'pip install ultralytics'"
            logger.error(self.error)
            return False
        try:
            self.model = YOLO(self.cfg.get("model", "yolov8n.pt"))
            self.available = True
            logger.info("YOLO model loaded: %s (device=%s)", self.cfg.get("model"), self.device)
            return True
        except Exception as exc:  # noqa: BLE001 — surface any load failure
            self.error = f"YOLO load failed: {exc}"
            logger.exception("YOLO model load failed")
            return False

    def update(self, frame) -> list[Detection]:
        """Run detection + tracking on one frame. Returns Detections with
        persistent track ids (ByteTrack, falling back to the IoU matcher)."""
        if self.model is None:
            return []
        try:
            result = self.model.track(
                frame,
                persist=self.persist,
                conf=self.conf,
                iou=self.iou,
                imgsz=self.imgsz,
                classes=self.vehicle_classes,
                tracker=self.tracker_cfg,
                device=self.device,
                verbose=False,
            )
        except Exception as exc:  # noqa: BLE001 — model must never kill the pipeline
            logger.warning("Tracking call failed: %s", exc)
            return []

        dets: list[Detection] = []
        box = result[0].boxes
        if box is None or len(box) == 0:
            return []

        try:
            ids = box.id
            xyxy = box.xyxy.cpu().numpy() if hasattr(box.xyxy, "cpu") else box.xyxy.numpy()
            confs = box.conf.cpu().numpy() if hasattr(box.conf, "cpu") else box.conf.numpy()
            clss = box.cls.cpu().numpy() if hasattr(box.cls, "cpu") else box.cls.numpy()
            for i, (x1, y1, x2, y2) in enumerate(xyxy):
                dets.append(Detection(
                    track_id=int(ids[i].item()) if ids is not None else -1,
                    cls=int(clss[i]),
                    cls_name=COCO_CLASSES.get(int(clss[i]), f"class{int(clss[i])}"),
                    conf=float(confs[i]),
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                ))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Detection post-processing failed: %s", exc)
            return []

        if any(d.track_id == -1 for d in dets):
            # Only fall back for the detections that actually lack an ID.
            # Re-running the IoU matcher over already-tracked detections would
            # rewrite their stable ByteTrack IDs mid-track (orphaning the
            # classifier caches / plate re-ID memory keyed by those IDs).
            with_id = [d for d in dets if d.track_id != -1]
            without_id = [d for d in dets if d.track_id == -1]
            if without_id:
                return with_id + self._iou_fallback.update(without_id)
            return with_id
        return dets

    def release(self) -> None:
        self.model = None
        logger.info("Tracker released")
