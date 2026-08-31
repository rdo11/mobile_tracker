"""
main.py — Mobile Tracker app entrypoint.

FastAPI + WebSockets real-time dashboard for a privacy-first vehicle
tracking/classification pipeline on a laptop dashcam.

Privacy guarantee: the frame that is written to disk and streamed to the UI
has every license-plate region pixelated/Gaussian-blurred by the
privacy engine BEFORE it reaches the recorder or the broadcaster.

Run:  python main.py          (config: config.yaml)
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import signal
import threading
import time
from dataclasses import asdict

import cv2
import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from anpr_privacy import AsyncPlateReader, PlateRegion, PrivacyEngine
from classifier import AsyncClassifier, VehicleClassifier
from grok_classifier import GrokClassifier
from gemini_classifier import GeminiClassifier
from deepseek_classifier import DeepSeekClassifier
from recorder import SessionLog, VideoRecorder
from road_context import AsyncRoadContext
from tracker import VehicleTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("mobile_tracker")

BASE_DIR = __file__ and __file__.rsplit("/", 1)[0] or "."
DEFAULT_CONFIG = f"{BASE_DIR}/config.yaml"

# Coarse labels for non-car classes: YOLO already names them, so no
# classifier / Grok calls are ever made for them (no API cost, no
# generation-level classes for trucks, motorbikes, buses, cyclists...).
COARSE_LABELS = {
    "person": "Pedestrian",
    "bicycle": "Cyclist",
    "motorcycle": "Motorbike",
    "bus": "Bus",
    "truck": "Truck",
}

# Classes that can carry a readable plate worth OCR'ing (async). Pedestrians
# and other COCO classes are skipped — scanning them wastes the expensive
# plate model + OCR on every frame.
PLATE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}

app = FastAPI(title="Mobile Tracker", version="1.0.0")


class DashboardState:
    """Shared state between the capture loop and the API/WS endpoints."""

    def __init__(self, config: dict):
        self.config = config
        self.cam_cfg = config.get("camera", {})
        self.det_cfg = config.get("detection", {})
        self.cls_cfg = config.get("classification", {})
        self.anpr_cfg = config.get("anpr", {})
        self.priv_cfg = config.get("privacy", {})
        self.rec_cfg = config.get("recorder", {})
        self.srv_cfg = config.get("server", {})
        self.grok_cfg = config.get("grok", {})
        self.gemini_cfg = config.get("gemini", {})
        self.deepseek_cfg = config.get("deepseek", {})
        self.road_cfg = config.get("road", {})

        self.tracker = VehicleTracker(self.det_cfg)
        # Road context (lights + speed signs) is opt-in: when enabled, the
        # shared YOLO tracker also follows COCO class 9 (traffic light).
        if self.road_cfg.get("enabled", False):
            if 9 not in self.tracker.vehicle_classes:
                self.tracker.vehicle_classes.append(9)
        self.road: AsyncRoadContext | None = None
        self.privacy = PrivacyEngine(self.anpr_cfg, self.priv_cfg)
        self.classifier = VehicleClassifier(self.cls_cfg)
        self.grok_cls = GrokClassifier(self.grok_cfg)
        self.gemini_cls = GeminiClassifier(self.gemini_cfg)
        self.deepseek_cls = DeepSeekClassifier(self.deepseek_cfg)
        # Deep make/model inference runs on a background thread (AsyncClassifier);
        # the capture loop only reads the cached label, so the ~40 ms ResNet50
        # call never stalls the video/recording.
        self.cls_worker = AsyncClassifier(
            self.classifier,
            float(self.cls_cfg.get("classify_refresh_interval", 1.5)))
        # Plate detection + OCR run on a background thread (AsyncPlateReader);
        # the capture loop only reads cached results so OCR never stalls the
        # video pipeline / recording.
        self.plate_reader = AsyncPlateReader(
            self.privacy, self.anpr_cfg,
            float(self.anpr_cfg.get("plate_refresh_interval", 1.5)))
        # Road context worker (lights + speed signs) — only when enabled.
        if self.road_cfg.get("enabled", False):
            self.road = AsyncRoadContext(self.road_cfg,
                                         ocr_reader=self.privacy.ocr_reader)

        self.recorder = VideoRecorder(self.rec_cfg, float(self.cam_cfg.get("fps", 30)))
        self.log = SessionLog(
            self.rec_cfg.get("session_log", "storage/session_vehicles.sqlite"),
            self.rec_cfg.get("session_retention_days", 0))
        self.grok_cls.on_result = lambda tid, r: self.log.update_grok(
            tid, r.get("label", "Unknown"), r.get("year_range", "Unknown"),
            r.get("confidence", 0.0)) if r.get("confidence", 0) >= 0.5 else None
        self.gemini_cls.on_result = self.grok_cls.on_result
        self.deepseek_cls.on_result = self.grok_cls.on_result

        self.capture = None
        self.running = False
        self.fps_now = 0.0
        self.frame_count = 0
        self.frames_blurred = 0
        self.error = ""
        self.signal_lost = False

        self._latest_jpeg: bytes | None = None
        self._jpeg_lock = threading.Lock()
        self.clients: set[WebSocket] = set()
        self._ws_lock = threading.Lock()
        self._last_stream_ts = 0.0
        # Plate-based re-ID memory: plate text -> saved vehicle attrs.
        # A car seen again with the same plate reuses these and skips the
        # classifier. In-memory only; plates themselves are never streamed.
        self.plate_memory: dict[str, dict] = {}
        self.plate_memory_ttl = float(self.anpr_cfg.get("plate_memory_ttl", 3600))

        # GDPR footgun guard: the code/docstrings promise blur-before-write,
        # but blur is opt-in. Fail hard when the operator asks for it, warn
        # loudly otherwise (recordings on disk contain readable plates).
        if not self.anpr_cfg.get("blur_plates", False):
            if self.priv_cfg.get("require_blur", False):
                raise SystemExit(
                    "privacy.require_blur=true but anpr.blur_plates=false — "
                    "refusing to run with readable plates on disk/stream")
            logger.warning(
                "blur_plates=false — recordings and the stream contain readable "
                "plates (GDPR: enable anpr.blur_plates before sharing)")

    # ------------------------------------------------------------- engines
    def load_engines(self) -> None:
        self.tracker.load()
        self.privacy.load()
        self.classifier.load()
        if self.road is not None:
            # OCR reader only exists after privacy.load() — refresh the
            # reference so digit reading for speed signs actually works.
            self.road.ocr = self.privacy.ocr_reader

    # -------------------------------------------------------------- stream
    def publish_jpeg(self, jpeg: bytes) -> None:
        with self._jpeg_lock:
            self._latest_jpeg = jpeg

    def snapshot(self) -> bytes | None:
        with self._jpeg_lock:
            return self._latest_jpeg

    def broadcast_jpeg(self, jpeg: bytes) -> None:
        with self._ws_lock:
            clients = list(self.clients)
        for client in clients:
            try:
                asyncio.run_coroutine_threadsafe(
                    client.send_bytes(jpeg), client.app_loop
                )
            except Exception:  # noqa: BLE001
                pass

    def broadcast_event(self, event: dict) -> None:
        payload = json.dumps(event, ensure_ascii=False).encode("utf-8")
        with self._ws_lock:
            clients = list(self.clients)
        for client in clients:
            try:
                asyncio.run_coroutine_threadsafe(
                    client.send_text(payload.decode("utf-8")), client.app_loop
                )
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------ telemetry
    def stats(self) -> dict:
        rec_elapsed = 0
        if self.recorder.rec_started is not None:
            rec_elapsed = int((time.time() - self.recorder.rec_started.timestamp()))
        return {
            "running": self.running,
            "fps": round(self.fps_now, 1),
            "frames": self.frame_count,
            "frames_blurred": self.frames_blurred,
            "recording": self.recorder.rec_started is not None,
            "rec_elapsed": rec_elapsed,
            "recording_path": self.recorder.path,
            "signal_lost": self.signal_lost,
            "tracker": {"available": self.tracker.available, "error": self.tracker.error},
            "classifier": {"available": self.classifier.available, "error": self.classifier.error},
            "privacy": {"ocr_available": self.privacy.ocr_available,
                        "plate_model": self.privacy._plate_model_loaded,
                        "mode": self.priv_cfg.get("mode", "pixel"),
                        "plate_memory": len(self.plate_memory)},
            "grok": {"available": self.grok_cls.available, "enabled": self.grok_cls.enabled,
                     "model": self.grok_cls.model,
                     "requested": self.grok_cls.requested, "responded": self.grok_cls.responded,
                     "last_label": self.grok_cls.last_label},
            "gemini": {"available": self.gemini_cls.available, "enabled": self.gemini_cls.enabled,
                       "model": self.gemini_cls.model, "batch_size": self.gemini_cls.batch_size,
                       "requested": self.gemini_cls.requested,
                       "responded": self.gemini_cls.responded,
                       "batches": self.gemini_cls.batches_sent,
                       "last_label": self.gemini_cls.last_label},
            "deepseek": {"available": self.deepseek_cls.available,
                         "enabled": self.deepseek_cls.enabled,
                         "model": self.deepseek_cls.model,
                         "requested": self.deepseek_cls.requested,
                         "responded": self.deepseek_cls.responded,
                         "last_label": self.deepseek_cls.last_label},
            "clients": len(self.clients),
            "road": self.road.display_state() if getattr(state, "road", None) else None,
            "error": self.error,
        }


def _draw_vehicle(state: DashboardState, frame, det, attrs, plate_status: str, color,
                  plate_country: str = "", plate_country_conf: float = 0.0) -> None:
    x1, y1, x2, y2 = [int(v) for v in det.bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    model_pct = f" {attrs.model_confidence:.0%}" if attrs.model_confidence > 0 else ""
    color_pct = f" {attrs.color_confidence:.0%}" if attrs.color_confidence > 0 else ""
    yr = f" ({attrs.year_range})" if attrs.year_range else ""
    label = f"[ID: #{det.track_id} | {attrs.make_model}{model_pct}{yr} | {attrs.color}{color_pct}]"
    if plate_country:
        plate_status = f"{plate_status} [{plate_country}{plate_country_conf:.0%}]"
    badge = f" {det.cls_name} {det.conf:.0%} | PLATE: {plate_status}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    (bw, _), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    y0 = max(th + 10, y1 - th - 8)
    cv2.rectangle(frame, (x1, y0 - th - 6), (x1 + max(tw, bw) + 10, y0 + 2), (30, 30, 30), -1)
    cv2.putText(frame, label, (x1 + 5, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, badge, (x1 + 5, y0 + th - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)


# ------------------------------------------------------------- capture loop
def open_camera(state: DashboardState) -> cv2.VideoCapture | None:
    src = state.cam_cfg.get("source", 0)
    width = int(state.cam_cfg.get("width", 1280))
    height = int(state.cam_cfg.get("height", 720))
    transport = state.cam_cfg.get("rtsp_transport", "")
    if isinstance(src, str) and src.lower().startswith("rtsp://") and transport:
        src = src + ("&" if "?" in src else "?") + f"rtsp_transport={transport}"
    cap = cv2.VideoCapture(src)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        return cap
    cap.release()
    return None


def is_replay_source(state: DashboardState) -> bool:
    """True when the camera source is a local video file (replay mode)."""
    src = state.cam_cfg.get("source", 0)
    if not isinstance(src, str):
        return False
    lower = src.lower()
    return (lower.startswith("file://") or lower.endswith((".mp4", ".mov", ".avi", ".mkv"))
            or "/" in src)


class FrameSlot:
    """Latest-frame slot shared between the reader thread and capture loop."""

    def __init__(self) -> None:
        self.frame = None
        self.ts = 0.0
        self._lock = threading.Lock()

    def put(self, frame) -> None:
        with self._lock:
            self.frame = frame
            self.ts = time.time()

    def get(self):
        with self._lock:
            return self.frame, self.ts


def reader_thread(state: DashboardState, slot: FrameSlot) -> None:
    """Owns the camera handle; reads continuously even when the feed stalls.

    A dead phone feed makes VideoCapture.read() block for seconds. Running it
    in its own thread keeps the dashboard + recorder responsive regardless.
    In replay mode (file source) reads are throttled to the file's real fps
    and the video loops at EOF.
    """
    width = int(state.cam_cfg.get("width", 1280))
    height = int(state.cam_cfg.get("height", 720))
    max_fail = int(state.cam_cfg.get("max_read_failures", 50))
    failures = 0
    replay = is_replay_source(state)
    replay_fps = 0.0
    replay_interval = 0.0
    replay_prev = time.time()

    while state.running:
        cap = state.capture
        if cap is None or not cap.isOpened():
            state.error = "WAITING FOR CAMERA - check the Iriun app on phone and Mac"
            if not replay:
                time.sleep(3.0)
            cap2 = open_camera(state)
            state.capture = cap2
            if cap2 is not None:
                failures = 0
                state.error = ""
                logger.info("Camera opened")
                if replay:
                    state.error = f"REPLAY MODE - {state.cam_cfg.get('source')}"
                    fps = cap2.get(cv2.CAP_PROP_FPS) or 30.0
                    replay_fps = fps
                    replay_interval = 1.0 / max(1.0, fps)
                    replay_prev = time.time()
            if replay:
                time.sleep(1.0)
            continue
        if replay and replay_interval > 0:
            now = time.time()
            wait = replay_interval - (now - replay_prev)
            if wait > 0:
                time.sleep(min(wait, 0.5))
                continue
            replay_prev = now
        ok, frame = cap.read()
        if ok:
            failures = 0
            slot.put(frame)
        else:
            if replay:
                # EOF -> loop the video
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                logger.info("Replay looped (EOF)")
                time.sleep(0.3)
                continue
            failures += 1
            if failures >= max_fail:
                logger.warning("Camera read failed repeatedly - reconnecting...")
                cap.release()
                state.capture = None
                failures = 0
            time.sleep(0.05)


def capture_loop(state: DashboardState) -> None:
    source = state.cam_cfg.get("source", 0)
    stream_fps = float(state.srv_cfg.get("max_stream_fps", 20))
    jpeg_q = int(state.srv_cfg.get("jpeg_quality", 70))

    slot = FrameSlot()
    state.running = True  # must be set BEFORE the reader thread starts
    threading.Thread(target=reader_thread, args=(state, slot), daemon=True).start()

    frame_interval = 1.0 / max(1.0, stream_fps)
    prev_ts = time.time()
    fps_counter = 0
    fps_ts = time.time()
    last_frame_ts = 0.0
    classify_every = int(state.cls_cfg.get("classify_every_n_frames", 30)) or 1
    min_display_conf = float(state.cls_cfg.get("min_display_conf", 0.45))
    det_stride = max(1, int(state.det_cfg.get("detect_every_n_frames", 2)))
    last_dets: list = []
    last_classify: dict[int, int] = {}
    last_attrs: dict[int, object] = {}
    last_best: dict[int, tuple] = {}
    last_best_col: dict[int, tuple] = {}
    last_grok_req: dict[int, int] = {}

    try:
        while state.running:
            frame, ts = slot.get()
            if frame is None:
                state.error = "WAITING FOR CAMERA - check the Iriun app on phone and Mac"
                time.sleep(0.5)
                continue
            if ts == last_frame_ts:
                # No fresh frame since last iteration: the feed is stalled
                # (dead phone). Don't re-process the same frame.
                time.sleep(0.002)
                continue
            last_frame_ts = ts
            state.error = ""
            state.frame_count += 1

            # Black-frame guard: when the phone feed dies (screen lock,
            # app backgrounded, WiFi hiccup) the Iriun virtual camera emits
            # pure black frames. Skip the whole pipeline — don't record,
            # don't detect — and stream a "NO SIGNAL" overlay instead.
            if cv2.mean(frame)[0] < 8.0:
                if state.frame_count % 60 == 0:
                    logger.warning("Black frame - phone feed down")
                state.signal_lost = True
                cv2.putText(frame, "NO SIGNAL - open Iriun on the phone",
                            (40, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 0, 255), 2, cv2.LINE_AA)
                now = time.time()
                if now - prev_ts >= frame_interval:
                    prev_ts = now
                    okj, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])
                    if okj:
                        jpeg = buf.tobytes()
                        state.publish_jpeg(jpeg)
                        state.broadcast_jpeg(jpeg)
                fps_counter += 1
                now_fps = time.time()
                if now_fps - fps_ts >= 1.0:
                    state.fps_now = fps_counter / (now_fps - fps_ts)
                    fps_counter = 0
                    fps_ts = now_fps
                    state.broadcast_event({"type": "stats", "stats": state.stats()})
                continue

            state.signal_lost = False
            if state.frame_count % det_stride == 0:
                det_scale = float(state.det_cfg.get("detect_scale", 1.0))
                if det_scale < 1.0:
                    det_frame = cv2.resize(frame, None, fx=det_scale, fy=det_scale,
                                           interpolation=cv2.INTER_AREA)
                else:
                    det_frame = frame
                last_dets = state.tracker.update(det_frame)
                if det_scale < 1.0:
                    for d in last_dets:
                        x1, y1, x2, y2 = d.bbox
                        d.bbox = (x1 / det_scale, y1 / det_scale,
                                  x2 / det_scale, y2 / det_scale)
            dets = last_dets
            road = getattr(state, "road", None)
            if road is not None:
                road.request_sign_scan(frame)   # internally rate-limited

            # Privacy-first order: OCR on the RAW frame, then blur the OUTPUT buffer.
            # classifier sees the raw ROI (better signal) but the annotated output
            # frame is blurred before it is recorded or streamed.
            frame_out = frame.copy()
            # Recording copy: blurred exactly like frame_out but WITHOUT overlay
            # graphics — recordings stay reusable for future crop mining
            # (burned-in boxes made past recordings unusable for training).
            rec_clean = bool(state.rec_cfg.get("clean_frames", True))
            rec_frame = frame.copy() if rec_clean else frame_out
            telemetry: list[dict] = []
            # Plate-based re-ID: a plate seen before re-uses the stored
            # make/model/color and SKIPS the heavy classifier entirely.
            plate_memory: dict[str, dict] = getattr(state, "plate_memory", {})
            for det in dets:
                x1, y1, x2, y2 = [int(v) for v in det.bbox]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                roi = frame[y1:y2, x1:x2]
                frame_no = state.frame_count
                # Traffic lights: classify state on the background worker and
                # SKIP the whole vehicle path (no classifier/plates/db/draw).
                if road is not None and det.cls_name == "traffic light":
                    if max(x2 - x1, y2 - y1) >= int(state.road_cfg.get("min_light_side", 24)):
                        road.request_light(str(det.track_id), roi, (x1, y1, x2, y2))
                    continue
                min_area = int(state.cls_cfg.get("min_classify_area", 0))
                min_side = int(state.cls_cfg.get("min_box_side", 0))
                # Coarse-labeled classes (person/bike/truck/bus) never need the
                # deep model, so skip the size gates for them — a distant
                # pedestrian should still say "Pedestrian", not "Unknown"
                # (that was the old footgun).
                coarse_cls = COARSE_LABELS.get(det.cls_name) if det.cls_name != "car" else None
                # Motorcycle/person confusion fix: distant motorcycles are
                # often detected as low-confidence, wide "person" boxes. A real
                # pedestrian is TALLER than wide; a motorbike is wider than
                # tall. Relabel obvious cases and drop the mushiest ones.
                if (det.cls_name == "person" and coarse_cls is not None
                        and (x2 - x1) > 0 and (y2 - y1) > 0):
                    ratio = (x2 - x1) / max(1, y2 - y1)
                    if ratio > 1.1 and det.conf < 0.65:
                        det.cls_name = "motorcycle"
                        coarse_cls = "Motorbike"
                do_classify = (
                    (
                        roi.size >= min_area
                        and max(x2 - x1, y2 - y1) >= min_side
                    )
                    or coarse_cls is not None
                ) and (
                    frame_no - last_classify.get(det.track_id, -10**9) >= classify_every
                )
                # ANPR/privacy: plate regions come from the async worker's
                # cached result (re-anchored onto the car's current box so the
                # blur follows it smoothly). OCR/plate-YOLO never block here.
                plate_regions = []
                plate_text = ""
                if state.anpr_cfg.get("enabled", True):
                    snap = state.plate_reader.snapshot(det.track_id)
                    if snap is not None:
                        dx = (x1 + x2) / 2 - (snap["bbox"][0] + snap["bbox"][2]) / 2
                        dy = (y1 + y2) / 2 - (snap["bbox"][1] + snap["bbox"][3]) / 2
                        for r in snap["regions"]:
                            nr = PlateRegion(
                                (r.bbox[0] + dx, r.bbox[1] + dy,
                                 r.bbox[2] + dx, r.bbox[3] + dy),
                                source=r.source, confidence=r.confidence)
                            nr.text = r.text
                            nr.country = r.country
                            nr.country_confidence = r.country_confidence
                            plate_regions.append(nr)
                            if nr.text:
                                plate_text = nr.text
                    else:
                        # First sighting of a track with blur ON: a plate must
                        # not hit the recording unblurred while the async worker
                        # warms up, so do ONE synchronous read (blur is only
                        # enabled in shipping mode anyway).
                        if state.anpr_cfg.get("blur_plates", False):
                            plate_regions = state.privacy.find_plates(frame, det.bbox)
                            for region in plate_regions:
                                if not region.text:
                                    region.text = state.privacy.extract_plate_text(frame, region)
                                if region.text:
                                    plate_text = region.text
                                    break
                    # Queue an async refresh for the track (rate-limited).
                    if det.cls_name in PLATE_CLASSES and max(x2 - x1, y2 - y1) >= min_side:
                        state.plate_reader.request(det.track_id, frame, det.bbox)
                n = state.privacy.anonymize(frame_out, plate_regions) \
                    if state.anpr_cfg.get("blur_plates", False) else 0
                if n:
                    state.frames_blurred += n
                    if rec_clean:
                        # keep the recording copy anonymized too (GDPR: the
                        # disk buffer must never hold readable plates)
                        state.privacy.anonymize(rec_frame, plate_regions)
                    if state.priv_cfg.get("show_blur_overlay", True):
                        state.privacy.draw_blur_overlay(frame_out, plate_regions)

                # Re-ID short-circuit: if we already know this plate, reuse the
                # stored attrs and skip the classifier + API fallback entirely.
                plate_mem = None
                if det.cls_name == "car" and plate_text \
                        and state.anpr_cfg.get("plate_reid", True):
                    plate_mem = plate_memory.get(plate_text)
                if plate_mem is not None:
                    attrs = type("A", (), {k: v for k, v in plate_mem.items() if k != "_ts"})()
                    last_attrs[det.track_id] = attrs
                    last_classify[det.track_id] = frame_no
                elif do_classify:
                    # Color runs synchronously (~1ms). The deep make/model runs
                    # on AsyncClassifier — pull the last cached label if one has
                    # arrived, otherwise request one and show Unknown for now.
                    attrs = state.classifier.color_only(roi)
                    cached = state.cls_worker.get(det.track_id)
                    if cached is not None:
                        label, conf, _ts = cached
                        attrs.make_model, attrs.year_range = VehicleClassifier._split_label(label)
                        attrs.model_confidence = conf
                    if det.cls_name == "car":
                        state.cls_worker.request(det.track_id, roi)
                    if det.cls_name != "car":
                        # Trucks/bikes/pedestrians: coarse YOLO label only,
                        # keep HSV color — never sent to classifier/API.
                        attrs.make_model = COARSE_LABELS.get(det.cls_name, det.cls_name)
                        attrs.year_range = ""
                        attrs.model_confidence = 0.0
                    if det.cls_name == "car":
                        # Track-level stability: keep the best (highest-confidence)
                        # label seen for this track, so labels don't flip-flop
                        # between random guesses every frame.
                        best = last_best.get(det.track_id)
                        if (best is not None
                                and attrs.model_confidence <= best[2] * 1.05
                                and best[2] >= min_display_conf):
                            attrs.make_model, attrs.year_range, attrs.model_confidence = best
                        else:
                            last_best[det.track_id] = (
                                attrs.make_model, attrs.year_range, attrs.model_confidence)
                        # Display gate: below min_display_conf, no random guesses
                        if attrs.model_confidence < min_display_conf:
                            attrs.make_model, attrs.year_range = "Unknown", ""
                            attrs.model_confidence = 0.0
                        # Color stability: keep the highest-confidence color seen
                        # for this track (body color doesn't change while parked).
                        bcol = last_best_col.get(det.track_id)
                        if (bcol is not None
                                and attrs.color_confidence <= bcol[1] * 1.05
                                and bcol[1] >= 0.4):
                            attrs.color, attrs.color_confidence = bcol
                        elif attrs.color_confidence >= 0.4:
                            last_best_col[det.track_id] = (attrs.color, attrs.color_confidence)
                    # API fallback provider: Gemini (batched) wins if enabled,
                    # otherwise DeepSeek, otherwise Grok. All disabled = pure
                    # local, low-confidence cars stay "Unknown".
                    fb = (state.gemini_cls if state.gemini_cls.enabled
                          else (state.deepseek_cls if state.deepseek_cls.enabled
                                else (state.grok_cls if state.grok_cls.enabled else None)))
                    if fb is not None and det.cls_name == "car":
                        local_conf = attrs.model_confidence if attrs else 0.0
                        local_ok = (attrs is not None
                                    and local_conf >= fb.fallback_conf
                                    and attrs.make_model not in ("Unknown", ""))
                        if not local_ok:
                            fb.maybe_retry(det.track_id, frame, det.bbox)
                            last_grok_req[det.track_id] = frame_no
                    last_attrs[det.track_id] = attrs
                    last_classify[det.track_id] = frame_no
                else:
                    attrs = last_attrs.get(det.track_id)
                    # Pick up a fresh async label for this track even outside
                    # its classify window, so the make/model appears as soon as
                    # the worker returns instead of up to a window later.
                    if attrs is not None and det.cls_name == "car":
                        cached = state.cls_worker.get(det.track_id)
                        if cached is not None:
                            label, conf, _ts = cached
                            if conf >= min_display_conf:
                                best = last_best.get(det.track_id)
                                if best is None or conf > best[2]:
                                    attrs.make_model, attrs.year_range = VehicleClassifier._split_label(label)
                                    attrs.model_confidence = conf
                                    last_best[det.track_id] = (
                                        attrs.make_model, attrs.year_range, conf)
                                else:
                                    attrs.make_model, attrs.year_range, attrs.model_confidence = best
                if attrs is not None:
                    fb = (state.gemini_cls if state.gemini_cls.enabled
                          else (state.deepseek_cls if state.deepseek_cls.enabled
                                else (state.grok_cls if state.grok_cls.enabled else None)))
                    g = fb.get(det.track_id) if fb is not None else None
                    if g and g.get("label") and g.get("confidence", 0) >= 0.5:
                        attrs.make_model = g["label"]
                        attrs.year_range = g.get("year_range", "Unknown")
                        attrs.model_confidence = g.get("confidence", 0.0)
                    elif fb is not None and det.track_id in last_grok_req \
                            and attrs.make_model in ("Unknown", ""):
                        attrs.make_model = "identifying..."
                plate_status = "none"
                plate_country = ""
                plate_country_conf = 0.0
                db_match = False
                if state.anpr_cfg.get("enabled", True) and plate_text:
                    plate_status = "ANONYMIZED"
                    for region in plate_regions:
                        if not region.text:
                            continue
                        if region.country and region.country_confidence > plate_country_conf:
                            plate_country = region.country
                            plate_country_conf = region.country_confidence
                        if state.privacy.plate_db.lookup(region.text):
                            db_match = True
                            plate_status = "ANONYMIZED (DB MATCH)"
                    # Re-ID memory store: a confident label + readable plate
                    # teaches the system so the next sighting is free.
                    if attrs is not None and det.cls_name == "car" \
                            and state.anpr_cfg.get("plate_reid", True) \
                            and attrs.make_model not in ("Unknown", "", "identifying...") \
                            and attrs.model_confidence >= float(state.cls_cfg.get("min_display_conf", 0.45)):
                        plate_memory[plate_text] = {
                            "make_model": attrs.make_model,
                            "year_range": attrs.year_range,
                            "color": attrs.color,
                            "color_confidence": attrs.color_confidence,
                            "model_confidence": attrs.model_confidence,
                            "country": plate_country,
                            "_ts": time.time(),
                        }

                cls_name = det.cls_name
                _draw_vehicle(
                    state, frame_out, det,
                    attrs or type("A", (), {"make_model": "Unknown", "year_range": "Unknown",
                                            "color": "Unknown", "model_confidence": 0.0,
                                            "color_confidence": 0.0})(),
                    plate_status, (0, 255, 0),
                    plate_country, plate_country_conf,
                )
                if attrs:
                    state.log.upsert({
                        "track_id": det.track_id, "cls_name": cls_name,
                        "cls_conf": det.conf,
                        "make_model": attrs.make_model, "year_range": attrs.year_range,
                        "color": attrs.color, "color_conf": attrs.color_confidence,
                        "model_conf": attrs.model_confidence,
                        "plate_text": plate_text, "plate_status": plate_status,
                        "plate_db_match": db_match,
                        "plate_country": plate_country,
                        "plate_country_conf": plate_country_conf,
                    })
                telemetry.append({
                    "timestamp": time.strftime("%H:%M:%S"),
                    "track_id": det.track_id, "cls_name": cls_name, "conf": det.conf,
                    "make_model": attrs.make_model if attrs else "Unknown",
                    "year_range": attrs.year_range if attrs else "Unknown",
                    "color": attrs.color if attrs else "Unknown",
                    "plate_status": plate_status,  # NO raw plate text on the stream
                })

            state.recorder.write(rec_frame)  # blurred, overlay-free frame -> disk

            # Draw traffic-light state + confirmed speed limit ON the streamed
            # frame (not just the corner badge) so the driver sees them live.
            if road is not None:
                try:
                    rs = road.display_state()
                    light = rs.get("light")
                    lbbox = rs.get("light_bbox")
                    if light and lbbox:
                        cx, cy, cx2, cy2 = lbbox
                        color = {"RED": (0, 0, 255), "YELLOW": (0, 255, 255),
                                 "GREEN": (0, 255, 0)}.get(light, (255, 255, 255))
                        cv2.rectangle(frame_out, (cx, cy), (cx2, cy2), color, 3)
                        # lit-lamp indicator strip on top of the box
                        cv2.rectangle(frame_out, (cx, cy - 14), (cx + 120, cy),
                                      color, -1)
                        cv2.putText(frame_out, light, (cx + 8, cy - 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (0, 0, 0), 2, cv2.LINE_AA)
                    maxsp = rs.get("maxspeed")
                    if maxsp is not None:
                        label = f"MAX {maxsp}"
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                                      1.0, 3)
                        x0, y0 = 30, 40
                        cv2.rectangle(frame_out, (x0 - 10, y0 - th - 12),
                                      (x0 + tw + 10, y0 + 8), (30, 30, 30), -1)
                        cv2.putText(frame_out, label, (x0, y0),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                    (255, 255, 255), 3, cv2.LINE_AA)
                except Exception:  # noqa: BLE001
                    pass

            if len(last_attrs) > 400:
                cutoff = frame_no - 3000
                for t in [t for t, f in last_classify.items() if f < cutoff]:
                    last_attrs.pop(t, None)
                    last_best.pop(t, None)
                    last_best_col.pop(t, None)
                    last_classify.pop(t, None)

            # Plate re-ID memory TTL: forget plates not seen for a while so
            # the dict (and the stored personal data) can't grow unbounded.
            if state.frame_count % 300 == 0 and plate_memory:
                ttl = state.plate_memory_ttl
                now = time.time()
                for k in [k for k, v in plate_memory.items()
                          if now - v.get("_ts", 0) > ttl]:
                    plate_memory.pop(k, None)

            now = time.time()
            if now - prev_ts >= frame_interval:  # throttle stream to max_stream_fps
                prev_ts = now
                ok, buf = cv2.imencode(".jpg", frame_out, [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])
                if ok:
                    jpeg = buf.tobytes()
                    state.publish_jpeg(jpeg)
                    state.broadcast_jpeg(jpeg)
            if telemetry and state.clients:
                state.broadcast_event({"type": "detections", "detections": telemetry})

            fps_counter += 1
            now_fps = time.time()
            if now_fps - fps_ts >= 1.0:
                state.fps_now = fps_counter / (now_fps - fps_ts)
                fps_counter = 0
                fps_ts = now_fps
                if road is not None:
                    rs = road.display_state()
                    if rs != getattr(state, "_last_road", None):
                        state._last_road = rs
                        state.broadcast_event({"type": "road", "road": rs})
                state.broadcast_event({"type": "stats", "stats": state.stats()})
    finally:
        state.running = False
        if state.capture is not None:
            state.capture.release()
        state.recorder.stop()
        state.log.close()
        logger.info("Capture loop ended")


# ---------------------------------------------------------------- endpoints
@app.get("/")
async def index():
    return FileResponse(f"{BASE_DIR}/frontend/index.html")


@app.get("/api/health")
async def health():
    return {"status": "ok", "running": state.running}


@app.get("/api/stats")
async def stats():
    return state.stats()


@app.get("/api/vehicles")
async def vehicles(limit: int = 50):
    rows = state.log.recent(limit)
    for r in rows:
        # Raw plate text is personal data — it lives in the local DB only and
        # must never be served over HTTP (the frontend shows status, not text).
        r.pop("plate_text", None)
    return rows


@app.get("/api/frame")
async def frame():
    jpeg = state.snapshot()
    if jpeg is None:
        return JSONResponse({"error": "no frame yet"}, status_code=503)
    import urllib.parse

    return {"image": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()}


@app.post("/api/record")
async def record_control(payload: dict):
    action = payload.get("action")
    if action == "start":
        if state.recorder.rec_started is None:
            state.recorder.start()
        state.broadcast_event({"type": "stats", "stats": state.stats()})
        return {"ok": True, "recording": True, "path": state.recorder.path}
    if action == "stop":
        state.recorder.stop()
        state.broadcast_event({"type": "stats", "stats": state.stats()})
        return {"ok": True, "recording": False}
    return JSONResponse({"error": "action must be 'start' or 'stop'"}, status_code=400)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    ws.app_loop = asyncio.get_running_loop()
    with state._ws_lock:
        state.clients.add(ws)
    try:
        while True:
            msg = await ws.receive_text()
            if msg == "ping":
                await ws.send_text("pong")
    except WebSocketDisconnect:
        pass
    finally:
        with state._ws_lock:
            state.clients.discard(ws)


app.mount("/frontend", StaticFiles(directory=f"{BASE_DIR}/frontend"), name="frontend")


# ------------------------------------------------------------------- main
def main() -> None:
    global state  # noqa: PLW0603
    cfg_path = DEFAULT_CONFIG
    with open(cfg_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    state = DashboardState(config)
    state.load_engines()

    capture_thread = threading.Thread(target=capture_loop, args=(state,), daemon=True)
    capture_thread.start()

    host = config.get("server", {}).get("host", "0.0.0.0")
    port = int(config.get("server", {}).get("port", 8500))

    stopping = threading.Event()

    def _stop(signum, frame):  # noqa: ARG001
        logger.info("Shutting down...")
        state.running = False
        stopping.set()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    import uvicorn

    try:
        uvicorn.run(app, host=host, port=port, log_level="info")
    finally:
        # Let the capture loop finish its current frame so recorder.stop()
        # finalizes the .mp4 (moov atom) before the process exits.
        state.running = False
        capture_thread.join(timeout=15)
        logger.info("Capture loop joined — recording finalized")


if __name__ == "__main__":
    main()
