"""
recorder.py — anonymized video writer + session telemetry log.

GDPR contract: this module accepts ONLY already-anonymized frames (the
privacy engine blurs plates on the output buffer BEFORE write/stream), so
the saved .mp4 never contains a readable license plate. The raw plate text
is stored only in the local SQLite session log, never sent to the stream.
"""

from __future__ import annotations

import datetime as dt
import logging
import os
import sqlite3
import threading
import time

import cv2

logger = logging.getLogger("mobile_tracker.recorder")


class VideoRecorder:
    """Writes the anonymized output stream to disk at native camera FPS."""

    def __init__(self, rec_cfg: dict, fps: float):
        self.cfg = rec_cfg
        self.fps = fps
        self.writer: cv2.VideoWriter | None = None
        self.path: str = ""
        self.rec_started: dt.datetime | None = None
        self._lock = threading.Lock()
        self._pending_codec = None
        self._pending_path = ""
        self._last_ts = 0.0
        self._dup_acc = 0.0

    def start(self) -> bool:
        directory = self.cfg.get("recordings_dir", "storage/recordings")
        os.makedirs(directory, exist_ok=True)
        ext = self.cfg.get("output_ext", "mp4")
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(directory, f"session_{stamp}.{ext}")
        codec = cv2.VideoWriter_fourcc(*self.cfg.get("codec", "mp4v"))
        # Writer is (re)opened lazily once the first frame size is known
        self._pending_codec = codec
        self._pending_path = self.path
        self.writer = None
        self.rec_started = dt.datetime.now()
        logger.info("Recorder armed: %s", self.path)
        return True

    def write(self, frame) -> None:
        if not self.enabled:
            return
        with self._lock:
            if self.writer is None and not self._pending_path:
                return  # recording not started (manual REC button)
            if self.writer is None and frame is not None:
                h, w = frame.shape[:2]
                self.writer = cv2.VideoWriter(
                    self._pending_path, self._pending_codec, max(1.0, self.fps), (w, h)
                )
                if not self.writer.isOpened():
                    logger.error("VideoWriter could not open %s", self._pending_path)
                    self.writer = None
                    return
                logger.info("Recording started: %s (%dx%d @ %.1f fps)",
                            self._pending_path, w, h, self.fps)
                self._last_ts = time.time()
                self._dup_acc = 0.0
            if self.writer is not None:
                # Real-time playback: the writer runs at the camera fps, but a
                # CPU-bound pipeline may deliver fewer frames than that. Duplicate
                # frames with an accumulator (dt*fps accumulates fractional copies)
                # so the .mp4 always spans the true recording duration instead of
                # fast-forwarding or freezing.
                now = time.time()
                elapsed = now - self._last_ts
                self._last_ts = now
                if elapsed > 0:
                    self._dup_acc += elapsed * self.fps
                    copies = int(self._dup_acc)
                    self._dup_acc -= copies
                else:
                    copies = 1
                for _ in range(max(1, copies)):
                    self.writer.write(frame)

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.get("enabled", True))

    def stop(self) -> None:
        with self._lock:
            self.rec_started = None
            # Clear the pending path so a later write() cannot reopen the
            # just-finalized file and truncate/corrupt it.
            self._pending_path = ""
            self._pending_codec = None
            if self.writer is not None:
                self.writer.release()
                self.writer = None
                logger.info("Recording stopped: %s", self.path)


class SessionLog:
    """SQLite telemetry log (upsert per track). Plate text is internal only."""

    def __init__(self, db_path: str, retention_days: int = 0):
        self.db_path = db_path
        self.retention_days = int(retention_days or 0)
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_schema()
        self.prune()

    def _create_schema(self) -> None:
        with self._lock:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS vehicles (
                    track_id      TEXT PRIMARY KEY,
                    first_seen    TEXT NOT NULL,
                    last_seen     TEXT NOT NULL,
                    cls           TEXT,
                    cls_conf      REAL,
                    make_model    TEXT,
                    year_range    TEXT,
                    color         TEXT,
                    color_conf    REAL,
                    model_conf    REAL,
                    plate_text    TEXT,
                    plate_status  TEXT DEFAULT 'none',
                    plate_db_match INTEGER DEFAULT 0,
                    frames_seen   INTEGER DEFAULT 1
                )
            """)
            try:
                self._conn.execute("ALTER TABLE vehicles ADD COLUMN model_conf REAL")
            except sqlite3.OperationalError:
                pass  # column already exists
            self._conn.commit()

    def prune(self) -> None:
        """Drop session rows older than retention_days (0 = keep forever).
        Plates are personal data; a bounded log is the GDPR-safe default."""
        if not self.retention_days:
            return
        cutoff = (dt.datetime.now()
                  - dt.timedelta(days=self.retention_days)).isoformat(timespec="seconds")
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM vehicles WHERE last_seen < ?", (cutoff,))
            self._conn.commit()
            if cur.rowcount:
                logger.info("pruned %d stale session rows (retention %d d)",
                            cur.rowcount, self.retention_days)

    def upsert(self, entry: dict) -> None:
        now = dt.datetime.now().isoformat(timespec="seconds")
        with self._lock:
            self._conn.execute("""
                INSERT INTO vehicles
                    (track_id, first_seen, last_seen, cls, cls_conf, make_model,
                     year_range, color, color_conf, model_conf, plate_text,
                     plate_status, plate_db_match, frames_seen)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                ON CONFLICT(track_id) DO UPDATE SET
                    last_seen = excluded.last_seen,
                    cls = excluded.cls,
                    make_model = excluded.make_model,
                    year_range = excluded.year_range,
                    color = excluded.color,
                    color_conf = excluded.color_conf,
                    model_conf = excluded.model_conf,
                    plate_text = excluded.plate_text,
                    plate_status = excluded.plate_status,
                    plate_db_match = excluded.plate_db_match,
                    frames_seen = vehicles.frames_seen + 1
            """, (
                str(entry["track_id"]), entry.get("first_seen", now), now,
                entry.get("cls_name"), entry.get("cls_conf"),
                entry.get("make_model"), entry.get("year_range"),
                entry.get("color"), entry.get("color_conf"),
                entry.get("model_conf"),
                entry.get("plate_text") or "",
                entry.get("plate_status", "none"),
                1 if entry.get("plate_db_match") else 0,
            ))
            self._conn.commit()

    def recent(self, limit: int = 50) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM vehicles ORDER BY last_seen DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def update_grok(self, track_id: int, make_model: str, year_range: str,
                    confidence: float) -> None:
        """Apply a Grok make/model result to a track even after it left the
        frame (async response arriving late)."""
        with self._lock:
            self._conn.execute(
                "UPDATE vehicles SET make_model = ?, year_range = ?, "
                "model_conf = MAX(COALESCE(model_conf, 0), ?) WHERE track_id = ?",
                (make_model, year_range, confidence, str(track_id)),
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
