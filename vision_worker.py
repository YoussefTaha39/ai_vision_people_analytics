from __future__ import annotations
import threading
import time
import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
from ultralytics import YOLO
import torch

import database
from config import (
    YOLO_MODEL, TRACKER_CONFIG_PATH, CAMERA_WIDTH, CAMERA_HEIGHT,
    IDENTITY_GRACE_SECONDS, GHOST_BOX_SECONDS, FACE_MATCH_INTERVAL_SECONDS,
    OCCUPANCY_SAMPLE_INTERVAL_SECONDS
)
from face_service import FaceService

class VisionWorker(QThread):
    frame_ready = Signal(object, dict)  # RGB frame, metrics
    status_changed = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = dict(settings)
        self._stop_event = threading.Event()
        self._registry_reload = threading.Event()
        self._registry_lock = threading.Lock()
        self._known_visitors = []
        self._face_service = None

    def stop(self):
        self._stop_event.set()
        self.wait(6000)

    def reload_registry(self):
        self._registry_reload.set()

    @staticmethod
    def _draw_label(frame, text, x, y, color):
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.58
        thickness = 2
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        y_top = max(0, y - th - baseline - 8)
        x_left = max(0, x)
        cv2.rectangle(frame, (x_left, y_top), (x_left + tw + 10, y_top + th + baseline + 8), color, -1)
        cv2.putText(frame, text, (x_left + 5, y_top + th + 2), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    def run(self):
        cap = None
        try:
            use_gpu = bool(self.settings.get("use_gpu", True)) and torch.cuda.is_available()
            device = "0" if use_gpu else "cpu"
            device_label = f"NVIDIA GPU: {torch.cuda.get_device_name(0)}" if use_gpu else "CPU fallback"
            self.status_changed.emit(f"Loading YOLO on {device_label} ...")
            model = YOLO(YOLO_MODEL)
            cap = cv2.VideoCapture(int(self.settings.get("camera_index", 0)), cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(int(self.settings.get("camera_index", 0)))
            if not cap.isOpened():
                raise RuntimeError(
                    f"Cannot open camera index {self.settings.get('camera_index', 0)}. "
                    "Close other camera apps or try camera index 1."
                )
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, 30)

            face_enabled = bool(self.settings.get("face_matching", False))
            face_threshold = float(self.settings.get("face_threshold", 0.42))
            if face_enabled:
                self._face_service = FaceService()
                self._face_service.load()
                self._known_visitors = database.list_visitors()

            self.status_changed.emit(f"Running · {device_label}")
            previous_tick = time.perf_counter()
            smooth_fps = 0.0
            last_occupancy = 0.0
            seen_tracks = set()
            recognized_tracks = set()
            # track cache entries retain last box and identity briefly to reduce visual flicker.
            track_cache = {}
            last_match = {}
            last_recognition_logged = {}
            while not self._stop_event.is_set():
                if self._registry_reload.is_set() and face_enabled:
                    self._known_visitors = database.list_visitors()
                    self._registry_reload.clear()
                ok, frame = cap.read()
                if not ok:
                    self.msleep(20)
                    continue
                frame = cv2.flip(frame, 1)
                result = model.track(
                    source=frame,
                    persist=True,
                    tracker=str(TRACKER_CONFIG_PATH),
                    imgsz=int(self.settings.get("inference_size", 512)),
                    conf=float(self.settings.get("confidence", 0.25)),
                    classes=[0],
                    device=device,
                    verbose=False,
                )[0]
                annotated = frame.copy()
                boxes = result.boxes
                current_track_ids = set()
                current_count = 0
                if boxes is not None and len(boxes) > 0:
                    coords = boxes.xyxy.cpu().numpy().astype(int)
                    confs = boxes.conf.cpu().numpy().tolist()
                    ids_obj = boxes.id
                    ids_list = ids_obj.int().cpu().tolist() if ids_obj is not None else [-1] * len(coords)
                    current_count = len(coords)
                    now = time.time()
                    for tid, box, det_conf in zip(ids_list, coords, confs):
                        x1, y1, x2, y2 = [int(v) for v in box.tolist()]
                        x1 = max(0, min(x1, frame.shape[1]-1))
                        x2 = max(0, min(x2, frame.shape[1]-1))
                        y1 = max(0, min(y1, frame.shape[0]-1))
                        y2 = max(0, min(y2, frame.shape[0]-1))
                        if x2 <= x1 or y2 <= y1:
                            continue
                        tid_int = int(tid) if tid >= 0 else None
                        if tid_int is not None:
                            current_track_ids.add(tid_int)
                            if tid_int not in seen_tracks:
                                seen_tracks.add(tid_int)
                                database.log_event("track_started", track_id=tid_int, note="New tracking ID observed in this camera session")
                            cache = track_cache.setdefault(tid_int, {})
                            cache["box"] = (x1, y1, x2, y2)
                            cache["last_seen"] = now
                            cache.setdefault("name", None)
                            cache.setdefault("visitor_id", None)
                            cache.setdefault("match_score", None)
                            cache.setdefault("last_match_success", 0.0)
                            cache.setdefault("color", (255, 145, 50))

                            # Match only periodically; use upper person crop to emphasize the face.
                            if face_enabled and self._face_service and (now - last_match.get(tid_int, 0) >= FACE_MATCH_INTERVAL_SECONDS):
                                last_match[tid_int] = now
                                person_h = y2 - y1
                                face_crop_y2 = min(y2, y1 + max(80, int(person_h * 0.62)))
                                crop = frame[y1:face_crop_y2, x1:x2]
                                if crop.size > 0 and crop.shape[0] >= 50 and crop.shape[1] >= 50:
                                    try:
                                        visitor, score = self._face_service.match_face(
                                            crop, self._known_visitors, threshold=face_threshold
                                        )
                                        if visitor is not None:
                                            cache["name"] = visitor["name"]
                                            cache["visitor_id"] = int(visitor["id"])
                                            cache["match_score"] = score
                                            cache["last_match_success"] = now
                                            cache["color"] = (40, 200, 80)
                                            recognized_tracks.add(tid_int)
                                            if last_recognition_logged.get(tid_int) != int(visitor["id"]):
                                                database.log_event(
                                                    "recognized", visitor_id=int(visitor["id"]),
                                                    visitor_name=visitor["name"], track_id=tid_int,
                                                    confidence=score, note="Local face match"
                                                )
                                                last_recognition_logged[tid_int] = int(visitor["id"])
                                        # If the current frame is not conclusive, retain the last match only temporarily.
                                    except Exception as exc:
                                        self.status_changed.emit(f"Face matching warning: {exc}")

                            name = cache.get("name")
                            matched_at = float(cache.get("last_match_success", 0.0))
                            if name and (now - matched_at) <= IDENTITY_GRACE_SECONDS:
                                label = f"{name} · {float(cache.get('match_score') or 0):.2f}"
                                color = (40, 200, 80)
                            else:
                                label = f"Person {tid_int} · {det_conf:.2f}"
                                color = (255, 145, 50)
                            cache["label"] = label
                            cache["color"] = color
                        else:
                            label = f"Person · {det_conf:.2f}"
                            color = (255, 145, 50)

                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                        self._draw_label(annotated, label, x1, y1, color)

                # Draw last known boxes for only a very short grace period; never count ghosts as visible people.
                now = time.time()
                for tid, cache in list(track_cache.items()):
                    if tid in current_track_ids:
                        continue
                    age = now - float(cache.get("last_seen", now))
                    if age <= GHOST_BOX_SECONDS and cache.get("box"):
                        x1, y1, x2, y2 = cache["box"]
                        color = tuple(int(c * 0.65) for c in cache.get("color", (255, 145, 50)))
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)
                        label = cache.get("label", f"Person {tid} · tracking")
                        self._draw_label(annotated, label + " · tracking", x1, y1, color)
                    elif age > IDENTITY_GRACE_SECONDS + 5:
                        track_cache.pop(tid, None)
                        last_match.pop(tid, None)

                tick = time.perf_counter()
                instant_fps = 1.0 / max(tick - previous_tick, 1e-6)
                previous_tick = tick
                smooth_fps = instant_fps if smooth_fps <= 0 else 0.85 * smooth_fps + 0.15 * instant_fps
                recognized_visible = sum(
                    1 for tid in current_track_ids
                    if tid in track_cache and track_cache[tid].get("name")
                    and (now - float(track_cache[tid].get("last_match_success", 0.0))) <= IDENTITY_GRACE_SECONDS
                )
                cv2.rectangle(annotated, (8, 8), (360, 82), (25, 28, 35), -1)
                cv2.putText(annotated, f"Visible people: {current_count}", (18, 31),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(annotated, f"Track IDs: {len(current_track_ids)} | FPS: {smooth_fps:.1f}",
                            (18, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(annotated, f"Recognized: {recognized_visible}", (18, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 240, 210), 1, cv2.LINE_AA)

                if now - last_occupancy >= OCCUPANCY_SAMPLE_INTERVAL_SECONDS:
                    database.log_occupancy(current_count, recognized_visible)
                    last_occupancy = now
                rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                stats = {
                    "count": current_count,
                    "track_ids": sorted(current_track_ids),
                    "unique_tracks": len(seen_tracks),
                    "recognized_tracks": len(recognized_tracks),
                    "recognized_visible": recognized_visible,
                    "fps": round(smooth_fps, 1),
                    "device": device_label,
                    "timestamp": now,
                    "status": "Running",
                }
                self.frame_ready.emit(rgb, stats)
        except Exception as exc:
            self.error_occurred.emit(str(exc))
        finally:
            if cap is not None:
                cap.release()
            self.status_changed.emit("Camera stopped")
