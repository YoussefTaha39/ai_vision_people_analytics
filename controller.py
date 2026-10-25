from __future__ import annotations
from PySide6.QtCore import QObject, Signal
import torch
import database
from config import DEFAULT_SETTINGS
from vision_worker import VisionWorker

class AppController(QObject):
    frame_ready = Signal(object, dict)
    stats_updated = Signal(dict)
    status_changed = Signal(str)
    error_occurred = Signal(str)
    camera_running_changed = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings = database.get_settings(DEFAULT_SETTINGS)
        self.worker = None
        self.latest_stats = {
            "count": 0, "track_ids": [], "unique_tracks": 0,
            "recognized_tracks": 0, "recognized_visible": 0,
            "fps": 0.0, "device": self.gpu_label(), "timestamp": 0, "status": "Stopped"
        }

    @staticmethod
    def gpu_label():
        try:
            return f"NVIDIA GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU fallback"
        except Exception:
            return "CPU fallback"

    def is_running(self):
        return self.worker is not None and self.worker.isRunning()

    def start_camera(self):
        if self.is_running():
            return
        self.worker = VisionWorker(self.settings)
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.status_changed.connect(self.status_changed.emit)
        self.worker.error_occurred.connect(self.error_occurred.emit)
        self.worker.finished.connect(lambda: self.camera_running_changed.emit(False))
        self.worker.start()
        self.camera_running_changed.emit(True)

    def stop_camera(self):
        if self.worker is not None:
            worker = self.worker
            if worker.isRunning():
                worker.stop()
            self.worker = None
        self.camera_running_changed.emit(False)
        self.status_changed.emit("Camera stopped")

    def apply_settings(self, values):
        values = dict(values)
        self.settings.update(values)
        database.save_settings(self.settings)
        was_running = self.is_running()
        if was_running:
            self.stop_camera()
            self.start_camera()
        self.status_changed.emit("Settings saved" + (" · camera restarted" if was_running else ""))

    def reload_visitor_registry(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.reload_registry()

    def _on_frame(self, rgb, stats):
        self.latest_stats = dict(stats)
        self.frame_ready.emit(rgb, stats)
        self.stats_updated.emit(stats)
