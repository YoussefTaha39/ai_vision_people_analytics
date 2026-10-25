from __future__ import annotations
import time
from datetime import datetime, date
import pandas as pd
from PySide6.QtCore import Qt, QTimer, Signal, QPointF
from PySide6.QtGui import QPainter, QPen, QFont, QPainterPath, QColor, QLinearGradient, QBrush
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QFormLayout, QGroupBox, QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog, QMessageBox,
    QSplitter, QApplication, QTabWidget, QSizePolicy
)
import database
from config import APP_NAME, DEFAULT_SETTINGS, DB_PATH, BASE_DIR
from face_service import FaceService
from camera_window import DARK_STYLE

class MetricCard(QGroupBox):
    def __init__(self, title, accent_color="#60a5fa", subtitle=""):
        super().__init__(title)
        self.accent_color = accent_color
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(4)

        self.value = QLabel("0")
        self.value.setStyleSheet(f"font-size:32px;font-weight:700;color:{accent_color};letter-spacing:-0.5px;")
        layout.addWidget(self.value)

        if subtitle:
            self.sub = QLabel(subtitle)
            self.sub.setStyleSheet("font-size:11px;color:#7e8b9f;")
            layout.addWidget(self.sub)

    def set_value(self, val, is_active=False):
        self.value.setText(str(val))
        if is_active and self.accent_color == "#10b981":
            # Active people present: glowing emerald
            self.value.setStyleSheet("font-size:32px;font-weight:700;color:#34d399;")
        elif self.accent_color == "#10b981" and str(val) == "0":
            self.value.setStyleSheet("font-size:32px;font-weight:700;color:#64748b;")
        else:
            self.value.setStyleSheet(f"font-size:32px;font-weight:700;color:{self.accent_color};")

class PeopleOccupancyChart(QWidget):
    """
    Dedicated high-performance real-time chart displaying
    the count of people present over time with gradient area fill,
    live pulsing head, peak reference, and integer Y-axis.
    """
    def __init__(self):
        super().__init__()
        self.samples = []
        self.time_window = 300  # Default to last 5 minutes (300s)
        self.setMinimumHeight(240)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background:#111624;border:1px solid #222d40;border-radius:8px;")

    def set_time_window(self, seconds: int):
        self.time_window = seconds
        self.update()

    def set_samples(self, samples):
        self.samples = list(samples)
        self.update()

    def add_live_sample(self, timestamp: float, people_count: int, recognized_count: int = 0):
        # Avoid duplicate timestamps
        if self.samples and abs(self.samples[-1]["timestamp"] - timestamp) < 0.5:
            self.samples[-1]["people_count"] = int(people_count)
            self.samples[-1]["recognized_count"] = int(recognized_count)
        else:
            self.samples.append({
                "timestamp": timestamp,
                "people_count": int(people_count),
                "recognized_count": int(recognized_count)
            })
        # Keep maximum 1200 samples in memory
        if len(self.samples) > 1200:
            self.samples = self.samples[-1000:]
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Background
        painter.fillRect(self.rect(), QColor("#111624"))

        # Drawing area with margins for axes
        area = self.rect().adjusted(54, 28, -24, -38)
        if area.width() <= 20 or area.height() <= 20:
            painter.end()
            return

        now = time.time()
        # Filter samples by time window if specified
        if self.time_window > 0:
            cutoff = now - self.time_window
            active_samples = [s for s in self.samples if s["timestamp"] >= cutoff]
            # If no samples in window but we have recent ones, keep last 30
            if not active_samples and self.samples:
                active_samples = self.samples[-30:]
        else:
            active_samples = self.samples

        # Draw Title & Y-Axis Label
        painter.setPen(QPen(QColor("#7e8b9f"), 1))
        painter.drawText(12, 20, "People Present (أفراد)")

        # Grid box borders
        painter.drawLine(area.left(), area.bottom(), area.right(), area.bottom())
        painter.drawLine(area.left(), area.top(), area.left(), area.bottom())

        if not active_samples:
            painter.setPen(QColor("#64748b"))
            painter.drawText(area, Qt.AlignmentFlag.AlignCenter, "Waiting for live camera occupancy data…")
            painter.end()
            return

        values = [int(s["people_count"]) for s in active_samples]
        timestamps = [float(s["timestamp"]) for s in active_samples]
        current_count = values[-1]
        peak_count = max(values)
        avg_count = sum(values) / max(1, len(values))

        # Dynamic integer Y-scale with minimum of 3
        max_y = max(3, peak_count + 1)

        # Draw horizontal grid lines & Y labels (integers only)
        grid_steps = min(5, max_y)
        for i in range(grid_steps + 1):
            val_at_grid = int(round(i * max_y / grid_steps))
            y_pos = area.bottom() - (val_at_grid / max_y) * area.height()
            if i > 0:
                painter.setPen(QPen(QColor("#1f293d"), 1, Qt.PenStyle.DashLine))
                painter.drawLine(area.left(), int(y_pos), area.right(), int(y_pos))
            painter.setPen(QColor("#7e8b9f"))
            painter.drawText(8, int(y_pos + 4), f"{val_at_grid:>3d}")

        # Compute point coordinates
        n = len(active_samples)
        t_start = timestamps[0]
        t_end = timestamps[-1]
        t_span = max(1.0, t_end - t_start)

        points = []
        for s in active_samples:
            t = float(s["timestamp"])
            val = int(s["people_count"])
            # X proportional to time
            if n > 1 and t_span > 1.0:
                x = area.left() + ((t - t_start) / t_span) * area.width()
            else:
                x = area.center().x()
            y = area.bottom() - (val / max_y) * area.height()
            points.append((x, y))

        # Draw Peak Reference Line
        if peak_count > 0:
            peak_y = area.bottom() - (peak_count / max_y) * area.height()
            painter.setPen(QPen(QColor("#f59e0b"), 1, Qt.PenStyle.DotLine))
            painter.drawLine(area.left(), int(peak_y), area.right(), int(peak_y))
            painter.setPen(QColor("#f59e0b"))
            painter.drawText(area.right() - 65, int(peak_y - 4), f"Peak: {peak_count}")

        # Draw Area Fill Under Curve
        if len(points) > 1:
            area_path = QPainterPath()
            area_path.moveTo(points[0][0], area.bottom())
            for px, py in points:
                area_path.lineTo(px, py)
            area_path.lineTo(points[-1][0], area.bottom())
            area_path.closeSubpath()

            gradient = QLinearGradient(0, area.top(), 0, area.bottom())
            gradient.setColorAt(0.0, QColor(16, 185, 129, 90))   # Emerald glow at top
            gradient.setColorAt(1.0, QColor(16, 185, 129, 5))    # Transparent at bottom
            painter.setBrush(QBrush(gradient))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawPath(area_path)

            # Draw Main Graph Line
            line_path = QPainterPath()
            line_path.moveTo(points[0][0], points[0][1])
            for px, py in points[1:]:
                line_path.lineTo(px, py)
            painter.setPen(QPen(QColor("#10b981"), 2.5))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(line_path)

        # Draw Sample Dots (nodes)
        painter.setBrush(QColor("#10b981"))
        painter.setPen(Qt.PenStyle.NoPen)
        sample_subset = points if len(points) <= 40 else points[-40:]
        for px, py in sample_subset[:-1]:
            painter.drawEllipse(int(px) - 2, int(py) - 2, 4, 4)

        # Draw Glowing Real-time Head on the latest point
        if points:
            lx, ly = points[-1]
            # Outer halo
            painter.setBrush(QColor(16, 185, 129, 80))
            painter.drawEllipse(int(lx) - 8, int(ly) - 8, 16, 16)
            # Inner solid circle
            painter.setBrush(QColor("#ffffff"))
            painter.drawEllipse(int(lx) - 4, int(ly) - 4, 8, 8)

            # Callout badge for current live value
            badge_text = f"Now: {current_count}"
            bx = min(area.right() - 65, max(area.left() + 5, int(lx - 32)))
            by = max(area.top() + 5, int(ly - 26))
            painter.setBrush(QColor("#064e3b"))
            painter.setPen(QPen(QColor("#10b981"), 1))
            painter.drawRoundedRect(bx, by, 64, 20, 5, 5)
            painter.setPen(QColor("#34d399"))
            painter.drawText(bx, by, 64, 20, Qt.AlignmentFlag.AlignCenter, badge_text)

        # X-Axis Time Labels
        painter.setPen(QColor("#7e8b9f"))
        t_first_str = datetime.fromtimestamp(timestamps[0]).strftime("%H:%M:%S")
        t_last_str = datetime.fromtimestamp(timestamps[-1]).strftime("%H:%M:%S")
        painter.drawText(area.left(), self.height() - 12, t_first_str)
        if len(timestamps) > 1 and timestamps[-1] - timestamps[0] > 60:
            t_mid_str = datetime.fromtimestamp((timestamps[0] + timestamps[-1]) / 2).strftime("%H:%M:%S")
            painter.drawText(area.center().x() - 25, self.height() - 12, t_mid_str)
        painter.drawText(area.right() - 55, self.height() - 12, t_last_str)

        # Summary HUD in top-left
        hud_text = f"Current: {current_count}  |  Peak: {peak_count}  |  Average: {avg_count:.1f}"
        painter.setPen(QColor("#a7f3d0"))
        painter.drawText(area.left() + 10, area.top() + 18, hud_text)

        painter.end()


class AnalyticsWindow(QMainWindow):
    closing_requested = Signal()

    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.setWindowTitle(f"{APP_NAME} — Statistics & Analytics Dashboard")
        self.resize(1250, 830)
        self.setStyleSheet(DARK_STYLE)
        self._build_ui()
        self.load_settings_to_ui()

        self.controller.stats_updated.connect(self.update_metrics)
        self.controller.status_changed.connect(self.on_status_changed)
        self.controller.error_occurred.connect(self.on_error)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_data)
        self.timer.start(2000)
        self.refresh_data()

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        main_layout = QVBoxLayout(root)
        main_layout.setContentsMargins(14, 12, 14, 12)
        main_layout.setSpacing(10)

        # Top Header Bar
        header = QHBoxLayout()
        header.setSpacing(12)

        titles = QVBoxLayout()
        titles.setSpacing(2)
        title = QLabel("PEOPLE ANALYTICS & OCCUPANCY DASHBOARD")
        title.setStyleSheet("font-size:20px;font-weight:700;color:#ffffff;letter-spacing:0.5px;")
        subtitle = QLabel("Live Occupancy Analytics · Historical Footfall · Visitor Event Intelligence")
        subtitle.setStyleSheet("color:#7e8b9f;font-size:12px;")
        titles.addWidget(title)
        titles.addWidget(subtitle)
        header.addLayout(titles)

        header.addStretch()

        # Status badge
        self.status_badge = QLabel("○ Camera Idle")
        self.status_badge.setStyleSheet(
            "background:#1b2332;color:#94a3b8;border:1px solid #2e3a4e;border-radius:14px;padding:5px 13px;font-size:12px;font-weight:600;"
        )
        header.addWidget(self.status_badge)

        # Export buttons in header
        export_btn = QPushButton("📥 Export Excel")
        export_btn.setObjectName("secondaryBtn")
        export_btn.clicked.connect(self.export_excel)
        csv_btn = QPushButton("📄 Export CSV")
        csv_btn.setObjectName("secondaryBtn")
        csv_btn.clicked.connect(self.export_csv)
        header.addWidget(export_btn)
        header.addWidget(csv_btn)

        main_layout.addLayout(header)

        # 4 High-Impact KPI Metric Cards
        metrics_row = QHBoxLayout()
        metrics_row.setSpacing(10)

        self.visible_card = MetricCard("People Present Now", accent_color="#10b981", subtitle="In camera view right now")
        self.peak_card = MetricCard("Peak Occupancy", accent_color="#f59e0b", subtitle="Highest count detected today")
        self.unique_card = MetricCard("Unique Persons Tracked", accent_color="#06b6d4", subtitle="Distinct tracking IDs in session")
        self.recognized_card = MetricCard("Recognized Visitors", accent_color="#a855f7", subtitle="Face matches confirmed")

        for card in (self.visible_card, self.peak_card, self.unique_card, self.recognized_card):
            metrics_row.addWidget(card)
        main_layout.addLayout(metrics_row)

        # HERO ELEMENT: People Occupancy Chart Section
        chart_box = QWidget()
        chart_layout = QVBoxLayout(chart_box)
        chart_layout.setContentsMargins(0, 0, 0, 0)
        chart_layout.setSpacing(6)

        chart_header = QHBoxLayout()
        chart_title = QLabel("Occupancy & People Presence Over Time (جراف عدد الأشخاص في الوقت الفعلي)")
        chart_title.setStyleSheet("font-size:14px;font-weight:700;color:#e2e8f0;")
        chart_header.addWidget(chart_title)
        chart_header.addStretch()

        # Time range filter buttons
        self.btn_5m = QPushButton("Last 5 Min")
        self.btn_15m = QPushButton("15 Min")
        self.btn_1h = QPushButton("1 Hour")
        self.btn_all = QPushButton("All Data")

        for btn in (self.btn_5m, self.btn_15m, self.btn_1h, self.btn_all):
            btn.setObjectName("secondaryBtn")
            btn.setStyleSheet("padding:4px 10px;font-size:11px;")
            chart_header.addWidget(btn)

        self.btn_5m.clicked.connect(lambda: self._set_chart_filter(300, self.btn_5m))
        self.btn_15m.clicked.connect(lambda: self._set_chart_filter(900, self.btn_15m))
        self.btn_1h.clicked.connect(lambda: self._set_chart_filter(3600, self.btn_1h))
        self.btn_all.clicked.connect(lambda: self._set_chart_filter(0, self.btn_all))
        self.btn_5m.setStyleSheet("padding:4px 10px;font-size:11px;background:#2563eb;color:#fff;")

        chart_layout.addLayout(chart_header)

        self.chart = PeopleOccupancyChart()
        chart_layout.addWidget(self.chart, 1)

        main_layout.addWidget(chart_box, 3)

        # Lower Tabs: Activity Log & Settings
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)

        # Tab 1: Real-time Activity Log
        events_tab = QWidget()
        events_layout = QVBoxLayout(events_tab)
        events_layout.setContentsMargins(4, 6, 4, 4)

        self.events_table = QTableWidget(0, 6)
        self.events_table.setHorizontalHeaderLabels(["Time", "Event Type", "Visitor Name", "Track ID", "Score", "Notes"])
        self.events_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.events_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.events_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        events_layout.addWidget(self.events_table)

        self.tabs.addTab(events_tab, "📋 Activity Log · Recent Detections")

        # Tab 2: Settings & Configuration
        settings_tab = QWidget()
        settings_layout = QHBoxLayout(settings_tab)
        settings_layout.setContentsMargins(12, 12, 12, 12)
        settings_layout.setSpacing(16)

        # Settings Left Column
        camera_group = QGroupBox("Camera & AI Model Configuration")
        form = QFormLayout(camera_group)
        form.setVerticalSpacing(10)

        self.camera_index = QSpinBox()
        self.camera_index.setRange(0, 8)
        self.confidence = QDoubleSpinBox()
        self.confidence.setRange(0.10, 0.90)
        self.confidence.setSingleStep(0.05)
        self.confidence.setDecimals(2)
        self.inference_size = QComboBox()
        self.inference_size.addItems(["416", "512", "640"])
        self.face_threshold = QDoubleSpinBox()
        self.face_threshold.setRange(0.25, 0.80)
        self.face_threshold.setSingleStep(0.01)
        self.face_threshold.setDecimals(2)
        self.face_enabled = QCheckBox("Identify registered visitors (Face Recognition)")
        self.use_gpu = QCheckBox("Use NVIDIA GPU acceleration when available")

        form.addRow("Camera Device Index:", self.camera_index)
        form.addRow("Detection Confidence:", self.confidence)
        form.addRow("Inference Input Size:", self.inference_size)
        form.addRow("Face Match Threshold:", self.face_threshold)
        form.addRow("", self.face_enabled)
        form.addRow("", self.use_gpu)

        apply_btn = QPushButton("💾 Save & Apply Settings")
        apply_btn.setStyleSheet("padding:9px;font-size:13px;background:#2563eb;")
        apply_btn.clicked.connect(self.apply_settings)
        form.addRow("", apply_btn)

        settings_layout.addWidget(camera_group, 5)

        # Settings Right Column: System info & Exports
        system_group = QGroupBox("System Information & Database")
        system_layout = QVBoxLayout(system_group)
        system_layout.setSpacing(10)

        self.gpu_status = QLabel("Active Device: checking…")
        self.gpu_status.setStyleSheet("color:#aab6c8;")
        system_layout.addWidget(self.gpu_status)

        self.face_models_label = QLabel()
        self.face_models_label.setStyleSheet("color:#aab6c8;")
        system_layout.addWidget(self.face_models_label)

        self.db_path_label = QLabel(f"SQLite Database:\n{DB_PATH}")
        self.db_path_label.setStyleSheet("color:#7e8b9f;font-size:11px;")
        system_layout.addWidget(self.db_path_label)

        system_layout.addStretch()

        exp_box = QHBoxLayout()
        export_btn_2 = QPushButton("📥 Export Full Excel Workbook")
        export_btn_2.setObjectName("secondaryBtn")
        export_btn_2.clicked.connect(self.export_excel)
        csv_btn_2 = QPushButton("📄 Export Events CSV")
        csv_btn_2.setObjectName("secondaryBtn")
        csv_btn_2.clicked.connect(self.export_csv)
        exp_box.addWidget(export_btn_2)
        exp_box.addWidget(csv_btn_2)
        system_layout.addLayout(exp_box)

        settings_layout.addWidget(system_group, 5)

        self.tabs.addTab(settings_tab, "⚙️ Camera && AI Settings")

        main_layout.addWidget(self.tabs, 2)

        self.gpu_status.setText("Active Device: " + self.controller.gpu_label())
        self.face_models_label.setText("Face Models: " + ("Ready" if FaceService.models_available() else "Not downloaded"))

    def _set_chart_filter(self, seconds: int, active_btn: QPushButton):
        self.chart.set_time_window(seconds)
        for btn in (self.btn_5m, self.btn_15m, self.btn_1h, self.btn_all):
            btn.setStyleSheet("padding:4px 10px;font-size:11px;")
        active_btn.setStyleSheet("padding:4px 10px;font-size:11px;background:#2563eb;color:#fff;")

    def on_status_changed(self, status: str):
        if "Running" in status or self.controller.is_running():
            self.status_badge.setText("● Live Monitoring Active")
            self.status_badge.setStyleSheet(
                "background:#064e3b;color:#34d399;border:1px solid #059669;border-radius:14px;padding:5px 13px;font-size:12px;font-weight:600;"
            )
        else:
            self.status_badge.setText("○ Camera Idle")
            self.status_badge.setStyleSheet(
                "background:#1b2332;color:#94a3b8;border:1px solid #2e3a4e;border-radius:14px;padding:5px 13px;font-size:12px;font-weight:600;"
            )

    def load_settings_to_ui(self):
        settings = self.controller.settings
        self.camera_index.setValue(int(settings.get("camera_index", DEFAULT_SETTINGS["camera_index"])))
        self.confidence.setValue(float(settings.get("confidence", DEFAULT_SETTINGS["confidence"])))
        self.inference_size.setCurrentText(str(settings.get("inference_size", DEFAULT_SETTINGS["inference_size"])))
        self.face_threshold.setValue(float(settings.get("face_threshold", DEFAULT_SETTINGS["face_threshold"])))
        self.face_enabled.setChecked(bool(settings.get("face_matching", False)))
        self.use_gpu.setChecked(bool(settings.get("use_gpu", True)))

    def apply_settings(self):
        values = {
            "camera_index": int(self.camera_index.value()),
            "confidence": float(self.confidence.value()),
            "inference_size": int(self.inference_size.currentText()),
            "face_threshold": float(self.face_threshold.value()),
            "face_matching": bool(self.face_enabled.isChecked()),
            "use_gpu": bool(self.use_gpu.isChecked()),
        }
        if values["face_matching"] and not FaceService.models_available():
            QMessageBox.warning(
                self, "Face Models Not Installed",
                "Face matching is enabled, but the models are not installed. Download models from the Visitor Registration tab on the Camera window, then apply settings again.",
                QMessageBox.StandardButton.Ok
            )
            return
        self.controller.apply_settings(values)
        QMessageBox.information(self, "Settings Saved", "Settings saved successfully. If the camera was running, it has restarted with the new parameters.")

    def update_metrics(self, stats):
        count = stats.get("count", 0)
        self.visible_card.set_value(count, is_active=(count > 0))
        self.unique_card.set_value(stats.get("unique_tracks", 0))
        self.recognized_card.set_value(stats.get("recognized_tracks", 0))

        # Peak occupancy check
        current_peak = database.get_peak_occupancy()
        self.peak_card.set_value(max(current_peak, count))

        device_str = str(stats.get("device", "Not running"))
        self.gpu_status.setText(f"Active Device: {device_str} ({stats.get('fps', 0):.1f} FPS)")

        # Real-time instant chart update
        now = stats.get("timestamp", time.time())
        recognized_visible = stats.get("recognized_visible", 0)
        self.chart.add_live_sample(now, count, recognized_visible)

        if self.controller.is_running():
            self.on_status_changed("Running")

    def refresh_data(self):
        # Refresh chart samples from SQLite database
        samples = database.get_occupancy(600)
        if samples:
            self.chart.set_samples(samples)

        # Update peak card
        peak = database.get_peak_occupancy()
        self.peak_card.set_value(peak)

        # Refresh recent events table
        events = database.get_events(150)
        self.events_table.setRowCount(len(events))
        for r, item in enumerate(events):
            timestamp = datetime.fromtimestamp(item["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
            vals = [
                timestamp,
                item["event_type"],
                item["visitor_name"] or "—",
                item["track_id"] if item["track_id"] is not None else "—",
                f"{item['confidence']:.2f}" if item["confidence"] is not None else "—",
                item["note"] or ""
            ]
            for c, val in enumerate(vals):
                cell = QTableWidgetItem(str(val))
                if c == 1 and val == "recognized":
                    cell.setForeground(Qt.GlobalColor.green)
                self.events_table.setItem(r, c, cell)

        self.face_models_label.setText(
            "Face Models: " + ("Ready" if FaceService.models_available() else "Not downloaded")
        )

    def export_excel(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Complete Excel Workbook", "ai_vision_analytics.xlsx",
            "Excel Workbook (*.xlsx)"
        )
        if not path:
            return
        try:
            events = database.get_events(1_000_000)
            occupancy = database.get_occupancy(1_000_000)
            visitors = database.list_visitors()
            event_rows = []
            for row in events:
                clean = {k: v for k, v in row.items() if k != "embedding"}
                clean["time_local"] = datetime.fromtimestamp(row["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
                event_rows.append(clean)
            occupancy_rows = [
                {**row, "time_local": datetime.fromtimestamp(row["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")}
                for row in occupancy
            ]
            visitor_rows = [{
                "id": v["id"], "name": v["name"], "external_ref": v["external_ref"],
                "photo_path": v["photo_path"], "consent_confirmed": v["consent_confirmed"],
                "has_face_template": v.get("embedding_array") is not None,
                "created_at_local": datetime.fromtimestamp(v["created_at"]).strftime("%Y-%m-%d %H:%M:%S"),
            } for v in visitors]
            with pd.ExcelWriter(path, engine="openpyxl") as writer:
                pd.DataFrame(event_rows).to_excel(writer, sheet_name="Events", index=False)
                pd.DataFrame(occupancy_rows).to_excel(writer, sheet_name="Occupancy", index=False)
                pd.DataFrame(visitor_rows).to_excel(writer, sheet_name="Visitors", index=False)
            QMessageBox.information(self, "Export Complete", f"Workbook saved successfully:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Failed", str(exc))

    def export_csv(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Events to CSV", "ai_vision_events.csv", "CSV (*.csv)"
        )
        if not path:
            return
        try:
            rows = database.get_events(1_000_000)
            clean = []
            for row in rows:
                item = dict(row)
                item["time_local"] = datetime.fromtimestamp(row["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
                clean.append(item)
            pd.DataFrame(clean).to_csv(path, index=False, encoding="utf-8-sig")
            QMessageBox.information(self, "Export Complete", f"CSV saved successfully:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Failed", str(exc))

    def on_error(self, message):
        QMessageBox.critical(self, "Camera / AI Error", message)

    def closeEvent(self, event):
        self.closing_requested.emit()
        event.accept()
