from __future__ import annotations
import shutil, time
from pathlib import Path
import cv2
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QLineEdit, QCheckBox, QFileDialog, QMessageBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QGroupBox, QFormLayout, QSplitter, QApplication, QProgressDialog,
    QTabWidget, QSizePolicy
)
import database
from config import APP_NAME, PHOTOS_DIR, BASE_DIR
from face_service import FaceService
from model_setup_worker import ModelSetupWorker

DARK_STYLE = """
QMainWindow, QWidget {
    background: #0f131c;
    color: #edf2f7;
}

QTabWidget::pane {
    border: 1px solid #252f40;
    background: #131824;
    border-radius: 8px;
    padding: 2px;
}

QTabBar::tab {
    background: #192030;
    color: #94a3b8;
    padding: 9px 20px;
    font-size: 13px;
    font-weight: 600;
    border: 1px solid #252f40;
    border-bottom: none;
    border-top-left-radius: 7px;
    border-top-right-radius: 7px;
    margin-right: 4px;
}

QTabBar::tab:selected {
    background: #2563eb;
    color: #ffffff;
    border-color: #3b82f6;
}

QTabBar::tab:hover:!selected {
    background: #222c40;
    color: #e2e8f0;
}

QGroupBox {
    border: 1px solid #2a3447;
    border-radius: 8px;
    margin-top: 14px;
    padding: 14px;
    background: #151b29;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: #60a5fa;
    font-weight: 600;
}

QLineEdit, QTableWidget {
    background: #1a2232;
    border: 1px solid #2e3a4e;
    border-radius: 5px;
    padding: 7px;
    color: #f4f7fb;
}

QLineEdit:focus {
    border: 1px solid #3b82f6;
}

QHeaderView::section {
    background: #202a3c;
    padding: 7px;
    border: 0;
    font-weight: 600;
    color: #94a3b8;
}

QPushButton {
    background: #2563eb;
    border: 0;
    border-radius: 6px;
    padding: 8px 14px;
    color: white;
    font-weight: 600;
}

QPushButton:hover {
    background: #1d4ed8;
}

QPushButton:disabled {
    background: #2e3848;
    color: #64748b;
}

QPushButton#startBtn {
    background: #059669;
}
QPushButton#startBtn:hover {
    background: #047857;
}

QPushButton#stopBtn {
    background: #dc2626;
}
QPushButton#stopBtn:hover {
    background: #b91c1c;
}

QPushButton#tabActionBtn {
    background: #3b82f6;
    border: 1px solid #60a5fa;
}
QPushButton#tabActionBtn:hover {
    background: #2563eb;
}

QPushButton#deleteBtn {
    background: #7f1d1d;
    border: 1px solid #991b1b;
    color: #fca5a5;
}
QPushButton#deleteBtn:hover {
    background: #991b1b;
    color: #ffffff;
}

QPushButton#secondaryBtn {
    background: #242e40;
    border: 1px solid #38465d;
    color: #cbd5e1;
}
QPushButton#secondaryBtn:hover {
    background: #313e54;
    color: #ffffff;
}

QCheckBox {
    spacing: 8px;
    color: #cbd5e1;
}
QCheckBox::indicator {
    width: 17px;
    height: 17px;
    border: 1px solid #38465d;
    border-radius: 4px;
    background: #1a2232;
}
QCheckBox::indicator:checked {
    background: #2563eb;
    border-color: #3b82f6;
}
"""

class CameraWindow(QMainWindow):
    closing_requested = Signal()

    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.latest_photo = None
        self.model_worker = None
        self.last_pixmap = None
        self.setWindowTitle(f"{APP_NAME} — Camera & Visitor Registration")
        self.resize(1250, 830)
        self.setStyleSheet(DARK_STYLE)
        self._build_ui()
        self.refresh_visitors()

        self.controller.frame_ready.connect(self.on_frame)
        self.controller.status_changed.connect(self.on_status_update)
        self.controller.error_occurred.connect(self.on_error)
        self.controller.camera_running_changed.connect(self.on_camera_running_changed)

        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh_visitors)
        self.refresh_timer.start(5000)

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)

        # Header Bar
        header = QHBoxLayout()
        header.setSpacing(12)

        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        title = QLabel("LIVE CAMERA")
        title.setStyleSheet("font-size:20px;font-weight:700;color:#ffffff;letter-spacing:0.5px;")
        subtitle = QLabel("High-Precision Person Tracking & Local Face Recognition")
        subtitle.setStyleSheet("color:#7e8b9f;font-size:12px;")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header.addLayout(title_box)

        header.addStretch()

        # Status badge in header (replaces noisy bottom labels)
        self.status_badge = QLabel("○ Camera Inactive")
        self.status_badge.setStyleSheet(
            "background:#1b2332;color:#94a3b8;border:1px solid #2e3a4e;border-radius:14px;padding:5px 13px;font-size:12px;font-weight:600;"
        )
        header.addWidget(self.status_badge)

        # Quick action: Switch to Add User Tab
        self.add_user_tab_btn = QPushButton("＋ Add Visitor")
        self.add_user_tab_btn.setObjectName("tabActionBtn")
        self.add_user_tab_btn.clicked.connect(self.switch_to_visitor_tab)
        header.addWidget(self.add_user_tab_btn)

        # Camera Start/Stop controls
        self.start_button = QPushButton("▶ Start Camera")
        self.start_button.setObjectName("startBtn")
        self.stop_button = QPushButton("■ Stop Camera")
        self.stop_button.setObjectName("stopBtn")
        self.stop_button.setEnabled(False)
        self.start_button.clicked.connect(self.controller.start_camera)
        self.stop_button.clicked.connect(self.controller.stop_camera)
        header.addWidget(self.start_button)
        header.addWidget(self.stop_button)

        layout.addLayout(header)

        # Main Tab Widget (Tabs at the top for clean, dedicated screens)
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)

        # TAB 1: Live Camera (The screen is dedicated 100% to the camera)
        camera_tab = QWidget()
        camera_tab_layout = QVBoxLayout(camera_tab)
        camera_tab_layout.setContentsMargins(6, 6, 6, 6)
        camera_tab_layout.setSpacing(0)

        self.video_label = QLabel("Camera is offline\n\nClick '▶ Start Camera' in the header to begin stream")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(480, 360)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_label.setStyleSheet(
            "background:#070a10;border:1px solid #1f2736;border-radius:8px;color:#6b788c;font-size:16px;line-height:1.5;"
        )
        camera_tab_layout.addWidget(self.video_label, 1)

        self.tabs.addTab(camera_tab, "🎥 Live Camera Feed")

        # TAB 2: Visitor Registration (Dedicated tab for adding & managing users)
        visitor_tab = QWidget()
        visitor_tab_layout = QHBoxLayout(visitor_tab)
        visitor_tab_layout.setContentsMargins(12, 12, 12, 12)
        visitor_tab_layout.setSpacing(16)

        # Left Column: Add Visitor Form
        left_col = QWidget()
        left_layout = QVBoxLayout(left_col)
        left_layout.setContentsMargins(0, 0, 0, 0)

        add_group = QGroupBox("Register New Visitor (إضافة مستخدم جديد)")
        form = QFormLayout(add_group)
        form.setVerticalSpacing(12)
        form.setHorizontalSpacing(14)

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Full name")
        self.ref_input = QLineEdit()
        self.ref_input.setPlaceholderText("Optional visitor / employee ID")

        # Photo selection row with thumbnail preview
        photo_box = QVBoxLayout()
        photo_actions = QHBoxLayout()
        choose_btn = QPushButton("📁 Choose photo…")
        choose_btn.setObjectName("secondaryBtn")
        choose_btn.clicked.connect(self.choose_photo)
        self.photo_label = QLabel("No photo selected")
        self.photo_label.setStyleSheet("color:#94a3b8;font-size:12px;")
        photo_actions.addWidget(choose_btn)
        photo_actions.addWidget(self.photo_label, 1)
        photo_box.addLayout(photo_actions)

        # Image preview label
        self.photo_preview = QLabel()
        self.photo_preview.setFixedSize(80, 80)
        self.photo_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.photo_preview.setStyleSheet("background:#111622;border:1px dashed #334155;border-radius:6px;color:#64748b;font-size:11px;")
        self.photo_preview.setText("Preview")
        photo_box.addWidget(self.photo_preview)

        self.consent_box = QCheckBox("I confirm appropriate consent / authorization for face matching")
        self.consent_box.setChecked(True)

        self.add_button = QPushButton("＋ Save & Register Visitor")
        self.add_button.setText("＋ Save & Register Visitor")
        self.add_button.setStyleSheet("padding:10px;font-size:14px;background:#2563eb;")
        self.add_button.clicked.connect(self.add_visitor)

        form.addRow("Name:", self.name_input)
        form.addRow("Reference:", self.ref_input)
        form.addRow("Photo:", photo_box)
        form.addRow("", self.consent_box)
        form.addRow("", self.add_button)

        left_layout.addWidget(add_group)

        # Face models box
        models_group = QGroupBox("Face Recognition Templates")
        models_layout = QVBoxLayout(models_group)
        self.face_model_status = QLabel("Status: " + ("Ready" if FaceService.models_available() else "Models not downloaded"))
        self.face_model_status.setStyleSheet("color:#94a3b8;padding:2px;")
        self.models_button = QPushButton("Download / Rebuild Face Templates")
        self.models_button.setObjectName("secondaryBtn")
        self.models_button.clicked.connect(self.download_models)
        models_layout.addWidget(self.face_model_status)
        models_layout.addWidget(self.models_button)
        left_layout.addWidget(models_group)
        left_layout.addStretch()

        visitor_tab_layout.addWidget(left_col, 4)

        # Right Column: Registered Visitors Table
        right_col = QWidget()
        right_layout = QVBoxLayout(right_col)
        right_layout.setContentsMargins(0, 0, 0, 0)

        table_group = QGroupBox("Registered Visitors (المستخدمين المسجلين)")
        table_layout = QVBoxLayout(table_group)

        self.visitor_table = QTableWidget(0, 4)
        self.visitor_table.setHorizontalHeaderLabels(["ID", "Name", "Photo File", "Face Template"])
        self.visitor_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.visitor_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.visitor_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        table_layout.addWidget(self.visitor_table, 1)

        btn_row = QHBoxLayout()
        delete_btn = QPushButton("🗑 Delete Selected Visitor")
        delete_btn.setObjectName("deleteBtn")
        delete_btn.clicked.connect(self.delete_selected_visitor)

        refresh_btn = QPushButton("🔄 Refresh List")
        refresh_btn.setObjectName("secondaryBtn")
        refresh_btn.clicked.connect(self.refresh_visitors)

        back_to_camera_btn = QPushButton("← Return to Camera")
        back_to_camera_btn.setObjectName("secondaryBtn")
        back_to_camera_btn.clicked.connect(lambda: self.tabs.setCurrentIndex(0))

        btn_row.addWidget(delete_btn)
        btn_row.addWidget(refresh_btn)
        btn_row.addStretch()
        btn_row.addWidget(back_to_camera_btn)
        table_layout.addLayout(btn_row)

        right_layout.addWidget(table_group)
        visitor_tab_layout.addWidget(right_col, 6)

        self.tabs.addTab(visitor_tab, "👤 Visitor Registration && Management")

        layout.addWidget(self.tabs, 1)

    def switch_to_visitor_tab(self):
        self.tabs.setCurrentIndex(1)
        self.name_input.setFocus()

    def on_status_update(self, status_text: str):
        if "Running" in status_text or self.controller.is_running():
            gpu = self.controller.gpu_label()
            self.status_badge.setText(f"● Live ({gpu})")
            self.status_badge.setStyleSheet(
                "background:#064e3b;color:#34d399;border:1px solid #059669;border-radius:14px;padding:5px 13px;font-size:12px;font-weight:600;"
            )
        else:
            self.status_badge.setText("○ Camera Stopped")
            self.status_badge.setStyleSheet(
                "background:#1b2332;color:#94a3b8;border:1px solid #2e3a4e;border-radius:14px;padding:5px 13px;font-size:12px;font-weight:600;"
            )

    def on_camera_running_changed(self, running):
        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running)
        if running:
            self.on_status_update("Running")
        else:
            self.status_badge.setText("○ Camera Stopped")
            self.status_badge.setStyleSheet(
                "background:#1b2332;color:#94a3b8;border:1px solid #2e3a4e;border-radius:14px;padding:5px 13px;font-size:12px;font-weight:600;"
            )
            self.video_label.setText("Camera is offline\n\nClick '▶ Start Camera' in the header to begin stream")

    def on_error(self, message):
        self.status_badge.setText("⚠ Error")
        self.status_badge.setStyleSheet(
            "background:#450a0a;color:#f87171;border:1px solid #991b1b;border-radius:14px;padding:5px 13px;font-size:12px;font-weight:600;"
        )
        QMessageBox.critical(self, "Camera / AI Error", message)

    def on_frame(self, rgb, stats):
        h, w, ch = rgb.shape
        qimage = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
        self.last_pixmap = QPixmap.fromImage(qimage)
        self._update_video_display()

    def _update_video_display(self):
        if self.last_pixmap is None or self.video_label.size().width() <= 10:
            return
        pixmap = self.last_pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.video_label.setPixmap(pixmap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_video_display()

    def choose_photo(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose a clear single-person photo", str(Path.home()),
            "Images (*.jpg *.jpeg *.png *.bmp *.webp)"
        )
        if path:
            self.latest_photo = path
            self.photo_label.setText(Path(path).name)
            pix = QPixmap(path)
            if not pix.isNull():
                self.photo_preview.setPixmap(
                    pix.scaled(
                        self.photo_preview.size(),
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation
                    )
                )

    def add_visitor(self):
        name = self.name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Missing Name", "Please enter the visitor's name.")
            return
        if not self.latest_photo:
            QMessageBox.warning(self, "Missing Photo", "Please choose a photo for the visitor.")
            return
        if not self.consent_box.isChecked():
            QMessageBox.warning(self, "Permission Needed", "Confirm appropriate consent/authorization before storing biometric face data.")
            return

        image = cv2.imread(self.latest_photo)
        if image is None:
            QMessageBox.warning(self, "Invalid Photo", "The selected image could not be opened.")
            return

        embedding = None
        if FaceService.models_available():
            try:
                service = FaceService()
                features = service.embeddings_from_image(image)
                if len(features) != 1:
                    QMessageBox.warning(
                        self, "Photo Issue",
                        f"Expected one clear face, found {len(features)}. Please select an image with exactly one clear face."
                    )
                    return
                embedding = features[0][0]
            except Exception as exc:
                QMessageBox.critical(self, "Face Template Error", str(exc))
                return

        PHOTOS_DIR.mkdir(parents=True, exist_ok=True)
        suffix = Path(self.latest_photo).suffix.lower() or ".jpg"
        safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)[:50]
        destination = PHOTOS_DIR / f"{int(time.time() * 1000)}_{safe_name}{suffix}"

        try:
            shutil.copy2(self.latest_photo, destination)
            database.add_visitor(name, str(destination), self.ref_input.text().strip(), embedding, True)
            self.latest_photo = None
            self.name_input.clear()
            self.ref_input.clear()
            self.photo_label.setText("No photo selected")
            self.photo_preview.clear()
            self.photo_preview.setText("Preview")
            self.consent_box.setChecked(True)
            self.refresh_visitors()
            self.controller.reload_visitor_registry()

            if embedding is None:
                QMessageBox.information(
                    self, "Visitor Saved",
                    "Visitor saved. Download face models and rebuild templates to enable face recognition."
                )
            else:
                QMessageBox.information(self, "Visitor Saved", "Visitor and face template saved successfully.")
        except Exception as exc:
            QMessageBox.critical(self, "Save Failed", str(exc))

    def refresh_visitors(self):
        if not hasattr(self, "visitor_table"):
            return
        visitors = database.list_visitors()
        self.visitor_table.setRowCount(len(visitors))
        for row, visitor in enumerate(visitors):
            values = [
                visitor["id"], visitor["name"], Path(visitor["photo_path"]).name,
                "Ready" if visitor.get("embedding_array") is not None else "Missing"
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if col == 3:
                    item.setForeground(Qt.GlobalColor.green if value == "Ready" else Qt.GlobalColor.yellow)
                self.visitor_table.setItem(row, col, item)

        self.face_model_status.setText(
            "Status: " + ("Ready" if FaceService.models_available() else "Models not downloaded")
        )

    def delete_selected_visitor(self):
        row = self.visitor_table.currentRow()
        if row < 0:
            QMessageBox.information(self, "Select Visitor", "Please select a visitor from the table first.")
            return
        visitor_id = int(self.visitor_table.item(row, 0).text())
        confirm = QMessageBox.question(
            self, "Delete Visitor",
            "Are you sure you want to delete this visitor, photo, and face template?"
        )
        if confirm == QMessageBox.StandardButton.Yes:
            database.delete_visitor(visitor_id)
            self.refresh_visitors()
            self.controller.reload_visitor_registry()

    def download_models(self):
        if self.model_worker and self.model_worker.isRunning():
            return
        self.models_button.setEnabled(False)
        self.model_worker = ModelSetupWorker()
        self.model_worker.progress.connect(self.face_model_status.setText)
        self.model_worker.completed.connect(self.on_models_ready)
        self.model_worker.failed.connect(self.on_models_failed)
        self.model_worker.start()

    def on_models_ready(self, built, skipped):
        self.models_button.setEnabled(True)
        self.refresh_visitors()
        self.controller.reload_visitor_registry()
        QMessageBox.information(
            self, "Face Models Ready",
            f"Face models installed.\nTemplates built: {built}\nPhotos skipped (no single clear face): {skipped}"
        )

    def on_models_failed(self, message):
        self.models_button.setEnabled(True)
        QMessageBox.critical(self, "Model Setup Failed", message)

    def closeEvent(self, event):
        if self.controller.is_running():
            self.controller.stop_camera()
        self.closing_requested.emit()
        event.accept()
