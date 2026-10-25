from __future__ import annotations
import sys
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication
from controller import AppController
from camera_window import CameraWindow
from analytics_window import AnalyticsWindow
from database import init_db
from config import APP_NAME

def place_windows(camera_window, analytics_window, app):
    screens = app.screens()
    primary = app.primaryScreen().availableGeometry()
    if len(screens) >= 2:
        # Camera and visitor registration on the primary display; analytics/settings on the second.
        camera_window.setGeometry(screens[0].availableGeometry())
        analytics_window.setGeometry(screens[1].availableGeometry())
    else:
        # With one monitor, open two independent, tiled windows.
        width = max(720, primary.width() // 2)
        camera_window.setGeometry(primary.x(), primary.y(), width, primary.height())
        analytics_window.setGeometry(primary.x() + width, primary.y(), max(640, primary.width() - width), primary.height())

def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    app.setApplicationName(APP_NAME)
    app.setQuitOnLastWindowClosed(True)
    init_db()
    controller = AppController()
    camera = CameraWindow(controller)
    analytics = AnalyticsWindow(controller)
    place_windows(camera, analytics, app)

    closing = {"busy": False}
    def close_both():
        if closing["busy"]:
            return
        closing["busy"] = True
        controller.stop_camera()
        camera.close()
        analytics.close()
        app.quit()

    camera.closing_requested.connect(close_both)
    analytics.closing_requested.connect(close_both)

    camera.show()
    analytics.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
