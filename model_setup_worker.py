from __future__ import annotations
import cv2
from PySide6.QtCore import QThread, Signal
import database
from face_service import FaceService

class ModelSetupWorker(QThread):
    progress = Signal(str)
    completed = Signal(int, int)  # embeddings built, skipped
    failed = Signal(str)

    def run(self):
        try:
            FaceService.download_models(self.progress.emit)
            service = FaceService()
            service.load()
            built = skipped = 0
            visitors = database.list_visitors()
            for index, visitor in enumerate(visitors, start=1):
                self.progress.emit(f"Preparing face template {index}/{len(visitors)}: {visitor['name']}")
                image = cv2.imread(visitor["photo_path"])
                if image is None:
                    skipped += 1
                    continue
                features = service.embeddings_from_image(image)
                if len(features) != 1:
                    skipped += 1
                    continue
                database.update_visitor_embedding(visitor["id"], features[0][0])
                built += 1
            self.completed.emit(built, skipped)
        except Exception as exc:
            self.failed.emit(str(exc))
