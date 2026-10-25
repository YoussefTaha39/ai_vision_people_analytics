from __future__ import annotations
import urllib.request
import cv2
import numpy as np
from config import (
    MODELS_DIR, FACE_DETECTOR_PATH, FACE_RECOGNIZER_PATH,
    FACE_DETECTOR_URL, FACE_RECOGNIZER_URL
)

class FaceService:
    """Local face detection/embedding/matching using OpenCV Zoo YuNet + SFace."""

    @staticmethod
    def models_available():
        return (
            FACE_DETECTOR_PATH.exists() and FACE_RECOGNIZER_PATH.exists()
            and FACE_DETECTOR_PATH.stat().st_size > 100_000
            and FACE_RECOGNIZER_PATH.stat().st_size > 100_000
        )

    @staticmethod
    def download_models(progress=None):
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        for url, path in (
            (FACE_DETECTOR_URL, FACE_DETECTOR_PATH),
            (FACE_RECOGNIZER_URL, FACE_RECOGNIZER_PATH),
        ):
            if path.exists() and path.stat().st_size > 100_000:
                continue
            if progress:
                progress(f"Downloading {path.name} ...")
            request = urllib.request.Request(url, headers={"User-Agent": "AI-Vision-People-Analytics"})
            with urllib.request.urlopen(request, timeout=90) as response, open(path, "wb") as output:
                output.write(response.read())
            if path.stat().st_size <= 100_000:
                path.unlink(missing_ok=True)
                raise RuntimeError(f"Model download looks incomplete: {path.name}")
        if progress:
            progress("Face models downloaded.")

    def __init__(self):
        self.detector = None
        self.recognizer = None

    def load(self):
        if not self.models_available():
            raise FileNotFoundError("Face models are missing. Download them from the camera/visitor window.")
        self.detector = cv2.FaceDetectorYN.create(
            str(FACE_DETECTOR_PATH), "", (320, 320), 0.82, 0.30, 5000
        )
        self.recognizer = cv2.FaceRecognizerSF.create(str(FACE_RECOGNIZER_PATH), "")
        return True

    def embeddings_from_image(self, image):
        if self.detector is None or self.recognizer is None:
            self.load()
        if image is None or image.size == 0:
            return []
        h, w = image.shape[:2]
        # Keep YuNet input within a manageable size while preserving face detail.
        scale = min(1.0, 960.0 / max(w, h))
        working = cv2.resize(image, (int(w * scale), int(h * scale))) if scale < 1 else image
        wh = (working.shape[1], working.shape[0])
        self.detector.setInputSize(wh)
        _, faces = self.detector.detect(working)
        if faces is None or len(faces) == 0:
            return []
        output = []
        for face in faces:
            aligned = self.recognizer.alignCrop(working, face)
            feature = self.recognizer.feature(aligned).flatten().astype(np.float32)
            norm = np.linalg.norm(feature)
            if norm:
                feature /= norm
            confidence = float(face[14]) if len(face) > 14 else 1.0
            output.append((feature, confidence))
        return output

    def match_face(self, person_crop, known_visitors, threshold=0.42):
        features = self.embeddings_from_image(person_crop)
        if not features:
            return None, None
        query, face_quality = max(features, key=lambda item: item[1])
        best_visitor, best_score = None, -1.0
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return None, None
        for visitor in known_visitors:
            ref = visitor.get("embedding_array")
            if ref is None or len(ref) != len(query):
                continue
            score = float(np.dot(query, ref) / ((query_norm * np.linalg.norm(ref)) + 1e-9))
            if score > best_score:
                best_visitor, best_score = visitor, score
        if best_visitor is not None and best_score >= float(threshold):
            return best_visitor, best_score
        return None, best_score if best_score >= 0 else None
