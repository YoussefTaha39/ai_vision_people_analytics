from pathlib import Path

APP_NAME = "AI Vision & People Analytics"
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PHOTOS_DIR = BASE_DIR / "visitor_photos"
MODELS_DIR = BASE_DIR / "models"
DB_PATH = DATA_DIR / "vision_analytics.sqlite3"

YOLO_MODEL = "yolo11n.pt"
TRACKER_CONFIG_PATH = BASE_DIR / "bytetrack_custom.yaml"
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
DEFAULT_CAMERA_INDEX = 0
DEFAULT_CONFIDENCE = 0.25
DEFAULT_INFERENCE_SIZE = 512
DEFAULT_FACE_THRESHOLD = 0.42
DEFAULT_FACE_MATCHING = False
DEFAULT_USE_GPU = True
IDENTITY_GRACE_SECONDS = 6.0
GHOST_BOX_SECONDS = 0.35
FACE_MATCH_INTERVAL_SECONDS = 0.8
OCCUPANCY_SAMPLE_INTERVAL_SECONDS = 2.0

FACE_DETECTOR_URL = (
    "https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
FACE_RECOGNIZER_URL = (
    "https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/"
    "face_recognition_sface/face_recognition_sface_2021dec.onnx"
)
FACE_DETECTOR_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
FACE_RECOGNIZER_PATH = MODELS_DIR / "face_recognition_sface_2021dec.onnx"

DEFAULT_SETTINGS = {
    "camera_index": DEFAULT_CAMERA_INDEX,
    "confidence": DEFAULT_CONFIDENCE,
    "inference_size": DEFAULT_INFERENCE_SIZE,
    "face_threshold": DEFAULT_FACE_THRESHOLD,
    "face_matching": DEFAULT_FACE_MATCHING,
    "use_gpu": DEFAULT_USE_GPU,
}
