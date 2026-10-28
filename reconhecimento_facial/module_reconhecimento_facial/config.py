import os
from pathlib import Path

from dotenv import load_dotenv

# Definido antes de importar o DeepFace nos outros módulos.
os.environ["DEEPFACE_BACKEND_ENGINE"] = "tensorflow"

PROJ_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJ_ROOT / ".env")

AUTHORIZED_DIR = PROJ_ROOT / "data" / "authorized"
TEST_DIR = PROJ_ROOT / "data" / "test"
OUTPUTS_DIR = PROJ_ROOT / "outputs"

MODEL_NAME = "Facenet512"
DETECTOR_BACKEND = "retinaface"
DISTANCE_METRIC = "cosine"
MIN_FACE_CONFIDENCE = 0.5

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}