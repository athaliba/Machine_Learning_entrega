import numpy as np

from module_reconhecimento_facial.config import (
    DETECTOR_BACKEND,
    DISTANCE_METRIC,
    MIN_FACE_CONFIDENCE,
    MODEL_NAME,
)
from module_reconhecimento_facial.database import RegisteredFace

from deepface import DeepFace
from deepface.modules.verification import find_threshold


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calcula a distância cosseno: quanto menor, mais semelhantes."""
    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        raise ValueError("Não é possível comparar um embedding nulo.")

    similarity = np.dot(a, b) / denominator
    return float(1.0 - np.clip(similarity, -1.0, 1.0))


def recognize_faces(
    image: np.ndarray,
    database: list[RegisteredFace],
) -> list[dict]:
    """Compara cada rosto detectado com as fotos cadastradas."""
    if not database:
        raise ValueError("O cadastro está vazio.")

    threshold = float(find_threshold(MODEL_NAME, DISTANCE_METRIC))

    faces = DeepFace.represent(
        img_path=image,
        model_name=MODEL_NAME,
        detector_backend=DETECTOR_BACKEND,
        enforce_detection=False,
        align=True,
    )

    results = []

    for face in faces:
        confidence = float(face["face_confidence"])

        # Descarta inclusive o resultado sem detecção real de rosto.
        if confidence < MIN_FACE_CONFIDENCE:
            continue

        embedding = np.asarray(face["embedding"], dtype=np.float64)

        if not np.isfinite(embedding).all():
            raise ValueError("O modelo retornou um embedding inválido.")

        distances = [
            cosine_distance(embedding, registered.embedding)
            for registered in database
        ]

        best_index = int(np.argmin(distances))
        distance = distances[best_index]
        authorized = distance <= threshold

        results.append({
            "name": database[best_index].name if authorized else "desconhecido",
            "authorized": authorized,
            "distance": distance,
            "threshold": threshold,
            "face_confidence": confidence,
            "box": face["facial_area"],
        })

    return results