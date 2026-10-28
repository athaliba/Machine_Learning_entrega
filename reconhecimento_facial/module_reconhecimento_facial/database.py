from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger

from module_reconhecimento_facial.config import (
    AUTHORIZED_DIR,
    DETECTOR_BACKEND,
    MIN_FACE_CONFIDENCE,
    MODEL_NAME,
)
from module_reconhecimento_facial.image_io import list_images, read_image

# A configuração do backend precisa ser carregada antes do DeepFace.
from deepface import DeepFace


@dataclass
class RegisteredFace:
    name: str
    embedding: np.ndarray
    source: Path


def build_database() -> list[RegisteredFace]:
    """Gera um embedding para cada foto válida do cadastro."""
    database = []

    for path in list_images(AUTHORIZED_DIR):
        relative = path.relative_to(AUTHORIZED_DIR)

        if len(relative.parts) < 2:
            logger.warning(
                "Coloque a foto em uma pasta com o nome da pessoa: {}",
                path.name,
            )
            continue

        name = relative.parts[0]

        try:
            image = read_image(path)
            faces = DeepFace.represent(
                img_path=image,
                model_name=MODEL_NAME,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=True,
                align=True,
            )

            # O cadastro exige exatamente um rosto por foto.
            if len(faces) != 1:
                logger.warning(
                    "{}: encontrados {} rostos; foto ignorada.",
                    path.name,
                    len(faces),
                )
                continue

            face = faces[0]

            if face["face_confidence"] < MIN_FACE_CONFIDENCE:
                logger.warning(
                    "{}: confiança de detecção insuficiente.",
                    path.name,
                )
                continue

            embedding = np.asarray(face["embedding"], dtype=np.float64)

            if (
                embedding.ndim != 1
                or not np.isfinite(embedding).all()
                or np.linalg.norm(embedding) == 0
            ):
                raise ValueError("Embedding inválido.")

            database.append(
                RegisteredFace(
                    name=name,
                    embedding=embedding,
                    source=path,
                )
            )
            logger.info("Cadastro carregado: {} | {}", name, path.name)

        except (ValueError, OSError) as error:
            logger.warning("{}: {}", path.name, error)

    if not database:
        raise ValueError("Nenhuma foto válida foi encontrada no cadastro.")

    logger.success(
        "Cadastro pronto: {} fotos de {} pessoas.",
        len(database),
        len({face.name for face in database}),
    )
    return database