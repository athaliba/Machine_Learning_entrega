from pathlib import Path

import cv2
import numpy as np

from module_reconhecimento_facial.config import IMAGE_EXTENSIONS


def list_images(directory: Path) -> list[Path]:
    """Lista imagens, incluindo as que estão em subpastas."""
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def read_image(path: Path) -> np.ndarray:
    """Lê uma imagem em BGR, aceitando caminhos com acentos."""
    data = np.fromfile(path, dtype=np.uint8)

    if data.size == 0:
        raise ValueError(f"Arquivo vazio: {path}")

    image = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Não foi possível decodificar: {path}")

    return image


def save_image(path: Path, image: np.ndarray) -> None:
    """Salva uma imagem, aceitando caminhos com acentos."""
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(path.suffix, image)

    if not success:
        raise ValueError(f"Não foi possível codificar: {path}")

    encoded.tofile(path)