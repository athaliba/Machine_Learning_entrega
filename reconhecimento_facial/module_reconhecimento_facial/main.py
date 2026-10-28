import csv

import cv2
from loguru import logger

from module_reconhecimento_facial.config import OUTPUTS_DIR, TEST_DIR
from module_reconhecimento_facial.database import build_database
from module_reconhecimento_facial.image_io import (
    list_images,
    read_image,
    save_image,
)
from module_reconhecimento_facial.recognizer import recognize_faces


def draw_results(image, results):
    """Desenha os rostos e as decisões em uma cópia da imagem."""
    annotated = image.copy()
    scale = max(0.4, min(image.shape[:2]) / 900)
    thickness = max(1, round(scale * 2))

    for result in results:
        box = result["box"]
        x, y, w, h = (int(box[key]) for key in ("x", "y", "w", "h"))

        authorized = result["authorized"]
        color = (0, 180, 0) if authorized else (0, 0, 220)
        label = (
            f"{result['name']} - CADASTRADO"
            if authorized
            else "DESCONHECIDO"
        )

        cv2.rectangle(
            annotated, (x, y), (x + w, y + h), color, thickness
        )
        cv2.putText(
            annotated,
            label,
            (max(0, x), max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    return annotated


def main():
    images = list_images(TEST_DIR)

    if not images:
        raise ValueError("Nenhuma imagem encontrada em data/test.")

    database = build_database()
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for path in images:
        relative = path.relative_to(TEST_DIR)
        image_name = relative.as_posix()

        try:
            image = read_image(path)
            results = recognize_faces(image, database)

            # Preserva subpastas e a extensão original no nome,
            # evitando colisões entre exemplo.jpg e exemplo.png.
            output_path = (
                OUTPUTS_DIR
                / relative.parent
                / f"{relative.name}.resultado.jpg"
            )
            save_image(output_path, draw_results(image, results))

            if not results:
                rows.append({
                    "image": image_name,
                    "status": "sem_rosto",
                })
                logger.warning("{}: nenhum rosto elegível.", image_name)
                continue

            for face_index, result in enumerate(results, start=1):
                status = (
                    "cadastrado"
                    if result["authorized"]
                    else "desconhecido"
                )
                rows.append({
                    "image": image_name,
                    "face_index": face_index,
                    "status": status,
                    "name": result["name"],
                    "distance": result["distance"],
                    "threshold": result["threshold"],
                    "face_confidence": result["face_confidence"],
                })
                logger.info(
                    "{} | {} | distância={:.4f}",
                    image_name,
                    status,
                    result["distance"],
                )

        except (ValueError, OSError, cv2.error) as error:
            rows.append({
                "image": image_name,
                "status": "erro",
                "error": str(error),
            })
            logger.error("{}: {}", image_name, error)

    report_path = OUTPUTS_DIR / "recognition_results.csv"
    columns = [
        "image",
        "face_index",
        "status",
        "name",
        "distance",
        "threshold",
        "face_confidence",
        "error",
    ]

    with report_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    logger.success("Resultados salvos em: {}", OUTPUTS_DIR)


if __name__ == "__main__":
    main()