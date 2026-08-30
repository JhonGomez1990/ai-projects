import cv2
import json
import zipfile
from datetime import datetime
from io import BytesIO
from pathlib import Path

class ResultStorage:

    @staticmethod
    def save_result(original_file, result_image, clusters, processing_time):
        # Identificador único de la ejecución
        timestamp = datetime.now()

        execution_id = timestamp.strftime("run_%Y%m%d_%H%M%S")

        # Crear carpeta de la ejecución
        execution_path = Path("data") / execution_id
        execution_path.mkdir(parents=True, exist_ok=True)

        # Guardar imagen original
        original_name = getattr(original_file, "name", "original.jpg")

        extension = Path(original_name).suffix

        if extension.lower() not in [".jpg", ".jpeg", ".png"]:
            extension = ".jpg"

        original_path = (
            execution_path /
            f"original{extension}"
        )

        with open(original_path, "wb") as file:
            file.write(
                original_file.getvalue()
            )

        # Guardar imagen procesada
        result_path = (
            execution_path /
            "resultado.jpg"
        )

        # OpenCV guarda utilizando BGR.
        result_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)

        cv2.imwrite(str(result_path), result_bgr)

        # Crear metadata
        height, width, channels = (result_image.shape)

        metadata = {
            "execution_id": execution_id,
            "original_filename": original_name,
            "clusters": clusters,
            "processing_time_seconds": round(
                processing_time,
                2
            ),
            "result_dimensions": {
                "height": height,
                "width": width,
                "channels": channels
            },
            "created_at": timestamp.isoformat()
        }

        metadata_path = (
            execution_path /
            "metadata.json"
        )

        with open(metadata_path, "w", encoding="utf-8") as file:

            json.dump(
                metadata,
                file,
                indent=4,
                ensure_ascii=False
            )

        # Generar ZIP
        zip_buffer = BytesIO()

        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:

            zip_file.write(
                original_path,
                original_path.name
            )

            zip_file.write(
                result_path,
                result_path.name
            )

            zip_file.write(
                metadata_path,
                metadata_path.name
            )

        zip_buffer.seek(0)

        return (execution_id, zip_buffer.getvalue())