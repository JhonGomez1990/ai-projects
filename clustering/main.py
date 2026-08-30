import time
import streamlit as st
from src.camera import CameraInput
from src.image_clustering import ImageClustering
from src.result_storage import ResultStorage

def main():
    st.set_page_config(
        page_title="Image Clustering",
        page_icon="📷",
        layout="centered"
    )

    st.title("Image Clustering")

    st.write("Captura una imagen usando la cámara de tu dispositivo.")

    # Capturar imagen desde la cámara
    picture = CameraInput.camera_input()

    if picture is not None:

        k = st.number_input("Ingrese el número de conjuntos deseados", min_value=2, max_value=10, value=3, step=1)

        if st.button("Procesar imagen", key="process_image"):

            start_time = time.time()

            with st.spinner("Procesando imagen con KMeans..."):
                Img_resultado = ImageClustering.process(picture, int(k))

            end_time = time.time()

            processing_time = end_time - start_time

            st.success(
                f"Procesamiento completado en "
                f"{processing_time:.2f} segundos"
            )

            st.subheader("Imagen resultante")

            st.image(Img_resultado, caption=f"Resultado con {k} clusters")

            # Guardar resultados
            execution_id, zip_file = ResultStorage.save_result(
                picture,
                Img_resultado,
                int(k),
                processing_time
            )

            st.success(f"Resultado almacenado: {execution_id}")

            # Descargar resultados
            st.download_button(
                label="Descargar resultados",
                data=zip_file,
                file_name=f"{execution_id}.zip",
                mime="application/zip"
            )

if __name__ == "__main__":
    main()