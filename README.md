from pathlib import Path

content = """# AI Challenges

Repositorio destinado al desarrollo de los retos propuestos durante el curso.

Actualmente contiene tres proyectos independientes.

---

## Projects

### 1. Image Clustering App

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![Camera](https://img.shields.io/badge/Camera-Ready-brightgreen)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

Aplicación para captura y procesamiento de imágenes utilizando técnicas de clustering.

Funcionalidades:

- Captura de imágenes desde la cámara.
- Selección de cantidad de clusters.
- Procesamiento de imágenes mediante clustering.
- Clasificación manual.
- Organización y descarga de resultados.

📁 [Ver proyecto](./clustering)

---

### 2. Segundo Proyecto

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

Bot de preguntas y respuestas que evalúa síntomas marcados por el usuario
contra una base de conocimiento estructurada de trastornos, y genera un
reporte preliminar de coincidencias mediante reglas de inferencia.


📁 [Ver proyecto](./sistemas_expertos)

---

### 3. Customer Churn Classification

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-150458?logo=pandas&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![Status](https://img.shields.io/badge/Status-In%20Development-yellow)

Proyecto de clasificación para predecir si un cliente abandonará o permanecerá en el servicio a partir de sus características.

Funcionalidades:

- Carga de datasets de entrenamiento y prueba.
- Exploración de datos.
- Limpieza y preparación de datos.
- Codificación de variables categóricas.
- Generación de datasets procesados.
- Implementación del modelo K-Nearest Neighbors (KNN).
- Evaluación del modelo mediante accuracy.
- Consolidación de resultados de los modelos.

📁 [Ver proyecto](./customer-churn)

---
"""

path = Path("/mnt/data/README.md")
path.write_text(content, encoding="utf-8")
print(path)
