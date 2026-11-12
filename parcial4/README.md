# Parcial 4 — Clasificación de clientes bancarios con representación visual y Deep Learning

Sistema que clasifica clientes del dataset **Bank Loan** en *Fully Paid* (paga el préstamo) o *Charged Off* (incumple). Cada cliente (registro tabular) se convierte en una **imagen** y se clasifica con redes convolucionales.

## Contenido

| Archivo | Descripción |
|---|---|
| `parcial4_bank_loan.ipynb` | Notebook con exploración, representaciones visuales, resultados y prueba del modelo |
| `train.py` | Entrena 2 representaciones × 3 esquemas (desde cero, transfer learning, fine tuning) y guarda métricas, figuras y el mejor modelo |
| `app.py` | Aplicación Streamlit para evaluar nuevos clientes |
| `src/preprocessing.py` | Limpieza, ingeniería de variables y normalización |
| `src/representations.py` | Representaciones visuales: matriz de características y Gramian Angular Field |
| `src/models.py` | CNN desde cero y ResNet-18 (transfer learning / fine tuning) |
| `informe_tecnico.md` / `informe_tecnico.docx` | Informe técnico |
| `guion_video.md` | Guion para el video del equipo |
| `results/` | Métricas (`metricas.csv`), historial de entrenamiento y figuras |
| `models/` | Mejor modelo (`best_model.pt`), preprocesador y metadatos |

## Uso

```bash
pip install -r requirements.txt
python train.py          # reentrena los 6 modelos (~30 min con GPU)
streamlit run app.py     # abre la aplicación
```

El modelo ya entrenado está incluido en `models/`, por lo que la aplicación funciona sin volver a entrenar.

Para PyTorch con GPU en Windows: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126`.
