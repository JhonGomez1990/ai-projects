# Medical Cost Prediction

Modelo de regresión que predice el costo médico anual (`charges`) de una persona a partir de su edad, sexo, BMI, número de hijos, hábito de fumar y región.

Dataset: [Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance) (Kaggle) — 1338 registros, incluido en `data/raw/insurance.csv`.

## Estructura

```text
medical-cost/
├── main.py                     # Pipeline: exploración, preparación, entrenamiento y comparación
├── app.py                      # Interfaz gráfica (Streamlit)
├── data/
│   ├── raw/insurance.csv       # Dataset original
│   ├── processed/              # Train / test ya codificados (se generan con main.py)
│   └── results/                # Métricas de los modelos (se generan con main.py)
├── reports/figures/            # Gráficas para el informe (se generan con main.py)
└── src/
    ├── interfaces/abstract.py  # Contrato RegressionModel + métricas (R², MAE, RMSE, MAPE)
    ├── models/                 # Un archivo por modelo + registry.py con la lista de modelos
    ├── service/                # Carga, exploración, preparación y resultados
    └── use_case/model_runner.py
```

## Preparación de los datos

- Sin valores nulos; se elimina 1 registro duplicado → 1337 registros.
- `sex` y `smoker` se codifican como 0/1.
- `region` se codifica con one-hot (`northeast` es la categoría base) porque no tiene orden.
- División 80 % entrenamiento / 20 % prueba, estratificada por `smoker` (`random_state=42`).

## Modelos

| Modelo | Archivo | Responsable |
| --- | --- | --- |
| Regresión lineal | `src/models/linear_regression_model.py` | Jhon |
| Regresión polinómica | `src/models/polynomial_regression_model.py` | Miguel |
| Regresión con Machine Learning | `src/models/random_forest_model.py` | Laura |

Todos implementan `RegressionModel` (`train`, `predict`, `evaluate`). Para agregar o cambiar un modelo basta con editar la lista en `src/models/registry.py`; `main.py` y `app.py` la usan automáticamente.

El grado del polinomio se eligió con validación cruzada de 5 particiones: el grado 2 obtiene el mejor R² de validación; los grados 3 y 4 se sobreajustan.

## Resultados (conjunto de prueba)

| Modelo | R² | MAE (USD) | RMSE (USD) | R² CV |
| --- | --- | --- | --- | --- |
| Random Forest | 0.913 | 2,212 | 3,536 | 0.834 |
| Regresión polinómica (grado 2) | 0.892 | 2,623 | 3,944 | 0.821 |
| Regresión lineal | 0.820 | 3,593 | 5,087 | 0.722 |

La variable más determinante es `smoker`. La regresión polinómica mejora mucho a la lineal porque incluye el producto `bmi × smoker`: el BMI solo dispara el costo en fumadores.

## Uso

Desde la carpeta `medical-cost`:

```bash
pip install -r requirements.txt
```

Entrenar, comparar los modelos y generar gráficas:

```bash
python main.py
```

Abrir la interfaz gráfica:

```bash
streamlit run app.py
```

La interfaz tiene tres pestañas: **Predicción** (formulario con los datos de la persona y el costo estimado por cada modelo), **Comparación de modelos** (métricas y gráficas) y **Datos** (dataset y exploración).
