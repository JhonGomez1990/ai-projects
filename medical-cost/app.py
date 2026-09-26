from pathlib import Path
import pandas as pd
import streamlit as st
from src.models.registry import get_models
from src.service.data_exploration import load_data
from src.service.data_preprocessing import build_input, clean_data, split_data, split_features_target
from src.use_case.model_runner import run_model

BASE_DIR = Path(__file__).resolve().parent
RAW_FILE = BASE_DIR / "data" / "raw" / "insurance.csv"
FIGURES_DIR = BASE_DIR / "reports" / "figures"

SEX_OPTIONS = {"Femenino": "female", "Masculino": "male"}
SMOKER_OPTIONS = {"No": "no", "Sí": "yes"}
REGION_OPTIONS = {
    "Noreste (northeast)": "northeast",
    "Noroeste (northwest)": "northwest",
    "Sureste (southeast)": "southeast",
    "Suroeste (southwest)": "southwest",
}

st.set_page_config(page_title="Predicción de costos médicos", page_icon="🩺", layout="wide")

@st.cache_resource(show_spinner="Entrenando los modelos...")
def train_models():
    """Entrena todos los modelos una sola vez y los deja en memoria."""

    df = load_data(RAW_FILE)
    train_df, test_df = split_data(clean_data(df))

    x_train, y_train = split_features_target(train_df)
    x_test, y_test = split_features_target(test_df)

    models = get_models()
    results = pd.DataFrame([run_model(model, x_train, y_train, x_test, y_test) for model in models])

    return models, results, df

def bmi_category(bmi: float) -> str:
    if bmi < 18.5:
        return "bajo peso"
    if bmi < 25:
        return "peso normal"
    if bmi < 30:
        return "sobrepeso"
    return "obesidad"

models, results, raw_df = train_models()
best_model_name = results.sort_values("r2_test", ascending=False).iloc[0]["model"]

st.title("🩺 Predicción de costos médicos")
st.caption(
    "Estima el costo anual del seguro médico de una persona a partir de sus características, "
    "usando tres modelos de regresión entrenados con el Medical Cost Personal Dataset (Kaggle)."
)

prediction_tab, comparison_tab, data_tab = st.tabs(["Predicción", "Comparación de modelos", "Datos"])

with prediction_tab:
    form_column, result_column = st.columns([1, 2], gap="large")

    with form_column:
        st.subheader("Datos de la persona")

        age = st.slider("Edad", min_value=18, max_value=64, value=35)
        sex_label = st.radio("Sexo", list(SEX_OPTIONS), horizontal=True)
        smoker_label = st.radio("¿Fuma?", list(SMOKER_OPTIONS), horizontal=True)
        children = st.number_input("Número de hijos", min_value=0, max_value=5, value=0, step=1)
        region_label = st.selectbox("Región de residencia (EE. UU.)", list(REGION_OPTIONS))

        use_height_weight = st.toggle("Calcular el BMI con peso y estatura")

        if use_height_weight:
            weight = st.number_input("Peso (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.5)
            height = st.number_input("Estatura (cm)", min_value=120.0, max_value=220.0, value=170.0, step=0.5)
            bmi = round(weight / (height / 100) ** 2, 2)
        else:
            bmi = st.number_input("Índice de masa corporal (BMI)", min_value=15.0, max_value=55.0, value=27.0, step=0.1)

        st.caption(f"BMI: **{bmi:.1f}** ({bmi_category(bmi)})")

        if not 15.9 <= bmi <= 53.2:
            st.warning("El BMI está fuera del rango de los datos de entrenamiento (16–53); la predicción es menos confiable.")

    person = build_input(
        age=age,
        sex=SEX_OPTIONS[sex_label],
        bmi=bmi,
        children=int(children),
        smoker=SMOKER_OPTIONS[smoker_label],
        region=REGION_OPTIONS[region_label],
    )

    predictions = {model.name: max(float(model.predict(person)[0]), 0.0) for model in models}

    with result_column:
        st.subheader("Costo estimado")

        best_prediction = predictions[best_model_name]
        best_mae = results.set_index("model").loc[best_model_name, "mae_test"]

        st.metric(f"Mejor modelo: {best_model_name}", f"${best_prediction:,.0f} USD / año")
        st.caption(f"En el conjunto de prueba este modelo se equivoca en promedio ±${best_mae:,.0f}.")

        st.markdown("**Predicción de cada modelo**")

        prediction_columns = st.columns(len(models))

        for column, (name, value) in zip(prediction_columns, predictions.items()):
            r2 = results.set_index("model").loc[name, "r2_test"]
            column.metric(name, f"${value:,.0f}", help=f"R² en prueba: {r2:.3f}")

        st.bar_chart(
            pd.DataFrame({"Modelo": list(predictions), "Costo (USD)": list(predictions.values())}),
            x="Modelo",
            y="Costo (USD)",
            horizontal=True,
        )

        # Contexto: personas parecidas en el dataset.
        similar = raw_df[
            (raw_df["smoker"] == SMOKER_OPTIONS[smoker_label])
            & (raw_df["age"].between(age - 5, age + 5))
        ]

        if len(similar) > 0:
            st.info(
                f"En el dataset, las personas de {age - 5} a {age + 5} años con el mismo hábito de fumar "
                f"({len(similar)} registros) pagan en promedio **${similar['charges'].mean():,.0f}**."
            )

with comparison_tab:
    st.subheader("Métricas en el conjunto de prueba")
    st.caption(
        "R²: proporción de la variación del costo que explica el modelo (1 = perfecto). "
        "MAE: error promedio en dólares. RMSE: similar al MAE pero castiga más los errores grandes. "
        "R² CV: promedio de validación cruzada de 5 particiones sobre el entrenamiento."
    )

    table = results.sort_values("r2_test", ascending=False).rename(
        columns={
            "model": "Modelo",
            "r2_train": "R² entrenamiento",
            "r2_test": "R² prueba",
            "r2_cv_mean": "R² CV (media)",
            "r2_cv_std": "R² CV (desv.)",
            "mae_test": "MAE (USD)",
            "rmse_test": "RMSE (USD)",
            "mape_test": "MAPE (%)",
        }
    )

    st.dataframe(
        table.style.format(
            {
                "R² entrenamiento": "{:.4f}",
                "R² prueba": "{:.4f}",
                "R² CV (media)": "{:.4f}",
                "R² CV (desv.)": "{:.4f}",
                "MAE (USD)": "${:,.0f}",
                "RMSE (USD)": "${:,.0f}",
                "MAPE (%)": "{:.1f}%",
            }
        ),
        hide_index=True,
        width="stretch",
    )

    for figure, caption in [
        ("05_comparacion_metricas.png", "Comparación de métricas"),
        ("06_real_vs_predicho.png", "Costo real vs. predicho (la línea roja es la predicción perfecta)"),
        ("07_seleccion_grado_polinomio.png", "Selección del grado del polinomio"),
    ]:
        if (FIGURES_DIR / figure).exists():
            st.image(str(FIGURES_DIR / figure), caption=caption)

    if not (FIGURES_DIR / "05_comparacion_metricas.png").exists():
        st.caption("Ejecute `python main.py` para generar las gráficas de comparación.")

with data_tab:
    st.subheader("Medical Cost Personal Dataset")
    st.caption(f"{len(raw_df)} registros · 6 variables de entrada · variable objetivo: charges (costo en USD)")
    st.dataframe(raw_df, width="stretch", height=300)

    for figure, caption in [
        ("01_distribucion_costo.png", "Distribución del costo"),
        ("02_costo_por_categoria.png", "Costo por sexo, hábito de fumar y región"),
        ("03_costo_vs_edad_bmi.png", "Costo vs. edad y BMI"),
        ("04_matriz_correlacion.png", "Matriz de correlación"),
    ]:
        if (FIGURES_DIR / figure).exists():
            st.image(str(FIGURES_DIR / figure), caption=caption)
