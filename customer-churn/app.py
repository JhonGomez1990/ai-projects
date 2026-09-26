import streamlit as st
import pandas as pd
from pathlib import Path

from src.service.data_preprocessing import clean_data
from src.models.knn_model import KNNModel
from src.models.logistic_model import LogisticModel
from src.models.decision_tree_model import DecisionTreeModel
from src.models.svm_model import SVMModel


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

TRAIN_FILE = (
    BASE_DIR
    / "data"
    / "raw"
    / "customer_churn_dataset-training-master.csv"
)


# ============================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================

st.set_page_config(
    page_title="Customer Churn",
    page_icon="📊",
    layout="centered",
)


# ============================================================
# TÍTULO
# ============================================================

st.title("📊 Customer Churn Prediction")

st.write(
    "Ingrese las características de un cliente para predecir "
    "si abandonará o permanecerá en el servicio."
)


# ============================================================
# CARGAR Y PREPARAR LOS DATOS
# ============================================================

@st.cache_data
def load_training_data():
    df = pd.read_csv(TRAIN_FILE)
    return clean_data(df)


# ============================================================
# ENTRENAR LOS MODELOS
# ============================================================

@st.cache_resource
def train_models(df):

    x_train = df.drop(columns=["Churn"])
    y_train = df["Churn"]

    models = {
        "K-Nearest Neighbors": KNNModel(k=5),
        "Regresión Logística": LogisticModel(),
        "Árbol de decisión": DecisionTreeModel(),
        "Support Vector Machine (SVM)": SVMModel(),
    }

    for model in models.values():
        model.train(x_train, y_train)

    return models


# ============================================================
# CARGAR DATOS
# ============================================================

try:

    train_df = load_training_data()
    models = train_models(train_df)

except Exception as e:

    st.error(
        "No fue posible cargar o entrenar los modelos."
    )

    st.exception(e)

    st.stop()


# ============================================================
# FORMULARIO DE DATOS DEL CLIENTE
# ============================================================

st.header("Datos del cliente")

age = st.number_input(
    "Edad",
    min_value=18,
    max_value=100,
    value=30,
)

gender = st.selectbox(
    "Género",
    ["Female", "Male"],
)

tenure = st.number_input(
    "Antigüedad (Tenure)",
    min_value=0,
    max_value=100,
    value=30,
)

usage_frequency = st.number_input(
    "Frecuencia de uso",
    min_value=0,
    max_value=100,
    value=14,
)

support_calls = st.number_input(
    "Llamadas de soporte",
    min_value=0,
    max_value=100,
    value=5,
)

payment_delay = st.number_input(
    "Retraso en pagos",
    min_value=0,
    max_value=100,
    value=5,
)

subscription_type = st.selectbox(
    "Tipo de suscripción",
    ["Basic", "Standard", "Premium"],
)

contract_length = st.selectbox(
    "Duración del contrato",
    ["Monthly", "Quarterly", "Annual"],
)

total_spend = st.number_input(
    "Gasto total",
    min_value=0.0,
    max_value=10000.0,
    value=500.0,
)

last_interaction = st.number_input(
    "Última interacción",
    min_value=0,
    max_value=100,
    value=15,
)


# ============================================================
# SELECCIÓN DEL MODELO
# ============================================================

st.header("Modelo de clasificación")

model_name = st.selectbox(
    "Seleccione el modelo que desea utilizar:",
    list(models.keys()),
)


# ============================================================
# BOTÓN DE PREDICCIÓN
# ============================================================

if st.button("🔮 Predecir Churn", use_container_width=True):

    # Crear los datos del cliente
    customer_data = pd.DataFrame(
        [
            {
                "Age": age,
                "Gender": gender,
                "Tenure": tenure,
                "Usage Frequency": usage_frequency,
                "Support Calls": support_calls,
                "Payment Delay": payment_delay,
                "Subscription Type": subscription_type,
                "Contract Length": contract_length,
                "Total Spend": total_spend,
                "Last Interaction": last_interaction,
            }
        ]
    )

    # Aplicar las mismas transformaciones utilizadas
    # durante el entrenamiento
    customer_data["Gender"] = customer_data["Gender"].map(
        {
            "Female": 0,
            "Male": 1,
        }
    )

    customer_data["Subscription Type"] = customer_data[
        "Subscription Type"
    ].map(
        {
            "Basic": 0,
            "Standard": 1,
            "Premium": 2,
        }
    )

    customer_data["Contract Length"] = customer_data[
        "Contract Length"
    ].map(
        {
            "Monthly": 0,
            "Quarterly": 1,
            "Annual": 2,
        }
    )

    # Obtener el modelo seleccionado
    model = models[model_name]

    # Mostrar los datos que se están enviando al modelo
    st.subheader("Datos enviados al modelo")
    st.write(customer_data)

    # Realizar la predicción
    prediction = model.predict(customer_data)[0]

    st.write("Predicción interna:", prediction)

    # ========================================================
    # MOSTRAR RESULTADO
    # ========================================================

    st.header("Resultado")

    if prediction == 1:

        st.error(
            "⚠️ El modelo predice que el cliente ABANDONARÁ el servicio."
        )

    else:

        st.success(
            "✅ El modelo predice que el cliente PERMANECERÁ en el servicio."
        )

    st.write(f"**Modelo utilizado:** {model_name}")

    st.write(
        f"**Predicción:** {'Abandona (1)' if prediction == 1 else 'Permanece (0)'}"
    )