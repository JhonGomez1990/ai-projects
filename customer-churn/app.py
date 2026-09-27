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
    layout="wide",
)

st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(180deg, #f5f7ff 0%, #eef4ff 100%);
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .title-box {
        background: linear-gradient(135deg, #1f3a8a, #2563eb);
        border-radius: 18px;
        padding: 1.5rem 1.7rem;
        color: white;
        box-shadow: 0 12px 28px rgba(37, 99, 235, 0.18);
    }
    .section-card {
        background: rgba(255,255,255,0.85);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 18px;
        padding: 1rem 1.2rem;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.05);
    }
    .metric-card {
        background: linear-gradient(135deg, #ffffff, #f8fbff);
        border: 1px solid #dbeafe;
        border-radius: 16px;
        padding: 1rem;
        box-shadow: 0 6px 14px rgba(59,130,246,0.08);
    }
    div[data-testid="stForm"] {
        background: transparent;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

header_col, badge_col = st.columns([3, 1])

with header_col:
    st.markdown(
        """
        <div class="title-box">
            <h1 style="margin:0; font-size:2.2rem;">📊 Customer Churn Prediction</h1>
            <p style="margin:0.5rem 0 0; font-size:1rem; opacity:0.9;">
                Evalúa el riesgo de abandono de clientes con un panel de modelos predictivos.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with badge_col:
    st.markdown(
        """
        <div class="metric-card">
            <div style="font-size:0.8rem; color:#475569;">Estado</div>
            <div style="font-size:1.8rem; font-weight:700; color:#0f172a;">Live</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

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

    with st.container():
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Modelos", len(models.keys()))
        col_b.metric("Entrada", "10 campos")
        col_c.metric("Objetivo", "Churn")

except Exception as e:

    st.error(
        "No fue posible cargar o entrenar los modelos."
    )

    st.exception(e)

    st.stop()


# ============================================================
# FORMULARIO DE DATOS DEL CLIENTE
# ============================================================

st.markdown("<div class='section-card'><h3 style='margin-top:0;'>🧾 Datos del cliente</h3></div>", unsafe_allow_html=True)

with st.form("churn_prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Edad",
            min_value=18,
            max_value=100,
            value=None,
            placeholder="Ingrese la edad",
        )

        gender = st.selectbox(
            "Género",
            ["Seleccione...", "Female", "Male"],
        )

        tenure = st.number_input(
            "Antigüedad (Tenure)",
            min_value=0,
            max_value=100,
            value=None,
            placeholder="Ingrese la antigüedad",
        )

        usage_frequency = st.number_input(
            "Frecuencia de uso",
            min_value=0,
            max_value=100,
            value=None,
            placeholder="Ingrese la frecuencia de uso",
        )

        support_calls = st.number_input(
            "Llamadas de soporte",
            min_value=0,
            max_value=100,
            value=None,
            placeholder="Ingrese las llamadas de soporte",
        )

    with col2:
        payment_delay = st.number_input(
            "Retraso en pagos",
            min_value=0,
            max_value=100,
            value=None,
            placeholder="Ingrese el retraso en pagos",
        )

        subscription_type = st.selectbox(
            "Tipo de suscripción",
            ["Seleccione...", "Basic", "Standard", "Premium"],
        )

        contract_length = st.selectbox(
            "Duración del contrato",
            ["Seleccione...", "Monthly", "Quarterly", "Annual"],
        )

        total_spend = st.number_input(
            "Gasto total",
            min_value=0.0,
            max_value=10000.0,
            value=None,
            placeholder="Ingrese el gasto total",
        )

        last_interaction = st.number_input(
            "Última interacción",
            min_value=0,
            max_value=100,
            value=None,
            placeholder="Ingrese los días desde la última interacción",
        )

    st.markdown("---")

    model_name = st.selectbox(
        "Seleccione el modelo que desea utilizar:",
        list(models.keys()),
    )

    st.write("")
    submitted = st.form_submit_button("🔮 Predecir Churn", use_container_width=True)

    if submitted:
        if (
            age is None
            or tenure is None
            or usage_frequency is None
            or support_calls is None
            or payment_delay is None
            or total_spend is None
            or last_interaction is None
            or gender == "Seleccione..."
            or subscription_type == "Seleccione..."
            or contract_length == "Seleccione..."
        ):
            st.warning("⚠️ Por favor, complete todos los campos antes de realizar la predicción.")
        else:
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

            customer_data["Gender"] = customer_data["Gender"].map({"Female": 0, "Male": 1})
            customer_data["Subscription Type"] = customer_data["Subscription Type"].map({"Basic": 0, "Standard": 1, "Premium": 2})
            customer_data["Contract Length"] = customer_data["Contract Length"].map({"Monthly": 0, "Quarterly": 1, "Annual": 2})

            model = models[model_name]

            st.markdown("---")
            st.subheader("📊 Datos enviados al modelo")
            st.dataframe(customer_data, use_container_width=True)

            prediction = model.predict(customer_data)[0]

            st.markdown("---")
            st.subheader("📌 Resultado de la predicción")

            if prediction == 1:
                st.error("⚠️ El modelo predice que el cliente ABANDONARÁ el servicio.")
            else:
                st.success("✅ El modelo predice que el cliente PERMANECERÁ en el servicio.")

            st.write(f"**Modelo utilizado:** {model_name}")
            st.write(f"**Predicción:** {'Abandona (1)' if prediction == 1 else 'Permanece (0)'}")

            st.write("Predicción interna:", prediction)


