"""Aplicación Streamlit para evaluar nuevos clientes con el mejor modelo.

Ejecutar:  streamlit run app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

from src.models import SCHEMES, build_model
from src.preprocessing import CLASSES, FEATURES, engineer
from src.representations import REPRESENTATIONS, make_images

ROOT = Path(__file__).parent
MODELS = ROOT / "models"
CLASS_ES = {"Fully Paid": "Pagará el préstamo (Fully Paid)", "Charged Off": "Riesgo de impago (Charged Off)"}

st.set_page_config(page_title="Clasificación de clientes bancarios", page_icon="🏦", layout="wide")


@st.cache_resource
def load():
    info = json.loads((MODELS / "model_info.json").read_text(encoding="utf-8"))
    prep = joblib.load(MODELS / "preprocessor.joblib")
    model = build_model(info["scheme"], pretrained=False)
    model.load_state_dict(torch.load(MODELS / "best_model.pt", map_location="cpu"))
    model.eval()
    return info, prep, model


@st.cache_data
def test_clients():
    df = pd.read_csv(ROOT / "data" / "credit_test.csv").dropna(subset=["Loan ID"])
    return df.drop_duplicates(subset=["Loan ID"]).reset_index(drop=True)


info, prep, model = load()
order = np.array(info["order"])

st.title("🏦 Clasificación de clientes bancarios con Deep Learning")
st.caption(
    f"Modelo seleccionado: **{REPRESENTATIONS[info['representation']]}** + **{SCHEMES[info['scheme']]}** "
    f"· ROC-AUC prueba {info['test_metrics']['roc_auc']:.3f} · F1 macro {info['test_metrics']['f1_macro']:.3f}"
)

# ---------- Entrada de datos ----------
with st.sidebar:
    st.header("Cliente a evaluar")
    if st.button("🎲 Cargar cliente aleatorio de credit_test.csv"):
        st.session_state["client"] = test_clients().sample(1).iloc[0].to_dict()
    c = st.session_state.get("client", {})

    def num(label, key, default, step=1.0, fmt="%.0f"):
        v = c.get(key, default)
        v = default if pd.isna(v) else float(v)
        return st.number_input(label, value=v, step=step, format=fmt)

    def sel(label, key, options, default):
        v = c.get(key, default)
        return st.selectbox(label, options, index=options.index(v) if v in options else options.index(default))

    term = sel("Plazo (Term)", "Term", ["Short Term", "Long Term"], "Short Term")
    loan = num("Monto actual del préstamo", "Current Loan Amount", 300000.0, 1000.0)
    score = num("Puntaje de crédito (Credit Score)", "Credit Score", 720.0)
    income = num("Ingreso anual", "Annual Income", 1200000.0, 1000.0)
    years = sel("Años en el empleo actual", "Years in current job",
                ["< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years", "6 years", "7 years",
                 "8 years", "9 years", "10+ years"], "10+ years")
    home = sel("Vivienda", "Home Ownership", ["Home Mortgage", "Rent", "Own Home", "HaveMortgage"], "Home Mortgage")
    purpose = sel("Propósito", "Purpose",
                  ["Debt Consolidation", "Home Improvements", "Business Loan", "Buy a Car", "Medical Bills",
                   "Buy House", "Take a Trip", "major_purchase", "small_business", "moving", "wedding",
                   "vacation", "Educational Expenses", "renewable_energy", "other", "Other"],
                  "Debt Consolidation")
    debt = num("Deuda mensual", "Monthly Debt", 18000.0, 100.0, "%.2f")
    hist = num("Años de historial crediticio", "Years of Credit History", 18.0, 0.5, "%.1f")
    never = st.checkbox("Nunca ha estado en mora", value=pd.isna(c.get("Months since last delinquent", np.nan)))
    delinq = np.nan if never else num("Meses desde la última mora", "Months since last delinquent", 30.0)
    open_acc = num("Cuentas abiertas", "Number of Open Accounts", 11.0)
    problems = num("Problemas de crédito", "Number of Credit Problems", 0.0)
    balance = num("Saldo de crédito actual", "Current Credit Balance", 250000.0, 1000.0)
    max_credit = num("Crédito máximo abierto", "Maximum Open Credit", 600000.0, 1000.0)
    bankrupt = num("Bancarrotas", "Bankruptcies", 0.0)
    liens = num("Gravámenes fiscales (Tax Liens)", "Tax Liens", 0.0)

raw = pd.DataFrame([{
    "Current Loan Amount": loan, "Term": term, "Credit Score": score, "Annual Income": income,
    "Years in current job": years, "Home Ownership": home, "Purpose": purpose, "Monthly Debt": debt,
    "Years of Credit History": hist, "Months since last delinquent": delinq,
    "Number of Open Accounts": open_acc, "Number of Credit Problems": problems,
    "Current Credit Balance": balance, "Maximum Open Credit": max_credit, "Bankruptcies": bankrupt,
    "Tax Liens": liens,
}])

# ---------- Predicción ----------
x = torch.tensor(prep.transform(engineer(raw)))
img = make_images(x, info["representation"], order)
with torch.no_grad():
    probs = torch.softmax(model(img), 1)[0].numpy()
pred = int(probs.argmax())
label = CLASSES[pred]
conf = float(probs[pred])

col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("Representación visual del cliente")
    big = make_images(x, info["representation"], order, size=256)
    st.image(big[0].permute(1, 2, 0).numpy(), width=320, clamp=True,
             caption=f"{REPRESENTATIONS[info['representation']]} (entrada de la red, 32×32)")
    other = [r for r in REPRESENTATIONS if r != info["representation"]][0]
    with st.expander(f"Ver también: {REPRESENTATIONS[other]}"):
        st.image(make_images(x, other, order, size=256)[0].permute(1, 2, 0).numpy(), width=240, clamp=True)
with col2:
    st.subheader("Resultado")
    if pred == 1:
        st.error(f"### {CLASS_ES[label]}")
    else:
        st.success(f"### {CLASS_ES[label]}")
    st.metric("Nivel de confianza", f"{conf:.1%}")
    st.progress(conf)
    fig, ax = plt.subplots(figsize=(5, 1.8))
    ax.barh(["Fully Paid", "Charged Off"], probs, color=["#2e7d32", "#c62828"])
    ax.set_xlim(0, 1)
    for i, p in enumerate(probs):
        ax.text(min(p + 0.02, 0.9), i, f"{p:.1%}", va="center")
    ax.set_xlabel("Probabilidad")
    fig.tight_layout()
    st.pyplot(fig)
    st.caption("La confianza es la probabilidad softmax del modelo; no es una garantía de acierto.")

with st.expander("Variables transformadas (entrada al modelo, escala 0-1)"):
    st.dataframe(pd.DataFrame({"variable": FEATURES, "valor": x[0].numpy()}).round(3), use_container_width=True)
