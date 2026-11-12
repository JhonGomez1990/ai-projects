"""Limpieza y transformación del dataset Bank Loan a un vector numérico en [0, 1]."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from sklearn.preprocessing import QuantileTransformer

TARGET = "Loan Status"
CLASSES = ["Fully Paid", "Charged Off"]  # índice 0 = paga, 1 = incumple

PURPOSE_GROUPS = {
    "Debt Consolidation": "purpose_debt",
    "Home Improvements": "purpose_home",
    "Business Loan": "purpose_business",
    "small_business": "purpose_business",
    "other": "purpose_other",
    "Other": "purpose_other",
}
HOME_GROUPS = {
    "Home Mortgage": "home_mortgage",
    "HaveMortgage": "home_mortgage",
    "Rent": "home_rent",
    "Own Home": "home_own",
}

FEATURES = [
    "loan_amount", "term_long", "credit_score", "annual_income", "years_job",
    "monthly_debt", "credit_history_years", "months_since_delinquent", "never_delinquent",
    "open_accounts", "credit_problems", "credit_balance", "max_open_credit",
    "bankruptcies", "tax_liens", "debt_to_income", "credit_utilization",
    "loan_to_income", "income_missing",
    "home_mortgage", "home_rent", "home_own",
    "purpose_debt", "purpose_home", "purpose_business", "purpose_other", "purpose_personal",
]

# Campos que el usuario ingresa en la aplicación (formato original del CSV).
RAW_NUMERIC = [
    "Current Loan Amount", "Credit Score", "Annual Income", "Monthly Debt",
    "Years of Credit History", "Months since last delinquent", "Number of Open Accounts",
    "Number of Credit Problems", "Current Credit Balance", "Maximum Open Credit",
    "Bankruptcies", "Tax Liens",
]
RAW_CATEGORICAL = ["Term", "Years in current job", "Home Ownership", "Purpose"]


def load_clean(path: str) -> pd.DataFrame:
    """Lee el CSV de entrenamiento, quita filas vacías, sin etiqueta y duplicadas."""
    df = pd.read_csv(path)
    df = df.dropna(subset=[TARGET])
    df = df.drop_duplicates()
    df = df.drop_duplicates(subset=["Loan ID"], keep="first")
    return df.reset_index(drop=True)


def _years_job(v) -> float:
    if pd.isna(v):
        return np.nan
    s = str(v)
    if s.startswith("<"):
        return 0.0
    return float("".join(ch for ch in s if ch.isdigit()) or "nan")


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte las columnas originales en las 27 variables numéricas de FEATURES."""
    out = pd.DataFrame(index=df.index)
    loan = df["Current Loan Amount"].astype(float).replace(99999999.0, np.nan)  # valor centinela
    score = df["Credit Score"].astype(float)
    score = score.where(score <= 850, score / 10)  # algunos puntajes vienen multiplicados por 10
    income = df["Annual Income"].astype(float)
    debt = df["Monthly Debt"].astype(float)
    balance = df["Current Credit Balance"].astype(float)
    max_credit = df["Maximum Open Credit"].astype(float)

    out["loan_amount"] = np.log1p(loan)
    out["term_long"] = (df["Term"] == "Long Term").astype(float)
    out["credit_score"] = score
    out["annual_income"] = np.log1p(income)
    out["years_job"] = df["Years in current job"].map(_years_job)
    out["monthly_debt"] = np.log1p(debt)
    out["credit_history_years"] = df["Years of Credit History"].astype(float)
    delinquent = df["Months since last delinquent"].astype(float)
    out["never_delinquent"] = delinquent.isna().astype(float)
    out["months_since_delinquent"] = delinquent.fillna(200.0)
    out["open_accounts"] = df["Number of Open Accounts"].astype(float)
    out["credit_problems"] = df["Number of Credit Problems"].astype(float)
    out["credit_balance"] = np.log1p(balance)
    out["max_open_credit"] = np.log1p(max_credit)
    out["bankruptcies"] = df["Bankruptcies"].astype(float)
    out["tax_liens"] = df["Tax Liens"].astype(float)
    out["debt_to_income"] = (12 * debt / income.replace(0, np.nan)).clip(0, 2)
    out["credit_utilization"] = (balance / max_credit.replace(0, np.nan)).clip(0, 2)
    out["loan_to_income"] = (loan / income.replace(0, np.nan)).clip(0, 3)
    out["income_missing"] = income.isna().astype(float)

    home = df["Home Ownership"].map(HOME_GROUPS)
    for col in ["home_mortgage", "home_rent", "home_own"]:
        out[col] = (home == col).astype(float)
    purpose = df["Purpose"].map(PURPOSE_GROUPS).fillna("purpose_personal")
    for col in ["purpose_debt", "purpose_home", "purpose_business", "purpose_other", "purpose_personal"]:
        out[col] = (purpose == col).astype(float)
    return out[FEATURES]


def labels(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == "Charged Off").astype(int).to_numpy()


class Preprocessor:
    """Imputación por mediana + QuantileTransformer -> cada variable en [0, 1].

    También guarda el orden de variables obtenido por clustering jerárquico de la
    correlación, usado por la representación de matriz de características.
    """

    def fit(self, feats: pd.DataFrame) -> "Preprocessor":
        self.medians_ = feats.median()
        filled = feats.fillna(self.medians_)
        self.qt_ = QuantileTransformer(n_quantiles=1000, output_distribution="uniform",
                                       subsample=200_000, random_state=42)
        self.qt_.fit(filled.to_numpy())
        corr = np.nan_to_num(np.corrcoef(self.qt_.transform(filled.to_numpy()).T))
        dist = np.clip(1 - np.abs(corr), 0, None)
        np.fill_diagonal(dist, 0)
        self.order_ = leaves_list(linkage(squareform(dist, checks=False), "average"))
        return self

    def transform(self, feats: pd.DataFrame) -> np.ndarray:
        x = self.qt_.transform(feats[FEATURES].fillna(self.medians_).to_numpy())
        return np.clip(x, 0, 1).astype(np.float32)
