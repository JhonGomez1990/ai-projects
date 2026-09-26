from pathlib import Path

import pandas as pd

CATEGORY_MAPPINGS = {
    "Gender": {
        "Female": 0,
        "Male": 1,
    },
    "Subscription Type": {
        "Basic": 0,
        "Standard": 1,
        "Premium": 2,
    },
    "Contract Length": {
        "Monthly": 0,
        "Quarterly": 1,
        "Annual": 2,
    },
}

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia y prepara el dataset para los modelos."""

    df = df.copy()

    # Eliminar filas con valores nulos
    df = df.dropna()

    # Eliminar registros duplicados
    df = df.drop_duplicates()

    # CustomerID identifica al cliente,
    # pero no aporta información para predecir Churn.
    df = df.drop(columns=["CustomerID"])

    # Convertir variables categóricas a valores numéricos.
    for column, mapping in CATEGORY_MAPPINGS.items():
        df[column] = df[column].map(mapping)

    # Churn debe ser una variable entera: 0 o 1.
    df["Churn"] = df["Churn"].astype(int)

    return df

def save_data(df: pd.DataFrame, output_path: str) -> None:
    """Guarda el dataset procesado."""

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        output_path,
        index=False,
    )

def preprocess_data(
    df: pd.DataFrame,
    output_path: str,
) -> pd.DataFrame:
    """Ejecuta el proceso completo de preparación."""

    clean_df = clean_data(df)

    save_data(
        clean_df,
        output_path,
    )

    return clean_df