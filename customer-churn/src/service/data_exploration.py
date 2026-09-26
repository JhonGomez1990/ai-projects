import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    """Carga un archivo CSV y retorna un DataFrame."""
    return pd.read_csv(file_path)

def explore_data(df: pd.DataFrame, dataset_name: str) -> None:
    """Muestra información básica para explorar un dataset."""

    print(f"\n{'=' * 50}")
    print(f"EXPLORACIÓN: {dataset_name}")
    print("=" * 50)

    print(f"\nFilas: {df.shape[0]}")
    print(f"Columnas: {df.shape[1]}")

    print("\nPrimeros 5 registros:")
    print(df.head())

    print("\nColumnas:")
    print(df.columns.tolist())

    print("\nTipos de datos:")
    print(df.dtypes)

    print("\nValores nulos:")
    print(df.isnull().sum())

    print("\nRegistros duplicados:")
    print(df.duplicated().sum())

    print("\nDistribución de Churn:")
    print(df["Churn"].value_counts(dropna=False))

    print("\nValores de variables categóricas:")

    categorical_columns = [
        "Gender",
        "Subscription Type",
        "Contract Length",
    ]

    for column in categorical_columns:
        print(f"\n{column}:")
        print(df[column].value_counts(dropna=False))