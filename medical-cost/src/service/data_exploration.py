from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

CATEGORICAL_COLUMNS = ["sex", "smoker", "region"]
NUMERIC_COLUMNS = ["age", "bmi", "children"]
TARGET_COLUMN = "charges"

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

    print("\nTipos de datos:")
    print(df.dtypes)

    print("\nValores nulos:")
    print(df.isnull().sum())

    print("\nRegistros duplicados:")
    print(df.duplicated().sum())

    print("\nEstadísticas descriptivas:")
    print(df.describe())

    print("\nValores de variables categóricas:")

    for column in CATEGORICAL_COLUMNS:
        print(f"\n{column}:")
        print(df[column].value_counts(dropna=False))

    print("\nCosto promedio por categoría:")

    for column in CATEGORICAL_COLUMNS:
        print(f"\n{column}:")
        print(df.groupby(column)[TARGET_COLUMN].mean().round(2))

    print("\nCorrelación de variables numéricas con el costo:")
    print(df[NUMERIC_COLUMNS + [TARGET_COLUMN]].corr()[TARGET_COLUMN].round(3))

def save_exploration_figures(df: pd.DataFrame, output_dir: Path) -> None:
    """Genera las gráficas de exploración que se usan en el informe."""

    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # Distribución del costo: está sesgada a la derecha (pocos costos muy altos).
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df[TARGET_COLUMN], bins=40, kde=True, ax=ax)
    ax.set_title("Distribución del costo médico")
    ax.set_xlabel("Costo (USD)")
    ax.set_ylabel("Personas")
    fig.tight_layout()
    fig.savefig(output_dir / "01_distribucion_costo.png", dpi=120)
    plt.close(fig)

    # Costo por variable categórica.
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, column in zip(axes, CATEGORICAL_COLUMNS):
        sns.boxplot(data=df, x=column, y=TARGET_COLUMN, ax=ax)
        ax.set_title(f"Costo según {column}")
        ax.set_ylabel("Costo (USD)")

    fig.tight_layout()
    fig.savefig(output_dir / "02_costo_por_categoria.png", dpi=120)
    plt.close(fig)

    # Edad y BMI contra el costo, separando fumadores: muestra la interacción
    # que el modelo polinómico logra capturar.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.scatterplot(data=df, x="age", y=TARGET_COLUMN, hue="smoker", alpha=0.6, ax=axes[0])
    axes[0].set_title("Costo vs. edad")
    axes[0].set_ylabel("Costo (USD)")

    sns.scatterplot(data=df, x="bmi", y=TARGET_COLUMN, hue="smoker", alpha=0.6, ax=axes[1])
    axes[1].axvline(30, color="gray", linestyle="--", linewidth=1)
    axes[1].set_title("Costo vs. BMI (línea: BMI = 30, obesidad)")
    axes[1].set_ylabel("Costo (USD)")

    fig.tight_layout()
    fig.savefig(output_dir / "03_costo_vs_edad_bmi.png", dpi=120)
    plt.close(fig)

    # Matriz de correlación con las variables ya codificadas.
    encoded = df.copy()
    encoded["sex"] = encoded["sex"].map({"female": 0, "male": 1})
    encoded["smoker"] = encoded["smoker"].map({"no": 0, "yes": 1})
    encoded = encoded.drop(columns=["region"])

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(encoded.corr(), annot=True, fmt=".2f", cmap="Blues", ax=ax)
    ax.set_title("Matriz de correlación")
    fig.tight_layout()
    fig.savefig(output_dir / "04_matriz_correlacion.png", dpi=120)
    plt.close(fig)
