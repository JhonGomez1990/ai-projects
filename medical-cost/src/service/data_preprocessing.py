from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

TARGET_COLUMN = "charges"

BINARY_MAPPINGS = {
    "sex": {
        "female": 0,
        "male": 1,
    },
    "smoker": {
        "no": 0,
        "yes": 1,
    },
}

# region es nominal (no tiene orden), por eso se codifica con one-hot
# en lugar de asignarle números 0, 1, 2, 3.
REGIONS = ["northeast", "northwest", "southeast", "southwest"]

# Orden fijo de columnas que reciben todos los modelos.
# northeast queda como categoría base (drop_first) para evitar multicolinealidad
# en la regresión lineal.
FEATURE_COLUMNS = [
    "age",
    "sex",
    "bmi",
    "children",
    "smoker",
    "region_northwest",
    "region_southeast",
    "region_southwest",
]

TEST_SIZE = 0.2
RANDOM_STATE = 42

def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte las variables categóricas a numéricas."""

    df = df.copy()

    for column, mapping in BINARY_MAPPINGS.items():
        df[column] = df[column].map(mapping)

    # Se fija la lista de categorías para que una sola fila (la de la interfaz)
    # genere exactamente las mismas columnas que el dataset completo.
    df["region"] = pd.Categorical(df["region"], categories=REGIONS)
    df = pd.get_dummies(df, columns=["region"], drop_first=True, dtype=int)

    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia y prepara el dataset para los modelos."""

    df = df.copy()

    # El dataset no tiene nulos, pero se valida por si cambia la fuente.
    df = df.dropna()

    # Hay un registro duplicado exacto; se elimina para que no quede
    # a la vez en entrenamiento y prueba.
    df = df.drop_duplicates()

    df = encode_features(df)

    return df[FEATURE_COLUMNS + [TARGET_COLUMN]]

def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separa el dataset en entrenamiento (80 %) y prueba (20 %).

    Se estratifica por smoker porque es la variable que más influye en el costo:
    así ambos conjuntos tienen la misma proporción de fumadores.
    """

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["smoker"],
    )

    return train_df, test_df

def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa las variables de entrada (X) de la variable objetivo (y)."""

    return df[FEATURE_COLUMNS], df[TARGET_COLUMN]

def build_input(age: int, sex: str, bmi: float, children: int, smoker: str, region: str) -> pd.DataFrame:
    """Construye una fila con el mismo formato de entrada de los modelos (la usa la interfaz)."""

    row = pd.DataFrame(
        [
            {
                "age": age,
                "sex": sex,
                "bmi": bmi,
                "children": children,
                "smoker": smoker,
                "region": region,
            }
        ]
    )

    return encode_features(row)[FEATURE_COLUMNS]

def save_data(df: pd.DataFrame, output_path: Path) -> None:
    """Guarda el dataset procesado."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

def preprocess_data(
    df: pd.DataFrame,
    train_output_path: Path,
    test_output_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ejecuta el proceso completo de preparación y guarda los conjuntos resultantes."""

    clean_df = clean_data(df)

    train_df, test_df = split_data(clean_df)

    save_data(train_df, train_output_path)
    save_data(test_df, test_output_path)

    return train_df, test_df
