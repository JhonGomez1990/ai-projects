from pathlib import Path
import pandas as pd
from src.models.registry import get_models
from src.service.data_exploration import explore_data, load_data, save_exploration_figures
from src.service.data_preprocessing import preprocess_data, split_features_target
from src.service.result_service import print_comparison, save_comparison_figures, save_results
from src.use_case.model_runner import run_model, select_polynomial_degree

BASE_DIR = Path(__file__).resolve().parent

RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
RESULTS_DATA_DIR = BASE_DIR / "data" / "results"
FIGURES_DIR = BASE_DIR / "reports" / "figures"

RAW_FILE = RAW_DATA_DIR / "insurance.csv"

TRAIN_PROCESSED_FILE = PROCESSED_DATA_DIR / "insurance_train_clean.csv"
TEST_PROCESSED_FILE = PROCESSED_DATA_DIR / "insurance_test_clean.csv"

RESULTS_FILE = RESULTS_DATA_DIR / "model_results.csv"
DEGREE_RESULTS_FILE = RESULTS_DATA_DIR / "polynomial_degree_selection.csv"

def main() -> None:

    # 1. Cargar datos
    df = load_data(RAW_FILE)

    # 2. Explorar datos
    explore_data(df, "MEDICAL COST")
    save_exploration_figures(df, FIGURES_DIR)

    # 3. Limpiar, preparar y dividir en entrenamiento / prueba
    train_df, test_df = preprocess_data(df, TRAIN_PROCESSED_FILE, TEST_PROCESSED_FILE)

    x_train, y_train = split_features_target(train_df)
    x_test, y_test = split_features_target(test_df)

    print(f"\nEntrenamiento: {len(train_df)} registros | Prueba: {len(test_df)} registros")

    # 4. Elegir el grado del polinomio con validación cruzada
    degree_results = select_polynomial_degree(x_train, y_train)

    print("\n" + "=" * 50)
    print("SELECCIÓN DEL GRADO DEL POLINOMIO")
    print("=" * 50)
    print(degree_results.round(4).to_string(index=False))

    # 5. Entrenar y evaluar cada modelo
    models = get_models()

    results = pd.DataFrame(
        [run_model(model, x_train, y_train, x_test, y_test) for model in models]
    )

    # 6. Comparar resultados
    print_comparison(results)

    # 7. Guardar resultados y gráficas
    save_results(results, RESULTS_FILE)
    save_results(degree_results, DEGREE_RESULTS_FILE)
    save_comparison_figures(results, models, x_test, y_test, degree_results, FIGURES_DIR)

    print(f"\nResultados guardados en: {RESULTS_DATA_DIR}")
    print(f"Gráficas guardadas en: {FIGURES_DIR}")

if __name__ == "__main__":
    main()
