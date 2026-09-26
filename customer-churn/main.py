from pathlib import Path
from src.service.data_exploration import explore_data, load_data
from src.service.data_preprocessing import preprocess_data
from src.service.result_service import save_result
from src.models.knn_model import KNNModel
from src.models.svm_model import SVMModel
from src.models.decision_tree_model import DecisionTreeModel
from src.models.logistic_model import LogisticModel
from src.use_case.model_runner import run_model

BASE_DIR = Path(__file__).resolve().parent

RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
RESULTS_DATA_DIR = BASE_DIR / "data" / "results"

TRAIN_FILE = (
    RAW_DATA_DIR
    / "customer_churn_dataset-training-master.csv"
)

TEST_FILE = (
    RAW_DATA_DIR
    / "customer_churn_dataset-testing-master.csv"
)

TRAIN_PROCESSED_FILE = (
    PROCESSED_DATA_DIR
    / "customer_churn_train_clean.csv"
)

TEST_PROCESSED_FILE = (
    PROCESSED_DATA_DIR
    / "customer_churn_test_clean.csv"
)

RESULTS_FILE = (
    RESULTS_DATA_DIR
    / "model_results.csv"
)

def main() -> None:
    
    # 1. Cargar datos
    train_df = load_data(TRAIN_FILE)
    test_df = load_data(TEST_FILE)

    # 2. Explorar datos
    explore_data(train_df, "TRAINING")

    explore_data(test_df, "TESTING")

    # 3. Limpiar y preparar
    train_clean = preprocess_data(train_df, TRAIN_PROCESSED_FILE)

    test_clean = preprocess_data(test_df, TEST_PROCESSED_FILE)

    # 4. Crear modelos
    models = [
        KNNModel(k=5),
        LogisticModel(),
        DecisionTreeModel(),
        SVMModel(),
    ]

    for model in models:

        # 5. Entrenar y evaluar
        accuracy = run_model(model, train_clean, test_clean)

        # 6. Mostrar resultado
        print("\n" + "=" * 50)
        print("RESULTADO DEL MODELO")
        print("=" * 50)

        print(f"\nModelo: {model.name}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Accuracy (%): {accuracy:.2%}")

        # 7. Guardar resultado
        save_result(model.name, accuracy, RESULTS_FILE)

if __name__ == "__main__":
    main()