import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold, cross_val_score
from src.interfaces.abstract import RegressionModel
from src.models.polynomial_regression_model import PolynomialRegressionModel

CV_FOLDS = 5
RANDOM_STATE = 42

def cross_validate_r2(model: RegressionModel, x_train: pd.DataFrame, y_train: pd.Series) -> pd.Series:
    """R2 en validación cruzada de 5 particiones sobre el conjunto de entrenamiento."""

    folds = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    scores = cross_val_score(clone(model.model), x_train, y_train, cv=folds, scoring="r2")

    return pd.Series(scores)

def run_model(
    model: RegressionModel,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Entrena y evalúa cualquier modelo que implemente RegressionModel."""

    cv_scores = cross_validate_r2(model, x_train, y_train)

    model.train(x_train, y_train)

    train_metrics = model.evaluate(x_train, y_train)
    test_metrics = model.evaluate(x_test, y_test)

    return {
        "model": model.name,
        "r2_train": train_metrics["r2"],
        "r2_test": test_metrics["r2"],
        "r2_cv_mean": cv_scores.mean(),
        "r2_cv_std": cv_scores.std(),
        "mae_test": test_metrics["mae"],
        "rmse_test": test_metrics["rmse"],
        "mape_test": test_metrics["mape"],
    }

def select_polynomial_degree(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    degrees: range = range(1, 5),
) -> pd.DataFrame:
    """
    Compara grados del polinomio con validación cruzada.
    Grados altos se ajustan mejor al entrenamiento pero generalizan peor (sobreajuste).
    """

    rows = []

    for degree in degrees:
        model = PolynomialRegressionModel(degree=degree)

        cv_scores = cross_validate_r2(model, x_train, y_train)

        model.train(x_train, y_train)

        rows.append(
            {
                "degree": degree,
                "r2_train": model.evaluate(x_train, y_train)["r2"],
                "r2_cv_mean": cv_scores.mean(),
                "r2_cv_std": cv_scores.std(),
            }
        )

    return pd.DataFrame(rows)
