from src.interfaces.abstract import RegressionModel
from src.models.linear_regression_model import LinearRegressionModel
from src.models.polynomial_regression_model import PolynomialRegressionModel
from src.models.random_forest_model import RandomForestModel

# Grado elegido con validación cruzada (ver select_polynomial_degree en main.py).
POLYNOMIAL_DEGREE = 2

def get_models() -> list[RegressionModel]:
    """
    Lista única de modelos que usan main.py (comparación) y app.py (interfaz).
    Para agregar o cambiar un modelo basta con modificar esta lista.
    """

    return [
        LinearRegressionModel(),
        PolynomialRegressionModel(degree=POLYNOMIAL_DEGREE),
        RandomForestModel(),
    ]
