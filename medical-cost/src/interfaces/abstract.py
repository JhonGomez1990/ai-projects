from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class RegressionModel(ABC):
    """Contrato que deben cumplir todos los modelos de regresión."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Nombre del modelo."""
        pass

    @property
    @abstractmethod
    def estimator(self):
        """Estimador de scikit-learn (se usa para validación cruzada)."""
        pass

    @abstractmethod
    def train(self, x_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Entrena el modelo utilizando los datos de entrenamiento."""
        pass

    @abstractmethod
    def predict(self, x_test: pd.DataFrame) -> np.ndarray:
        """Realiza predicciones utilizando el modelo entrenado."""
        pass

    def evaluate(self, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """
        Evalúa el modelo y retorna sus métricas.

        - R2: proporción de la varianza del costo explicada por el modelo (1 = perfecto).
        - MAE: error absoluto promedio, en dólares.
        - RMSE: raíz del error cuadrático medio, en dólares; castiga más los errores grandes.
        - MAPE: error porcentual absoluto promedio.
        """

        predictions = self.predict(x_test)

        return {
            "r2": r2_score(y_test, predictions),
            "mae": mean_absolute_error(y_test, predictions),
            "rmse": np.sqrt(mean_squared_error(y_test, predictions)),
            "mape": np.mean(np.abs((y_test - predictions) / y_test)) * 100,
        }
