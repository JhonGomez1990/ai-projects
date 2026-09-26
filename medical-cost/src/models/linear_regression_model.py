import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from src.interfaces.abstract import RegressionModel

class LinearRegressionModel(RegressionModel):
    """Modelo de regresión lineal múltiple."""

    def __init__(self) -> None:
        self.model = LinearRegression()

    @property
    def name(self) -> str:
        return "Regresión Lineal"

    def train(self, x_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Entrena el modelo con los datos de entrenamiento."""
        self.model.fit(x_train, y_train)

    def predict(self, x_test: pd.DataFrame) -> np.ndarray:
        """Predice los costos médicos."""
        return self.model.predict(x_test)