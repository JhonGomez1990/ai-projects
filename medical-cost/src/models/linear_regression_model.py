# PROVISIONAL: implementación base para que la comparación y la interfaz funcionen.
# Jhon la reemplaza por su versión manteniendo la misma clase e interfaz (RegressionModel).

import pandas as pd
from sklearn.linear_model import LinearRegression
from src.interfaces.abstract import RegressionModel

class LinearRegressionModel(RegressionModel):
    """Regresión lineal múltiple."""

    def __init__(self):
        self.model = LinearRegression()

    @property
    def name(self) -> str:
        return "Regresión lineal"

    @property
    def estimator(self):
        return self.model

    def train(self, x_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Entrena la regresión lineal."""

        self.model.fit(x_train, y_train)

    def predict(self, x_test: pd.DataFrame):
        """Realiza predicciones con la regresión lineal."""

        return self.model.predict(x_test)
