# PROVISIONAL: implementación base para que la comparación y la interfaz funcionen.
# Laura la reemplaza por su modelo de Machine Learning manteniendo la interfaz RegressionModel
# (puede cambiar el algoritmo; solo hay que actualizar el import en main.py y en app.py).

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from src.interfaces.abstract import RegressionModel

class RandomForestModel(RegressionModel):
    """Random Forest: promedio de muchos árboles de decisión."""

    def __init__(self, n_estimators: int = 300, min_samples_leaf: int = 5, random_state: int = 42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1,
        )

    @property
    def name(self) -> str:
        return "Random Forest"

    @property
    def estimator(self):
        return self.model

    def train(self, x_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Entrena el bosque aleatorio."""

        self.model.fit(x_train, y_train)

    def predict(self, x_test: pd.DataFrame):
        """Realiza predicciones promediando los árboles."""

        return self.model.predict(x_test)
