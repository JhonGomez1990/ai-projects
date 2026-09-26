import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 
from sklearn.preprocessing import StandardScaler

from src.interfaces.abstract import ClassificationModel

class LogisticModel (ClassificationModel):
    """Implementación del modelo de Regresión Logística."""
    def __init__(self):
        self.scaler = StandardScaler()

        self.model = LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    @property
    def name (self) -> str:
        return "Regresion Logística"
    def train (
            self, 
            x_train: pd.DataFrame,
            y_train: pd.Series
    ) -> None:
        """Normaliza los datos y entrena el modelo."""
        x_train_scaled = self.scaler.fit_transform(x_train)

        self.model.fit(
            x_train_scaled,
            y_train
        )
    def predict(
        self, 
        x_test: pd.DataFrame
    ) -> np.ndarray:
        """Realiza predicicones utilizadno regresion logística."""
        x_test_scaled = self.scaler.transform(x_test)
        return self.model.predict(x_test_scaled)
    
    def evaluate(
        self,
        x_test: pd.DataFrame,
        y_test: pd.Series
    ) -> float:
        """Calcula el accuracy del modelo."""

        predictions = self.predict (x_test)
        return accuracy_score(
            y_test,
            predictions
        )
    def predict_proba(
        self,
        x_test: pd.DataFrame
    ):
        """Obtiene las probabilidades de cada clase"""
        x_test_scaled=self.scaler.transform(x_test)
        return self.model.predict_proba(
            x_test_scaled
        )