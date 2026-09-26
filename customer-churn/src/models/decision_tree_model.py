import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from src.interfaces.abstract import ClassificationModel

class DecisionTreeModel (ClassificationModel):
    """Implementación del modelo árbol de decisión"""
    def __init__(self):
        self.model=DecisionTreeClassifier(
            max_depth=5,
            random_state=42
        )
    @property
    def name (self) -> str:
        return "Árbol de decisión"

    def train (
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series
    ) -> None:
        """Entrena el arbol de decisión"""
        self.model.fit(
            x_train,
            y_train
        )
    def predict(
        self, 
        x_test: pd.DataFrame
    )-> np.ndarray:
        """Realiza predicciones"""
        return  self.model.predict(
            x_test
        )
    def evaluate(
            self, 
            x_test: pd.DataFrame,
            y_test: pd.Series
        )-> float:
        """Calcula el accuracy del modelo."""
        predictions = self.predict(x_test)
        return accuracy_score(
            y_test,
            predictions
        )
    def predict_proba(
        self,
        x_test: pd.DataFrame
    ):
        """Obtiene las probabilidades de cada clase."""
        return self.model.predict_proba(
            x_test
        )

