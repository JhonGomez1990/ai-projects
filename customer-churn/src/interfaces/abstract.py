from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class ClassificationModel(ABC):
    """Contrato que deben cumplir todos los modelos de clasificación."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Nombre del modelo."""
        pass
    
    @abstractmethod
    def train(self, x_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Entrena el modelo utilizando los datos de entrenamiento."""
        pass

    @abstractmethod
    def predict(self, x_test: pd.DataFrame) -> np.ndarray:
        """Realiza predicciones utilizando el modelo entrenado."""
        pass

    @abstractmethod
    def evaluate(self, x_test: pd.DataFrame, y_test: pd.Series) -> float:
        """Evalúa el modelo y retorna su accuracy."""
        pass