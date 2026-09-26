import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from src.interfaces.abstract import ClassificationModel

class KNNModel(ClassificationModel):
    """Implementación del modelo K-Nearest Neighbors."""

    def __init__(self, k: int = 5):
        self.scaler = StandardScaler()
        self.model = KNeighborsClassifier(n_neighbors=k)
        
    @property
    def name(self) -> str:
        return "K-Nearest Neighbors"

    def train(self, x_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Normaliza los datos y entrena el modelo KNN."""

        x_train_scaled = self.scaler.fit_transform(x_train)

        self.model.fit(x_train_scaled, y_train)

    def predict(self, x_test: pd.DataFrame):
        """Realiza predicciones utilizando KNN."""

        x_test_scaled = self.scaler.transform(x_test)

        return self.model.predict(x_test_scaled)

    def evaluate(self, x_test: pd.DataFrame, y_test: pd.Series) -> float:
        """Calcula el accuracy del modelo."""

        predictions = self.predict(x_test)

        return accuracy_score(y_test, predictions)