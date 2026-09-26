import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from src.interfaces.abstract import ClassificationModel

class SVMModel(ClassificationModel):
    """
    Implementación del modelo Support Vector Machine (SVM).

    Busca la frontera que separa a los clientes que abandonan de los que se quedan
    dejando el mayor margen posible entre ambos grupos. El kernel RBF permite que
    esa frontera sea curva.
    """

    def __init__(
        self,
        kernel: str = "rbf",
        c: float = 1.0,
        sample_size: int = 30000,
        random_state: int = 42,
    ):
        # El tiempo de entrenamiento de SVC crece aproximadamente con el cuadrado
        # del número de registros: con los ~440.000 del dataset tardaría horas.
        # Se entrena con una muestra estratificada, que conserva la proporción de Churn.
        self.sample_size = sample_size
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.model = SVC(kernel=kernel, C=c)

    @property
    def name(self) -> str:
        return "Support Vector Machine (SVM)"

    def train(self, x_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Toma una muestra, normaliza los datos y entrena el modelo SVM."""

        if len(x_train) > self.sample_size:
            x_train, _, y_train, _ = train_test_split(
                x_train,
                y_train,
                train_size=self.sample_size,
                stratify=y_train,
                random_state=self.random_state,
            )

        # SVM depende de distancias, por eso necesita variables en la misma escala.
        x_train_scaled = self.scaler.fit_transform(x_train)

        self.model.fit(x_train_scaled, y_train)

    def predict(self, x_test: pd.DataFrame):
        """Realiza predicciones utilizando SVM."""

        x_test_scaled = self.scaler.transform(x_test)

        return self.model.predict(x_test_scaled)

    def evaluate(self, x_test: pd.DataFrame, y_test: pd.Series) -> float:
        """Calcula el accuracy del modelo."""

        predictions = self.predict(x_test)

        return accuracy_score(y_test, predictions)
