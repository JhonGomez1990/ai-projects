import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.interfaces.abstract import RegressionModel


class RandomForestModel(RegressionModel):
    """
    Modelo de Machine Learning basado en Random Forest.

    Random Forest combina múltiples árboles de decisión
    para realizar una predicción del costo médico.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        min_samples_leaf: int = 5,
        random_state: int = 42,
    ):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1,
        )

    @property
    def name(self) -> str:
        return "Random Forest"

    def train(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> None:
        """
        Entrena el modelo utilizando los datos de entrenamiento.
        """

        self.model.fit(x_train, y_train)

    def predict(
        self,
        x_test: pd.DataFrame,
    ):
        """
        Predice el costo médico para los datos de entrada.
        """

        return self.model.predict(x_test)