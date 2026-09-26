import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from src.interfaces.abstract import RegressionModel

class PolynomialRegressionModel(RegressionModel):
    """
    Regresión polinómica.

    Agrega potencias y productos entre las variables (edad², bmi × smoker, ...)
    y luego ajusta una regresión lineal sobre ellas. Los productos entre variables
    le permiten capturar que el BMI solo encarece mucho el costo cuando la persona fuma.
    """

    def __init__(self, degree: int = 2):
        self.degree = degree

        # Se escala antes de elevar a potencias para que las variables
        # queden en rangos comparables y el ajuste sea numéricamente estable.
        self.model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
                ("regressor", LinearRegression()),
            ]
        )

    @property
    def name(self) -> str:
        return f"Regresión polinómica (grado {self.degree})"

    def train(self, x_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Genera las variables polinómicas y entrena la regresión."""

        self.model.fit(x_train, y_train)

    def predict(self, x_test: pd.DataFrame):
        """Realiza predicciones con el modelo polinómico."""

        return self.model.predict(x_test)
