import pandas as pd
from src.interfaces.abstract import ClassificationModel

def run_model(model: ClassificationModel, train_df: pd.DataFrame, test_df: pd.DataFrame) -> float:
    """Entrena y evalúa cualquier modelo que implemente ClassificationModel."""

    x_train = train_df.drop(columns=["Churn"])

    y_train = train_df["Churn"]

    x_test = test_df.drop(columns=["Churn"])

    y_test = test_df["Churn"]

    model.train(x_train, y_train,)

    return model.evaluate(x_test, y_test)