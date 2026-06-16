import joblib
import mlflow
import pandas as pd
from functools import lru_cache
import os
from src.config import MLFLOW_TRACKING_URI
# model = None
LABELS = {
    0: "setosa",
    1: "versicolor",
    2: "virginica"
}

mlflow.set_tracking_uri(
    MLFLOW_TRACKING_URI
)
@lru_cache(maxsize=1)
def get_model():
    return mlflow.pyfunc.load_model(
        "models:/iris_classifier@production"
    )

# def load_model():

#     global model

#     if model is None:
#         model = joblib.load("models/model.pkl")

#     return model


def predict(features):

    df = pd.DataFrame(
        [features],
        columns=[
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width"
        ]
    )

    model = get_model()

    prediction = model.predict(df)

    prediction = prediction[0]

    prediction = int(prediction)

    return LABELS[prediction]