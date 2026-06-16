import os

import mlflow
import pandas as pd


MODEL_NAME = "iris_classifier"
MODEL_ALIAS = "production"


def test_model_loading():

    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://localhost:5000"
    )

    mlflow.set_tracking_uri(
        tracking_uri
    )

    model = mlflow.pyfunc.load_model(
        f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    )

    sample = pd.DataFrame(
        [[5.1, 3.5, 1.4, 0.2]],
        columns=[
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
        ]
    )

    prediction = model.predict(
        sample
    )

    assert len(prediction) == 1