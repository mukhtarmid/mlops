import os

import mlflow
from mlflow import MlflowClient


MODEL_NAME = "iris_classifier"


def test_model_registration():

    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://localhost:5000"
    )

    mlflow.set_tracking_uri(
        tracking_uri
    )

    client = MlflowClient()

    model = client.get_registered_model(
        MODEL_NAME
    )

    assert model.name == MODEL_NAME