import os

import mlflow


def test_mlflow_connection():

    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://localhost:5000"
    )

    mlflow.set_tracking_uri(
        tracking_uri
    )

    experiments = mlflow.search_experiments()

    assert experiments is not None