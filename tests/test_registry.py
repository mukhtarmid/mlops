import mlflow


def test_model_loads():

    model = mlflow.pyfunc.load_model(
        "models:/iris_classifier@production"
    )

    assert model is not None