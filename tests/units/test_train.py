from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

from src.train import (
    load_data,
    train_model,
)


def build_model():
    X_train, _, y_train, _ = load_data()

    return train_model(
        X_train,
        y_train
    )


def test_model_training():

    model = build_model()

    assert model is not None


def test_model_type():

    model = build_model()

    assert isinstance(
        model,
        RandomForestClassifier
    )


def test_model_is_fitted():

    model = build_model()

    assert hasattr(
        model,
        "feature_importances_"
    )


def test_model_accuracy():

    X_train, X_test, y_train, y_test = load_data()

    model = build_model()

    score = model.score(
        X_test,
        y_test
    )

    assert score > 0.90